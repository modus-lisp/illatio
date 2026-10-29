// qwen35.metal -- the kernels for a Qwen3.5 (qwen35) token on the GPU.
//
// Each is one op of src/qwen35.lisp's forward pass, on f32 activations.  The
// Q6_K matrix-vector product and row fetch read the GGUF's own blocks; they
// follow ggml's Metal kernels (kernel_mul_mv_q6_K_f32, dequantize_q6_K; ggml
// is MIT, Copyright (c) 2023-2026 The ggml authors).  The rest are written
// from the Lisp: the arithmetic is the same, but sums are reduced in a
// different order, so the GPU is compared with llama.cpp and with the CPU
// statistically, as the CPU is with llama.cpp (inspect/vectors-test.lisp).

#include <metal_stdlib>
using namespace metal;

#define QK_K 256

typedef struct {
    uint8_t ql[QK_K/2];      // low 4 bits
    uint8_t qh[QK_K/4];      // high 2 bits
    int8_t  scales[QK_K/16]; // one per 16 weights
    half    d;
} block_q6_K;

// The sum (or max) of V over a threadgroup of NTG threads, every thread
// getting it.  SH holds 32 floats.
static inline float tg_sum(float v, threadgroup float * sh, uint tid, uint ntg) {
    v = simd_sum(v);
    const uint lane = tid % 32, sg = tid / 32, nsg = (ntg + 31) / 32;
    if (lane == 0) sh[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? sh[lane] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

static inline float tg_max(float v, threadgroup float * sh, uint tid, uint ntg) {
    v = simd_max(v);
    const uint lane = tid % 32, sg = tid / 32, nsg = (ntg + 31) / 32;
    if (lane == 0) sh[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? sh[lane] : -INFINITY;
    r = simd_max(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

static inline float silu(float x)    { return x / (1.0f + exp(-x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// --- Q6_K ------------------------------------------------------------------------

struct mv_args { uint ncols; uint nrows; uint row_bytes; uint x_stride; uint y_stride; };

// Y = W X for each column of X (grid y): two simdgroups a threadgroup, two
// rows each, as ggml's kernel_mul_mv_q6_K_f32.
#define MV_NR0 2
#define MV_NSG 2
kernel void mv_q6k(constant mv_args & a [[buffer(0)]],
                   device const char  * w [[buffer(1)]],
                   device const float * x [[buffer(2)]],
                   device       float * y [[buffer(3)]],
                   uint3  tgpig [[threadgroup_position_in_grid]],
                   ushort tiisg [[thread_index_in_simdgroup]],
                   ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint8_t kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
    const int nb = a.ncols / QK_K;
    const int first_row = (tgpig.x * MV_NSG + sgitg) * MV_NR0;
    const uint r1 = tgpig.y;

    device const block_q6_K * xb = (device const block_q6_K *) (w + (ulong)first_row * a.row_bytes);
    device const float * yy = x + (ulong)r1 * a.x_stride;

    float sumf[MV_NR0] = { 0.f };
    float yl[16];

    const short tid = tiisg / 2;
    const short ix  = tiisg % 2;
    const short ip  = tid / 8;
    const short il  = tid % 8;
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;

    const short y_offset   = 128 * ip + l0;
    const short q_offset_l =  64 * ip + l0;
    const short q_offset_h =  32 * ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = xb[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = xb[i].qh + q_offset_h;
        device const int8_t  * sc = xb[i].scales + is;
        device const half    * dh = &xb[i].d;

        device const float * yv = yy + i * QK_K + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = yv[l +  0];
            yl[4*l + 1] = yv[l + 32];
            yl[4*l + 2] = yv[l + 64];
            yl[4*l + 3] = yv[l + 96];
        }

        for (short row = 0; row < MV_NR0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
            q1 += a.row_bytes; q2 += a.row_bytes; qh += a.row_bytes; sc += a.row_bytes;
            dh += a.row_bytes / 2;
        }
    }

    device float * dst = y + (ulong)r1 * a.y_stride;
    for (int row = 0; row < MV_NR0 && first_row + row < (int)a.nrows; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0) dst[first_row + row] = s;
    }
}

struct row_args { uint ncols; uint row_bytes; uint row; };

// Y = row ROW of W, dequantized (the embedding), one thread a weight.
kernel void row_q6k(constant row_args & a [[buffer(0)]],
                    device const char * w [[buffer(1)]],
                    device float * y [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= a.ncols) return;
    device const block_q6_K * b = (device const block_q6_K *) (w + (ulong)a.row * a.row_bytes) + i / QK_K;
    const uint j = i % QK_K, n = j / 128, r = j % 128, quarter = r / 32, l = r % 32;
    device const uint8_t * ql = b->ql + 64 * n;
    device const uint8_t * qh = b->qh + 32 * n;
    device const int8_t  * sc = b->scales + 8 * n;
    int q;
    switch (quarter) {
        case 0:  q = (ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4); break;
        case 1:  q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4); break;
        case 2:  q = (ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4); break;
        default: q = (ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4); break;
    }
    y[i] = (float)b->d * sc[l / 16 + 2 * quarter] * (float)(q - 32);
}

// --- norms and elementwise ops ---------------------------------------------------

struct norm_args { uint n; float eps; uint w_stride; };

// Y = rmsnorm(X) * W per row of N (a threadgroup a row); W moves on by
// W_STRIDE a row (0: one weight vector for all).
kernel void rmsnorm(constant norm_args & a [[buffer(0)]],
                    device const float * x [[buffer(1)]],
                    device const float * w [[buffer(2)]],
                    device       float * y [[buffer(3)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint ntg [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    device const float * xr = x + (ulong)tg * a.n;
    device const float * wr = w + (ulong)tg * a.w_stride;
    device       float * yr = y + (ulong)tg * a.n;
    float s = 0.f;
    for (uint i = tid; i < a.n; i += ntg) s += xr[i] * xr[i];
    s = tg_sum(s, sh, tid, ntg);
    const float scale = 1.0f / sqrt(s / a.n + a.eps);
    for (uint i = tid; i < a.n; i += ntg) yr[i] = (xr[i] * scale) * wr[i];
}

struct n_args { uint n; };

kernel void add(constant n_args & a [[buffer(0)]],
                device float * x [[buffer(1)]], device const float * y [[buffer(2)]],
                uint i [[thread_position_in_grid]]) {
    if (i < a.n) x[i] += y[i];
}

kernel void swiglu(constant n_args & a [[buffer(0)]],
                   device float * g [[buffer(1)]], device const float * u [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < a.n) g[i] = silu(g[i]) * u[i];
}

kernel void gate_mul(constant n_args & a [[buffer(0)]],
                     device float * x [[buffer(1)]], device const float * gate [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < a.n) x[i] *= sigmoid(gate[i]);
}

// --- the gated delta net (a linear layer) -----------------------------------------

struct gates_args { uint vh; };

// BETA = sigmoid(BETA); ALPHA = softplus(ALPHA + dt) * A, per V head.
kernel void q35_gates(constant gates_args & a [[buffer(0)]],
                      device float * alpha [[buffer(1)]], device float * beta [[buffer(2)]],
                      device const float * A [[buffer(3)]], device const float * dt [[buffer(4)]],
                      uint j [[thread_position_in_grid]]) {
    if (j >= a.vh) return;
    beta[j] = sigmoid(beta[j]);
    const float s = alpha[j] + dt[j];
    alpha[j] = (s > 20.0f ? s : log(1.0f + exp(s))) * A[j];
}

struct conv_args { uint ch; };

// C = silu(causal conv of each channel's last 4 inputs); STATE moves on.
kernel void q35_conv(constant conv_args & a [[buffer(0)]],
                     device const float * qkv [[buffer(1)]], device const float * kern [[buffer(2)]],
                     device float * state [[buffer(3)]], device float * c [[buffer(4)]],
                     uint ci [[thread_position_in_grid]]) {
    if (ci >= a.ch) return;
    device float * st = state + ci * 3;
    device const float * k = kern + ci * 4;
    const float x0 = st[0], x1 = st[1], x2 = st[2], x3 = qkv[ci];
    float s = 0.f;
    s += x0 * k[0]; s += x1 * k[1]; s += x2 * k[2]; s += x3 * k[3];
    c[ci] = silu(s);
    st[0] = x1; st[1] = x2; st[2] = x3;
}

struct l2_args { uint n; float eps; };

// X /= max(|X|, eps) per row of N, in place (a threadgroup a row).
kernel void l2norm(constant l2_args & a [[buffer(0)]],
                   device float * x [[buffer(1)]],
                   uint tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_position_in_threadgroup]],
                   uint ntg [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    device float * xr = x + (ulong)tg * a.n;
    float s = 0.f;
    for (uint i = tid; i < a.n; i += ntg) s += xr[i] * xr[i];
    s = tg_sum(s, sh, tid, ntg);
    const float scale = 1.0f / max(sqrt(s), a.eps);
    for (uint i = tid; i < a.n; i += ntg) xr[i] *= scale;
}

struct delta_args { uint kh; uint kd; uint vh; uint vd; float scale; };

// The delta rule for one V head (a threadgroup), one value v a thread.  S is
// k-major, S[k][v] at h*kd*vd + k*vd + v, as on the CPU, so a k's values
// are adjacent and the threads read them together.
kernel void q35_delta(constant delta_args & a [[buffer(0)]],
                      device const float * c [[buffer(1)]],
                      device const float * beta [[buffer(2)]],
                      device const float * g [[buffer(3)]],
                      device float * S [[buffer(4)]],
                      device float * o [[buffer(5)]],
                      uint h [[threadgroup_position_in_grid]],
                      uint v [[thread_position_in_threadgroup]],
                      uint ntg [[threads_per_threadgroup]]) {
    threadgroup float kv[256];
    threadgroup float qv[256];
    const uint kq = (h % a.kh) * a.kd;                      // K heads tile over V heads
    const uint kk = a.kh * a.kd + (h % a.kh) * a.kd;
    const uint vv = 2 * a.kh * a.kd + h * a.vd;
    for (uint t = v; t < a.kd; t += ntg) { kv[t] = c[kk + t]; qv[t] = c[kq + t] * a.scale; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (v >= a.vd) return;
    const float decay = exp(g[h]), b = beta[h];
    device float * Sh = S + (ulong)h * a.kd * a.vd + v;
    float sk = 0.f;
    for (uint k = 0; k < a.kd; ++k) {
        const float s = Sh[k * a.vd] * decay;
        Sh[k * a.vd] = s;
        sk += s * kv[k];
    }
    const float d = (c[vv + v] - sk) * b;
    float ov = 0.f;
    for (uint k = 0; k < a.kd; ++k) {
        const float s = Sh[k * a.vd] + kv[k] * d;
        Sh[k * a.vd] = s;
        ov += s * qv[k];
    }
    o[h * a.vd + v] = ov;
}

struct outnorm_args { uint vd; float eps; };

// O = rmsnorm(O) * ssm_norm * silu(Z), per V head (a threadgroup).
kernel void q35_outnorm(constant outnorm_args & a [[buffer(0)]],
                        device float * o [[buffer(1)]], device const float * w [[buffer(2)]],
                        device const float * z [[buffer(3)]],
                        uint h [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint ntg [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    device float * oh = o + h * a.vd;
    device const float * zh = z + h * a.vd;
    float s = 0.f;
    for (uint i = tid; i < a.vd; i += ntg) s += oh[i] * oh[i];
    s = tg_sum(s, sh, tid, ntg);
    const float scale = 1.0f / sqrt(s / a.vd + a.eps);
    for (uint i = tid; i < a.vd; i += ntg) oh[i] = ((oh[i] * scale) * w[i]) * silu(zh[i]);
}

// --- a full (gated attention) layer -----------------------------------------------

struct qk_args { uint nh; uint nkv; uint hd; uint rope_dims; float eps; uint pos; };

// Per head (a threadgroup, a thread a dimension): Q's heads come out of QG
// ([q | gate] per head), K's from K; each is normed, then its first
// ROPE_DIMS rotated as NEOX pairs by the table ROPE ([cos | sin]); K and V go
// to the caches at POS.
kernel void q35_qk(constant qk_args & a [[buffer(0)]],
                   device const float * qg [[buffer(1)]], device const float * kin [[buffer(2)]],
                   device const float * vin [[buffer(3)]],
                   device const float * qn [[buffer(4)]], device const float * kn [[buffer(5)]],
                   device const float * rope [[buffer(6)]],
                   device float * q [[buffer(7)]], device float * gate [[buffer(8)]],
                   device float * kc [[buffer(9)]], device float * vc [[buffer(10)]],
                   uint u [[threadgroup_position_in_grid]],
                   uint t [[thread_position_in_threadgroup]],
                   uint ntg [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    threadgroup float xs[512];
    const uint hd = a.hd, kvd = a.nkv * hd;
    const bool isq = u < a.nh;
    const uint j = isq ? u : u - a.nh;
    device const float * src = isq ? qg + j * 2 * hd : kin + j * hd;
    device const float * nw  = isq ? qn : kn;
    device float * dst = isq ? q + j * hd : kc + (ulong)a.pos * kvd + j * hd;
    float ss = 0.f;
    for (uint i = t; i < hd; i += ntg) { const float x = src[i]; xs[i] = x; ss += x * x; }
    ss = tg_sum(ss, sh, t, ntg);
    const float scale = 1.0f / sqrt(ss / hd + a.eps);
    for (uint i = t; i < hd; i += ntg) xs[i] = (xs[i] * scale) * nw[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_ = a.rope_dims / 2;
    for (uint i = t; i < hd; i += ntg) {
        float r;
        if (i < half_) {
            r = xs[i] * rope[i] - xs[i + half_] * rope[half_ + i];
        } else if (i < a.rope_dims) {
            const uint k = i - half_;
            r = xs[k] * rope[half_ + k] + xs[i] * rope[k];
        } else {
            r = xs[i];
        }
        dst[i] = r;
        if (isq) gate[j * hd + i] = qg[j * 2 * hd + hd + i];
        else     vc[(ulong)a.pos * kvd + j * hd + i] = vin[j * hd + i];
    }
}

struct attn_args { uint nh; uint nkv; uint hd; uint n; float scale; };

// One query head (a threadgroup): softmax(q.K / sqrt(hd)) . V over positions
// 0..N-1; SC (threadgroup memory, N floats) holds the scores.
kernel void attend(constant attn_args & a [[buffer(0)]],
                   device const float * q [[buffer(1)]],
                   device const float * kc [[buffer(2)]], device const float * vc [[buffer(3)]],
                   device float * out [[buffer(4)]],
                   threadgroup float * sc [[threadgroup(0)]],
                   uint h [[threadgroup_position_in_grid]],
                   uint t [[thread_position_in_threadgroup]],
                   uint ntg [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    const uint hd = a.hd, kvd = a.nkv * hd, kvh = h / (a.nh / a.nkv);
    device const float * qh = q + h * hd;
    float mx = -INFINITY;
    for (uint p = t; p < a.n; p += ntg) {
        device const float * kr = kc + (ulong)p * kvd + kvh * hd;
        float s = 0.f;
        for (uint i = 0; i < hd; ++i) s += qh[i] * kr[i];
        s *= a.scale;
        sc[p] = s;
        mx = max(mx, s);
    }
    mx = tg_max(mx, sh, t, ntg);
    float sum = 0.f;
    for (uint p = t; p < a.n; p += ntg) { const float e = exp(sc[p] - mx); sc[p] = e; sum += e; }
    sum = tg_sum(sum, sh, t, ntg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / sum;
    for (uint i = t; i < hd; i += ntg) {
        float acc = 0.f;
        for (uint p = 0; p < a.n; ++p) acc += sc[p] * vc[(ulong)p * kvd + kvh * hd + i];
        out[h * hd + i] = acc * inv;
    }
}
