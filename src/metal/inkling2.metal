// inkling2.metal — Inkling 2 Mini on the GPU (src/metal-inkling2.lisp),
// one token at a time: every weight read where it lies in the mapped
// checkpoint (BF16, NVFP4, MXFP8; all 16-byte aligned there).  Each kernel
// is a step of inkling2.lisp's CPU reference.

static inline float ink_bf(ushort h) { return as_type<float>(uint(h) << 16); }
static inline float4 ink_bf4(ushort4 h) {
    return float4(ink_bf(h.x), ink_bf(h.y), ink_bf(h.z), ink_bf(h.w));
}

constant float ink_fp4[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

constant float ink_e4m3_lut[256] = {0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f, 0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f, 0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f, 0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f, 0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f, 0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f, 0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f, 1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f, 2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f, 4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f, 32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f, 64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f, 128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f, 256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 0.0f, -0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f, -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f, -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f, -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f, -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f, -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f, -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f, -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f, -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f, -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f, -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f, -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f, -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f, -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f, -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f, -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, 0.0f};

static inline float ink_e4m3(uchar b) {
    const float s = (b & 0x80) ? -1.f : 1.f;
    const int e = (b >> 3) & 15, m = b & 7;
    return e == 0 ? s * float(m) * 0.001953125f : s * (1.f + float(m) * 0.125f) * exp2(float(e - 7));
}

struct ink_mv_args { uint ncols; uint nrows; };

// Y = W X, W BF16 row-major: a simdgroup a row, 4 rows a threadgroup.
// grid ceil(rows/4), threads 128.
kernel void ink_mv_bf16(constant ink_mv_args & a [[buffer(0)]], device const ushort * w [[buffer(1)]],
                        device const float * x [[buffer(2)]], device float * y [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = tg * 4 + sg;
    if (r >= a.nrows) return;
    device const ushort4 * w4 = (device const ushort4 *)(w + (ulong)r * a.ncols);
    device const float4 * x4 = (device const float4 *)x;
    float s = 0.f;
    for (uint i = lane; i < a.ncols / 4; i += 32) s += dot(ink_bf4(w4[i]), x4[i]);
    s = simd_sum(s);
    if (lane == 0) y[r] = s;
}

struct ink_xmv_args { uint ncols; uint nrows; uint x_stride; uint y_stride; float den; };

// Y_k = W_{ids[k]} X_k for the 8 chosen experts, NVFP4: two E2M1 codes a
// byte (the even element low), an E4M3 scale per 16, the expert's outer
// scale amax/den.  A simdgroup two rows (X's 16 floats a block shared by
// them), lanes over the 16-blocks.  grid (ceil(rows/8), 8), threads 128.
#define INK_NR 2
kernel void ink_xmv_nvfp4(constant ink_xmv_args & a [[buffer(0)]], device const uchar * w [[buffer(1)]],
                          device const uchar * sc [[buffer(2)]], device const float * amax [[buffer(3)]],
                          device const uint * ids [[buffer(4)]], device const float * x [[buffer(5)]],
                          device float * y [[buffer(6)]],
                          uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    const uint r0 = (tg.x * 4 + sg) * INK_NR, k = tg.y;
    if (r0 >= a.nrows) return;
    const uint e = ids[k];
    const uint nb = a.ncols / 16;
    device const float * xk = x + k * a.x_stride;
    float s[INK_NR] = {0.f, 0.f};
    for (uint b = lane; b < nb; b += 32) {
        device const float4 * x4 = (device const float4 *)(xk + b * 16);
        const float4 x0 = x4[0], x1 = x4[1], x2 = x4[2], x3 = x4[3];
        for (short rr = 0; rr < INK_NR; ++rr) {
            const ulong row = (ulong)e * a.nrows + r0 + rr;
            const uint2 q = *(device const uint2 *)(w + row * (a.ncols / 2) + b * 8);
            const float4 v0 = float4(ink_fp4[q.x & 15], ink_fp4[(q.x >> 4) & 15], ink_fp4[(q.x >> 8) & 15], ink_fp4[(q.x >> 12) & 15]);
            const float4 v1 = float4(ink_fp4[(q.x >> 16) & 15], ink_fp4[(q.x >> 20) & 15], ink_fp4[(q.x >> 24) & 15], ink_fp4[q.x >> 28]);
            const float4 v2 = float4(ink_fp4[q.y & 15], ink_fp4[(q.y >> 4) & 15], ink_fp4[(q.y >> 8) & 15], ink_fp4[(q.y >> 12) & 15]);
            const float4 v3 = float4(ink_fp4[(q.y >> 16) & 15], ink_fp4[(q.y >> 20) & 15], ink_fp4[(q.y >> 24) & 15], ink_fp4[q.y >> 28]);
            s[rr] += ink_e4m3_lut[sc[row * nb + b]] * (dot(v0, x0) + dot(v1, x1) + dot(v2, x2) + dot(v3, x3));
        }
    }
    const float outer = amax[e] / a.den;
    for (short rr = 0; rr < INK_NR; ++rr) {
        const float t = simd_sum(s[rr]);
        if (lane == 0 && r0 + rr < a.nrows) y[k * a.y_stride + r0 + rr] = t * outer;
    }
}

// The same for MXFP8 (layers 20-23): E4M3 values, an E8M0 scale per 32.
kernel void ink_xmv_mxfp8(constant ink_xmv_args & a [[buffer(0)]], device const uchar * w [[buffer(1)]],
                          device const uchar * sc [[buffer(2)]], device const float * unused [[buffer(3)]],
                          device const uint * ids [[buffer(4)]], device const float * x [[buffer(5)]],
                          device float * y [[buffer(6)]],
                          uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = tg.x * 4 + sg, k = tg.y;
    if (r >= a.nrows) return;
    const uint e = ids[k];
    const ulong row = (ulong)e * a.nrows + r;
    device const uchar * wr = w + row * a.ncols;
    device const uchar * sr = sc + row * (a.ncols / 32);
    device const float * xk = x + k * a.x_stride;
    float s = 0.f;
    for (uint b = lane; b < a.ncols / 32; b += 32) {
        float t = 0.f;
        for (short i = 0; i < 32; ++i) t += ink_e4m3_lut[wr[b * 32 + i]] * xk[b * 32 + i];
        s += exp2(float(int(sr[b]) - 127)) * t;
    }
    s = simd_sum(s);
    if (lane == 0) y[k * a.y_stride + r] = s;
}

static float ink_tg_sum(float v, threadgroup float * red, ushort tid, ushort nthreads) {
    v = simd_sum(v);
    const ushort lane = tid % 32, sg = tid / 32, nsg = (nthreads + 31) / 32;
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = lane < nsg ? red[lane] : 0.f;
    t = simd_sum(t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

static float ink_tg_max(float v, threadgroup float * red, ushort tid, ushort nthreads) {
    v = simd_max(v);
    const ushort lane = tid % 32, sg = tid / 32, nsg = (nthreads + 31) / 32;
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = lane < nsg ? red[lane] : -INFINITY;
    t = simd_max(t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

struct ink_depth_args { uint d; uint n; uint s[8]; float eps; };

// x = sum_j softmax_j(q . vhat_j / sqrt d) vhat_j, vhat = v / RMS(v), over
// the N stack slots S (Goodfire's depth-wise read).  One threadgroup of 1024.
kernel void ink_depth(constant ink_depth_args & a [[buffer(0)]], device const float * stack [[buffer(1)]],
                      device const float * q [[buffer(2)]], device float * out [[buffer(3)]],
                      threadgroup float * red [[threadgroup(0)]],
                      ushort tid [[thread_index_in_threadgroup]]) {
    float inv[8], lg[8];
    for (uint j = 0; j < a.n; ++j) {
        device const float * v = stack + a.s[j] * a.d;
        float ss = 0.f, qd = 0.f;
        for (uint i = tid; i < a.d; i += 1024) { const float x = v[i]; ss += x * x; qd += q[i] * x; }
        ss = ink_tg_sum(ss, red, tid, 1024);
        qd = ink_tg_sum(qd, red, tid, 1024);
        inv[j] = 1.f / sqrt(ss / a.d + a.eps);
        lg[j] = qd * inv[j] / sqrt(float(a.d));
    }
    float m = lg[0];
    for (uint j = 1; j < a.n; ++j) m = max(m, lg[j]);
    float z = 0.f;
    for (uint j = 0; j < a.n; ++j) { lg[j] = exp(lg[j] - m); z += lg[j]; }
    for (uint i = tid; i < a.d; i += 1024) {
        float x = 0.f;
        for (uint j = 0; j < a.n; ++j) x += lg[j] / z * inv[j] * stack[a.s[j] * a.d + i];
        out[i] = x;
    }
}

struct ink_rms_args { uint n; uint gain; float eps; };

// OUT = X / RMS(X), times GAIN (BF16) when a.gain.  One threadgroup of 1024
// per vector; grid G vectors of N, each at its own offset (a head's norm).
kernel void ink_rms(constant ink_rms_args & a [[buffer(0)]], device const float * x [[buffer(1)]],
                    device const ushort * g [[buffer(2)]], device float * out [[buffer(3)]],
                    threadgroup float * red [[threadgroup(0)]],
                    uint tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                    ushort nt [[threads_per_threadgroup]]) {
    device const float * xv = x + tg * a.n;
    float ss = 0.f;
    for (uint i = tid; i < a.n; i += nt) ss += xv[i] * xv[i];
    ss = ink_tg_sum(ss, red, tid, nt);
    const float k = 1.f / sqrt(ss / a.n + a.eps);
    for (uint i = tid; i < a.n; i += nt) out[tg * a.n + i] = xv[i] * k * (a.gain ? ink_bf(g[i]) : 1.f);
}

struct ink_n_args { uint n; };

// Cross-pass: OUT = (1 - tanh(FZ)) X + tanh(GZ) M
kernel void ink_mix(constant ink_n_args & a [[buffer(0)]], device const float * x [[buffer(1)]],
                    device const float * m [[buffer(2)]], device const float * fz [[buffer(3)]],
                    device const float * gz [[buffer(4)]], device float * out [[buffer(5)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    out[i] = (1.f - precise::tanh(fz[i])) * x[i] + precise::tanh(gz[i]) * m[i];
}

struct ink_conv_args { uint n; uint dyn; };

// Y = X + the causal depthwise conv (width 4, taps oldest first) of the last
// three inputs (HIST) and X, then HIST shifts.  Taps: N x 4 BF16 (static),
// or with DYN static 4 x N (BF16) plus a per-head part DV (8 heads x 4).
kernel void ink_conv(constant ink_conv_args & a [[buffer(0)]], device float * hist [[buffer(1)]],
                     device const float * x [[buffer(2)]], device const ushort * w [[buffer(3)]],
                     device const float * dv [[buffer(4)]], device float * y [[buffer(5)]],
                     uint c [[thread_position_in_grid]]) {
    if (c >= a.n) return;
    float t[4];
    for (short k = 0; k < 4; ++k)
        t[k] = a.dyn ? ink_bf(w[k * a.n + c]) + dv[(c / 128) * 4 + k] : ink_bf(w[c * 4 + k]);
    const float x0 = hist[c], x1 = hist[a.n + c], x2 = hist[2 * a.n + c], x3 = x[c];
    y[c] = x3 + t[0] * x0 + t[1] * x1 + t[2] * x2 + t[3] * x3;
    hist[c] = x1; hist[a.n + c] = x2; hist[2 * a.n + c] = x3;
}

struct ink_clip_args { uint hd; float thr; uint heads; uint mode; };

// Each head of V (HD wide) scaled to L2 norm at most THR, then times SCALE
// (BF16, heads x HD).  One threadgroup a head, threads HD.
kernel void ink_clip(constant ink_clip_args & a [[buffer(0)]], device float * v [[buffer(1)]],
                     device const ushort * scale [[buffer(2)]], threadgroup float * red [[threadgroup(0)]],
                     uint h [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                     ushort nt [[threads_per_threadgroup]]) {
    const uint i = h * a.hd + tid;
    const float x = v[i];
    const float n = sqrt(ink_tg_sum(x * x, red, tid, nt));
    // mode 0: clipped to THR, then scaled; 1: RMS-normed, then scaled; 2: only scaled
    const float sc = ink_bf(scale[(h % a.heads) * a.hd + tid]);
    if (a.mode == 3) {                              // scaled, then clipped
        const float n2 = sqrt(ink_tg_sum(x * sc * x * sc, red, tid, nt));
        v[i] = x * sc * (n2 > a.thr ? a.thr / n2 : 1.f);
        return;
    }
    const float k = a.mode == 0 ? (n > a.thr ? a.thr / n : 1.f) : a.mode == 1 ? sqrt(float(a.hd)) / max(n, 1e-12f) : 1.f;
    v[i] = x * k * sc;
}

struct ink_attn_args { uint pos; uint first; uint cap; float scale; uint extent; uint ctx; };

// One token's attention, a threadgroup a query head (24): logits over the
// cached positions FIRST..POS (a ring of CAP slots) plus the relative bias
// r_h . P[:, pos - j] within EXTENT, and a zero-value sink with logit
// SINK[h]; softmax; the weighted V.  SCRATCH: 24 x CTX floats.
kernel void ink_attend(constant ink_attn_args & a [[buffer(0)]], device const float * q [[buffer(1)]],
                       device const float * r [[buffer(2)]], device const float * sink [[buffer(3)]],
                       device const float * kc [[buffer(4)]], device const float * vc [[buffer(5)]],
                       device const ushort * P [[buffer(6)]], device float * lg [[buffer(7)]],
                       device float * out [[buffer(8)]], threadgroup float * red [[threadgroup(0)]],
                       uint h [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint g = h / 3, n = a.pos - a.first + 1;
    device float * l = lg + h * a.ctx;
    device const float4 * q4 = (device const float4 *)(q + h * 128);
    float rr[16];
    for (short i = 0; i < 16; ++i) rr[i] = r[h * 16 + i];
    float mx = sink[h];
    for (uint jj = tid; jj < n; jj += 256) {
        const uint j = a.first + jj, slot = j % a.cap, dist = a.pos - j;
        device const float4 * k4 = (device const float4 *)(kc + slot * 1024 + g * 128);
        float d = 0.f;
        for (short i = 0; i < 32; ++i) d += dot(q4[i], k4[i]);
        float b = 0.f;
        if (dist < a.extent) for (short i = 0; i < 16; ++i) b += rr[i] * ink_bf(P[i * a.extent + dist]);
        const float x = d * a.scale + b;
        l[jj] = x; mx = max(mx, x);
    }
    mx = ink_tg_max(mx, red, tid, 256);
    float z = 0.f;
    for (uint jj = tid; jj < n; jj += 256) { const float e = exp(l[jj] - mx); l[jj] = e; z += e; }
    z = ink_tg_sum(z, red, tid, 256) + exp(sink[h] - mx);
    threadgroup_barrier(mem_flags::mem_device);
    // the weighted V: thread = (dim, half of the positions)
    const uint dim = tid % 128, half_ = tid / 128;
    float acc = 0.f;
    for (uint jj = half_; jj < n; jj += 2) acc += l[jj] * vc[((a.first + jj) % a.cap) * 1024 + g * 128 + dim];
    red[32 + tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128) out[h * 128 + dim] = (red[32 + tid] + red[32 + 128 + tid]) / z;
}

struct ink_gate_args { uint n; uint mode; };

// O *= sigmoid(G) (mode 1: 2 sigmoid(G); 2: G sigmoid(G))
kernel void ink_gate(constant ink_gate_args & a [[buffer(0)]], device float * o [[buffer(1)]],
                     device const float * g [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    const float s = 1.f / (1.f + exp(-g[i]));
    o[i] = o[i] * (a.mode == 0 ? s : a.mode == 1 ? 2.f * s : g[i] * s);
}

// Y += X
kernel void ink_add(constant ink_n_args & a [[buffer(0)]], device float * y [[buffer(1)]],
                    device const float * x [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    y[i] += x[i];
}

struct ink_route_args { float scale; };

// The router's choice: sigmoid scores, the 8 best by score + bias, their
// scores normalized to sum 1 and times SCALE.  One threadgroup of 128.
kernel void ink_route(constant ink_route_args & a [[buffer(0)]], device const float * logits [[buffer(1)]],
                      device const float * bias [[buffer(2)]], device uint * ids [[buffer(3)]],
                      device float * wts [[buffer(4)]], threadgroup float * red [[threadgroup(0)]],
                      ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float key[128];
    threadgroup float scv[128];
    const float s = 1.f / (1.f + exp(-logits[tid]));
    scv[tid] = s; key[tid] = s + bias[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot = 0.f; uint pick[8];
        for (short k = 0; k < 8; ++k) {
            uint best = 0; float bv = -INFINITY;
            for (uint e = 0; e < 128; ++e) if (key[e] > bv) { bv = key[e]; best = e; }
            pick[k] = best; key[best] = -INFINITY; tot += scv[best];
        }
        for (short k = 0; k < 8; ++k) { ids[k] = pick[k]; wts[k] = scv[pick[k]] / tot * a.scale; }
    }
}

// The experts' activation: A[k][i] = silu(4 tanh(g/4)) * 25 tanh(u/25),
// g = H[k][2i], u = H[k][2i+1].  grid 8 x FFN.
kernel void ink_act(constant ink_n_args & a [[buffer(0)]], device const float * h [[buffer(1)]],
                    device float * out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= 8 * a.n) return;
    const uint k = i / a.n, j = i % a.n;
    const float g = 4.f * precise::tanh(h[k * 2 * a.n + 2 * j] / 4.f);
    const float u = 25.f * precise::tanh(h[k * 2 * a.n + 2 * j + 1] / 25.f);
    out[i] = g / (1.f + exp(-g)) * u;
}

// OUT = sum_k W[k] Y[k]
kernel void ink_combine(constant ink_n_args & a [[buffer(0)]], device const float * y [[buffer(1)]],
                        device const float * w [[buffer(2)]], device float * out [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    float s = 0.f;
    for (short k = 0; k < 8; ++k) s += w[k] * y[k * a.n + i];
    out[i] = s;
}

struct ink_embed_args { uint d; uint token; };

kernel void ink_embed(constant ink_embed_args & a [[buffer(0)]], device const ushort * e [[buffer(1)]],
                      device float * out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.d) return;
    out[i] = ink_bf(e[(ulong)a.token * a.d + i]);
}

struct ink_cap_args { uint n; float cap; };

kernel void ink_softcap(constant ink_cap_args & a [[buffer(0)]], device float * x [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    x[i] = a.cap * precise::tanh(x[i] / a.cap);
}

// Y = X (N floats), or Y = 0 when X is the same buffer and a.n has its top bit
kernel void ink_copy(constant ink_n_args & a [[buffer(0)]], device const float * x [[buffer(1)]],
                     device float * y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    y[i] = x[i];
}

kernel void ink_zero(constant ink_n_args & a [[buffer(0)]], device float * y [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    y[i] = 0.f;
}

// --- a chunk of tokens (prefill) -----------------------------------------------------
//
// The same steps for N tokens at once: a token's vectors are N apart (or a
// slot's N x D block in the stack), weights read once per chunk where it pays.

struct ink_depthn_args { uint d; uint n; uint s[8]; float eps; uint ntok; float temp; };

// INK_DEPTH for token TG of N: slot k's row t at (k * ntok + t) * d.
kernel void ink_depth_n(constant ink_depthn_args & a [[buffer(0)]], device const float * stack [[buffer(1)]],
                        device const float * q [[buffer(2)]], device float * out [[buffer(3)]],
                        threadgroup float * red [[threadgroup(0)]],
                        uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    float inv[8], lg[8];
    for (uint j = 0; j < a.n; ++j) {
        device const float * v = stack + ((ulong)a.s[j] * a.ntok + t) * a.d;
        float ss = 0.f, qd = 0.f;
        for (uint i = tid; i < a.d; i += 1024) { const float x = v[i]; ss += x * x; qd += q[i] * x; }
        ss = ink_tg_sum(ss, red, tid, 1024);
        qd = ink_tg_sum(qd, red, tid, 1024);
        inv[j] = 1.f / sqrt(ss / a.d + a.eps);
        lg[j] = qd * inv[j] * a.temp;
    }
    float m = lg[0];
    for (uint j = 1; j < a.n; ++j) m = max(m, lg[j]);
    float z = 0.f;
    for (uint j = 0; j < a.n; ++j) { lg[j] = exp(lg[j] - m); z += lg[j]; }
    for (uint i = tid; i < a.d; i += 1024) {
        float x = 0.f;
        for (uint j = 0; j < a.n; ++j) x += lg[j] / z * inv[j] * stack[((ulong)a.s[j] * a.ntok + t) * a.d + i];
        out[(ulong)t * a.d + i] = x;
    }
}

struct ink_mvn_args { uint ncols; uint nrows; uint ntok; };

// INK_MV_BF16 for N tokens, 8 at a time: a simdgroup a row, its BF16
// weights read once for the 8.  grid (ceil(rows/4), ceil(N/8)); token t's
// X at t*ncols, Y at t*nrows.
kernel void ink_mv_bf16_n(constant ink_mvn_args & a [[buffer(0)]], device const ushort * w [[buffer(1)]],
                          device const float * x [[buffer(2)]], device float * y [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = tg.x * 4 + sg, t0 = tg.y * 8;
    if (r >= a.nrows) return;
    const uint nt = min(8u, a.ntok - t0);
    device const ushort4 * w4 = (device const ushort4 *)(w + (ulong)r * a.ncols);
    float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (uint i = lane; i < a.ncols / 4; i += 32) {
        const float4 wv = ink_bf4(w4[i]);
        for (uint k = 0; k < nt; ++k) s[k] += dot(wv, ((device const float4 *)(x + (ulong)(t0 + k) * a.ncols))[i]);
    }
    for (uint k = 0; k < nt; ++k) {
        const float v = simd_sum(s[k]);
        if (lane == 0) y[(ulong)(t0 + k) * a.nrows + r] = v;
    }
}

struct ink_convn_args { uint n; uint dyn; uint ntok; uint plain; };

// INK_CONV for N tokens: token t's window is the chunk's own earlier rows,
// then HIST's.  HIST is left alone (INK_HIST_N moves it on after).
kernel void ink_conv_n(constant ink_convn_args & a [[buffer(0)]], device const float * hist [[buffer(1)]],
                       device const float * x [[buffer(2)]], device const ushort * w [[buffer(3)]],
                       device const float * dv [[buffer(4)]], device float * y [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, t = gid.y;
    if (c >= a.n || t >= a.ntok) return;
    float xs[4];
    for (int k = 0; k < 4; ++k) {
        const int src = int(t) - 3 + k;                  // the chunk row, or HIST's (3 + src)
        xs[k] = src >= 0 ? x[(ulong)src * a.n + c] : hist[(3 + src) * a.n + c];
    }
    float s = a.plain ? 0.f : xs[3];
    for (short k = 0; k < 4; ++k) {
        const float sw = a.dyn ? ink_bf(w[k * a.n + c]) : 0.f;
        const float dw = a.dyn == 0 ? 0.f : a.dyn >= 5 ? dv[(ulong)t * 32 + k * 8 + c / 128] : dv[(ulong)t * 32 + (c / 128) * 4 + k];
        // dyn 1: static + per-token; 2: per-token only; 3: static only; 4: static x (1 + per-token)
        const float tap = a.dyn == 0 ? ink_bf(w[c * 4 + k]) : (a.dyn == 1 || a.dyn == 5) ? sw + dw : a.dyn == 2 ? dw : a.dyn == 3 ? sw
                        : a.dyn == 6 ? sw * (1.f + dw) : a.dyn == 7 ? sw * dw : sw * (1.f + dw);
        s += tap * xs[k];
    }
    y[(ulong)t * a.n + c] = s;
}

// HIST becomes the last three inputs, the chunk's (and before it HIST's own).
kernel void ink_hist_n(constant ink_convn_args & a [[buffer(0)]], device float * hist [[buffer(1)]],
                       device const float * x [[buffer(2)]], uint c [[thread_position_in_grid]]) {
    if (c >= a.n) return;
    float h[3] = {hist[c], hist[a.n + c], hist[2 * a.n + c]};
    float nh[3];
    for (int k = 0; k < 3; ++k) {
        const int src = int(a.ntok) - 3 + k;
        nh[k] = src >= 0 ? x[(ulong)src * a.n + c] : h[3 + src];
    }
    hist[c] = nh[0]; hist[a.n + c] = nh[1]; hist[2 * a.n + c] = nh[2];
}

struct ink_kvn_args { uint pos; uint cap; float eps; uint mode; float thr; };

// Each token's K (or V), normed per head (no gain), into its cache slot
// (pos + t) % cap.  grid (8 heads, N), threads 128.
kernel void ink_kv_n(constant ink_kvn_args & a [[buffer(0)]], device const float * x [[buffer(1)]],
                     device float * cache [[buffer(2)]], threadgroup float * red [[threadgroup(0)]],
                     uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint h = tg.x, t = tg.y;
    const float v = x[(ulong)t * 1024 + h * 128 + tid];
    const float ss = ink_tg_sum(v * v, red, tid, 128);
    // mode 0: RMS-normed; 1: L2-clipped to THR; 2: as it is
    const float k = a.mode == 0 ? 1.f / sqrt(ss / 128.f + a.eps)
                  : a.mode == 1 ? (sqrt(ss) > a.thr ? a.thr / sqrt(ss) : 1.f) : 1.f;
    cache[(ulong)((a.pos + t) % a.cap) * 1024 + h * 128 + tid] = v * k;
}

struct ink_attnn_args { uint pos; uint cap; float scale; uint extent; uint window; float rel_scale; uint gqa_mod; uint sink; };

// Attention for the chunk's queries, a threadgroup a (head, token): the
// cached positions it may see, taken 256 at a time with a running max and
// sum (no scores kept), the relative bias within EXTENT, the sink.
// grid (24, N), threads 256.
kernel void ink_attend_n(constant ink_attnn_args & a [[buffer(0)]], device const float * q [[buffer(1)]],
                         device const float * r [[buffer(2)]], device const float * sink [[buffer(3)]],
                         device const float * kc [[buffer(4)]], device const float * vc [[buffer(5)]],
                         device const ushort * P [[buffer(6)]], device float * out [[buffer(7)]],
                         threadgroup float * red [[threadgroup(0)]],
                         uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float tile[256];
    threadgroup float acc2[256];
    const uint h = tg.x, t = tg.y, g = a.gqa_mod ? h % 8 : h / 3;
    const uint pos = a.pos + t;
    const uint first = a.window ? (pos + 1 > a.window ? pos + 1 - a.window : 0) : 0;
    device const float4 * q4 = (device const float4 *)(q + ((ulong)t * 24 + h) * 128);
    float rr[16];
    for (short i = 0; i < 16; ++i) rr[i] = r[((ulong)t * 24 + h) * 16 + i];
    const float s0 = sink[(ulong)t * 24 + h];
    float m = a.sink ? s0 : -INFINITY, z = a.sink ? 1.f : 0.f, acc = 0.f;   // the sink: logit s0, value 0
    const uint dim = tid % 128, half_ = tid / 128;
    for (uint j0 = first; j0 <= pos; j0 += 256) {
        const uint j = j0 + tid;
        float l = -INFINITY;
        if (j <= pos) {
            const uint slot = j % a.cap, dist = pos - j;
            device const float4 * k4 = (device const float4 *)(kc + (ulong)slot * 1024 + g * 128);
            float d = 0.f;
            for (short i = 0; i < 32; ++i) d += dot(q4[i], k4[i]);
            float b = 0.f;
            if (dist < a.extent) for (short i = 0; i < 16; ++i) b += rr[i] * ink_bf(P[i * a.extent + dist]);
            l = d * a.scale + b * a.rel_scale;
        }
        const float tm = ink_tg_max(l, red, tid, 256);
        const float mn = max(m, tm);
        const float e = j <= pos ? exp(l - mn) : 0.f;
        tile[tid] = e;
        const float ts = ink_tg_sum(e, red, tid, 256);
        const float sc = exp(m - mn);
        z = z * sc + ts;
        acc *= sc;
        const uint cnt = min(256u, pos + 1 - j0);
        for (uint jj = half_; jj < cnt; jj += 2)
            acc += tile[jj] * vc[(ulong)((j0 + jj) % a.cap) * 1024 + g * 128 + dim];
        m = mn;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    acc2[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128) out[((ulong)t * 24 + h) * 128 + dim] = (acc2[tid] + acc2[tid + 128]) / z;
}

struct ink_routen_args { float scale; uint bias_w; };

// INK_ROUTE for token TG: its 8 experts and weights at TG*8.
kernel void ink_route_n(constant ink_routen_args & a [[buffer(0)]], device const float * logits [[buffer(1)]],
                        device const float * bias [[buffer(2)]], device uint * ids [[buffer(3)]],
                        device float * wts [[buffer(4)]],
                        uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float key[128];
    threadgroup float scv[128];
    const float s = 1.f / (1.f + exp(-logits[(ulong)t * 128 + tid]));
    scv[tid] = s; key[tid] = s + bias[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot = 0.f; uint pick[8];
        for (short k = 0; k < 8; ++k) {
            uint best = 0; float bv = -INFINITY;
            for (uint e = 0; e < 128; ++e) if (key[e] > bv) { bv = key[e]; best = e; }
            pick[k] = best; key[best] = -INFINITY; tot += scv[best];
        }
        if (a.bias_w) { tot = 0.f; for (short k = 0; k < 8; ++k) tot += scv[pick[k]] + bias[pick[k]]; }
        for (short k = 0; k < 8; ++k) { ids[t * 8 + k] = pick[k]; wts[t * 8 + k] = (scv[pick[k]] + (a.bias_w ? bias[pick[k]] : 0.f)) / tot * a.scale; }
    }
}

struct ink_group_args { uint ntok; };

// The chunk's (token, k) pairs grouped by expert: COUNT[e], START[e] and,
// from START[e], the pairs (t * 8 + k) that chose e.  One thread.
kernel void ink_group(constant ink_group_args & a [[buffer(0)]], device const uint * ids [[buffer(1)]],
                      device uint * count [[buffer(2)]], device uint * start [[buffer(3)]],
                      device uint * pairs [[buffer(4)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    for (uint e = 0; e < 128; ++e) count[e] = 0;
    for (uint p = 0; p < a.ntok * 8; ++p) count[ids[p]]++;
    uint s = 0;
    for (uint e = 0; e < 128; ++e) { start[e] = s; s += count[e]; count[e] = 0; }
    for (uint p = 0; p < a.ntok * 8; ++p) { const uint e = ids[p]; pairs[start[e] + count[e]++] = p; }
}

struct ink_xmm_args { uint ncols; uint nrows; uint x_pair; float den; };

// Grouped expert products, NVFP4: threadgroup (row tile of 64, expert e,
// pair tile of 8): each of its 64 threads a row, its 8 pairs' X (from the
// pair's token, or the pair itself when X_PAIR) staged 16 columns at a
// time; a row's codes read once for the 8.  Y at pair * nrows.
kernel void ink_xmm_nvfp4(constant ink_xmm_args & a [[buffer(0)]], device const uchar * w [[buffer(1)]],
                          device const uchar * sc [[buffer(2)]], device const float * amax [[buffer(3)]],
                          device const uint * count [[buffer(4)]], device const uint * start [[buffer(5)]],
                          device const uint * pairs [[buffer(6)]], device const float * x [[buffer(7)]],
                          device float * y [[buffer(8)]],
                          uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint e = tg.y, p0 = tg.z * 8;
    const uint cnt = count[e];
    if (p0 >= cnt) return;
    const uint np = min(8u, cnt - p0);
    threadgroup float xs[8][16];
    uint pr[8];
    for (uint k = 0; k < 8; ++k) pr[k] = k < np ? pairs[start[e] + p0 + k] : 0;
    const uint r = tg.x * 64 + tid;
    const uint nb = a.ncols / 16;
    const ulong row = (ulong)e * a.nrows + min(r, a.nrows - 1);
    device const uchar * wr = w + row * (a.ncols / 2);
    device const uchar * sr = sc + row * nb;
    float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (uint b = 0; b < nb; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8 * 16; i += 64) {
            const uint k = i / 16, c = i % 16;
            const uint src = a.x_pair ? pr[k] : pr[k] / 8;
            xs[k][c] = k < np ? x[(ulong)src * a.ncols + b * 16 + c] : 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint2 q = *(device const uint2 *)(wr + b * 8);
        float v[16];
        for (short i = 0; i < 8; ++i) {
            const uint word = i < 4 ? q.x : q.y, sh = (i & 3) * 8;
            const uint byte = (word >> sh) & 255;
            v[2 * i] = ink_fp4[byte & 15]; v[2 * i + 1] = ink_fp4[byte >> 4];
        }
        const float bs = ink_e4m3_lut[sr[b]];
        for (uint k = 0; k < 8; ++k) {
            float t = 0.f;
            for (short i = 0; i < 16; ++i) t += v[i] * xs[k][i];
            s[k] += bs * t;
        }
    }
    if (r >= a.nrows) return;
    const float outer = amax[e] / a.den;
    for (uint k = 0; k < np; ++k) y[(ulong)pr[k] * a.nrows + r] = s[k] * outer;
}

// The same for MXFP8 (layers 20-23): 32-column blocks, an E8M0 scale each.
kernel void ink_xmm_mxfp8(constant ink_xmm_args & a [[buffer(0)]], device const uchar * w [[buffer(1)]],
                          device const uchar * sc [[buffer(2)]], device const float * unused [[buffer(3)]],
                          device const uint * count [[buffer(4)]], device const uint * start [[buffer(5)]],
                          device const uint * pairs [[buffer(6)]], device const float * x [[buffer(7)]],
                          device float * y [[buffer(8)]],
                          uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint e = tg.y, p0 = tg.z * 8;
    const uint cnt = count[e];
    if (p0 >= cnt) return;
    const uint np = min(8u, cnt - p0);
    threadgroup float xs[8][32];
    uint pr[8];
    for (uint k = 0; k < 8; ++k) pr[k] = k < np ? pairs[start[e] + p0 + k] : 0;
    const uint r = tg.x * 64 + tid;
    const uint nb = a.ncols / 32;
    const ulong row = (ulong)e * a.nrows + min(r, a.nrows - 1);
    device const uchar * wr = w + row * a.ncols;
    device const uchar * sr = sc + row * nb;
    float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (uint b = 0; b < nb; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8 * 32; i += 64) {
            const uint k = i / 32, c = i % 32;
            const uint src = a.x_pair ? pr[k] : pr[k] / 8;
            xs[k][c] = k < np ? x[(ulong)src * a.ncols + b * 32 + c] : 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float bs = exp2(float(int(sr[b]) - 127));
        float v[32];
        for (short i = 0; i < 32; ++i) v[i] = ink_e4m3_lut[wr[b * 32 + i]];
        for (uint k = 0; k < 8; ++k) {
            float t = 0.f;
            for (short i = 0; i < 32; ++i) t += v[i] * xs[k][i];
            s[k] += bs * t;
        }
    }
    if (r >= a.nrows) return;
    for (uint k = 0; k < np; ++k) y[(ulong)pr[k] * a.nrows + r] = s[k];
}

struct ink_actn_args { uint ffn; uint npairs; uint mode; uint halves; };

// INK_ACT over NPAIRS pairs: A[p][i] from H[p][2i], H[p][2i+1].
kernel void ink_act_n(constant ink_actn_args & a [[buffer(0)]], device const float * h [[buffer(1)]],
                      device float * out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.npairs * a.ffn) return;
    const uint p = i / a.ffn, j = i % a.ffn;
    const float g0 = a.halves ? h[(ulong)p * 2 * a.ffn + j] : h[(ulong)p * 2 * a.ffn + 2 * j];
    const float u0 = a.halves ? h[(ulong)p * 2 * a.ffn + a.ffn + j] : h[(ulong)p * 2 * a.ffn + 2 * j + 1];
    // mode 0: silu(4 tanh(g/4)) 25 tanh(u/25); 1: silu(g) u; 2: silu(g) 25 tanh(u/25); 3: 4 tanh(silu(g)/4) 25 tanh(u/25)
    const float g = (a.mode == 0) ? 4.f * precise::tanh(g0 / 4.f) : g0;
    const float u = (a.mode == 1) ? u0 : 25.f * precise::tanh(u0 / 25.f);
    const float sg = g / (1.f + exp(-g));
    out[i] = (a.mode == 3 ? 4.f * precise::tanh(sg / 4.f) : sg) * u;
}

struct ink_combn_args { uint d; uint ntok; };

// OUT[t] = sum_k W[t*8+k] Y[t*8+k]
kernel void ink_combine_n(constant ink_combn_args & a [[buffer(0)]], device const float * y [[buffer(1)]],
                          device const float * w [[buffer(2)]], device float * out [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, t = gid.y;
    if (i >= a.d || t >= a.ntok) return;
    float s = 0.f;
    for (short k = 0; k < 8; ++k) s += w[t * 8 + k] * y[((ulong)t * 8 + k) * a.d + i];
    out[(ulong)t * a.d + i] = s;
}

struct ink_embedn_args { uint d; };

kernel void ink_embed_n(constant ink_embedn_args & a [[buffer(0)]], device const ushort * e [[buffer(1)]],
                        device const uint * toks [[buffer(2)]], device float * out [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.d) return;
    out[(ulong)gid.y * a.d + gid.x] = ink_bf(e[(ulong)toks[gid.y] * a.d + gid.x]);
}

// A token's row of ZIN = [RMSNorm(X_t); RMSNorm(M_t)] (no gains), 2d wide.
struct ink_rms2_args { uint d; float eps; };
kernel void ink_rms2(constant ink_rms2_args & a [[buffer(0)]], device const float * x [[buffer(1)]],
                     device const float * m [[buffer(2)]], device float * out [[buffer(3)]],
                     threadgroup float * red [[threadgroup(0)]],
                     uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    device const float * xv = x + (ulong)t * a.d;
    device const float * mv = m + (ulong)t * a.d;
    float sx = 0.f, sm = 0.f;
    for (uint i = tid; i < a.d; i += 1024) { sx += xv[i] * xv[i]; sm += mv[i] * mv[i]; }
    sx = ink_tg_sum(sx, red, tid, 1024);
    sm = ink_tg_sum(sm, red, tid, 1024);
    const float kx = 1.f / sqrt(sx / a.d + a.eps), km = 1.f / sqrt(sm / a.d + a.eps);
    for (uint i = tid; i < a.d; i += 1024) {
        out[(ulong)t * 2 * a.d + i] = xv[i] * kx;
        out[(ulong)t * 2 * a.d + a.d + i] = mv[i] * km;
    }
}
