// gpt2.metal -- GPT-2's training step on the GPU (metal-gpt2-train.lisp).
//
// The layers of gpt2-train.lisp (llm.c's train_gpt2.c), in f32.  The matrix
// products, attention's included, are one kernel: C = alpha op(A) op(B)
// (+ C) (+ bias), op a transpose or not, batched over a grid's z with two
// levels of strides (attention's batch and head).  The rest are a thread an
// element, or a simdgroup a row where a row is reduced.

#include <metal_stdlib>
using namespace metal;

// --- the matrix product ------------------------------------------------------

struct g2_gemm_args {
    uint M, N, K, lda, ldb, ldc;
    float alpha; uint beta; uint bias;
    uint nb2; uint sa1, sa2, sb1, sb2, sc1, sc2;
};

// A 64x64 tile of C a threadgroup of four simdgroups, each 32x32 of it in
// sixteen 8x8 simdgroup matrices (ggml's kernel_mul_mm, in f32).  A's 64x32
// and B's 32x64 slices are staged in threadgroup memory a contiguous 8x8
// block at a time, so each simdgroup_load reads 64 floats in a row: no bank
// conflicts.  Where a slice is whole and VEC (every row 16-byte aligned) it
// is read as float4s, else a float at a time, zero past the edges.  C goes
// out through threadgroup memory, so ALPHA, BETA and BIAS apply there.
template <bool TA, bool TB, typename T, bool SPLIT>
kernel void g2_gemm(constant g2_gemm_args &p [[buffer(0)]],
                    device const float *A [[buffer(1)]],
                    device const float *B [[buffer(2)]],
                    device float *C [[buffer(3)]],
                    device const float *bias [[buffer(4)]],
                    threadgroup float *shf [[threadgroup(0)]],
                    uint3 tg [[threadgroup_position_in_grid]],
                    ushort tid [[thread_index_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM = 64, BN = 64, BK = 32;
    threadgroup T *sh = (threadgroup T *)shf;
    const uint z1 = tg.z / p.nb2, z2 = tg.z % p.nb2;
    A += z1 * p.sa1 + z2 * p.sa2;
    B += z1 * p.sb1 + z2 * p.sb2;
    C += z1 * p.sc1 + z2 * p.sc2;
    const uint m0 = tg.y * BM, n0 = tg.x * BN;
    threadgroup T *As = sh;             // block (kb, mb) at (kb*8 + mb)*64, then [m%8][k%8]
    threadgroup T *Bs = sh + BM * BK;   // block (kb, nb) at (kb*8 + nb)*64, then [k%8][n%8]
    // SPLIT: each value as hi + lo, the lo parts after; three products a step
    threadgroup T *Al = sh + 2 * BM * BK, *Bl = sh + 3 * BM * BK;
    auto put = [&](threadgroup T *tile, uint at, float v) {
        T hi = T(v);
        tile[at] = hi;
        if (SPLIT) tile[at + 2 * BM * BK] = T(v - float(hi));
    };
    auto put4 = [&](threadgroup T *tile, uint at, float4 v) {
        vec<T, 4> hi = vec<T, 4>(v);
        *(threadgroup vec<T, 4> *)(tile + at) = hi;
        if (SPLIT) *(threadgroup vec<T, 4> *)(tile + at + 2 * BM * BK) = vec<T, 4>(v - float4(hi));
    };
    const uint mb0 = (sg % 2) * 4, nb0 = (sg / 2) * 4;
    const bool vec = p.beta >> 1;           // bit 1: rows aligned for float4
    const bool mfull = m0 + BM <= p.M, nfull = n0 + BN <= p.N;

    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        const bool kfull = k0 + BK <= p.K;
        // A: sixteen elements a thread
        if (!TA) {   // k contiguous: row mm, k from kh
            const uint mm = tid / 2, kh = (tid % 2) * 16, gm = m0 + mm;
            threadgroup T *dst = As + (mm / 8) * 64 + (mm % 8) * 8;
            if (vec && mfull && kfull) {
                device const float4 *src = (device const float4 *)(A + gm * p.lda + k0 + kh);
                for (int q = 0; q < 4; q++) {
                    const uint kk = kh + 4 * q;
                    put4(dst, (kk / 8) * 512 + kk % 8, src[q]);
                }
            } else {
                for (int q = 0; q < 16; q++) {
                    const uint kk = kh + q, gk = k0 + kk;
                    put(dst, (kk / 8) * 512 + kk % 8, (gm < p.M && gk < p.K) ? A[gm * p.lda + gk] : 0.0f);
                }
            }
        } else {     // m contiguous: k row kk, m from mh
            const uint kk = tid / 4, mh = (tid % 4) * 16, gk = k0 + kk;
            threadgroup T *dst = As + (kk / 8) * 512 + kk % 8;
            if (vec && mfull && kfull) {
                device const float4 *src = (device const float4 *)(A + gk * p.lda + m0 + mh);
                for (int q = 0; q < 4; q++) {
                    float4 v = src[q];
                    for (int r = 0; r < 4; r++) {
                        const uint mm = mh + 4 * q + r;
                        put(dst, (mm / 8) * 64 + (mm % 8) * 8, v[r]);
                    }
                }
            } else {
                for (int q = 0; q < 16; q++) {
                    const uint mm = mh + q, gm = m0 + mm;
                    put(dst, (mm / 8) * 64 + (mm % 8) * 8, (gm < p.M && gk < p.K) ? A[gk * p.lda + gm] : 0.0f);
                }
            }
        }
        // B
        if (!TB) {   // n contiguous: k row kk, n from nh
            const uint kk = tid / 4, nh = (tid % 4) * 16, gk = k0 + kk;
            threadgroup T *dst = Bs + (kk / 8) * 512 + (kk % 8) * 8;
            if (vec && nfull && kfull) {
                device const float4 *src = (device const float4 *)(B + gk * p.ldb + n0 + nh);
                for (int q = 0; q < 4; q++) {
                    const uint nn = nh + 4 * q;
                    put4(dst, (nn / 8) * 64 + nn % 8, src[q]);
                }
            } else {
                for (int q = 0; q < 16; q++) {
                    const uint nn = nh + q, gn = n0 + nn;
                    put(dst, (nn / 8) * 64 + nn % 8, (gk < p.K && gn < p.N) ? B[gk * p.ldb + gn] : 0.0f);
                }
            }
        } else {     // k contiguous: n row nn, k from kh
            const uint nn = tid / 2, kh = (tid % 2) * 16, gn = n0 + nn;
            threadgroup T *dst = Bs + (nn / 8) * 64 + nn % 8;
            if (vec && nfull && kfull) {
                device const float4 *src = (device const float4 *)(B + gn * p.ldb + k0 + kh);
                for (int q = 0; q < 4; q++) {
                    float4 v = src[q];
                    for (int r = 0; r < 4; r++) {
                        const uint kk = kh + 4 * q + r;
                        put(dst, (kk / 8) * 512 + (kk % 8) * 8, v[r]);
                    }
                }
            } else {
                for (int q = 0; q < 16; q++) {
                    const uint kk = kh + q, gk = k0 + kk;
                    put(dst, (kk / 8) * 512 + (kk % 8) * 8, (gk < p.K && gn < p.N) ? B[gn * p.ldb + gk] : 0.0f);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kb = 0; kb < BK / 8; kb++) {
            simdgroup_matrix<T, 8, 8> a[4], b[4];
            for (int i = 0; i < 4; i++) simdgroup_load(a[i], As + (kb * 8 + mb0 + i) * 64, 8);
            for (int j = 0; j < 4; j++) simdgroup_load(b[j], Bs + (kb * 8 + nb0 + j) * 64, 8);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            if (SPLIT) {
                simdgroup_matrix<T, 8, 8> al[4], bl[4];
                for (int i = 0; i < 4; i++) simdgroup_load(al[i], Al + (kb * 8 + mb0 + i) * 64, 8);
                for (int j = 0; j < 4; j++) simdgroup_load(bl[j], Bl + (kb * 8 + nb0 + j) * 64, 8);
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], al[i], b[j], acc[i][j]);
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], bl[j], acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float *Cs = shf;  // BM x BN row-major, over the staging
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            simdgroup_store(acc[i][j], Cs + (8 * (mb0 + i)) * BN + 8 * (nb0 + j), BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool beta = p.beta & 1;
    if (vec && mfull && nfull) {
        for (uint e = tid; e < BM * BN / 4; e += 128) {
            const uint mm = e / (BN / 4), nn = (e % (BN / 4)) * 4;
            device float4 *d = (device float4 *)(C + (m0 + mm) * p.ldc + n0 + nn);
            float4 v = p.alpha * *(threadgroup float4 *)(Cs + mm * BN + nn);
            if (p.bias) v += *(device const float4 *)(bias + n0 + nn);
            if (beta) v += *d;
            *d = v;
        }
    } else {
        for (uint e = tid; e < BM * BN; e += 128) {
            const uint gm = m0 + e / BN, gn = n0 + e % BN;
            if (gm < p.M && gn < p.N) {
                float v = p.alpha * Cs[e];
                if (p.bias) v += bias[gn];
                if (beta) v += C[gm * p.ldc + gn];
                C[gm * p.ldc + gn] = v;
            }
        }
    }
}

template [[host_name("g2_gemm_nn")]] kernel void g2_gemm<false, false, float, false>(
    constant g2_gemm_args &, device const float *, device const float *, device float *, device const float *,
    threadgroup float *, uint3, ushort, ushort);
template [[host_name("g2_gemm_nt")]] kernel void g2_gemm<false, true, float, false>(
    constant g2_gemm_args &, device const float *, device const float *, device float *, device const float *,
    threadgroup float *, uint3, ushort, ushort);
template [[host_name("g2_gemm_tn")]] kernel void g2_gemm<true, false, float, false>(
    constant g2_gemm_args &, device const float *, device const float *, device float *, device const float *,
    threadgroup float *, uint3, ushort, ushort);
template [[host_name("g2_gemm_nn_b")]] kernel void g2_gemm<false, false, bfloat, false>(
    constant g2_gemm_args &, device const float *, device const float *, device float *, device const float *,
    threadgroup float *, uint3, ushort, ushort);
template [[host_name("g2_gemm_nt_b")]] kernel void g2_gemm<false, true, bfloat, false>(
    constant g2_gemm_args &, device const float *, device const float *, device float *, device const float *,
    threadgroup float *, uint3, ushort, ushort);
template [[host_name("g2_gemm_tn_b")]] kernel void g2_gemm<true, false, bfloat, false>(
    constant g2_gemm_args &, device const float *, device const float *, device float *, device const float *,
    threadgroup float *, uint3, ushort, ushort);
template [[host_name("g2_gemm_nn_b3")]] kernel void g2_gemm<false, false, bfloat, true>(
    constant g2_gemm_args &, device const float *, device const float *, device float *, device const float *,
    threadgroup float *, uint3, ushort, ushort);
template [[host_name("g2_gemm_nt_b3")]] kernel void g2_gemm<false, true, bfloat, true>(
    constant g2_gemm_args &, device const float *, device const float *, device float *, device const float *,
    threadgroup float *, uint3, ushort, ushort);
template [[host_name("g2_gemm_tn_b3")]] kernel void g2_gemm<true, false, bfloat, true>(
    constant g2_gemm_args &, device const float *, device const float *, device float *, device const float *,
    threadgroup float *, uint3, ushort, ushort);

// --- the rest ------------------------------------------------------------------

// args: n (rows), T, C
kernel void g2_encoder_fwd(constant uint *a [[buffer(0)]], device const int *inp [[buffer(1)]],
                           device const float *wte [[buffer(2)]], device const float *wpe [[buffer(3)]],
                           device float *out [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    const uint n = a[0], T = a[1], C = a[2];
    if (gid >= n * C) return;
    const uint bt = gid / C, i = gid % C;
    out[gid] = wte[inp[bt] * C + i] + wpe[(bt % T) * C + i];
}

// a thread a channel, over every row: the rows of a token add in order.
// dwte adds to the head's gradient, written first; dwpe is the first
// writer of its own unless ACC (gradients accumulating over micro-batches)
kernel void g2_encoder_bwd(constant uint *a [[buffer(0)]], device const int *inp [[buffer(1)]],
                           device float *dwte [[buffer(2)]], device float *dwpe [[buffer(3)]],
                           device const float *dout [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    const uint n = a[0], T = a[1], C = a[2], acc = a[3];
    if (i >= C) return;
    for (uint bt = 0; bt < n; bt++) {
        float d = dout[bt * C + i];
        dwte[inp[bt] * C + i] += d;
        if (bt < T && !acc) dwpe[bt * C + i] = d; else dwpe[(bt % T) * C + i] += d;
    }
}

// a simdgroup a row.  args: n, C
kernel void g2_layernorm_fwd(constant uint *a [[buffer(0)]], device const float *inp [[buffer(1)]],
                             device const float *w [[buffer(2)]], device const float *b [[buffer(3)]],
                             device float *out [[buffer(4)]], device float *mean [[buffer(5)]],
                             device float *rstd [[buffer(6)]],
                             uint r [[threadgroup_position_in_grid]], ushort l [[thread_index_in_simdgroup]]) {
    const uint n = a[0], C = a[1];
    if (r >= n) return;
    device const float *x = inp + r * C;
    float m = 0.0f;
    for (uint i = l; i < C; i += 32) m += x[i];
    m = simd_sum(m) / C;
    float v = 0.0f;
    for (uint i = l; i < C; i += 32) { float d = x[i] - m; v += d * d; }
    v = simd_sum(v) / C;
    const float s = 1.0f / precise::sqrt(v + 1e-5f);
    for (uint i = l; i < C; i += 32) out[r * C + i] = (s * (x[i] - m)) * w[i] + b[i];
    if (l == 0) { mean[r] = m; rstd[r] = s; }
}

// the input's gradient, a simdgroup a row: added to DINP if ADD, else written
kernel void g2_layernorm_bwd_x(constant uint *a [[buffer(0)]], device float *dinp [[buffer(1)]],
                               device const float *dout [[buffer(2)]], device const float *inp [[buffer(3)]],
                               device const float *w [[buffer(4)]], device const float *mean [[buffer(5)]],
                               device const float *rstd [[buffer(6)]],
                               uint r [[threadgroup_position_in_grid]], ushort l [[thread_index_in_simdgroup]]) {
    const uint n = a[0], C = a[1];
    if (r >= n) return;
    const float mu = mean[r], rs = rstd[r];
    device const float *x = inp + r * C;
    device const float *d = dout + r * C;
    float dm = 0.0f, dnm = 0.0f;
    for (uint i = l; i < C; i += 32) {
        float nrm = (x[i] - mu) * rs, dn = w[i] * d[i];
        dm += dn; dnm += dn * nrm;
    }
    dm = simd_sum(dm) / C; dnm = simd_sum(dnm) / C;
    for (uint i = l; i < C; i += 32) {
        float nrm = (x[i] - mu) * rs, dn = w[i] * d[i];
        float g = ((dn - dm) - nrm * dnm) * rs;
        dinp[r * C + i] = a[2] ? dinp[r * C + i] + g : g;
    }
}

// Column sums over N rows: a threadgroup of 1024 for 32 columns, 32 lanes
// of rows each (row r on lane r mod 32, so each 32 columns read in a row),
// the lanes' partial sums added in order through threadgroup memory.

// the layernorm weight's and bias's gradients.  args: n, C, acc
kernel void g2_layernorm_bwd_w(constant uint *a [[buffer(0)]], device float *dw [[buffer(1)]],
                               device float *db [[buffer(2)]], device const float *dout [[buffer(3)]],
                               device const float *inp [[buffer(4)]], device const float *mean [[buffer(5)]],
                               device const float *rstd [[buffer(6)]], threadgroup float *sh [[threadgroup(0)]],
                               uint g [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint n = a[0], C = a[1], acc = a[2];
    const uint c = g * 32 + tid % 32, lane = tid / 32;
    float gw = 0.0f, gb = 0.0f;
    if (c < C)
        for (uint r = lane; r < n; r += 32) {
            float d = dout[r * C + c];
            gb += d;
            gw += ((inp[r * C + c] - mean[r]) * rstd[r]) * d;
        }
    sh[lane * 33 + tid % 32] = gw;
    sh[32 * 33 + lane * 33 + tid % 32] = gb;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0 && c < C) {
        float sw = 0.0f, sb = 0.0f;
        for (uint k = 0; k < 32; k++) { sw += sh[k * 33 + tid]; sb += sh[32 * 33 + k * 33 + tid]; }
        dw[c] = acc ? dw[c] + sw : sw;
        db[c] = acc ? db[c] + sb : sb;
    }
}

// a bias's gradient: each column's sum.  args: n, OC, acc
kernel void g2_colsum(constant uint *a [[buffer(0)]], device float *db [[buffer(1)]],
                      device const float *dout [[buffer(2)]], threadgroup float *sh [[threadgroup(0)]],
                      uint g [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint n = a[0], OC = a[1], acc = a[2];
    const uint o = g * 32 + tid % 32, lane = tid / 32;
    float s = 0.0f;
    if (o < OC)
        for (uint r = lane; r < n; r += 32) s += dout[r * OC + o];
    sh[lane * 33 + tid % 32] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0 && o < OC) {
        float t = 0.0f;
        for (uint k = 0; k < 32; k++) t += sh[k * 33 + tid];
        db[o] = acc ? db[o] + t : t;
    }
}

// the causal softmax of each (b, h, t) row of scores, already scaled: a
// simdgroup a row.  args: rows, T
kernel void g2_attn_softmax(constant uint *a [[buffer(0)]], device const float *pre [[buffer(1)]],
                            device float *att [[buffer(2)]],
                            uint r [[threadgroup_position_in_grid]], ushort l [[thread_index_in_simdgroup]]) {
    const uint rows = a[0], T = a[1];
    if (r >= rows) return;
    const uint tq = r % T;
    device const float *p = pre + r * T;
    device float *o = att + r * T;
    float mx = -10000.0f;
    for (uint t = l; t <= tq; t += 32) mx = max(mx, p[t]);
    mx = simd_max(mx);
    float s = 0.0f;
    for (uint t = l; t <= tq; t += 32) s += precise::exp(p[t] - mx);
    s = simd_sum(s);
    const float inv = s == 0.0f ? 0.0f : 1.0f / s;
    for (uint t = l; t < T; t += 32) o[t] = t <= tq ? precise::exp(p[t] - mx) * inv : 0.0f;
}

// its gradient: dpre = att (datt - sum att datt)
kernel void g2_attn_softmax_bwd(constant uint *a [[buffer(0)]], device float *dpre [[buffer(1)]],
                                device const float *datt [[buffer(2)]], device const float *att [[buffer(3)]],
                                uint r [[threadgroup_position_in_grid]], ushort l [[thread_index_in_simdgroup]]) {
    const uint rows = a[0], T = a[1];
    if (r >= rows) return;
    const uint tq = r % T;
    device const float *p = att + r * T;
    device const float *d = datt + r * T;
    float s = 0.0f;
    for (uint t = l; t <= tq; t += 32) s += p[t] * d[t];
    s = simd_sum(s);
    for (uint t = l; t < T; t += 32) dpre[r * T + t] = t <= tq ? p[t] * (d[t] - s) : 0.0f;
}

constant float GELU_K = 0.7978845608028654f;  // sqrt(2/pi)

// tanh by one exp: 1 - 2/(e^2u + 1), exact at the limits (e^2u is 0 or inf)
static inline float g2_tanh(float u) { return 1.0f - 2.0f / (fast::exp(2.0f * u) + 1.0f); }

kernel void g2_gelu_fwd(constant uint *a [[buffer(0)]], device const float *inp [[buffer(1)]],
                        device float *out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= a[0]) return;
    float x = inp[i], cube = 0.044715f * x * x * x;
    out[i] = 0.5f * x * (1.0f + g2_tanh(GELU_K * (x + cube)));
}

kernel void g2_gelu_bwd(constant uint *a [[buffer(0)]], device float *dinp [[buffer(1)]],
                        device const float *inp [[buffer(2)]], device const float *dout [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= a[0]) return;
    float x = inp[i], cube = 0.044715f * x * x * x, arg = GELU_K * (x + cube);
    float th = g2_tanh(arg), sech = 1.0f - th * th;
    float g = 0.5f * (1.0f + th) + x * 0.5f * sech * GELU_K * (1.0f + 3.0f * 0.044715f * x * x);
    dinp[i] = g * dout[i];
}

kernel void g2_add(constant uint *a [[buffer(0)]], device float *out [[buffer(1)]],
                   device const float *x [[buffer(2)]], device const float *y [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= a[0]) return;
    out[i] = x[i] + y[i];
}

// a residual's gradient to both its inputs, their first writer
kernel void g2_copy2(constant uint *a [[buffer(0)]], device float *dx [[buffer(1)]],
                     device float *dy [[buffer(2)]], device const float *d [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= a[0]) return;
    dx[i] = d[i]; dy[i] = d[i];
}

kernel void g2_zero(constant uint *a [[buffer(0)]], device float *x [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < a[0]) x[i] = 0.0f;
}

// the vocabulary's softmax and each row's loss: a threadgroup of 1024 a row.
// args: n, V, Vp, whether there are targets
kernel void g2_softmax_ce(constant uint *a [[buffer(0)]], device const float *logits [[buffer(1)]],
                          device float *probs [[buffer(2)]], device const int *targets [[buffer(3)]],
                          device float *losses [[buffer(4)]], threadgroup float *red [[threadgroup(0)]],
                          uint r [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]], ushort l [[thread_index_in_simdgroup]]) {
    const uint V = a[1], Vp = a[2];
    device const float *x = logits + r * Vp;
    device float *p = probs + r * Vp;
    float mx = -10000.0f;
    for (uint i = tid; i < V; i += 1024) mx = max(mx, x[i]);
    mx = simd_max(mx);
    if (l == 0) red[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = simd_max(red[l]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint i = tid; i < V; i += 1024) { float e = precise::exp(x[i] - mx); p[i] = e; s += e; }
    s = simd_sum(s);
    if (l == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = simd_sum(red[l]);
    for (uint i = tid; i < V; i += 1024) p[i] = p[i] / s;
    for (uint i = V + tid; i < Vp; i += 1024) p[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0 && a[3]) losses[r] = -precise::log(p[targets[r]]);
}

// args: n, V, Vp, dloss (float)
kernel void g2_ce_bwd(constant uint *a [[buffer(0)]], device float *dlogits [[buffer(1)]],
                      device const float *probs [[buffer(2)]], device const int *targets [[buffer(3)]],
                      uint2 g [[thread_position_in_grid]]) {
    const uint V = a[1], Vp = a[2];
    const float dl = as_type<float>(a[3]);
    const uint i = g.x, r = g.y;
    if (i >= Vp || r >= a[0]) return;
    dlogits[r * Vp + i] = i < V ? (probs[r * Vp + i] - (i == (uint)targets[r] ? 1.0f : 0.0f)) * dl : 0.0f;
}

// AdamW.  args: count, lr, beta1, beta2, eps, weight decay, 1-beta1^t, 1-beta2^t
kernel void g2_adamw(constant uint *a [[buffer(0)]], device float *p [[buffer(1)]],
                     device const float *g [[buffer(2)]], device float *m [[buffer(3)]],
                     device float *v [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= a[0]) return;
    const float lr = as_type<float>(a[1]), b1 = as_type<float>(a[2]), b2 = as_type<float>(a[3]);
    const float eps = as_type<float>(a[4]), wd = as_type<float>(a[5]);
    const float c1 = as_type<float>(a[6]), c2 = as_type<float>(a[7]);
    float w = p[i], gr = g[i];
    float mi = b1 * m[i] + (1.0f - b1) * gr;
    float vi = b2 * v[i] + (1.0f - b2) * gr * gr;
    m[i] = mi; v[i] = vi;
    p[i] = w - lr * ((mi / c1) / (precise::sqrt(vi / c2) + eps) + wd * w);
}
