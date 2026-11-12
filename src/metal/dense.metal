// dense.metal — the dense models (qwen2, qwen3, qwen3vl's text model, llama)
// on the GPU, Q8_0: what metal-dense.lisp adds to the products of ds4f.metal
// (d4_mm_q8_0, d4_mv_q8_0) and qwen35.metal's rmsnorm, add and swiglu.
//
//   dn_row_q8(cols, row_bytes, token)        OUT = row TOKEN of a Q8_0 matrix, as floats
//   dn_bias(count, cols)                     Y += B, per column
//   dn_qk_rope(...)                          a head of Q or K: RMSNorm (qwen3), RoPE; Q to halves,
//                                            K and V to the cache (halves) at their positions
//   dn_attend_h64, dn_attend_h128            causal attention over the cache, GQA, flash style
//   dn_attend_split<D,G>, dn_attend_merge     one token's attention, split over the keys
//   dn_steer                                 directional steering of a layer's output rows
//   dn_mv_q4_k, dn_row_q4_k                  Q4_K: one token's product, an embedding row
//                                            (a prefill's products: ds4f.metal's d4_mm_q4_k)

#include <metal_stdlib>
using namespace metal;

struct dn_row_args { uint cols; uint row_bytes; uint token; };

kernel void dn_row_q8(constant dn_row_args & a [[buffer(0)]], device const char * w [[buffer(1)]],
                      device float * out [[buffer(2)]], uint j [[thread_position_in_grid]]) {
    if (j >= a.cols) return;
    device const char * blk = w + (ulong)a.token * a.row_bytes + (j / 32) * 34;
    const float d = (float)*(device const half *)blk;
    out[j] = d * (float)((device const char *)(blk + 2))[j % 32];
}

struct dn_bias_args { uint count; uint cols; };

kernel void dn_bias(constant dn_bias_args & a [[buffer(0)]], device float * y [[buffer(1)]],
                    device const float * b [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < a.count) y[i] += b[i % a.cols];
}

struct dn_qk_args { uint nh; uint nkv; uint hd; uint rd; uint pos0; float eps; uint neox; uint qnorm; uint knorm; };

// grid (nh + nkv, n), threads hd/2: head H of token T.  Q's heads to QH (rows
// of nh x hd halves); K's, with V's, to the caches at position POS0 + T.
// ROPE: rows of RD floats a position (cosines, then sines).
kernel void dn_qk_rope(constant dn_qk_args & a [[buffer(0)]],
                       device const float * q [[buffer(1)]], device const float * k [[buffer(2)]],
                       device const float * v [[buffer(3)]],
                       device const float * qn [[buffer(4)]], device const float * kn [[buffer(5)]],
                       device const float * rope [[buffer(6)]],
                       device half * qh [[buffer(7)]], device half * kc [[buffer(8)]], device half * vc [[buffer(9)]],
                       uint2 tg [[threadgroup_position_in_grid]], ushort i [[thread_index_in_threadgroup]],
                       uint2 ntg2 [[threads_per_threadgroup]]) {
    const ushort ntg = ntg2.x;
    threadgroup float xs[256];
    threadgroup float red[32];
    const uint head = tg.x, t = tg.y, pos = a.pos0 + t, hd = a.hd, half_ = a.rd / 2;
    const uint qd = a.nh * hd, kvd = a.nkv * hd;
    const bool isq = head < a.nh;
    device const float * x = isq ? q + (ulong)t * qd + head * hd : k + (ulong)t * kvd + (head - a.nh) * hd;
    for (uint j = i; j < hd; j += ntg) xs[j] = x[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (isq ? a.qnorm : a.knorm) {
        device const float * nw = isq ? qn : kn;
        float s = 0.f;
        for (uint j = i; j < hd; j += ntg) s += xs[j] * xs[j];
        s = simd_sum(s);
        if ((i & 31) == 0) red[i >> 5] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.f;
        for (uint w = 0; w < (ntg + 31) / 32; ++w) tot += red[w];
        const float sc = 1.0f / sqrt(tot / (float)hd + a.eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = i; j < hd; j += ntg) xs[j] = xs[j] * sc * nw[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device const float * cs = rope + (ulong)pos * a.rd;
    for (uint p = i; p < half_; p += ntg) {
        const uint ia = a.neox ? p : 2 * p, ib = a.neox ? p + half_ : 2 * p + 1;
        const float x0 = xs[ia], x1 = xs[ib], c = cs[p], s = cs[half_ + p];
        xs[ia] = x0 * c - x1 * s;
        xs[ib] = x0 * s + x1 * c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (isq) {
        for (uint j = i; j < hd; j += ntg) qh[(ulong)t * qd + head * hd + j] = (half)xs[j];
    } else {
        const uint kh = head - a.nh;
        const ulong o = (ulong)pos * kvd + kh * hd;
        for (uint j = i; j < hd; j += ntg) {
            kc[o + j] = (half)xs[j];
            vc[o + j] = (half)v[(ulong)t * kvd + kh * hd + j];
        }
    }
}

struct dn_attn_args { uint n; uint pos0; uint nh; uint nkv; float scale; };

// Causal attention, flash style, on the half simdgroup matrices: a
// threadgroup takes 64 query rows (tokens T0.., at positions POS0 + T) of
// one head, its eight simdgroups 8 rows each; the keys go 32 a step through
// threadgroup memory, shared by the eight; a query sees keys 0 .. its
// position.  S = Q K^T and O += P V are half products summed in floats, P
// rounded to halves, the softmax kept running per row (four lanes a row,
// eight keys each), O rescaled (a product with a diagonal) only when a
// row's maximum moved.  QH: rows of NH x D halves; KC, VC: the caches, rows
// of NKV x D halves; OUT rows of NH x D floats.  grid (nh, ceil(n/64)),
// threads 256, shared 4 KB * D / 32 + 16 KB.
template <short D>
kernel void dn_attend_h(constant dn_attn_args & a [[buffer(0)]],
                        device const half * q [[buffer(1)]], device const half * kc [[buffer(2)]],
                        device const half * vc [[buffer(3)]], device float * out [[buffer(4)]],
                        threadgroup char * shmem [[threadgroup(0)]],
                        uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr short KB = 32, SG = 8;
    const uint h = tgp.x, kvh = h / (a.nh / a.nkv), tq0 = tgp.y * 8 * SG, t0 = tq0 + sg * 8;
    const uint qd = a.nh * D, kvd = a.nkv * D;
    threadgroup half * ks = (threadgroup half *)shmem;                  // 32 x D
    threadgroup half * vs = ks + KB * D;                                // 32 x D
    threadgroup char * mine = shmem + 2 * KB * D * 2 + sg * 2048;       // this simdgroup's 2 KB
    threadgroup float * ss = (threadgroup float *)mine;                 // 8 x 32: S
    threadgroup float * dg = ss + 8 * KB;                               // 8 x 8: a diagonal
    threadgroup half * ps = (threadgroup half *)(dg + 64);              // 8 x 32: P

    simdgroup_half8x8 qm[D / 8];
    {
        threadgroup half * qs = (threadgroup half *)mine;              // 8 x D halves: at most 2 KB
        for (uint i = lane; i < 8 * D; i += 32) {
            const uint rr = min(t0 + i / D, a.n - 1);
            qs[i] = q[(ulong)rr * qd + h * D + (i % D)];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (short c = 0; c < D / 8; ++c) simdgroup_load(qm[c], qs + c * 8, D);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = lane; i < 64; i += 32) dg[i] = 0.f;

    simdgroup_float8x8 o[D / 8];
    for (short c = 0; c < D / 8; ++c) o[c] = make_filled_simdgroup_matrix<float, 8>(0.f);

    const uint r = lane / 4, part = lane % 4;
    const bool live = t0 + r < a.n;
    const uint lim = a.pos0 + min(t0 + r, a.n - 1);
    const uint last = a.pos0 + min(tq0 + 8 * SG - 1, a.n - 1);
    float m = -INFINITY, lsum = 0.f;

    for (uint kb = 0; kb <= last; kb += KB) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < KB * D / 8; i += 32 * SG) {
            const uint key = i / (D / 8), c = (i % (D / 8)) * 8, kp = kb + key;
            half4 k0 = half4(0.h), k1 = half4(0.h), v0 = half4(0.h), v1 = half4(0.h);
            if (kp <= last) {
                device const half4 * kq = (device const half4 *)(kc + (ulong)kp * kvd + kvh * D + c);
                device const half4 * vq = (device const half4 *)(vc + (ulong)kp * kvd + kvh * D + c);
                k0 = kq[0]; k1 = kq[1]; v0 = vq[0]; v1 = vq[1];
            }
            *(threadgroup half4 *)(ks + key * D + c) = k0;
            *(threadgroup half4 *)(ks + key * D + c + 4) = k1;
            *(threadgroup half4 *)(vs + key * D + c) = v0;
            *(threadgroup half4 *)(vs + key * D + c + 4) = v1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8>(0.f), s1 = s0, s2 = s0, s3 = s0;
        for (short c = 0; c < D / 8; ++c) {
            simdgroup_half8x8 k0, k1, k2, k3;
            simdgroup_load(k0, ks + c * 8,          D, 0, true);
            simdgroup_load(k1, ks + 8 * D + c * 8,  D, 0, true);
            simdgroup_load(k2, ks + 16 * D + c * 8, D, 0, true);
            simdgroup_load(k3, ks + 24 * D + c * 8, D, 0, true);
            simdgroup_multiply_accumulate(s0, qm[c], k0, s0);
            simdgroup_multiply_accumulate(s1, qm[c], k1, s1);
            simdgroup_multiply_accumulate(s2, qm[c], k2, s2);
            simdgroup_multiply_accumulate(s3, qm[c], k3, s3);
        }
        simdgroup_store(s0, ss,      KB);
        simdgroup_store(s1, ss + 8,  KB);
        simdgroup_store(s2, ss + 16, KB);
        simdgroup_store(s3, ss + 24, KB);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float v[8], mx = -INFINITY;
        for (short i = 0; i < 8; ++i) {
            const uint key = kb + part * 8 + i;
            v[i] = (live && key <= lim) ? ss[r * KB + part * 8 + i] * a.scale : -INFINITY;
            mx = max(mx, v[i]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float mn = max(m, mx);
        float ls = 0.f;
        for (short i = 0; i < 8; ++i) {
            const float e = mn == -INFINITY ? 0.f : precise::exp(v[i] - mn);
            ps[r * KB + part * 8 + i] = (half)e;
            ls += e;
        }
        ls += simd_shuffle_xor(ls, 1);
        ls += simd_shuffle_xor(ls, 2);
        const float f = mn == -INFINITY ? 1.f : precise::exp(m - mn);
        lsum = lsum * f + ls;
        m = mn;
        if (part == 0) dg[r * 8 + r] = f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_any(f != 1.f)) {
            simdgroup_float8x8 dm;
            simdgroup_load(dm, dg, 8);
            for (short c = 0; c < D / 8; ++c) simdgroup_multiply(o[c], dm, o[c]);
        }
        for (short kk = 0; kk < KB / 8; ++kk) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, ps + kk * 8, KB);
            for (short c = 0; c < D / 8; ++c) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, vs + kk * 8 * D + c * 8, D);
                simdgroup_multiply_accumulate(o[c], pm, vm, o[c]);
            }
        }
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0) dg[r * 8 + r] = lsum > 0.f ? 1.f / lsum : 0.f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dm;
    simdgroup_load(dm, dg, 8);
    for (short c = 0; c < D / 8; ++c) simdgroup_multiply(o[c], dm, o[c]);
    if (t0 + 8 <= a.n) {
        for (short c = 0; c < D / 8; ++c) simdgroup_store(o[c], out + (ulong)t0 * qd + h * D + c * 8, qd);
    } else if (t0 < a.n) {
        threadgroup float * stage = (threadgroup float *)mine;          // 8 x 64 floats
        for (short hf = 0; hf < D / 64; ++hf) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (short c = 0; c < 8; ++c) simdgroup_store(o[hf * 8 + c], stage + c * 8, 64);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < 8 * 64; i += 32) {
                const uint rr = t0 + i / 64;
                if (rr < a.n) out[(ulong)rr * qd + h * D + hf * 64 + (i % 64)] = stage[i];
            }
        }
    }
}

typedef decltype(dn_attend_h<64>) dn_attend_t;
template [[host_name("dn_attend_h64")]]  kernel dn_attend_t dn_attend_h<64>;
template [[host_name("dn_attend_h128")]] kernel dn_attend_t dn_attend_h<128>;

// a threadgroup's sum, in every thread (qwen35.metal's tg_sum, which this file is compiled before)
static inline float dn_tg_sum(float v, threadgroup float * sh, uint tid, uint ntg) {
    v = simd_sum(v);
    const uint lane = tid % 32, sg = tid / 32, nsg = (ntg + 31) / 32;
    if (lane == 0) sh[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? sh[lane] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

struct dn_steer_args { uint d; float scale; uint cap_row; uint cap; };

// Directional steering (steer.lisp), a row of D a threadgroup: row CAP_ROW
// first copied to CAPTURE when CAP (taken before steering), then each row
// y <- y - SCALE u (u . y), U the layer's unit direction (unread at SCALE 0).
kernel void dn_steer(constant dn_steer_args & a [[buffer(0)]], device float * y [[buffer(1)]],
                     device const float * u [[buffer(2)]], device float * capture [[buffer(3)]],
                     uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                     uint ntg [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    device float * row = y + (ulong)t * a.d;
    if (a.cap && t == a.cap_row)
        for (uint i = tid; i < a.d; i += ntg) capture[i] = row[i];
    if (a.scale == 0.f) return;
    float s = 0.f;
    for (uint i = tid; i < a.d; i += ntg) s += u[i] * row[i];
    const float k = a.scale * dn_tg_sum(s, sh, tid, ntg);
    for (uint i = tid; i < a.d; i += ntg) row[i] -= k * u[i];
}

// --- Q4_K: 256 weights a block in four chunks of 64 (the low nibbles of 32
// bytes, then their high nibbles), eight sub-blocks of 32 each with a 6-bit
// scale and min packed in 12 bytes: w = d sc q - dmin m ----------------------

typedef struct { half d; half dmin; uint8_t scales[12]; uint8_t qs[128]; } dn_block_q4_K;

// sub-block J's (scale, min), as ggml's get_scale_min_k4
static inline uchar2 dn_q4k_scale(int j, device const uint8_t * q) {
    return j < 4 ? uchar2(q[j] & 63, q[j + 4] & 63)
                 : uchar2((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4));
}

struct dn_mv_args { uint ncols; uint nrows; };

// Y = W X for one token, W's Q4_K rows, in ggml's layout (its
// kernel_mul_mv_q4_K_f32): a simdgroup two rows, eight lanes a block, four
// blocks a step.  Lane (IQ, IR) of a block takes 8 bytes of chunk IQ and 8
// of chunk IQ+2 at 8 IR, both nibbles of each: 32 weights of sub-blocks 2IQ,
// 2IQ+1, 2IQ+4, 2IQ+5, whose scales and mins come out of three 16-bit words
// by masks.  X's 32 floats a lane are shared by the two rows.
// grid ceil(rows/4), threads 64.
#define DN_NR 2
kernel void dn_mv_q4_k(constant dn_mv_args & a [[buffer(0)]], device const char * w [[buffer(1)]],
                       device const float * x [[buffer(2)]], device float * y [[buffer(3)]],
                       uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    const uint r0 = (tg * 2 + sg) * DN_NR;
    if (r0 >= a.nrows) return;
    const uint nb = a.ncols / 256;
    const ulong row_bytes = (ulong)nb * sizeof(dn_block_q4_K);
    const ushort ix = lane / 8, it = lane % 8, iq = it / 4, ir = it % 4;
    float s[DN_NR];
    for (short rr = 0; rr < DN_NR; ++rr) s[rr] = 0.f;
    float yl[16], yh[16];
    for (uint b = ix; b < nb; b += 4) {
        device const float * y4 = x + b * 256 + 64 * iq + 8 * ir;
        float4 sumy = 0.f;
        for (short i = 0; i < 8; ++i) {
            yl[i] = y4[i];        sumy[0] += yl[i];
            yl[i + 8] = y4[i + 32];  sumy[1] += yl[i + 8];
            yh[i] = y4[i + 128];  sumy[2] += yh[i];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }
        for (short rr = 0; rr < DN_NR; ++rr) {
            if (r0 + rr >= a.nrows) break;
            device const dn_block_q4_K * blk = (device const dn_block_q4_K *)(w + (r0 + rr) * row_bytes) + b;
            device const ushort * sc = (device const ushort *)blk->scales + iq;
            const ushort A = sc[0], B = sc[2], C = sc[4];
            ushort s16[4];
            s16[0] = A & 0x3f3f;                                     // scales of sub-blocks 2IQ, 2IQ+1
            s16[1] = B & 0x3f3f;                                     // their mins
            s16[2] = (C & 0x0f0f) | ((A & 0xc0c0) >> 2);             // scales of 2IQ+4, 2IQ+5
            s16[3] = ((C >> 4) & 0x0f0f) | ((B & 0xc0c0) >> 2);      // their mins
            thread const uchar * s8 = (thread const uchar *)s16;
            const uint2 q1 = *(device const uint2 *)(blk->qs + 32 * iq + 8 * ir);
            const uint2 q2 = *(device const uint2 *)(blk->qs + 32 * iq + 64 + 8 * ir);
            const float4 l1a = float4(as_type<uchar4>(q1.x & 0x0F0F0F0Fu)), l1b = float4(as_type<uchar4>(q1.y & 0x0F0F0F0Fu));
            const float4 h1a = float4(as_type<uchar4>((q1.x >> 4) & 0x0F0F0F0Fu)), h1b = float4(as_type<uchar4>((q1.y >> 4) & 0x0F0F0F0Fu));
            const float4 l2a = float4(as_type<uchar4>(q2.x & 0x0F0F0F0Fu)), l2b = float4(as_type<uchar4>(q2.y & 0x0F0F0F0Fu));
            const float4 h2a = float4(as_type<uchar4>((q2.x >> 4) & 0x0F0F0F0Fu)), h2b = float4(as_type<uchar4>((q2.y >> 4) & 0x0F0F0F0Fu));
            const float lo1 = dot(l1a, float4(yl[0], yl[1], yl[2], yl[3])) + dot(l1b, float4(yl[4], yl[5], yl[6], yl[7]));
            const float hi1 = dot(h1a, float4(yl[8], yl[9], yl[10], yl[11])) + dot(h1b, float4(yl[12], yl[13], yl[14], yl[15]));
            const float lo2 = dot(l2a, float4(yh[0], yh[1], yh[2], yh[3])) + dot(l2b, float4(yh[4], yh[5], yh[6], yh[7]));
            const float hi2 = dot(h2a, float4(yh[8], yh[9], yh[10], yh[11])) + dot(h2b, float4(yh[12], yh[13], yh[14], yh[15]));
            const half2 dd = *(device const half2 *)blk;
            s[rr] += (float)dd[0] * (s8[0] * lo1 + s8[1] * hi1 + s8[4] * lo2 + s8[5] * hi2)
                   - (float)dd[1] * (s8[2] * sumy[0] + s8[3] * sumy[1] + s8[6] * sumy[2] + s8[7] * sumy[3]);
        }
    }
    for (short rr = 0; rr < DN_NR; ++rr) {
        const float t = simd_sum(s[rr]);
        if (lane == 0 && r0 + rr < a.nrows) y[r0 + rr] = t;
    }
}

struct dn_rowk_args { uint cols; uint row_bytes; uint token; };

// OUT = row TOKEN of a Q4_K matrix, as floats (an embedding): a thread a weight
kernel void dn_row_q4_k(constant dn_rowk_args & a [[buffer(0)]], device const char * w [[buffer(1)]],
                        device float * out [[buffer(2)]], uint j [[thread_position_in_grid]]) {
    if (j >= a.cols) return;
    device const dn_block_q4_K * blk = (device const dn_block_q4_K *)(w + (ulong)a.token * a.row_bytes) + j / 256;
    const uint k = j % 256, c = k / 64, pos = k % 64;
    const uint8_t q = blk->qs[c * 32 + (pos % 32)];
    const uchar2 sm = dn_q4k_scale(k / 32, blk->scales);
    out[j] = (float)blk->d * sm[0] * (float)(pos >= 32 ? (q >> 4) : (q & 0xF)) - (float)blk->dmin * sm[1];
}

// --- decode attention, split over the keys -------------------------------------
//
// One token's attention as flash decoding: a threadgroup per KV head and
// chunk of positions, its GRP query heads at once (K and V read once for
// them).  Each of its four simdgroups walks every fourth position of the
// chunk, lanes across the head's dimensions (D/32 a lane, read 8 bytes at a
// time at 128), keeping each query head's softmax and output running; the
// four are then combined, and the chunk's unnormalized output, max and sum
// written per query head for dn_attend_merge.  grid (nkv, nsplit),
// threads 128, shared 4 * GRP * (D + 2) floats.

struct dn_split_args { uint nkv; uint grp; uint n; uint chunk; uint nsplit; float scale; };

template <short D, short G>
kernel void dn_attend_split(constant dn_split_args & a [[buffer(0)]],
                            device const half * q [[buffer(1)]],
                            device const half * kc [[buffer(2)]], device const half * vc [[buffer(3)]],
                            device float * part [[buffer(4)]], device float * ml [[buffer(5)]],
                            threadgroup float * shm [[threadgroup(0)]],
                            uint2 tgp [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr short PD = D / 32, NSG = 4;
    const uint g = tgp.x, c = tgp.y, p0 = c * a.chunk, p1 = min(p0 + a.chunk, a.n);
    const uint kvd = a.nkv * D;
    float qv[G][PD], o[G][PD], m[G], l[G];
    for (short hh = 0; hh < G; ++hh) {
        for (short i = 0; i < PD; ++i) { qv[hh][i] = (float)q[(g * G + hh) * D + lane * PD + i] * a.scale; o[hh][i] = 0.f; }
        m[hh] = -INFINITY; l[hh] = 0.f;
    }
    // UN positions' K and V loaded before any is used: more loads in flight a lane
    constexpr short UN = 4;
    for (uint pb = p0 + sg * UN; pb < p1; pb += NSG * UN) {
        float kv[UN][PD], vv[UN][PD];
        for (short u = 0; u < UN; ++u) {
            const uint p = min(pb + u, p1 - 1);
            device const half * kr = kc + (ulong)p * kvd + g * D + lane * PD;
            device const half * vr = vc + (ulong)p * kvd + g * D + lane * PD;
            for (short i = 0; i < PD; ++i) { kv[u][i] = (float)kr[i]; vv[u][i] = (float)vr[i]; }
        }
        for (short u = 0; u < UN && pb + u < p1; ++u) {
            for (short hh = 0; hh < G; ++hh) {
                float s = 0.f;
                for (short i = 0; i < PD; ++i) s += kv[u][i] * qv[hh][i];
                s = simd_sum(s);
                const float mn = max(m[hh], s), f = exp(m[hh] - mn), e = exp(s - mn);
                l[hh] = l[hh] * f + e;
                for (short i = 0; i < PD; ++i) o[hh][i] = o[hh][i] * f + e * vv[u][i];
                m[hh] = mn;
            }
        }
    }
    // the four simdgroups' partials: each writes its own, the first combines them
    threadgroup float * mine = shm + sg * G * (D + 2);
    for (short hh = 0; hh < G; ++hh) {
        for (short i = 0; i < PD; ++i) mine[hh * (D + 2) + lane * PD + i] = o[hh][i];
        if (lane == 0) { mine[hh * (D + 2) + D] = m[hh]; mine[hh * (D + 2) + D + 1] = l[hh]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    for (short hh = 0; hh < G; ++hh) {
        float M = -INFINITY;
        for (short k = 0; k < NSG; ++k) M = max(M, shm[(k * G + hh) * (D + 2) + D]);
        float L = 0.f, acc[PD];
        for (short i = 0; i < PD; ++i) acc[i] = 0.f;
        for (short k = 0; k < NSG; ++k) {
            threadgroup const float * pk = shm + (k * G + hh) * (D + 2);
            const float w = pk[D] == -INFINITY ? 0.f : exp(pk[D] - M);
            L += pk[D + 1] * w;
            for (short i = 0; i < PD; ++i) acc[i] += pk[lane * PD + i] * w;
        }
        const uint row = (g * G + hh) * a.nsplit + c;
        for (short i = 0; i < PD; ++i) part[row * D + lane * PD + i] = acc[i];
        if (lane == 0) { ml[row * 2] = M; ml[row * 2 + 1] = L; }
    }
}

// G, the query heads a KV head, a constant: the per-head arrays stay in registers
typedef decltype(dn_attend_split<64, 1>) dn_split_t;
#define DN_SPLIT(D, G) template [[host_name("dn_attend_split" #D "_" #G)]] kernel dn_split_t dn_attend_split<D, G>;
DN_SPLIT(64, 1) DN_SPLIT(64, 2) DN_SPLIT(64, 4) DN_SPLIT(64, 6) DN_SPLIT(64, 7) DN_SPLIT(64, 8)
DN_SPLIT(128, 1) DN_SPLIT(128, 2) DN_SPLIT(128, 4) DN_SPLIT(128, 6) DN_SPLIT(128, 7) DN_SPLIT(128, 8)

struct dn_merge_args { uint nsplit; uint d; };

// OUT's head H (grid x) = its chunks weighed together; threads D
kernel void dn_attend_merge(constant dn_merge_args & a [[buffer(0)]],
                            device const float * part [[buffer(1)]], device const float * ml [[buffer(2)]],
                            device float * out [[buffer(3)]],
                            uint h [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    device const float * mh = ml + h * a.nsplit * 2;
    float M = -INFINITY;
    for (uint c = 0; c < a.nsplit; ++c) M = max(M, mh[2 * c]);
    float L = 0.f, o = 0.f;
    for (uint c = 0; c < a.nsplit; ++c) {
        const float w = exp(mh[2 * c] - M);
        L += mh[2 * c + 1] * w;
        o += part[(h * a.nsplit + c) * a.d + i] * w;
    }
    out[h * a.d + i] = o / L;
}
