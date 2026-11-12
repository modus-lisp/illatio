// dense.metal — the dense models (qwen2, qwen3, qwen3vl's text model, llama)
// on the GPU, Q8_0: what metal-dense.lisp adds to the products of ds4f.metal
// (d4_mm_q8_0, d4_mv_q8_0) and qwen35.metal's rmsnorm, add and swiglu.
//
//   dn_row_q8(cols, row_bytes, token)        OUT = row TOKEN of a Q8_0 matrix, as floats
//   dn_bias(count, cols)                     Y += B, per column
//   dn_qk_rope(...)                          a head of Q or K: RMSNorm (qwen3), RoPE; Q to halves,
//                                            K and V to the cache (halves) at their positions
//   dn_attend_h64, dn_attend_h128            causal attention over the cache, GQA, flash style

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
