// qwen35moe.metal -- Qwen3.5/3.6 MoE (qwen35moe) on the GPU: the routed
// experts.  The rest of a layer is qwen35's.  A token's 8 experts are 8
// pairs (token t, slot k), p = t*8 + k: each reads its expert's rows (ids[p])
// and its token's row of X (or, X_PAIR, its own: the down projection's
// input), and writes Y's row p.  The matrix-vector kernels are qwen35's and
// the dense models' own (mv_q6k, dn_mv_q4_k, dn_mv_q5_k), generated from
// them, so their sums are those kernels' sums; grid (rows tiles, pairs).

struct q35x_args { uint ncols; uint nrows; uint x_stride; uint y_stride; uint x_pair; };
struct q35x6_args { uint ncols; uint nrows; uint row_bytes; uint x_stride; uint y_stride; uint x_pair; };

kernel void q35_xmv_q4_k(constant q35x_args & a [[buffer(0)]], device const char * w [[buffer(1)]],
                       device const float * x [[buffer(2)]], device float * y [[buffer(3)]], device const uint * ids [[buffer(4)]],
                       uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    const uint p = tg.y;
    w += (ulong)ids[p] * a.nrows * ((a.ncols / 256) * sizeof(dn_block_q4_K));
    x += (ulong)(a.x_pair ? p : p / 8) * a.x_stride;
    y += (ulong)p * a.y_stride;
    const uint r0 = (tg.x * 2 + sg) * DN_NR;
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

kernel void q35_xmv_q5_k(constant q35x_args & a [[buffer(0)]], device const char * w [[buffer(1)]],
                       device const float * x [[buffer(2)]], device float * y [[buffer(3)]], device const uint * ids [[buffer(4)]],
                       uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    const uint p = tg.y;
    w += (ulong)ids[p] * a.nrows * ((a.ncols / 256) * sizeof(dn_block_q5_K));
    x += (ulong)(a.x_pair ? p : p / 8) * a.x_stride;
    y += (ulong)p * a.y_stride;
    const uint r0 = (tg.x * 2 + sg) * DN_NR;
    if (r0 >= a.nrows) return;
    const uint nb = a.ncols / 256;
    const ulong row_bytes = (ulong)nb * sizeof(dn_block_q5_K);
    const uint k0 = lane * 8, c = k0 / 64, pos = k0 % 64, l = pos % 32;
    const bool high = pos >= 32;
    const uint shift = 2 * c + (high ? 1 : 0);
    float s[DN_NR];
    for (short rr = 0; rr < DN_NR; ++rr) s[rr] = 0.f;
    for (uint b = 0; b < nb; ++b) {
        device const float4 * xv = (device const float4 *)(x + b * 256 + k0);
        const float4 x0 = xv[0], x1 = xv[1];
        const float xs = x0[0] + x0[1] + x0[2] + x0[3] + x1[0] + x1[1] + x1[2] + x1[3];
        for (short rr = 0; rr < DN_NR; ++rr) {
            if (r0 + rr >= a.nrows) break;
            device const dn_block_q5_K * blk = (device const dn_block_q5_K *)(w + (r0 + rr) * row_bytes) + b;
            const uchar2 sm = dn_q4k_scale(k0 / 32, blk->scales);
            const uchar4 qa = *(device const uchar4 *)(blk->qs + c * 32 + l), qb = *(device const uchar4 *)(blk->qs + c * 32 + l + 4);
            const uchar4 ha = *(device const uchar4 *)(blk->qh + l),          hb = *(device const uchar4 *)(blk->qh + l + 4);
            const float4 wa = float4(high ? (qa >> 4) : (qa & 0xF)) + 16.f * float4((ha >> shift) & 1);
            const float4 wb = float4(high ? (qb >> 4) : (qb & 0xF)) + 16.f * float4((hb >> shift) & 1);
            s[rr] += (float)blk->d * sm[0] * (dot(wa, x0) + dot(wb, x1)) - (float)blk->dmin * sm[1] * xs;
        }
    }
    for (short rr = 0; rr < DN_NR; ++rr) {
        const float t = simd_sum(s[rr]);
        if (lane == 0 && r0 + rr < a.nrows) y[r0 + rr] = t;
    }
}

kernel void q35_xmv_q6k(constant q35x6_args & a [[buffer(0)]],
                   device const char  * w [[buffer(1)]],
                   device const float * x [[buffer(2)]],
                   device       float * y [[buffer(3)]],
                   device const uint * ids [[buffer(4)]],
                   uint3  tgpig [[threadgroup_position_in_grid]],
                   ushort tiisg [[thread_index_in_simdgroup]],
                   ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint8_t kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
    const int nb = a.ncols / QK_K;
    const int first_row = (tgpig.x * MV_NSG + sgitg) * MV_NR0;
    const uint p = tgpig.y;
    const uint r1 = a.x_pair ? p : p / 8;
    w += (ulong)ids[p] * a.nrows * a.row_bytes;

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

    device float * dst = y + (ulong)p * a.y_stride;
    for (int row = 0; row < MV_NR0 && first_row + row < (int)a.nrows; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0) dst[first_row + row] = s;
    }
}

struct q35_route_args { uint n_expert; };

// The router's choice for token TG: the 8 largest logits of N_EXPERT, their
// softmax weights renormalized over the 8 (llama.cpp's softmax gating with
// norm_w).  One threadgroup of 32.
kernel void q35_route(constant q35_route_args & a [[buffer(0)]], device const float * logits [[buffer(1)]],
                      device uint * ids [[buffer(2)]], device float * wts [[buffer(3)]],
                      uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    if (tid != 0) return;
    device const float * l = logits + (ulong)t * a.n_expert;
    uint pick[8]; float v[8];
    for (short k = 0; k < 8; ++k) {
        uint best = 0; float bv = -INFINITY;
        for (uint e = 0; e < a.n_expert; ++e) {
            bool taken = false;
            for (short j = 0; j < k; ++j) taken = taken || pick[j] == e;
            if (!taken && l[e] > bv) { bv = l[e]; best = e; }
        }
        pick[k] = best; v[k] = bv;
    }
    float z = 0.f; const float top = v[0];
    for (short k = 0; k < 8; ++k) { v[k] = exp(v[k] - top); z += v[k]; }
    for (short k = 0; k < 8; ++k) { ids[t * 8 + k] = pick[k]; wts[t * 8 + k] = v[k] / z; }
}

struct q35_mvf_args { uint ncols; uint nrows; uint ntok; };

// Y = W X for N tokens, W F32 row-major (the router; qwen35's ssm_alpha and
// ssm_beta in some files): a simdgroup a row, 8 tokens at a time.
// grid (ceil(rows/4), ceil(N/8)), threads 128.
kernel void q35_mv_f32_n(constant q35_mvf_args & a [[buffer(0)]], device const float * w [[buffer(1)]],
                         device const float * x [[buffer(2)]], device float * y [[buffer(3)]],
                         uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = tg.x * 4 + sg, t0 = tg.y * 8;
    if (r >= a.nrows) return;
    const uint nt = min(8u, a.ntok - t0);
    device const float4 * w4 = (device const float4 *)(w + (ulong)r * a.ncols);
    float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (uint i = lane; i < a.ncols / 4; i += 32) {
        const float4 wv = w4[i];
        for (uint k = 0; k < nt; ++k) s[k] += dot(wv, ((device const float4 *)(x + (ulong)(t0 + k) * a.ncols))[i]);
    }
    for (uint k = 0; k < nt; ++k) {
        const float v = simd_sum(s[k]);
        if (lane == 0) y[(ulong)(t0 + k) * a.nrows + r] = v;
    }
}

struct q35_dot_args { uint d; };

// S[t] = W . X[t] (the shared expert's gate): a threadgroup of 256 a token.
kernel void q35_dot(constant q35_dot_args & a [[buffer(0)]], device const float * w [[buffer(1)]],
                    device const float * x [[buffer(2)]], device float * s [[buffer(3)]],
                    uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[8];
    float v = 0.f;
    for (uint i = tid; i < a.d; i += 256) v += w[i] * x[(ulong)t * a.d + i];
    v = simd_sum(v);
    if (tid % 32 == 0) red[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float z = 0.f; for (short i = 0; i < 8; ++i) z += red[i]; s[t] = z; }
}

struct q35_comb_args { uint d; uint ntok; };

// Y[t] = sum_k W[t*8+k] YE[t*8+k] + sigmoid(SG[t]) SH[t]: the routed experts
// and the shared one, gated.
kernel void q35_moe_combine(constant q35_comb_args & a [[buffer(0)]], device const float * ye [[buffer(1)]],
                            device const float * w [[buffer(2)]], device const float * sh [[buffer(3)]],
                            device const float * sg [[buffer(4)]], device float * y [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, t = gid.y;
    if (i >= a.d || t >= a.ntok) return;
    float s = 0.f;
    for (short k = 0; k < 8; ++k) s += w[t * 8 + k] * ye[((ulong)t * 8 + k) * a.d + i];
    y[(ulong)t * a.d + i] = s + sh[(ulong)t * a.d + i] / (1.f + exp(-sg[t]));
}

// --- prefill: the pairs grouped by expert ------------------------------------------
//
// With many tokens each expert serves several pairs, so a prefill runs the
// routed experts as ds4f's does (ggml's mul_mm_id, ds4f.metal's d4_mm_id):
// Q35_GROUP orders the pairs by expert (COUNTS, BASES; slot BASES[e] + j is
// expert e's j-th pair, POS[p] pair p's slot, TOK[slot] its token), the
// tokens' rows are gathered into that order (d4_gather_rows), each expert
// multiplies its rows as a matrix product, and Q35_MOE_COMBINE_G sums each
// token's slots back.  One threadgroup, a thread an expert (N_EXPERT <= 1024).

struct q35_group_args { uint npairs; uint n_expert; };

kernel void q35_group(constant q35_group_args & a [[buffer(0)]], device const uint * ids [[buffer(1)]],
                      device uint * counts [[buffer(2)]], device uint * bases [[buffer(3)]],
                      device uint * pos [[buffer(4)]], device uint * tok [[buffer(5)]],
                      ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup uint cnt[1024];
    const uint e = tid;
    if (e < a.n_expert) {
        uint c = 0;
        for (uint p = 0; p < a.npairs; ++p) c += ids[p] == e ? 1u : 0u;
        cnt[e] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint b = 0;
        for (uint i = 0; i < a.n_expert; ++i) { const uint c = cnt[i]; cnt[i] = b; b += c; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e < a.n_expert) {
        const uint b = cnt[e];
        uint j = 0;
        for (uint p = 0; p < a.npairs; ++p)
            if (ids[p] == e) { pos[p] = b + j; tok[b + j] = p / 8; ++j; }
        counts[e] = j; bases[e] = b;
    }
}

// Q35_MOE_COMBINE with the experts' rows in grouped order (pair p's at POS[p]).
kernel void q35_moe_combine_g(constant q35_comb_args & a [[buffer(0)]], device const float * ye [[buffer(1)]],
                              device const float * w [[buffer(2)]], device const float * sh [[buffer(3)]],
                              device const float * sg [[buffer(4)]], device float * y [[buffer(5)]],
                              device const uint * pos [[buffer(6)]],
                              uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, t = gid.y;
    if (i >= a.d || t >= a.ntok) return;
    float s = 0.f;
    for (short k = 0; k < 8; ++k) s += w[t * 8 + k] * ye[(ulong)pos[t * 8 + k] * a.d + i];
    y[(ulong)t * a.d + i] = s + sh[(ulong)t * a.d + i] / (1.f + exp(-sg[t]));
}

template [[host_name("q35_mm_id_q4_k")]] kernel d4_mm_id_t d4_mm_id<dn_block_q4_K, 16, d4_deq_q4_K>;
template [[host_name("q35_mm_id_q5_k")]] kernel d4_mm_id_t d4_mm_id<dn_block_q5_K, 16, d4_deq_q5_K>;
// (qwen35.metal's dequantizer is static: a template argument wants it external)
void q35_deq_q6_K(device const block_q6_K * xb, short il, thread half4x4 & reg) { dequantize_q6_K(xb, il, reg); }
template [[host_name("q35_mm_id_q6_k")]] kernel d4_mm_id_t d4_mm_id<block_q6_K, 16, q35_deq_q6_K>;

// D4_MM_ID with 32-token tiles (MM_Q6K's, qwen35.metal: 64 rows by 32
// pairs a threadgroup): an expert's share of a 512-token pass is ~16 pairs,
// and a 64-wide tile would be three quarters empty.
template <typename block_q, short nl_, void (*deq)(device const block_q *, short, thread half4x4 &)>
kernel void q35_mm_id32(constant d4_mmid_args & a [[buffer(0)]],
                   device const char  * pool [[buffer(1)]],
                   device const ulong * offs [[buffer(2)]],
                   device const uint  * counts [[buffer(3)]],
                   device const uint  * bases [[buffer(4)]],
                   device const float * xin [[buffer(5)]],
                   device       float * yout [[buffer(6)]],
                   threadgroup  char  * shmem [[threadgroup(0)]],
                   uint3  tgpig [[threadgroup_position_in_grid]],
                   ushort tiitg [[thread_index_in_threadgroup]],
                   ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint ez = tgpig.z, ntok = counts[ez];
    if (tgpig.x * 32 >= ntok) return;
    device const char  * src0 = pool + offs[ez] + a.add;
    device const float * src1 = xin + (ulong)bases[ez] * a.x_stride;
    device       float * dst  = yout + (ulong)bases[ez] * a.y_stride;
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = nl_;

    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr0 = ((int)a.nrows - r0 < NR0) ? ((int)a.nrows - r0) : NR0;
    const short nr1 = ((int)ntok  - r1 < NR1) ? ((int)ntok  - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    device const block_q * x = (device const block_q *)(src0 + (ulong)a.row_bytes*(r0 + lr0)) + il0/nl;

    const short iy = 8*(tiitg % NL1);
    device const float * y = src1 + (ulong)a.x_stride*(r1 + lr1) + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint loop_k = 0; loop_k < a.ncols; loop_k += NK) {
        half4x4 temp_a;
        deq(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
        }

        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;
            const short ib = 4*sx + sy;
            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = (half2x4)(*((device float2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        for (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (r0 + NR0 <= (int)a.nrows && r1 + NR1 <= (int)ntok) {
        device float * C = dst + (r0 + 32*(sgitg & 1)) + (ulong)(r1 + 16*(sgitg >> 1)) * a.nrows;
        for (short i = 0; i < 8; i++) simdgroup_store(mc[i], C + 8*(i%4) + 8*a.nrows*(i/4), a.nrows, 0, false);
    } else {
        // a tile at the edge: through threadgroup memory, only what is inside
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float * D = dst + r0 + (ulong)(r1 + j)*a.nrows;
                threadgroup float * C = temp_str + (j*NR0);
                for (int i = 0; i < nr0; i++) D[i] = C[i];
            }
        }
    }
}


typedef decltype(q35_mm_id32<dn_block_q4_K, 16, d4_deq_q4_K>) q35_mm_id32_t;
template [[host_name("q35_mm_id32_q4_k")]] kernel q35_mm_id32_t q35_mm_id32<dn_block_q4_K, 16, d4_deq_q4_K>;
template [[host_name("q35_mm_id32_q5_k")]] kernel q35_mm_id32_t q35_mm_id32<dn_block_q5_K, 16, d4_deq_q5_K>;
template [[host_name("q35_mm_id32_q6_k")]] kernel q35_mm_id32_t q35_mm_id32<block_q6_K, 16, q35_deq_q6_K>;
