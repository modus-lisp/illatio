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

// Rows 16*IL..16*IL+15 of a block's 256 weights, as halves (ggml's
// dequantize_q6_K).
static void dequantize_q6_K(device const block_q6_K * xb, short il, thread half4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

struct mm_args { uint ncols; uint nrows; uint ntok; uint row_bytes; };

// Y[t][r] = W[r] . X[t] for NTOK tokens: ggml's kernel_mul_mm for Q6_K (its
// simdgroup-matrix path, without bounds checks on the input: NCOLS is a
// whole number of super-blocks).  A threadgroup of four simdgroups makes a
// 64-row by 32-token tile, its weights dequantized to halves and both
// operands staged in threadgroup memory (8 KB), 32 columns at a time.
kernel void mm_q6k(constant mm_args & a [[buffer(0)]],
                   device const char  * src0 [[buffer(1)]],
                   device const float * src1 [[buffer(2)]],
                   device       float * dst  [[buffer(3)]],
                   threadgroup  char  * shmem [[threadgroup(0)]],
                   uint3  tgpig [[threadgroup_position_in_grid]],
                   ushort tiitg [[thread_index_in_threadgroup]],
                   ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 16;                // QK_NL: a block is 16 groups of 16

    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr0 = ((int)a.nrows - r0 < NR0) ? ((int)a.nrows - r0) : NR0;
    const short nr1 = ((int)a.ntok  - r1 < NR1) ? ((int)a.ntok  - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    device const block_q6_K * x = (device const block_q6_K *)(src0 + (ulong)a.row_bytes*(r0 + lr0)) + il0/nl;

    const short iy = 8*(tiitg % NL1);
    device const float * y = src1 + (ulong)a.ncols*(r1 + lr1) + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint loop_k = 0; loop_k < a.ncols; loop_k += NK) {
        half4x4 temp_a;
        dequantize_q6_K(x, il, temp_a);

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

    if (r0 + NR0 <= (int)a.nrows && r1 + NR1 <= (int)a.ntok) {
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

// MM_Q6K with 64-token tiles: 64 rows by 64 tokens a threadgroup, each of its
// four simdgroups 32 by 32 (sixteen 8x8 accumulators).  A weight is rebuilt
// and a weight tile read half as often per token as with 32-token tiles.
kernel void mm_q6k_64(constant mm_args & a [[buffer(0)]],
                      device const char  * src0 [[buffer(1)]],
                      device const float * src1 [[buffer(2)]],
                      device       float * dst  [[buffer(3)]],
                      threadgroup  char  * shmem [[threadgroup(0)]],
                      uint3  tgpig [[threadgroup_position_in_grid]],
                      ushort tiitg [[thread_index_in_threadgroup]],
                      ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 64;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 16;

    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr0 = ((int)a.nrows - r0 < NR0) ? ((int)a.nrows - r0) : NR0;
    const short nr1 = ((int)a.ntok  - r1 < NR1) ? ((int)a.ntok  - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    // each thread stages two token rows: lr1 and lr1 + 32
    const short t0 = (short)tiitg/NL1, t1 = t0 + 32;
    const short lr1a = t0 < nr1 ? t0 : nr1 - 1;
    const short lr1b = t1 < nr1 ? t1 : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    device const block_q6_K * x = (device const block_q6_K *)(src0 + (ulong)a.row_bytes*(r0 + lr0)) + il0/nl;

    const short iy = 8*(tiitg % NL1);
    device const float * ya = src1 + (ulong)a.ncols*(r1 + lr1a) + iy;
    device const float * yb = src1 + (ulong)a.ncols*(r1 + lr1b) + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[4];
    simdgroup_float8x8 mc[16];
    for (short i = 0; i < 16; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint loop_k = 0; loop_k < a.ncols; loop_k += NK) {
        half4x4 temp_a;
        dequantize_q6_K(x, il, temp_a);

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
            *(threadgroup half2x4 *)(sb + 64*(8*sx + t0/8) + 8*(t0%8)) = (half2x4)(*((device float2x4 *) ya));
            *(threadgroup half2x4 *)(sb + 64*(8*sx + t1/8) + 8*(t1%8)) = (half2x4)(*((device float2x4 *) yb));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        ya += NK; yb += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 4*64*(sgitg/2));

        for (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 16; i++) simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            lsma += 8*64;
            lsmb += 8*64;
        }
    }

    if (r0 + NR0 <= (int)a.nrows && r1 + NR1 <= (int)a.ntok) {
        device float * C = dst + (r0 + 32*(sgitg & 1)) + (ulong)(r1 + 32*(sgitg >> 1)) * a.nrows;
        for (short i = 0; i < 16; i++) simdgroup_store(mc[i], C + 8*(i%4) + 8*a.nrows*(i/4), a.nrows, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (32*(sgitg >> 1))*NR0;
        for (short i = 0; i < 16; i++) simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += 32) {
                device float * D = dst + r0 + (ulong)(r1 + j)*a.nrows;
                threadgroup float * C = ((threadgroup float *) shmem) + j*NR0;
                for (int i = 0; i < nr0; i++) D[i] = C[i];
            }
        }
    }
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

struct gates_args { uint vh; uint n; };

// BETA = sigmoid(BETA); ALPHA = softplus(ALPHA + dt) * A, per V head of each
// of N tokens.
kernel void q35_gates(constant gates_args & a [[buffer(0)]],
                      device float * alpha [[buffer(1)]], device float * beta [[buffer(2)]],
                      device const float * A [[buffer(3)]], device const float * dt [[buffer(4)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= a.vh * a.n) return;
    const uint j = i % a.vh;
    beta[i] = sigmoid(beta[i]);
    const float s = alpha[i] + dt[j];
    alpha[i] = (s > 20.0f ? s : log(1.0f + exp(s))) * A[j];
}

struct conv_args { uint ch; uint n; };

// C = silu(causal conv of each channel's last 4 inputs) for N tokens in
// order (a thread a channel); STATE moves on by one a token.
kernel void q35_conv(constant conv_args & a [[buffer(0)]],
                     device const float * qkv [[buffer(1)]], device const float * kern [[buffer(2)]],
                     device float * state [[buffer(3)]], device float * c [[buffer(4)]],
                     uint ci [[thread_position_in_grid]]) {
    if (ci >= a.ch) return;
    device float * st = state + ci * 3;
    device const float * k = kern + ci * 4;
    float x0 = st[0], x1 = st[1], x2 = st[2];
    for (uint t = 0; t < a.n; ++t) {
        const float x3 = qkv[t * a.ch + ci];
        float s = 0.f;
        s += x0 * k[0]; s += x1 * k[1]; s += x2 * k[2]; s += x3 * k[3];
        c[t * a.ch + ci] = silu(s);
        x0 = x1; x1 = x2; x2 = x3;
    }
    st[0] = x0; st[1] = x1; st[2] = x2;
}

struct l2_args { uint n; float eps; uint rows_per_tok; uint tok_stride; };

// X /= max(|X|, eps) per row of N, in place (a threadgroup a row); row r is
// token r / ROWS_PER_TOK's row r % ROWS_PER_TOK, tokens TOK_STRIDE apart.
kernel void l2norm(constant l2_args & a [[buffer(0)]],
                   device float * x [[buffer(1)]],
                   uint tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_position_in_threadgroup]],
                   uint ntg [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    device float * xr = x + (ulong)(tg / a.rows_per_tok) * a.tok_stride + (tg % a.rows_per_tok) * a.n;
    float s = 0.f;
    for (uint i = tid; i < a.n; i += ntg) s += xr[i] * xr[i];
    s = tg_sum(s, sh, tid, ntg);
    const float scale = 1.0f / max(sqrt(s), a.eps);
    for (uint i = tid; i < a.n; i += ntg) xr[i] *= scale;
}

struct delta_args { uint kh; uint kd; uint vh; uint vd; float scale; uint n; uint ch; };

#define DELTA_KD 128                    // a K head's width: the host checks
#define DELTA_TG 32                     // values a threadgroup

// The delta rule for DELTA_TG values of one V head (grid x the head, y the
// block of values; a thread a value v), for N tokens in order (token t's C
// at t*CH, its gates at t*VH, its O at t*VH*VD).  A thread keeps its column
// of S -- S[k][v] for every k, the state is k-major as on the CPU -- in
// registers for the whole chunk; only K and Q are shared, a token at a time.
kernel void q35_delta(constant delta_args & a [[buffer(0)]],
                      device const float * c [[buffer(1)]],
                      device const float * beta [[buffer(2)]],
                      device const float * g [[buffer(3)]],
                      device float * S [[buffer(4)]],
                      device float * o [[buffer(5)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      uint2 tpg [[thread_position_in_threadgroup]]) {
    threadgroup float kv[DELTA_KD];
    threadgroup float qv[DELTA_KD];
    const uint h = tgp.x, tid = tpg.x, v = tgp.y * DELTA_TG + tid;
    const uint kq = (h % a.kh) * DELTA_KD;                  // K heads tile over V heads
    const uint kk = a.kh * DELTA_KD + (h % a.kh) * DELTA_KD;
    const uint vv = 2 * a.kh * DELTA_KD + h * a.vd;
    device float * Sh = S + (ulong)h * DELTA_KD * a.vd + v;
    float s[DELTA_KD];
    for (uint k = 0; k < DELTA_KD; ++k) s[k] = Sh[k * a.vd];
    for (uint t = 0; t < a.n; ++t) {
        device const float * ct = c + (ulong)t * a.ch;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < DELTA_KD; i += DELTA_TG) { kv[i] = ct[kk + i]; qv[i] = ct[kq + i] * a.scale; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float decay = exp(g[t * a.vh + h]), b = beta[t * a.vh + h];
        float sk = 0.f;
        for (uint k = 0; k < DELTA_KD; ++k) { s[k] *= decay; sk += s[k] * kv[k]; }
        const float d = (ct[vv + v] - sk) * b;
        float ov = 0.f;
        for (uint k = 0; k < DELTA_KD; ++k) { s[k] += kv[k] * d; ov += s[k] * qv[k]; }
        o[((ulong)t * a.vh + h) * a.vd + v] = ov;
    }
    for (uint k = 0; k < DELTA_KD; ++k) Sh[k * a.vd] = s[k];
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

struct qk_args { uint nh; uint nkv; uint hd; uint rope_dims; float eps; uint pos0; };

// Per head of each token (a threadgroup; grid x the head, y the token; a
// thread a dimension): Q's heads come out of QG ([q | gate] per head), K's
// from K; each is normed, then its first ROPE_DIMS rotated as NEOX pairs by
// the token's row of ROPE ([cos | sin]); K and V go to the caches at POS0+t.
kernel void q35_qk(constant qk_args & a [[buffer(0)]],
                   device const float * qg [[buffer(1)]], device const float * kin [[buffer(2)]],
                   device const float * vin [[buffer(3)]],
                   device const float * qn [[buffer(4)]], device const float * kn [[buffer(5)]],
                   device const float * rope [[buffer(6)]],
                   device float * q [[buffer(7)]], device float * gate [[buffer(8)]],
                   device float * kc [[buffer(9)]], device float * vc [[buffer(10)]],
                   uint2 tgp [[threadgroup_position_in_grid]],
                   uint2 tpg [[thread_position_in_threadgroup]],
                   uint2 ntg2 [[threads_per_threadgroup]]) {
    const uint t = tpg.x, ntg = ntg2.x;
    threadgroup float sh[32];
    threadgroup float xs[512];
    const uint u = tgp.x, tk = tgp.y, pos = a.pos0 + tk;
    const uint hd = a.hd, kvd = a.nkv * hd, qd = a.nh * hd;
    const bool isq = u < a.nh;
    const uint j = isq ? u : u - a.nh;
    device const float * qgt = qg + (ulong)tk * 2 * qd;
    device const float * src = isq ? qgt + j * 2 * hd : kin + (ulong)tk * kvd + j * hd;
    device const float * nw  = isq ? qn : kn;
    device float * dst = isq ? q + (ulong)tk * qd + j * hd : kc + (ulong)pos * kvd + j * hd;
    device const float * rp = rope + tk * a.rope_dims;
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
            r = xs[i] * rp[i] - xs[i + half_] * rp[half_ + i];
        } else if (i < a.rope_dims) {
            const uint k = i - half_;
            r = xs[k] * rp[half_ + k] + xs[i] * rp[k];
        } else {
            r = xs[i];
        }
        dst[i] = r;
        if (isq) gate[(ulong)tk * qd + j * hd + i] = qgt[j * 2 * hd + hd + i];
        else     vc[(ulong)pos * kvd + j * hd + i] = vin[(ulong)tk * kvd + j * hd + i];
    }
}

struct attn_args { uint nh; uint nkv; uint hd; uint n0; float scale; };

// One query head of one token (grid x the head, y the token): softmax(q.K /
// sqrt(hd)) . V over positions 0..N0+t-1 -- causality is the length; SC
// (threadgroup memory) holds the scores.
kernel void attend(constant attn_args & a [[buffer(0)]],
                   device const float * q [[buffer(1)]],
                   device const float * kc [[buffer(2)]], device const float * vc [[buffer(3)]],
                   device float * out [[buffer(4)]],
                   threadgroup float * sc [[threadgroup(0)]],
                   uint2 tgp [[threadgroup_position_in_grid]],
                   uint2 tpg [[thread_position_in_threadgroup]],
                   uint2 ntg2 [[threads_per_threadgroup]]) {
    const uint t = tpg.x, ntg = ntg2.x;
    threadgroup float sh[32];
    const uint h = tgp.x, tk = tgp.y, n = a.n0 + tk;
    const uint hd = a.hd, kvd = a.nkv * hd, kvh = h / (a.nh / a.nkv), qd = a.nh * hd;
    device const float * qh = q + (ulong)tk * qd + h * hd;
    // the scores: a simdgroup a position, its lanes across the dimensions
    const uint lane = t % 32, sg = t / 32, nsg = ntg / 32;
    float mx = -INFINITY;
    for (uint p = sg; p < n; p += nsg) {
        device const float * kr = kc + (ulong)p * kvd + kvh * hd;
        float s = 0.f;
        for (uint i = lane; i < hd; i += 32) s += qh[i] * kr[i];
        s = simd_sum(s) * a.scale;
        if (lane == 0) sc[p] = s;
        mx = max(mx, s);
    }
    mx = tg_max(mx, sh, t, ntg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.f;
    for (uint p = t; p < n; p += ntg) { const float e = exp(sc[p] - mx); sc[p] = e; sum += e; }
    sum = tg_sum(sum, sh, t, ntg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / sum;
    for (uint i = t; i < hd; i += ntg) {
        float acc = 0.f;
        for (uint p = 0; p < n; ++p) acc += sc[p] * vc[(ulong)p * kvd + kvh * hd + i];
        out[(ulong)tk * qd + h * hd + i] = acc * inv;
    }
}

struct fa_args { uint nh; uint nkv; uint n; float scale; uint pos0; };

#define FA_D 256                        // the head width: the host checks
#define FA_KB 32                        // keys a step

// Prefill attention, flash style: a threadgroup (one simdgroup) takes 8
// query rows (tokens tk0..tk0+7) of head H and walks the keys FA_KB at a
// time, causal per row, with S = Q K^T and O += P V as 8x8 simdgroup
// matrices and the softmax kept running (max M and sum L per row): K and V
// are read once for 8 rows, not once a row.  The four 8x8 score blocks of a
// step accumulate independently, so their multiplies overlap; the softmax
// runs four lanes a row.  Q is staged in threadgroup memory.  All in f32.
kernel void attend_fa(constant fa_args & a [[buffer(0)]],
                      device const float * q [[buffer(1)]],
                      device const float * kc [[buffer(2)]], device const float * vc [[buffer(3)]],
                      device float * out [[buffer(4)]],
                      threadgroup float * shmem [[threadgroup(0)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    const uint h = tgp.x, tk0 = tgp.y * 8;
    const uint qd = a.nh * FA_D, kvd = a.nkv * FA_D, kvh = h / (a.nh / a.nkv);
    threadgroup float * qs  = shmem;                  // 8 x FA_D
    threadgroup float * ps  = qs + 8 * FA_D;          // 8 x FA_KB: S, then P
    threadgroup float * dsh = ps + 8 * FA_KB;         // 8 x 8: diag(alpha), then diag(1/L)

    // Q's rows (a row past N repeats N-1's: it is computed, never stored)
    for (uint i = lane; i < 8 * FA_D; i += 32) {
        const uint r = i / FA_D, tk = min(tk0 + r, a.n - 1);
        qs[i] = q[(ulong)tk * qd + h * FA_D + (i % FA_D)];
    }
    for (uint i = lane; i < 64; i += 32) dsh[i] = 0.f;
    const uint r = lane / 4, part = lane % 4;          // this lane's row, and its 8 of a step's keys
    const uint lim = a.pos0 + min(tk0 + r, a.n - 1);   // the last key its row sees
    float m = -INFINITY, l = 0.f;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    device const float * kb = kc + kvh * FA_D;
    device const float * vb = vc + kvh * FA_D;
    simdgroup_float8x8 o[FA_D / 8];
    for (short c = 0; c < FA_D / 8; ++c) o[c] = make_filled_simdgroup_matrix<float, 8>(0.f);

    const uint last = a.pos0 + min(tk0 + 7, a.n - 1);
    for (uint pb = 0; pb <= last; pb += FA_KB) {
        simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8>(0.f), s1 = s0, s2 = s0, s3 = s0;
        device const float * kp = kb + (ulong)pb * kvd;
        for (short c = 0; c < FA_D / 8; ++c) {
            simdgroup_float8x8 qm, k0, k1, k2, k3;
            simdgroup_load(qm, qs + c * 8, FA_D);
            simdgroup_load(k0, kp + c * 8,           kvd, 0, true);
            simdgroup_load(k1, kp + 8 * kvd + c * 8,  kvd, 0, true);
            simdgroup_load(k2, kp + 16 * kvd + c * 8, kvd, 0, true);
            simdgroup_load(k3, kp + 24 * kvd + c * 8, kvd, 0, true);
            simdgroup_multiply_accumulate(s0, qm, k0, s0);
            simdgroup_multiply_accumulate(s1, qm, k1, s1);
            simdgroup_multiply_accumulate(s2, qm, k2, s2);
            simdgroup_multiply_accumulate(s3, qm, k3, s3);
        }
        simdgroup_store(s0, ps,      FA_KB);
        simdgroup_store(s1, ps + 8,  FA_KB);
        simdgroup_store(s2, ps + 16, FA_KB);
        simdgroup_store(s3, ps + 24, FA_KB);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // the running softmax: four lanes a row, eight keys each
        float v[8], mx = -INFINITY;
        for (short j = 0; j < 8; ++j) {
            const uint p = pb + part * 8 + j;
            v[j] = (p <= lim) ? ps[r * FA_KB + part * 8 + j] * a.scale : -INFINITY;
            mx = max(mx, v[j]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float mn = max(m, mx);
        const float alpha = exp(m - mn);
        float sum = 0.f;
        for (short j = 0; j < 8; ++j) { const float e = exp(v[j] - mn); ps[r * FA_KB + part * 8 + j] = e; sum += e; }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        l = l * alpha + sum;
        m = mn;
        if (part == 0) dsh[r * 8 + r] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 p0, p1, p2, p3, dg;
        simdgroup_load(p0, ps,      FA_KB);
        simdgroup_load(p1, ps + 8,  FA_KB);
        simdgroup_load(p2, ps + 16, FA_KB);
        simdgroup_load(p3, ps + 24, FA_KB);
        simdgroup_load(dg, dsh, 8);
        device const float * vp = vb + (ulong)pb * kvd;
        for (short c = 0; c < FA_D / 8; ++c) {
            simdgroup_float8x8 v0, v1, v2, v3;
            simdgroup_load(v0, vp + c * 8,            kvd);
            simdgroup_load(v1, vp + 8 * kvd + c * 8,  kvd);
            simdgroup_load(v2, vp + 16 * kvd + c * 8, kvd);
            simdgroup_load(v3, vp + 24 * kvd + c * 8, kvd);
            simdgroup_multiply(o[c], dg, o[c]);
            simdgroup_multiply_accumulate(o[c], p0, v0, o[c]);
            simdgroup_multiply_accumulate(o[c], p1, v1, o[c]);
            simdgroup_multiply_accumulate(o[c], p2, v2, o[c]);
            simdgroup_multiply_accumulate(o[c], p3, v3, o[c]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (part == 0) dsh[r * 8 + r] = 1.0f / l;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dg;
    simdgroup_load(dg, dsh, 8);
    // rows past N are stored too, into the scratch's unused rows (the host sizes it)
    device float * ob = out + (ulong)tk0 * qd + h * FA_D;
    for (short c = 0; c < FA_D / 8; ++c) {
        simdgroup_multiply(o[c], dg, o[c]);
        simdgroup_store(o[c], ob + c * 8, qd);
    }
}
