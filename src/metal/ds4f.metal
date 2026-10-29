// ds4f.metal -- DeepSeek V4 Flash's kernels on the GPU (metal-ds4f.lisp).
//
// The matrix product is MM_Q6K_64's (qwen35.metal), ggml's kernel_mul_mm
// with 64-token tiles, made a template over the weight block and its
// dequantizer, as ggml's is; the dequantizers and the IQ2_XXS tables are
// ggml's (MIT, Copyright (c) 2023-2026 The ggml authors).  Instances:
// Q8_0 and F16 (the non-routed weights), IQ2_XXS and Q2_K (the experts).

#include <metal_stdlib>
using namespace metal;

#ifndef QK_K
#define QK_K 256
#endif

typedef struct { half d; int8_t qs[32]; } d4_block_q8_0;
typedef struct { uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; half d; half dmin; } d4_block_q2_K;
typedef struct { half d; uint16_t qs[QK_K/8]; } d4_block_iq2_xxs;

constant uint64_t iq2xxs_grid[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL,
    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL,
    0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL,
    0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL,
    0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,
    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL,
    0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,
    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL,
    0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL,
    0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,
    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL,
    0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL,
    0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,
    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL,
    0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL,
    0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL,
    0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,
    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL,
    0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL,
    0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL,
    0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL,
    0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL,
    0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL,
    0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL,
    0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL,
    0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL,
    0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,
    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL,
    0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,
    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL,
    0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL,
    0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,
    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL,
    0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL,
    0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,
    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL,
    0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,
    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL,
    0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL,
    0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL,
    0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,
    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL,
    0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL,
    0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL,
    0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL,
    0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL,
    0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL,
    0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL,
    0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL,
    0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL,
};
constant uint8_t ksigns_iq2xs[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
constant uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

void d4_deq_q8_0(device const d4_block_q8_0 * xb, short il, thread half4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;
    float4x4 reg_f;
    for (int i = 0; i < 16; i++) reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    reg = (half4x4) reg_f;
}

void d4_deq_f16(device const half4x4 * src, short il, thread half4x4 & reg) {
    reg = *src;
}

void d4_deq_q2_K(device const d4_block_q2_K * xb, short il, thread half4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];
    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;
    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) reg[i/4][i%4] = dl * (q[i] & mask) - ml;
}

void d4_deq_iq2_xxs(device const d4_block_iq2_xxs * xb, short il, thread half4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
}

struct d4_mm_args { uint ncols; uint nrows; uint ntok; uint row_bytes; uint x_stride; uint y_stride; };

// Y[t][r] = W[r] . X[t] for NTOK tokens (token t's X at t*X_STRIDE, its Y
// at t*Y_STRIDE), 64 rows by 64 tokens a threadgroup
// (four simdgroups of 32 by 32), weights dequantized to halves in
// threadgroup memory 32 columns at a time.  A block holds 16*NL weights.
template <typename block_q, short nl, void (*deq)(device const block_q *, short, thread half4x4 &)>
static inline void d4_mm_body(uint ncols, uint nrows, uint ntok, uint row_bytes, uint x_stride, uint y_stride,
                              device const char * src0, device const float * src1, device float * dst,
                              threadgroup char * shmem, uint3 tgpig, ushort tiitg, ushort sgitg) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);
    constexpr int NR0 = 64, NR1 = 64, NK = 32, NL0 = NK/16, NL1 = NK/8;
    const int r0 = tgpig.y*NR0, r1 = tgpig.x*NR1;
    const short nr0 = ((int)nrows - r0 < NR0) ? ((int)nrows - r0) : NR0;
    const short nr1 = ((int)ntok  - r1 < NR1) ? ((int)ntok  - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short t0 = (short)tiitg/NL1, t1 = t0 + 32;
    const short lr1a = t0 < nr1 ? t0 : nr1 - 1;
    const short lr1b = t1 < nr1 ? t1 : nr1 - 1;
    const short il0 = (tiitg % NL0);
    short il = il0;
    device const block_q * x = (device const block_q *)(src0 + (ulong)row_bytes*(r0 + lr0)) + il0/nl;
    const short iy = 8*(tiitg % NL1);
    device const float * ya = src1 + (ulong)x_stride*(r1 + lr1a) + iy;
    device const float * yb = src1 + (ulong)x_stride*(r1 + lr1b) + iy;
    simdgroup_half8x8 ma[4], mb[4];
    simdgroup_float8x8 mc[16];
    for (short i = 0; i < 16; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint loop_k = 0; loop_k < ncols; loop_k += NK) {
        half4x4 temp_a;
        deq(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8, sy = (tiitg/NL0)/8, lx = (tiitg/NL0)%8, ly = i%8;
            *(sa + 64*(8*sx + sy) + 8*ly + lx) = temp_a[i/4][i%4];
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
            lsma += 8*64; lsmb += 8*64;
        }
    }
    if (r0 + NR0 <= (int)nrows && r1 + NR1 <= (int)ntok) {
        device float * C = dst + (r0 + 32*(sgitg & 1)) + (ulong)(r1 + 32*(sgitg >> 1)) * y_stride;
        for (short i = 0; i < 16; i++) simdgroup_store(mc[i], C + 8*(i%4) + 8*y_stride*(i/4), y_stride, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (32*(sgitg >> 1))*NR0;
        for (short i = 0; i < 16; i++) simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += 32) {
                device float * D = dst + r0 + (ulong)(r1 + j)*y_stride;
                threadgroup float * C = ((threadgroup float *) shmem) + j*NR0;
                for (int i = 0; i < nr0; i++) D[i] = C[i];
            }
        }
    }
}


template <typename block_q, short nl, void (*deq)(device const block_q *, short, thread half4x4 &)>
kernel void d4_mm(constant d4_mm_args & a [[buffer(0)]],
                  device const char  * src0 [[buffer(1)]],
                  device const float * src1 [[buffer(2)]],
                  device       float * dst  [[buffer(3)]],
                  threadgroup  char  * shmem [[threadgroup(0)]],
                  uint3  tgpig [[threadgroup_position_in_grid]],
                  ushort tiitg [[thread_index_in_threadgroup]],
                  ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    d4_mm_body<block_q, nl, deq>(a.ncols, a.nrows, a.ntok, a.row_bytes, a.x_stride, a.y_stride,
                                 src0, src1, dst, shmem, tgpig, tiitg, sgitg);
}

// The routed experts of a chunk at once (ggml's mul_mm_id, as ds4 runs
// them): grid z an expert, its COUNTS[z] rows of X from row BASES[z]
// (grouped by expert), its weights at POOL + OFFS[z] + ADD; tiles past an
// expert's rows do nothing.
struct d4_mmid_args { uint ncols; uint nrows; uint row_bytes; uint x_stride; uint y_stride; uint add; };

template <typename block_q, short nl, void (*deq)(device const block_q *, short, thread half4x4 &)>
kernel void d4_mm_id(constant d4_mmid_args & a [[buffer(0)]],
                     device const char  * pool [[buffer(1)]],
                     device const ulong * offs [[buffer(2)]],
                     device const uint  * counts [[buffer(3)]],
                     device const uint  * bases [[buffer(4)]],
                     device const float * x [[buffer(5)]],
                     device       float * y [[buffer(6)]],
                     threadgroup  char  * shmem [[threadgroup(0)]],
                     uint3  tgpig [[threadgroup_position_in_grid]],
                     ushort tiitg [[thread_index_in_threadgroup]],
                     ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint z = tgpig.z, count = counts[z];
    if (tgpig.x * 64 >= count) return;
    const uint base = bases[z];
    d4_mm_body<block_q, nl, deq>(a.ncols, a.nrows, count, a.row_bytes, a.x_stride, a.y_stride,
                                 pool + offs[z] + a.add, x + (ulong)base * a.x_stride, y + (ulong)base * a.y_stride,
                                 shmem, uint3(tgpig.x, tgpig.y, 0), tiitg, sgitg);
}

typedef decltype(d4_mm<d4_block_q8_0, 2, d4_deq_q8_0>) d4_mm_t;
template [[host_name("d4_mm_q8_0")]]    kernel d4_mm_t d4_mm<d4_block_q8_0, 2, d4_deq_q8_0>;
template [[host_name("d4_mm_f16")]]     kernel d4_mm_t d4_mm<half4x4, 1, d4_deq_f16>;
template [[host_name("d4_mm_q2_k")]]    kernel d4_mm_t d4_mm<d4_block_q2_K, 16, d4_deq_q2_K>;
template [[host_name("d4_mm_iq2_xxs")]] kernel d4_mm_t d4_mm<d4_block_iq2_xxs, 16, d4_deq_iq2_xxs>;
typedef decltype(d4_mm_id<d4_block_q2_K, 16, d4_deq_q2_K>) d4_mm_id_t;
template [[host_name("d4_mm_id_q2_k")]]    kernel d4_mm_id_t d4_mm_id<d4_block_q2_K, 16, d4_deq_q2_K>;
template [[host_name("d4_mm_id_iq2_xxs")]] kernel d4_mm_id_t d4_mm_id<d4_block_iq2_xxs, 16, d4_deq_iq2_xxs>;

struct d4_swiglu_args { uint n; uint ff; float clamp; };

// MID[r][j] = silu(clamp(G)) * clamp(U) * W[r], each routed row r (a token's
// slot) of FF values: the expert's mid activation, its routing weight in.
kernel void d4_swiglu_w(constant d4_swiglu_args & a [[buffer(0)]],
                        device const float * g [[buffer(1)]], device const float * u [[buffer(2)]],
                        device const float * w [[buffer(3)]], device float * mid [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= a.n * a.ff) return;
    float gv = g[i], uv = u[i];
    if (a.clamp > 1e-6f) { gv = min(gv, a.clamp); uv = clamp(uv, -a.clamp, a.clamp); }
    const float s = gv >= 0.f ? 1.f / (1.f + exp(-gv)) : exp(gv) / (1.f + exp(gv));
    mid[i] = (gv * s) * uv * w[i / a.ff];
}

// --- attention for a chunk --------------------------------------------------------

// RCAP: KR is a ring of RCAP rows (decode's raw window), 0 for rows in a
// line (prefill's); IDENT: a token's compressed rows are the first NCOMP,
// without IDX.
struct d4_attn_args { uint nh; uint hd; float scale; uint maxc; uint rcap; uint ident; };

#define D4_HD 512                       // the head width: the host checks
#define D4_KB 8                         // keys a step

// One token's attention (a threadgroup; 16 simdgroups: simdgroup s takes
// heads 8*(s/2).. and the half s%2 of the 512 dimensions), over its keys:
// RAW[t] rows of KR from RAWS[t], then NCOMP[t] rows of KC, the ones IDX
// lists (row t of IDX, MAXC wide).  Keys go D4_KB a step, gathered into
// threadgroup memory; S = Q K^T and O += P V are 8x8 simdgroup matrices; the
// softmax runs per head from the head's sink, as the CPU's does (the sink in
// the denominator).  The key is the value (one KV head).  All in f32.
kernel void d4_attend(constant d4_attn_args & a [[buffer(0)]],
                      device const float * q [[buffer(1)]],
                      device const float * kr [[buffer(2)]],
                      device const float * kc [[buffer(3)]],
                      device const uint  * raws [[buffer(4)]],
                      device const uint  * nraw [[buffer(5)]],
                      device const uint  * ncomp [[buffer(6)]],
                      device const uint  * idx [[buffer(7)]],
                      device const float * sinks [[buffer(8)]],
                      device float * out [[buffer(9)]],
                      threadgroup float * shmem [[threadgroup(0)]],
                      uint t [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float * ks  = shmem;                       // D4_KB x D4_HD
    threadgroup float * ps  = ks + D4_KB * D4_HD;          // 64 heads x D4_KB: S, then P
    threadgroup float * dg  = ps + 64 * D4_KB;             // 8 groups x 8x8: diag(alpha), then diag(1/L)
    threadgroup float * ml  = dg + 8 * 64;                 // 64 maxima, then 64 sums
    const uint hg = sg / 2, dh = sg % 2;
    const uint nr = nraw[t], nc = ncomp[t], total = nr + nc, r0 = raws[t];
    device const float * qt = q + (ulong)t * a.nh * D4_HD;
    for (uint i = tid; i < 8 * 64; i += 512) dg[i] = 0.f;
    if (tid < 64) { ml[tid] = sinks[tid]; ml[64 + tid] = 1.0f; }   // max = sink, sum = exp(0)
    simdgroup_float8x8 o[32];
    for (short c = 0; c < 32; ++c) o[c] = make_filled_simdgroup_matrix<float, 8>(0.f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kb = 0; kb < total; kb += D4_KB) {
        // gather this step's keys
        for (uint i = tid; i < D4_KB * D4_HD; i += 512) {
            const uint j = kb + i / D4_HD, dd = i % D4_HD;
            float v = 0.f;
            if (j < nr) v = kr[(ulong)(a.rcap ? (r0 + j) % a.rcap : r0 + j) * D4_HD + dd];
            else if (j < total) v = kc[(ulong)(a.ident ? j - nr : idx[(ulong)t * a.maxc + (j - nr)]) * D4_HD + dd];
            ks[i] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S for this group's 8 heads (the dh = 0 simdgroup of each pair)
        if (dh == 0) {
            simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8>(0.f);
            for (short c = 0; c < D4_HD / 8; ++c) {
                simdgroup_float8x8 qm, kt;
                simdgroup_load(qm, qt + hg * 8 * D4_HD + c * 8, D4_HD);
                simdgroup_load(kt, ks + c * 8, D4_HD, 0, true);
                simdgroup_multiply_accumulate(s, qm, kt, s);
            }
            simdgroup_store(s, ps + hg * 8 * D4_KB, D4_KB);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // the running softmax: thread (head, key) for 64 heads x 8 keys
        {
            const uint h = tid / D4_KB, j = tid % D4_KB;
            const float v = (kb + j < total) ? ps[h * D4_KB + j] * a.scale : -INFINITY;
            float mx = v;
            mx = max(mx, simd_shuffle_xor(mx, 1)); mx = max(mx, simd_shuffle_xor(mx, 2)); mx = max(mx, simd_shuffle_xor(mx, 4));
            const float m = ml[h], mn = max(m, mx);
            const float e = exp(v - mn);
            float sum = e;
            sum += simd_shuffle_xor(sum, 1); sum += simd_shuffle_xor(sum, 2); sum += simd_shuffle_xor(sum, 4);
            ps[h * D4_KB + j] = e;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (j == 0) {
                const float alpha = exp(m - mn);
                ml[h] = mn;
                ml[64 + h] = ml[64 + h] * alpha + sum;
                dg[(h / 8) * 64 + (h % 8) * 9] = alpha;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(alpha) O + P V for this simdgroup's heads and half
        {
            simdgroup_float8x8 p, al;
            simdgroup_load(p, ps + hg * 8 * D4_KB, D4_KB);
            simdgroup_load(al, dg + hg * 64, 8);
            for (short c = 0; c < 32; ++c) {
                simdgroup_float8x8 vm;
                simdgroup_load(vm, ks + dh * 256 + c * 8, D4_HD);
                simdgroup_multiply(o[c], al, o[c]);
                simdgroup_multiply_accumulate(o[c], p, vm, o[c]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 64) dg[(tid / 8) * 64 + (tid % 8) * 9] = 1.0f / ml[64 + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 inv;
    simdgroup_load(inv, dg + hg * 64, 8);
    device float * ot = out + (ulong)t * a.nh * D4_HD + hg * 8 * D4_HD + dh * 256;
    for (short c = 0; c < 32; ++c) {
        simdgroup_multiply(o[c], inv, o[c]);
        simdgroup_store(o[c], ot + c * 8, D4_HD);
    }
}

// --- the indexer's scores for a chunk ---------------------------------------------

struct d4_ixs_args { uint n; uint ncomp; uint nh; };

// ds4's kernel_dsv4_indexer_scores_tiled_f32 (MIT, antirez/ds4): SCORES[t][r]
// = sum over heads h of relu(Q[t][h] . K[r]) * W[t][h] (W scaled), for the
// rows token t sees (VIS[t]), -inf past them.  A threadgroup (4 simdgroups)
// takes 8 tokens x 32 rows; the 128-wide dots are 8x8 simdgroup matrices.
kernel void d4_ix_scores(constant d4_ixs_args & a [[buffer(0)]],
                         device const float * q [[buffer(1)]],
                         device const float * w [[buffer(2)]],
                         device const float * k [[buffer(3)]],
                         device const uint  * vis [[buffer(4)]],
                         device float * scores [[buffer(5)]],
                         threadgroup float * shmem [[threadgroup(0)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8, TN = 32, TS = 8, D = 128;
    const uint c0 = tg.x * TN, t0 = tg.y * TM;
    threadgroup float * qtg = shmem;            // [8][128]
    threadgroup float * ktg = qtg + TM * D;     // [32][128]
    threadgroup float * dot = ktg + TN * D;     // [8][32]
    uint maxvis = 0;
    for (uint r = 0; r < TM; r++) if (t0 + r < a.n) maxvis = max(maxvis, vis[t0 + r]);
    if (c0 >= maxvis) {
        for (uint i = tid; i < TM * TN; i += 128) {
            const uint t = t0 + i / TN, c = c0 + i % TN;
            if (t < a.n && c < a.ncomp) scores[(uint64_t)t * a.ncomp + c] = -INFINITY;
        }
        return;
    }
    for (uint i = tid; i < TN * D; i += 128) {
        const uint c = c0 + i / D;
        ktg[i] = c < a.ncomp ? k[(uint64_t)c * D + i % D] : 0.f;
    }
    const uint cell0 = lane, cell1 = lane + 32u;
    const uint row0 = cell0 >> 3, row1 = cell1 >> 3;
    const uint col0 = (uint)sg * TS + (cell0 & 7u), col1 = (uint)sg * TS + (cell1 & 7u);
    const uint tok0 = t0 + row0, tok1 = t0 + row1, cmp0 = c0 + col0, cmp1 = c0 + col1;
    float acc0 = 0.f, acc1 = 0.f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 0; h < a.nh; h++) {
        for (uint i = tid; i < TM * D; i += 128) {
            const uint t = t0 + i / D;
            qtg[i] = t < a.n ? q[((uint64_t)t * a.nh + h) * D + i % D] : 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 md = make_filled_simdgroup_matrix<float, 8>(0.f);
        for (uint db = 0; db < D / TS; db++) {
            simdgroup_float8x8 mq, mk;
            simdgroup_load(mq, qtg + db * TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db * TS, D, 0, true);
            simdgroup_multiply_accumulate(md, mq, mk, md);
        }
        simdgroup_store(md, dot + (uint)sg * TS, TN, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tok0 < a.n) acc0 += max(dot[row0 * TN + col0], 0.f) * w[tok0 * a.nh + h];
        if (tok1 < a.n) acc1 += max(dot[row1 * TN + col1], 0.f) * w[tok1 * a.nh + h];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tok0 < a.n && cmp0 < a.ncomp)
        scores[(uint64_t)tok0 * a.ncomp + cmp0] = cmp0 < vis[tok0] ? acc0 : -INFINITY;
    if (tok1 < a.n && cmp1 < a.ncomp)
        scores[(uint64_t)tok1 * a.ncomp + cmp1] = cmp1 < vis[tok1] ? acc1 : -INFINITY;
}

// --- the routed experts of one token (decode) ----------------------------------------

struct d4_mv_args { uint ncols; uint nrows; uint row_bytes; uint add; uint x_stride; uint y_stride; };

#define D4_MV_NSG 2                     // simdgroups a threadgroup (IQ2_XXS's tables load with 64 threads)
#define D4_MV_NR0 4                     // rows a simdgroup

// Y[z] = W[z] X[z] for each expert z of the dispatch (grid z), W[z]'s rows
// at STAGE + OFFS[z] + ADD, X[z] at z*X_STRIDE, Y[z] at z*Y_STRIDE: ggml's
// kernel_mul_mv_iq2_xxs_f32 (MIT, the ggml authors), from the expert cache.
kernel void d4_mv_iq2_xxs(constant d4_mv_args & a [[buffer(0)]],
                          device const char * stage [[buffer(1)]],
                          device const ulong * offs [[buffer(2)]],
                          device const float * src1 [[buffer(3)]],
                          device float * dst [[buffer(4)]],
                          threadgroup char * shmem [[threadgroup(0)]],
                          uint3 tgpig [[threadgroup_position_in_grid]],
                          ushort tiisg [[thread_index_in_simdgroup]],
                          ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int nb = a.ncols / QK_K, nb32 = nb * (QK_K / 32);
    const int first_row = (tgpig.x * D4_MV_NSG + sgitg) * D4_MV_NR0;
    const uint z = tgpig.z;
    device const d4_block_iq2_xxs * x =
        (device const d4_block_iq2_xxs *)(stage + offs[z] + a.add + (uint64_t)first_row * a.row_bytes);
    device const float * y = src1 + (uint64_t)z * a.x_stride;
    threadgroup uint64_t * svalues = (threadgroup uint64_t *)shmem;
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t *)(svalues + 256);
    {
        int pos = (32 * sgitg + tiisg) * 4;
        for (int i = 0; i < 4; ++i) svalues[pos + i] = iq2xxs_grid[pos + i];
        pos = (32 * sgitg + tiisg) * 2;
        for (int i = 0; i < 2; ++i) ssigns[pos + i] = ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float yl[32];
    float sumf[D4_MV_NR0] = {0.f};
    const uint rstride = a.row_bytes / 2;       // in halves / uint16s
    device const float * y4 = y + 32 * tiisg;
    for (int ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) yl[i] = y4[i];
        const int ibl = ib32 / (QK_K / 32), ib = ib32 % (QK_K / 32);
        device const d4_block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;
        for (short row = 0; row < D4_MV_NR0; row++) {
            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));
            float sum = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
                const uint8_t signs = ssigns[(aux32 >> 7 * l) & 127];
                for (short j = 0; j < 8; ++j) sum += yl[8 * l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            }
            sumf[row] += d * sum;
            dh += rstride; q2 += rstride;
        }
        y4 += 32 * 32;
    }
    device float * out = dst + (uint64_t)z * a.y_stride;
    for (int row = 0; row < D4_MV_NR0 && first_row + row < (int)a.nrows; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0) out[first_row + row] = s * 0.25f;
    }
}

// The same for Q2_K: ggml's kernel_mul_mv_q2_K_f32.
kernel void d4_mv_q2_k(constant d4_mv_args & a [[buffer(0)]],
                       device const char * stage [[buffer(1)]],
                       device const ulong * offs [[buffer(2)]],
                       device const float * src1 [[buffer(3)]],
                       device float * dst [[buffer(4)]],
                       uint3 tgpig [[threadgroup_position_in_grid]],
                       ushort tiisg [[thread_index_in_simdgroup]],
                       ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int nb = a.ncols / QK_K;
    const int first_row = (tgpig.x * D4_MV_NSG + sgitg) * D4_MV_NR0;
    const uint z = tgpig.z;
    device const d4_block_q2_K * x =
        (device const d4_block_q2_K *)(stage + offs[z] + a.add + (uint64_t)first_row * a.row_bytes);
    device const float * y = src1 + (uint64_t)z * a.x_stride;
    float yl[32];
    float sumf[D4_MV_NR0] = {0.f};
    const short ix = tiisg / 8, it = tiisg % 8, iq = it / 4, ir = it % 4, is = (8 * ir) / 16;
    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;
    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
            yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
            yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
            yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
        }
        device const uint8_t  * sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;
        for (short row = 0; row < D4_MV_NR0; row++) {
            float4 acc1 = {0.f, 0.f, 0.f, 0.f}, acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003); acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c); acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030); acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0); acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
            }
            const float dall = dh[0], dmin = dh[1] * 1.f / 16.f;
            sumf[row] += dall * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                 (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                 (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                 (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                 sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
            qs += a.row_bytes / 2; sc += a.row_bytes; dh += a.row_bytes / 2;
        }
        y4 += 4 * QK_K;
    }
    device float * out = dst + (uint64_t)z * a.y_stride;
    for (int row = 0; row < D4_MV_NR0 && first_row + row < (int)a.nrows; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0) out[first_row + row] = s;
    }
}

// --- a token's Q8_0 projections (decode) ---------------------------------------------

struct d4_mvq8_args { uint ncols; uint nrows; uint row_bytes; uint group_rows; uint group_stride; };

// Y = W X for one token, W's Q8_0 rows; row r reads X from (r / GROUP_ROWS)
// * GROUP_STRIDE (the grouped output projection; GROUP_ROWS = NROWS for a
// plain one).  A simdgroup takes 4 rows, its 32 lanes 8 blocks a step (4
// lanes a block, 8 quants each), as ggml's kernel_mul_mv_q8_0_f32 walks K;
// 2 simdgroups a threadgroup.
kernel void d4_mv_q8_0(constant d4_mvq8_args & a [[buffer(0)]],
                       device const char * w [[buffer(1)]],
                       device const float * x [[buffer(2)]],
                       device float * y [[buffer(3)]],
                       uint tg [[threadgroup_position_in_grid]],
                       ushort tiisg [[thread_index_in_simdgroup]],
                       ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int first_row = (tg * 2 + sgitg) * 4;
    if (first_row >= (int)a.nrows) return;
    const int nb = a.ncols / 32;
    const short ix = tiisg / 4, il = tiisg % 4;
    device const float * xr = x + (uint64_t)(first_row / a.group_rows) * a.group_stride;
    float sumf[4] = {0.f, 0.f, 0.f, 0.f};
    for (int ib = ix; ib < nb; ib += 8) {
        device const float4 * xb = (device const float4 *)(xr + ib * 32 + il * 8);
        const float4 y0 = xb[0], y1 = xb[1];
        for (short row = 0; row < 4; ++row) {
            device const d4_block_q8_0 * blk =
                (device const d4_block_q8_0 *)(w + (uint64_t)(first_row + row) * a.row_bytes) + ib;
            device const int8_t * qs = blk->qs + il * 8;
            const float sq = qs[0] * y0[0] + qs[1] * y0[1] + qs[2] * y0[2] + qs[3] * y0[3]
                           + qs[4] * y1[0] + qs[5] * y1[1] + qs[6] * y1[2] + qs[7] * y1[3];
            sumf[row] += sq * (float)blk->d;
        }
    }
    for (short row = 0; row < 4 && first_row + row < (int)a.nrows; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0) y[first_row + row] = s;
    }
}


// --- decode on the GPU: a token's layers, one command buffer a layer -----------------
//
// The CPU's ops (ds4f.lisp) one for one, for one token: RMS norms, the F32
// and Q8_0 products, the hyper-connections' split (Sinkhorn) and mixes,
// RoPE (its cos/sin from the CPU, as D4-ROPE! computes them), the FP8 and
// FP4 round trips, the raw window (a ring) and the compressors' windows and
// rows, the indexer's top-K, attention (D4_ATTEND).  Sums go in float, not
// the CPU's double; the vectors test holds them to ds4.

// RMS norm of GROUPS rows of N at X, each into OUT (times W when HAS_W): a
// threadgroup a row.
struct d4_rms_args { uint n; float eps; uint has_w; };

static float d4_tg_sum(float v, threadgroup float * red, ushort tid, ushort nthreads) {
    v = simd_sum(v);
    const ushort lane = tid % 32, sg = tid / 32, nsg = (nthreads + 31) / 32;
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = lane < nsg ? red[lane] : 0.f;
    t = simd_sum(t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

kernel void d4_rms(constant d4_rms_args & a [[buffer(0)]],
                   device const float * x [[buffer(1)]], device const float * w [[buffer(2)]],
                   device float * out [[buffer(3)]],
                   threadgroup float * red [[threadgroup(0)]],
                   uint g [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                   ushort nt [[threads_per_threadgroup]]) {
    device const float * xr = x + (uint64_t)g * a.n;
    device float * orow = out + (uint64_t)g * a.n;
    float ss = 0.f;
    for (uint i = tid; i < a.n; i += nt) { const float v = xr[i]; ss += v * v; }
    ss = d4_tg_sum(ss, red, tid, nt);
    const float scale = 1.f / sqrt(ss / (float)a.n + a.eps);
    for (uint i = tid; i < a.n; i += nt) orow[i] = a.has_w ? (xr[i] * scale) * w[i] : xr[i] * scale;
}

// Y = W X, W f32 rows: 4 rows a threadgroup, K split across its 8 simdgroups.
struct d4_mvf_args { uint ncols; uint nrows; };

kernel void d4_mv_f32(constant d4_mvf_args & a [[buffer(0)]],
                      device const float * w [[buffer(1)]], device const float * x [[buffer(2)]],
                      device float * y [[buffer(3)]],
                      threadgroup float * red [[threadgroup(0)]],
                      uint tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint r0 = tg * 4;
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    device const float4 * x4 = (device const float4 *)x;
    const uint n4 = a.ncols / 4;
    for (uint i = tid; i < n4; i += 256) {
        const float4 xv = x4[i];
        for (short r = 0; r < 4; ++r)
            if (r0 + r < a.nrows) s[r] += dot(((device const float4 *)(w + (uint64_t)(r0 + r) * a.ncols))[i], xv);
    }
    for (short r = 0; r < 4; ++r) {
        const float t = d4_tg_sum(s[r], red, tid, 256);
        if (tid == 0 && r0 + r < a.nrows) y[r0 + r] = t;
    }
}

// hc_split_sinkhorn_one and the pre-weighted sum: MIX (the N*(N+2) mixes)
// into PC (POST[N], then COMB[N*N]) and OUT[j] = sum_h STREAMS[h][j] PRE[h].
struct d4_hcpre_args { uint d; uint n; uint iters; float eps; };

kernel void d4_hc_pre_fin(constant d4_hcpre_args & a [[buffer(0)]],
                          device const float * mix [[buffer(1)]], device const float * scale [[buffer(2)]],
                          device const float * base [[buffer(3)]], device const float * streams [[buffer(4)]],
                          device float * out [[buffer(5)]], device float * pc [[buffer(6)]],
                          threadgroup float * sh [[threadgroup(0)]],
                          uint2 gid2 [[thread_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                          uint2 tg2 [[threadgroup_position_in_grid]]) {
    // a token a row of threadgroups (grid y): its mixes, streams, out and PC
    const uint tg = tg2.x, gid = gid2.x, tok = tg2.y;
    mix += tok * 32; streams += (uint64_t)tok * a.n * a.d; out += (uint64_t)tok * a.d; pc += tok * 32;
    // the first simdgroup: lane i < 16 holds COMB entry (src = i % 4, dst =
    // i / 4); a dst's row sums over lanes ^1 ^2, a src's column over ^4 ^8
    // (written for 4 streams: the host checks)
    threadgroup float * pre = sh;
    const float eps = a.eps;
    if (tid < 32) {
        const uint lane = tid, n = 4, src = lane % 4, dst = (lane / 4) % 4, idx = src + dst * n;
        float c = mix[2 * n + idx] * scale[2] + base[2 * n + idx];
        float mx = max(c, simd_shuffle_xor(c, 1)); mx = max(mx, simd_shuffle_xor(mx, 2));
        c = exp(c - mx);
        float sum = c + simd_shuffle_xor(c, 1); sum += simd_shuffle_xor(sum, 2);
        c = c * (1.f / sum) + eps;
        for (uint it = 0; it < a.iters; ++it) {
            if (it > 0) {
                float r = c + simd_shuffle_xor(c, 1); r += simd_shuffle_xor(r, 2);
                c *= 1.f / (r + eps);
            }
            float col = c + simd_shuffle_xor(c, 4); col += simd_shuffle_xor(col, 8);
            c *= 1.f / (col + eps);
        }
        if (tg == 0 && lane < 16) pc[n + idx] = c;
        if (lane < 4) pre[lane] = 1.f / (1.f + exp(-(mix[lane] * scale[0] + base[lane]))) + eps;
        if (tg == 0 && lane >= 4 && lane < 8) {
            const uint i = lane - 4;
            pc[i] = 2.f / (1.f + exp(-(mix[n + i] * scale[1] + base[n + i])));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid < a.d) {
        float acc = 0.f;
        for (uint h = 0; h < a.n; ++h) acc += streams[h * a.d + gid] * pre[h];
        out[gid] = acc;
    }
}

// hc_post_one: stream DST = BLOCK * POST[DST] + sum_src COMB[dst + src*n] * STREAMS[src].
kernel void d4_hc_post(constant d4_hcpre_args & a [[buffer(0)]],
                       device const float * block [[buffer(1)]], device const float * streams [[buffer(2)]],
                       device const float * pc [[buffer(3)]], device float * outs [[buffer(4)]],
                       uint gid [[thread_position_in_grid]]) {
    // GID over tokens x streams x D (decode: one token)
    const uint hcd = a.n * a.d, tok = gid / hcd, r = gid % hcd;
    if (tok >= a.iters) return;                 // ITERS: here, the tokens
    const uint dst = r / a.d, j = r % a.d, n = a.n;
    block += (uint64_t)tok * a.d; streams += (uint64_t)tok * hcd; pc += tok * 32;
    float v = block[j] * pc[dst];
    for (uint src = 0; src < n; ++src) v += pc[n + dst + src * n] * streams[src * a.d + j];
    outs[(uint64_t)tok * hcd + r] = v;
}

// RoPE on the last NROT dims of each of NH heads of HD at X, CS the pairs'
// (cos, sin) from the CPU (D4-ROPE-CS: a layer's, a position's, inverse or not).
struct d4_rope_args { uint nh; uint hd; uint nrot; uint ntok; };

kernel void d4_rope(constant d4_rope_args & a [[buffer(0)]],
                    device float * x [[buffer(1)]], device const float * cs [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    const uint half_rot = a.nrot / 2, per = a.nh * half_rot;
    if (gid >= a.ntok * per) return;
    const uint tok = gid / per, h = (gid % per) / half_rot, i = 2 * (gid % half_rot);
    cs += tok * a.nrot;                         // a token's table, then its heads
    device float * t = x + ((uint64_t)tok * a.nh + h) * a.hd + (a.hd - a.nrot);
    const float c = cs[i], s = cs[i + 1], x0 = t[i], x1 = t[i + 1];
    t[i] = x0 * c - x1 * s;
    t[i + 1] = x0 * s + x1 * c;
}

// E4M3 and E2M1 rounding, as E4M3-ROUND / E2M1-ROUND (ties to the even code).
constant float d4_e4m3[127] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f
};

static float d4_e4m3_round(float x) {
    const float sign = x < 0.f ? -1.f : 1.f, ax = min(fabs(x), 448.f);
    int lo = 0, hi = 126;
    while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (d4_e4m3[mid] <= ax) lo = mid; else hi = mid - 1; }
    int best = lo;
    if (best < 126) {
        const float bd = fabs(ax - d4_e4m3[best]), nd = fabs(ax - d4_e4m3[best + 1]);
        if (nd < bd || (nd == bd && ((best + 1) % 2 == 0) && (best % 2 == 1))) best++;
    }
    return sign * d4_e4m3[best];
}

constant float d4_e2m1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

static float d4_e2m1_round(float x) {
    const float sign = x < 0.f ? -1.f : 1.f, ax = min(fabs(x), 6.f);
    int best = 0; float bd = fabs(ax);
    for (int i = 1; i < 8; ++i) {
        const float dd = fabs(ax - d4_e2m1[i]);
        if (dd < bd || (dd == bd && (i % 2 == 0) && (best % 2 == 1))) { best = i; bd = dd; }
    }
    return sign * d4_e2m1[best];
}

// A row of HD in threadgroup memory: FP8 (blocks of 64 over the first HD -
// NROT dims, a power-of-two scale) when KIND is 0, else the indexer's QAT
// (a scaled 128-point Hadamard, then FP4 in blocks of 32).  HD threads.
static void d4_round_row(threadgroup float * v, uint hd, uint nrot, uint kind, ushort tid, threadgroup float * red) {
    if (kind == 0) {
        const uint nb = (hd - nrot) / 64;
        if (tid < nb) {
            float amax = 0.f;
            for (uint i = 0; i < 64; ++i) amax = max(amax, fabs(v[tid * 64 + i]));
            red[tid] = max(amax, 1e-4f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < hd - nrot) {
            const float scale = exp2(ceil(log2(red[tid / 64] / 448.f)));
            const float x = clamp(v[tid] / scale, -448.f, 448.f);
            v[tid] = d4_e4m3_round(x) * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        for (uint stride = 1; stride < 128; stride *= 2) {
            if (tid < 64) {
                const uint base = (tid / stride) * 2 * stride, i = tid % stride;
                const float x = v[base + i], y = v[base + stride + i];
                v[base + i] = x + y; v[base + stride + i] = x - y;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid < 128) v[tid] *= 0.08838834764831845f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 4) {
            float amax = 0.f;
            for (uint i = 0; i < 32; ++i) amax = max(amax, fabs(v[tid * 32 + i]));
            red[tid] = max(amax, 7.052966104933725e-38f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 128) {
            const float scale = exp2(ceil(log2(red[tid / 32] / 6.f)));
            const float x = clamp(v[tid] / scale, -6.f, 6.f);
            v[tid] = d4_e2m1_round(x) * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

static void d4_rope_row(threadgroup float * v, uint hd, uint nrot, device const float * cs, ushort tid) {
    if (tid < nrot / 2) {
        const uint i = 2 * tid;
        threadgroup float * t = v + (hd - nrot);
        const float c = cs[i], s = cs[i + 1], x0 = t[i], x1 = t[i + 1];
        t[i] = x0 * c - x1 * s; t[i + 1] = x0 * s + x1 * c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The KV row: RMS-normed (W), RoPE'd, through FP8, then through F16 into
// ring row SLOT of RAW.  HD threads.
struct d4_kv_args { uint hd; uint nrot; float eps; uint slot; };

kernel void d4_kv_fin(constant d4_kv_args & a [[buffer(0)]],
                      device const float * kv0 [[buffer(1)]], device const float * w [[buffer(2)]],
                      device const float * cs [[buffer(3)]], device float * raw [[buffer(4)]],
                      threadgroup float * sh [[threadgroup(0)]],
                      uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    // a threadgroup a token: row T of KV0 into row SLOT + T of RAW
    threadgroup float * v = sh;                 // hd
    threadgroup float * red = sh + a.hd;        // 32
    kv0 += (uint64_t)t * a.hd; cs += t * a.nrot;
    const float x = kv0[tid];
    const float ss = d4_tg_sum(x * x, red, tid, a.hd);
    const float scale = 1.f / sqrt(ss / (float)a.hd + a.eps);
    v[tid] = (x * scale) * w[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    d4_rope_row(v, a.hd, a.nrot, cs, tid);
    d4_round_row(v, a.hd, a.nrot, 0, tid, red);
    raw[(uint64_t)(a.slot + t) * a.hd + tid] = (float)(half)v[tid];
}

// compressor_decode_one: this token's projected row KV and score SC (plus
// APE's row POS_MOD) into window row ROW (WIDTH each).
struct d4_cst_args { uint width; uint row; uint pos_mod; uint ape_cols; };

kernel void d4_comp_store(constant d4_cst_args & a [[buffer(0)]],
                          device const float * kv [[buffer(1)]], device const float * sc [[buffer(2)]],
                          device const float * ape [[buffer(3)]], device float * st_kv [[buffer(4)]],
                          device float * st_score [[buffer(5)]], uint j [[thread_position_in_grid]]) {
    if (j >= a.width) return;
    st_kv[a.row * a.width + j] = kv[j];
    st_score[a.row * a.width + j] = sc[j] + ape[a.pos_mod * a.ape_cols + j];
}

// Every RATIO tokens: the window softmax-pooled per dim, RMS-normed (W),
// RoPE'd (CS: its first position), rounded (KIND 0: FP8, 1: the indexer's
// QAT), through F16 as compressed row DST of ROWS; with ratio 4, the
// current half then becomes the previous one.  HD threads.
struct d4_cpool_args { uint hd; uint ratio; uint width; float eps; uint kind; uint nrot; uint dst; };

kernel void d4_comp_pool(constant d4_cpool_args & a [[buffer(0)]],
                         device float * st_kv [[buffer(1)]], device float * st_score [[buffer(2)]],
                         device const float * w [[buffer(3)]], device const float * cs [[buffer(4)]],
                         device float * rows [[buffer(5)]],
                         threadgroup float * sh [[threadgroup(0)]],
                         ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float * v = sh;                 // hd
    threadgroup float * red = sh + a.hd;        // 32
    const uint j = tid, r4 = a.ratio == 4, width = a.width, hd = a.hd;
    float mx = -1e30f;
    for (uint r = 0; r < a.ratio; ++r) {
        mx = max(mx, st_score[r * width + j]);
        if (r4) mx = max(mx, st_score[(a.ratio + r) * width + hd + j]);
    }
    float pooled = 0.f;
    if (mx > -5e29f) {
        float denom = 0.f, sum = 0.f;
        for (uint r = 0; r < a.ratio; ++r) {
            if (r4) {
                const float wp = exp(st_score[r * width + j] - mx);
                const float wc = exp(st_score[(a.ratio + r) * width + hd + j] - mx);
                denom += wp + wc;
                sum += wp * st_kv[r * width + j];
                sum += wc * st_kv[(a.ratio + r) * width + hd + j];
            } else {
                const float ww = exp(st_score[r * width + j] - mx);
                denom += ww; sum += ww * st_kv[r * width + j];
            }
        }
        pooled = denom > 0.f ? sum / denom : 0.f;
    }
    const float ss = d4_tg_sum(pooled * pooled, red, tid, hd);
    const float scale = 1.f / sqrt(ss / (float)hd + a.eps);
    v[j] = (pooled * scale) * w[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    d4_rope_row(v, hd, a.nrot, cs, tid);
    d4_round_row(v, hd, a.nrot, a.kind, tid, red);
    rows[(uint64_t)a.dst * hd + j] = (float)(half)v[j];
    if (r4) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < a.ratio; ++r)
            for (uint k = j; k < width; k += hd) {
                st_kv[r * width + k] = st_kv[(a.ratio + r) * width + k];
                st_score[r * width + k] = st_score[(a.ratio + r) * width + k];
            }
    }
}

// The indexer's queries: each of NH heads of 128 RoPE'd (CS) then QAT'd; W,
// the head weights, scaled.  A threadgroup a head, 128 threads.
struct d4_ixq_args { uint nh; uint nrot; float wscale; };

kernel void d4_ixq_fin(constant d4_ixq_args & a [[buffer(0)]],
                       device float * q [[buffer(1)]], device const float * cs [[buffer(2)]],
                       device float * w [[buffer(3)]],
                       threadgroup float * sh [[threadgroup(0)]],
                       uint2 hg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    // threadgroup (head, token)
    const uint h = hg.x, tok = hg.y;
    threadgroup float * v = sh, * red = sh + 128;
    cs += tok * a.nrot; w += tok * a.nh;
    device float * qh = q + ((uint64_t)tok * a.nh + h) * 128;
    v[tid] = qh[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    d4_rope_row(v, 128, a.nrot, cs, tid);
    d4_round_row(v, 128, a.nrot, 1, tid, red);
    qh[tid] = v[tid];
    if (tid == 0) w[h] *= a.wscale;
}

// The indexer's pick: of N scores, the K highest, their rows in order into
// IDX (the earlier of equal scores first, as TOP-K-ROWS).  1024 threads.
struct d4_topk_args { uint n; uint k; uint stride; };

static uint d4_key(float f) { const uint u = as_type<uint>(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }

kernel void d4_topk(constant d4_topk_args & a [[buffer(0)]],
                    device const float * scores [[buffer(1)]], device uint * idx [[buffer(2)]],
                    device const uint * vis [[buffer(3)]],
                    threadgroup uint * sh [[threadgroup(0)]],
                    uint tok [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    // a threadgroup a token: its row of scores (STRIDE apart), VIS of them;
    // a token that sees K or fewer takes them all
    threadgroup uint * cnt = sh;                // 32 (simd partials)
    threadgroup uint * pg = sh + 32;            // 32: per-simdgroup counts above
    threadgroup uint * pe = sh + 64;            // 32: per-simdgroup counts equal
    const uint n = vis[tok];
    scores += (uint64_t)tok * a.stride; idx += (uint64_t)tok * a.k;
    if (n <= a.k) { for (uint i = tid; i < n; i += 1024) idx[i] = i; return; }
    const uint per = (n + 1023) / 1024, s0 = tid * per, s1 = min(n, s0 + per);
    // the largest T with at least K keys >= T, bit by bit
    uint t = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const uint cand = t | (1u << bit);
        uint c = 0;
        for (uint i = s0; i < s1; ++i) c += d4_key(scores[i]) >= cand;
        c = simd_sum(c);
        if (tid % 32 == 0) cnt[tid / 32] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tot = 0;
        for (uint s = 0; s < 32; ++s) tot += cnt[s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tot >= a.k) t = cand;
    }
    uint g = 0, e = 0;
    for (uint i = s0; i < s1; ++i) { const uint kk = d4_key(scores[i]); g += kk > t; e += kk == t; }
    // exclusive prefix sums over the threads (index order): within each
    // simdgroup, then over the simdgroups' totals
    const uint gx = simd_prefix_exclusive_sum(g), ex = simd_prefix_exclusive_sum(e);
    if (tid % 32 == 31) { pg[tid / 32] = gx + g; pe[tid / 32] = ex + e; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint gbefore = gx, ebefore = ex, gtot = 0;
    for (uint s = 0; s < 32; ++s) { if (s < tid / 32) { gbefore += pg[s]; ebefore += pe[s]; } gtot += pg[s]; }
    const uint ties = a.k - gtot;
    uint at = gbefore + min(ebefore, ties), et = ebefore;
    for (uint i = s0; i < s1; ++i) {
        const uint kk = d4_key(scores[i]);
        if (kk > t) idx[at++] = i;
        else if (kk == t && et < ties) { idx[at++] = i; et++; }
    }
}

// MOE[r] = sum of the K experts' rows of DOWN, in order, plus SHARED.
struct d4_msum_args { uint d; uint k; };

kernel void d4_moe_sum(constant d4_msum_args & a [[buffer(0)]],
                       device const float * down [[buffer(1)]], device const float * shared_ [[buffer(2)]],
                       device float * moe [[buffer(3)]], uint r [[thread_position_in_grid]]) {
    if (r >= a.d) return;
    float acc = 0.f;
    for (uint i = 0; i < a.k; ++i) acc += down[i * a.d + r];
    moe[r] = acc + shared_[r];
}

// One token's attention (decode): a threadgroup takes 4 heads (128
// threads), so the 64 heads spread over 16 cores.  Its keys: NRAW rows of
// the raw ring KR (RCAP rows) from R0, then NCOMP compressed rows of KC (the
// first NCOMP, or IDX's).  Each key row is read once for the 4 heads: first
// the scores (a thread a key), then the softmax from each head's sink (the
// sink in the denominator, as the CPU's), then the sum (a thread 4 dims of
// each head).  All in f32.
struct d4_att1_args { uint hd; float scale; uint rcap; uint r0; uint nraw; uint ncomp; uint ident; };

kernel void d4_attend1(constant d4_att1_args & a [[buffer(0)]],
                       device const float * q [[buffer(1)]], device const float * kr [[buffer(2)]],
                       device const float * kc [[buffer(3)]], device const uint * idx [[buffer(4)]],
                       device const float * sinks [[buffer(5)]], device float * out [[buffer(6)]],
                       threadgroup float * sh [[threadgroup(0)]],
                       uint g [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint hd = a.hd, total = a.nraw + a.ncomp, h0 = g * 4;
    threadgroup float * qs = sh;                    // 4 x hd
    threadgroup float * ps = sh + 4 * hd;           // 4 x total: scores, then weights
    threadgroup float * red = ps + 4 * total;       // 8
    for (uint i = tid; i < 4 * hd; i += 128) qs[i] = q[(uint64_t)h0 * hd + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = tid; j < total; j += 128) {
        device const float * row = j < a.nraw ? kr + (uint64_t)((a.r0 + j) % a.rcap) * hd
                                              : kc + (uint64_t)(a.ident ? j - a.nraw : idx[j - a.nraw]) * hd;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        for (uint d = 0; d < hd; d += 4) {
            const float4 k = *(device const float4 *)(row + d);
            s0 += dot(k, *(threadgroup const float4 *)(qs + d));
            s1 += dot(k, *(threadgroup const float4 *)(qs + hd + d));
            s2 += dot(k, *(threadgroup const float4 *)(qs + 2 * hd + d));
            s3 += dot(k, *(threadgroup const float4 *)(qs + 3 * hd + d));
        }
        ps[j] = s0 * a.scale; ps[total + j] = s1 * a.scale; ps[2 * total + j] = s2 * a.scale; ps[3 * total + j] = s3 * a.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // each head's max and denominator (simdgroup hh takes head hh)
    {
        const ushort hh = tid / 32, lane = tid % 32;
        threadgroup float * p = ps + hh * total;
        float mx = sinks[h0 + hh];
        for (uint j = lane; j < total; j += 32) mx = max(mx, p[j]);
        mx = simd_max(mx);
        float den = 0.f;
        for (uint j = lane; j < total; j += 32) { const float w = exp(p[j] - mx); p[j] = w; den += w; }
        den = simd_sum(den) + exp(sinks[h0 + hh] - mx);
        if (lane == 0) red[hh] = 1.f / den;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
    const uint d = tid * 4;
    for (uint j = 0; j < total; ++j) {
        device const float * row = j < a.nraw ? kr + (uint64_t)((a.r0 + j) % a.rcap) * hd
                                              : kc + (uint64_t)(a.ident ? j - a.nraw : idx[j - a.nraw]) * hd;
        const float4 v = *(device const float4 *)(row + d);
        o0 += ps[j] * v; o1 += ps[total + j] * v; o2 += ps[2 * total + j] * v; o3 += ps[3 * total + j] * v;
    }
    device float * ob = out + (uint64_t)h0 * hd + d;
    *(device float4 *)ob = o0 * red[0];
    *(device float4 *)(ob + hd) = o1 * red[1];
    *(device float4 *)(ob + 2 * hd) = o2 * red[2];
    *(device float4 *)(ob + 3 * hd) = o3 * red[3];
}

// --- prefill on the GPU: a chunk's layer, its activations staying there --------------

// KR[i] = RING[(R0 + i) % RCAP] for NROWS rows of HD: the window's rows,
// oldest first, ahead of the chunk's own.
struct d4_ring_args { uint hd; uint rcap; uint r0; uint k0; uint nrows; };

kernel void d4_ring_gather(constant d4_ring_args & a [[buffer(0)]],
                           device const float * ring [[buffer(1)]], device float * kr [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= a.nrows * a.hd) return;
    const uint i = gid / a.hd, j = gid % a.hd;
    kr[(uint64_t)(a.k0 + i) * a.hd + j] = ring[(uint64_t)((a.r0 + i) % a.rcap) * a.hd + j];
}

// RING[(R0 + i) % RCAP] = KR[K0 + i]: the chunk's last rows into the window.
kernel void d4_ring_store(constant d4_ring_args & a [[buffer(0)]],
                          device float * ring [[buffer(1)]], device const float * kr [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= a.nrows * a.hd) return;
    const uint i = gid / a.hd, j = gid % a.hd;
    ring[(uint64_t)((a.r0 + i) % a.rcap) * a.hd + j] = kr[(uint64_t)(a.k0 + i) * a.hd + j];
}

// A compressor over a chunk.  Token Q's projected row (WIDTH: for ratio 4 a
// previous-window half then a current one) and score (plus APE's row Q mod
// RATIO): the chunk's own (PROJ_KV, PROJ_SC) from POS0, else the window the
// context carries (ST_*: ratio 4, rows RATIO + Q mod RATIO for the current
// window, Q mod RATIO for the one before; ratio 128, Q mod RATIO).
struct d4_cb_args { uint hd; uint ratio; uint width; float eps; uint kind; uint nrot;
                    uint dst0; uint pos0; uint p0; uint wstart; uint ape_cols; };

static void d4_comp_src(constant d4_cb_args & a, int q, uint col,
                        device const float * pkv, device const float * psc, device const float * ape,
                        device const float * st_kv, device const float * st_sc, thread float & kv, thread float & sc) {
    if (q < 0) { kv = 0.f; sc = -1e30f; return; }
    const uint uq = (uint)q;
    if (uq >= a.pos0) {
        const uint64_t o = (uint64_t)(uq - a.pos0) * a.width + col;
        kv = pkv[o]; sc = psc[o] + ape[(uq % a.ratio) * a.ape_cols + col];
    } else {
        const uint row = (a.ratio == 4 && uq >= a.wstart) ? a.ratio + uq % a.ratio : uq % a.ratio;
        kv = st_kv[row * a.width + col]; sc = st_sc[row * a.width + col];
    }
}

// Every window that ends in the chunk (boundary B at P0 + B * RATIO), as
// D4_COMP_POOL pools it: compressed row DST0 + B.  A threadgroup a boundary,
// HD threads; CS its RoPE table (NROT a boundary).
kernel void d4_comp_pool_chunk(constant d4_cb_args & a [[buffer(0)]],
                               device const float * pkv [[buffer(1)]], device const float * psc [[buffer(2)]],
                               device const float * ape [[buffer(3)]], device const float * st_kv [[buffer(4)]],
                               device const float * st_sc [[buffer(5)]], device const float * w [[buffer(6)]],
                               device const float * cs [[buffer(7)]], device float * rows [[buffer(8)]],
                               threadgroup float * sh [[threadgroup(0)]],
                               uint b [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float * v = sh, * red = sh + a.hd;
    const uint j = tid, hd = a.hd, r4 = a.ratio == 4;
    const int p = (int)(a.p0 + b * a.ratio);
    float mx = -1e30f;
    for (uint r = 0; r < a.ratio; ++r) {
        float kv, sc;
        d4_comp_src(a, p - (int)a.ratio + 1 + (int)r, (r4 ? hd : 0) + j, pkv, psc, ape, st_kv, st_sc, kv, sc);
        mx = max(mx, sc);
        if (r4) { d4_comp_src(a, p - 2 * (int)a.ratio + 1 + (int)r, j, pkv, psc, ape, st_kv, st_sc, kv, sc); mx = max(mx, sc); }
    }
    float pooled = 0.f;
    if (mx > -5e29f) {
        float denom = 0.f, sum = 0.f;
        for (uint r = 0; r < a.ratio; ++r) {
            float kv, sc;
            if (r4) {
                float kp, sp;
                d4_comp_src(a, p - 2 * (int)a.ratio + 1 + (int)r, j, pkv, psc, ape, st_kv, st_sc, kp, sp);
                d4_comp_src(a, p - (int)a.ratio + 1 + (int)r, hd + j, pkv, psc, ape, st_kv, st_sc, kv, sc);
                const float wp = exp(sp - mx), wc = exp(sc - mx);
                denom += wp + wc; sum += wp * kp; sum += wc * kv;
            } else {
                d4_comp_src(a, p - (int)a.ratio + 1 + (int)r, j, pkv, psc, ape, st_kv, st_sc, kv, sc);
                const float ww = exp(sc - mx);
                denom += ww; sum += ww * kv;
            }
        }
        pooled = denom > 0.f ? sum / denom : 0.f;
    }
    const float ss = d4_tg_sum(pooled * pooled, red, tid, hd);
    const float scale = 1.f / sqrt(ss / (float)hd + a.eps);
    v[j] = (pooled * scale) * w[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    d4_rope_row(v, hd, a.nrot, cs + b * a.nrot, tid);
    d4_round_row(v, hd, a.nrot, a.kind, tid, red);
    rows[(uint64_t)(a.dst0 + b) * hd + j] = (float)(half)v[j];
}

// The window the context carries after the chunk (ending at E = POS0 + N - 1,
// in window [CW, CW + RATIO)), as the CPU's would hold it: the current
// window's rows for the chunk's tokens in it, and for ratio 4 the window
// before (or, when E ends a window, that window) as the previous half.  One
// threadgroup of 1024; the carried rows it reads are read before any write.
struct d4_cc_args { uint ratio; uint width; uint pos0; uint n; uint wstart; uint ape_cols; uint prev_from; uint set_prev; };

kernel void d4_comp_carry(constant d4_cc_args & a [[buffer(0)]],
                          device const float * pkv [[buffer(1)]], device const float * psc [[buffer(2)]],
                          device const float * ape [[buffer(3)]], device float * st_kv [[buffer(4)]],
                          device float * st_sc [[buffer(5)]],
                          ushort tid [[thread_index_in_threadgroup]]) {
    const uint e = a.pos0 + a.n - 1, cw = (e / a.ratio) * a.ratio, width = a.width, r4 = a.ratio == 4;
    // the previous half's new rows, gathered first (they may read rows being rewritten)
    float pk[4], ps[4];
    const uint np = r4 && a.set_prev ? a.ratio * width : 0;
    for (uint k = 0; k < 4; ++k) {
        const uint i = tid + k * 1024;
        if (i < np) {
            const uint r = i / width, col = i % width, q = a.prev_from + r;
            if (q >= a.pos0) {
                const uint64_t o = (uint64_t)(q - a.pos0) * width + col;
                pk[k] = pkv[o]; ps[k] = psc[o] + ape[(q % a.ratio) * a.ape_cols + col];
            } else {
                const uint row = q >= a.wstart ? a.ratio + q % a.ratio : q % a.ratio;
                pk[k] = st_kv[row * width + col]; ps[k] = st_sc[row * width + col];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint k = 0; k < 4; ++k) {
        const uint i = tid + k * 1024;
        if (i < np) { st_kv[i] = pk[k]; st_sc[i] = ps[k]; }
    }
    // the current window's rows for the chunk's tokens in it
    const uint q0 = max(cw, a.pos0);
    const uint nc = (e + 1 - q0) * width;
    for (uint i = tid; i < nc; i += 1024) {
        const uint q = q0 + i / width, col = i % width;
        const uint row = r4 ? a.ratio + q % a.ratio : q % a.ratio;
        const uint64_t o = (uint64_t)(q - a.pos0) * width + col;
        st_kv[row * width + col] = pkv[o];
        st_sc[row * width + col] = psc[o] + ape[(q % a.ratio) * a.ape_cols + col];
    }
}

// X[r] = SRC[MAP[r]], rows of D: the tokens an expert serves, gathered.
struct d4_gather_args { uint d; uint nrows; };

kernel void d4_gather_rows(constant d4_gather_args & a [[buffer(0)]],
                           device const float * src [[buffer(1)]], device const uint * map [[buffer(2)]],
                           device float * x [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= a.nrows * a.d) return;
    const uint r = gid / a.d, j = gid % a.d;
    x[gid] = src[(uint64_t)map[r] * a.d + j];
}

// MOE[t] = its NU slots' rows of DOWN (MAP[t * NU + slot]), in slot order,
// plus SHARED[t].
struct d4_msum2_args { uint d; uint nu; uint ntok; };

kernel void d4_moe_sum_rows(constant d4_msum2_args & a [[buffer(0)]],
                            device const float * down [[buffer(1)]], device const uint * map [[buffer(2)]],
                            device const float * shared_ [[buffer(3)]], device float * moe [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= a.ntok * a.d) return;
    const uint t = gid / a.d, j = gid % a.d;
    float acc = 0.f;
    for (uint s = 0; s < a.nu; ++s) acc += down[(uint64_t)map[t * a.nu + s] * a.d + j];
    moe[gid] = acc + shared_[gid];
}

// --- a prefill chunk's routing on the GPU ------------------------------------------------

// D4-ROUTE for each token (a threadgroup, a thread an expert): the hash
// table's experts (HASH), else the top NU of sqrt(softplus(logit)) + bias,
// the earlier expert first of equal ones (the CPU's insertion order, its
// slot order); weights the unbiased probabilities, renormalized, scaled.
struct d4_route_args { uint ne; uint nu; uint hash; float scale; uint has_bias; };

kernel void d4_route(constant d4_route_args & a [[buffer(0)]],
                     device const float * logits [[buffer(1)]], device const float * bias [[buffer(2)]],
                     device const int * tid2eid [[buffer(3)]], device const uint * tokens [[buffer(4)]],
                     device uint * sel [[buffer(5)]], device float * wout [[buffer(6)]],
                     threadgroup float * sh [[threadgroup(0)]],
                     uint t [[threadgroup_position_in_grid]], ushort e [[thread_index_in_threadgroup]]) {
    threadgroup float * probs = sh;                 // NE
    threadgroup float * sv = sh + a.ne;             // NE: the scores the pick runs over
    threadgroup float * rv = sh + 2 * a.ne;         // 32: simdgroup maxima
    threadgroup uint * ri = (threadgroup uint *)(sh + 2 * a.ne + 32);   // 32: their experts
    threadgroup uint * picked = (threadgroup uint *)(sh + 2 * a.ne + 64); // NU
    const float l = logits[(uint64_t)t * a.ne + e];
    // precise:: (closer to the CPU's, which rounds a double result; an ulp
    // apart still: the weights agree to about 1e-7, the experts exactly)
    const float sp = l > 20.f ? l : (l < -20.f ? precise::exp(l) : precise::log(1.f + precise::exp(l)));
    const float p = precise::sqrt(sp);
    probs[e] = p;
    sv[e] = p + (a.has_bias ? bias[e] : 0.f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (a.hash) {
        if (e < a.nu) picked[e] = (uint)tid2eid[(uint64_t)tokens[t] * a.nu + e];
    } else {
        for (uint k = 0; k < a.nu; ++k) {
            // the largest, the earlier of equal ones: reduce (value, -index)
            float v = sv[e]; uint idx = e;
            for (ushort off = 16; off > 0; off /= 2) {
                const float ov = simd_shuffle_down(v, off); const uint oi = simd_shuffle_down(idx, off);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (e % 32 == 0) { rv[e / 32] = v; ri[e / 32] = idx; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (e == 0) {
                float bv = rv[0]; uint bi = ri[0];
                for (uint s = 1; s < a.ne / 32; ++s)
                    if (rv[s] > bv || (rv[s] == bv && ri[s] < bi)) { bv = rv[s]; bi = ri[s]; }
                picked[k] = bi;
                sv[bi] = -INFINITY;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) {
        float sum = 0.f;
        for (uint i = 0; i < a.nu; ++i) sum += probs[picked[i]];
        if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
        for (uint i = 0; i < a.nu; ++i) {
            sel[t * a.nu + i] = picked[i];
            wout[t * a.nu + i] = precise::divide(probs[picked[i]], sum) * a.scale;
        }
    }
}

// The chunk's (token, slot) rows grouped by expert: each one's rank among
// its expert's (COUNT, atomically), the experts' first rows (SCAN), then each
// row's token and weight, and the way back from (token, slot) (SCATTER).
struct d4_group_args { uint n; uint ne; uint nu; };

kernel void d4_group_zero(device atomic_uint * counts [[buffer(1)]], uint e [[thread_position_in_grid]]) {
    atomic_store_explicit(counts + e, 0u, memory_order_relaxed);
}

kernel void d4_group_count(constant d4_group_args & a [[buffer(0)]],
                           device const uint * sel [[buffer(1)]], device atomic_uint * counts [[buffer(2)]],
                           device uint * rank [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= a.n) return;
    rank[gid] = atomic_fetch_add_explicit(counts + sel[gid], 1u, memory_order_relaxed);
}

kernel void d4_group_scan(constant d4_group_args & a [[buffer(0)]],
                          device const uint * counts [[buffer(1)]], device uint * bases [[buffer(2)]],
                          threadgroup uint * sh [[threadgroup(0)]], ushort e [[thread_index_in_threadgroup]]) {
    const uint c = e < a.ne ? counts[e] : 0u;
    const uint x = simd_prefix_exclusive_sum(c);
    if (e % 32 == 31) sh[e / 32] = x + c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = x;
    for (uint s = 0; s < e / 32; ++s) before += sh[s];
    if (e < a.ne) bases[e] = before;
}

kernel void d4_group_scatter(constant d4_group_args & a [[buffer(0)]],
                             device const uint * sel [[buffer(1)]], device const uint * rank [[buffer(2)]],
                             device const uint * bases [[buffer(3)]], device const float * w [[buffer(4)]],
                             device uint * rmap [[buffer(5)]], device uint * smap [[buffer(6)]],
                             device float * wrow [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= a.n) return;
    const uint row = bases[sel[gid]] + rank[gid];
    rmap[row] = gid / a.nu; smap[gid] = row; wrow[row] = w[gid];
}

// Y = W X, W's rows F16 (the file's): D4_MV_F32's shape, half the bytes.
kernel void d4_mv_f16(constant d4_mvf_args & a [[buffer(0)]],
                      device const half * w [[buffer(1)]], device const float * x [[buffer(2)]],
                      device float * y [[buffer(3)]],
                      threadgroup float * red [[threadgroup(0)]],
                      uint tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint r0 = tg * 4;
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    device const float4 * x4 = (device const float4 *)x;
    const uint n4 = a.ncols / 4;
    for (uint i = tid; i < n4; i += 256) {
        const float4 xv = x4[i];
        for (short r = 0; r < 4; ++r)
            if (r0 + r < a.nrows) s[r] += dot(float4(((device const half4 *)(w + (uint64_t)(r0 + r) * a.ncols))[i]), xv);
    }
    for (short r = 0; r < 4; ++r) {
        const float t = d4_tg_sum(s[r], red, tid, 256);
        if (tid == 0 && r0 + r < a.nrows) y[r0 + r] = t;
    }
}

// The output head's mix: W[h] = sigmoid(PRE[h] SCALE + BASE[h]) + EPS, OUT[j]
// = sum_h STREAMS[h][j] W[h] (output_logits_one, before its norm).
struct d4_hmix_args { uint d; uint n; float eps; };

kernel void d4_head_mix(constant d4_hmix_args & a [[buffer(0)]],
                        device const float * pre [[buffer(1)]], device const float * scale [[buffer(2)]],
                        device const float * base [[buffer(3)]], device const float * streams [[buffer(4)]],
                        device float * out [[buffer(5)]], uint j [[thread_position_in_grid]]) {
    if (j >= a.d) return;
    float acc = 0.f;
    for (uint h = 0; h < a.n; ++h) {
        const float z = pre[h] * scale[0] + base[h];
        const float sg = z >= 0.f ? 1.f / (1.f + exp(-z)) : exp(z) / (1.f + exp(z));
        acc += streams[h * a.d + j] * (sg + a.eps);
    }
    out[j] = acc;
}
