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
kernel void d4_mm(constant d4_mm_args & a [[buffer(0)]],
                  device const char  * src0 [[buffer(1)]],
                  device const float * src1 [[buffer(2)]],
                  device       float * dst  [[buffer(3)]],
                  threadgroup  char  * shmem [[threadgroup(0)]],
                  uint3  tgpig [[threadgroup_position_in_grid]],
                  ushort tiitg [[thread_index_in_threadgroup]],
                  ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);
    constexpr int NR0 = 64, NR1 = 64, NK = 32, NL0 = NK/16, NL1 = NK/8;
    const int r0 = tgpig.y*NR0, r1 = tgpig.x*NR1;
    const short nr0 = ((int)a.nrows - r0 < NR0) ? ((int)a.nrows - r0) : NR0;
    const short nr1 = ((int)a.ntok  - r1 < NR1) ? ((int)a.ntok  - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short t0 = (short)tiitg/NL1, t1 = t0 + 32;
    const short lr1a = t0 < nr1 ? t0 : nr1 - 1;
    const short lr1b = t1 < nr1 ? t1 : nr1 - 1;
    const short il0 = (tiitg % NL0);
    short il = il0;
    device const block_q * x = (device const block_q *)(src0 + (ulong)a.row_bytes*(r0 + lr0)) + il0/nl;
    const short iy = 8*(tiitg % NL1);
    device const float * ya = src1 + (ulong)a.x_stride*(r1 + lr1a) + iy;
    device const float * yb = src1 + (ulong)a.x_stride*(r1 + lr1b) + iy;
    simdgroup_half8x8 ma[4], mb[4];
    simdgroup_float8x8 mc[16];
    for (short i = 0; i < 16; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint loop_k = 0; loop_k < a.ncols; loop_k += NK) {
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
    if (r0 + NR0 <= (int)a.nrows && r1 + NR1 <= (int)a.ntok) {
        device float * C = dst + (r0 + 32*(sgitg & 1)) + (ulong)(r1 + 32*(sgitg >> 1)) * a.y_stride;
        for (short i = 0; i < 16; i++) simdgroup_store(mc[i], C + 8*(i%4) + 8*a.y_stride*(i/4), a.y_stride, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (32*(sgitg >> 1))*NR0;
        for (short i = 0; i < 16; i++) simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += 32) {
                device float * D = dst + r0 + (ulong)(r1 + j)*a.y_stride;
                threadgroup float * C = ((threadgroup float *) shmem) + j*NR0;
                for (int i = 0; i < nr0; i++) D[i] = C[i];
            }
        }
    }
}

typedef decltype(d4_mm<d4_block_q8_0, 2, d4_deq_q8_0>) d4_mm_t;
template [[host_name("d4_mm_q8_0")]]    kernel d4_mm_t d4_mm<d4_block_q8_0, 2, d4_deq_q8_0>;
template [[host_name("d4_mm_f16")]]     kernel d4_mm_t d4_mm<half4x4, 1, d4_deq_f16>;
template [[host_name("d4_mm_q2_k")]]    kernel d4_mm_t d4_mm<d4_block_q2_K, 16, d4_deq_q2_K>;
template [[host_name("d4_mm_iq2_xxs")]] kernel d4_mm_t d4_mm<d4_block_iq2_xxs, 16, d4_deq_iq2_xxs>;

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

struct d4_attn_args { uint nh; uint hd; float scale; uint maxc; };

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
            if (j < nr) v = kr[(ulong)(r0 + j) * D4_HD + dd];
            else if (j < total) v = kc[(ulong)idx[(ulong)t * a.maxc + (j - nr)] * D4_HD + dd];
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
