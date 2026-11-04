// ds4f_mm.cu — DS4F's matrix products for a chunk of tokens, on the tensor
// cores (sm_80 and up: prefill runs on the first device only).
//
// Metal's D4_MM (ggml's kernel_mul_mm, as ds4 runs it): the weights
// dequantized to halves 32 columns at a time, the activations rounded to
// halves, their products summed in floats.  A block takes 64 rows x 64
// tokens, its four warps 32 x 32 each, as m16n8k16 products from shared
// memory (rows 40 halves apart: no bank conflicts on the fragments' loads).
// A thread dequantizes 16 consecutive weights of a row -- one Q8_0 half
// block, one Q2_K or IQ2_XXS sub-block of 16 (a single scale), or 16 F16s
// -- and rounds 16 of a token's activations.
//
// The formats (F): 0 Q8_0 repacked (int8s at W, a row's f16 scales at WD,
// as the decode's products take them), 1 F16, 2 Q2_K, 3 IQ2_XXS (the
// file's blocks, ROW_BYTES a row).
//
//   d4_mm_<f>(ncols, nrows, ntok, row_bytes, x_stride, y_stride, w, wd, x, y)
//     Y[t*y_stride + r] = W[r] . X[t*x_stride ..]; grid (ceil(ntok/64),
//     ceil(nrows/64)), 128 threads.  WD only for Q8_0.
//   d4_mm_id_<f>(ncols, nrows, row_bytes, x_stride, y_stride, add, ptrs,
//                counts, bases, x, y)
//     The routed experts at once: grid z an expert, its COUNTS[z] rows of X
//     from row BASES[z], its weights at PTRS[z] + ADD (device addresses);
//     grid (ceil(max count/64), ceil(nrows/64), experts), 128 threads.

#define D4M_R 64
#define D4M_T 64
#define D4M_LD 40

// A and B rounded to halves, A in the low half (the lower column).
__device__ __forceinline__ unsigned d4m_h2(float a, float b) {
  return (unsigned)f2h(a) | ((unsigned)f2h(b) << 16);
}

// Row R's weights K .. K+15 (K a multiple of 16): their bytes loaded
// (D4M_RAW), then decoded to floats (D4M_DEC) -- apart, so a step's loads
// are in flight while the step before is multiplied.
struct d4m_raw { uint4 q, q2; unsigned a, b, c; };

template <int F>
__device__ __forceinline__ void d4m_load(const unsigned char* __restrict__ w, const u16* __restrict__ wd,
                                         long long r, int row_bytes, int ncols, int k, d4m_raw& raw) {
  if (F == 0) {
    raw.q = __ldg((const uint4*)(w + r * ncols + k));
    raw.a = __ldg(wd + r * (ncols >> 5) + (k >> 5));
  } else if (F == 1) {
    const uint4* p = (const uint4*)(w + r * row_bytes + 2LL * k);
    raw.q = __ldg(p); raw.q2 = __ldg(p + 1);
  } else if (F == 2) {
    const unsigned char* blk = w + r * row_bytes + (long long)(k >> 8) * 84;
    const int il = (k & 255) >> 4;
    const unsigned* qp = (const unsigned*)(blk + 16 + 32 * (il >> 3) + 16 * (il & 1));   // blocks are 4-aligned
    raw.q = make_uint4(__ldg(qp), __ldg(qp + 1), __ldg(qp + 2), __ldg(qp + 3));
    raw.a = __ldg((const unsigned*)(blk + 80));                                           // d, dmin
    raw.b = __ldg(blk + il);
  } else {
    const unsigned char* blk = w + r * row_bytes + (long long)(k >> 8) * 66;
    const u16* q2 = (const u16*)(blk + 2) + 4 * ((k & 255) >> 5);
    raw.a = __ldg(q2) | ((unsigned)__ldg(q2 + 1) << 16);
    raw.b = __ldg(q2 + 2) | ((unsigned)__ldg(q2 + 3) << 16);
    raw.c = __ldg((const u16*)blk);
  }
}

__device__ __forceinline__ unsigned d4m_hmul2(unsigned a, unsigned b) {
  unsigned r; asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r;
}

// RAW decoded straight to eight half pairs.  IQ2_XXS: its grid's eight
// magnitudes as halves and its signs as masks of the halves' sign bits are
// tables in shared memory (GRIDH, SIGNM: four words an entry), the scale
// multiplied in as halves -- one rounding more than dequantizing in floats
// (the scale's).  Q2_K: the four values a sub-block's 2-bit quants can
// take, each rounded once, chosen from.
template <int F>
__device__ __forceinline__ void d4m_dec(const d4m_raw& raw, int k, const unsigned* gridh,
                                        const unsigned* signm, unsigned* out) {
  if (F == 0) {
    const float d = h2f((u16)raw.a);
    const unsigned u[4] = {raw.q.x, raw.q.y, raw.q.z, raw.q.w};
#pragma unroll
    for (int i = 0; i < 8; ++i)
      out[i] = d4m_h2((float)(signed char)(u[i >> 1] >> (16 * (i & 1))) * d,
                      (float)(signed char)(u[i >> 1] >> (16 * (i & 1) + 8)) * d);
  } else if (F == 1) {
    out[0] = raw.q.x; out[1] = raw.q.y; out[2] = raw.q.z; out[3] = raw.q.w;
    out[4] = raw.q2.x; out[5] = raw.q2.y; out[6] = raw.q2.z; out[7] = raw.q2.w;
  } else if (F == 2) {
    const int il = (k & 255) >> 4;
    const float d = h2f(raw.a & 0xffff), dmin = h2f(raw.a >> 16);
    const float dl = d * (raw.b & 0xF), ml = dmin * (raw.b >> 4);
    const int shift = 2 * ((il >> 1) & 3);
    const unsigned h[4] = {f2h(-ml), f2h(dl - ml), f2h(dl * 2 - ml), f2h(dl * 3 - ml)};
    const unsigned u[4] = {raw.q.x, raw.q.y, raw.q.z, raw.q.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const unsigned w = u[i >> 1] >> (16 * (i & 1) + shift);
      out[i] = h[w & 3] | (h[(w >> 8) & 3] << 16);
    }
  } else {
    const int h = ((k & 255) >> 4) & 1;
    const unsigned dl = f2h(h2f((u16)raw.c) * (0.5f + (raw.b >> 28)) * 0.25f);
    const unsigned dl2 = dl | (dl << 16);
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const uint4 gv = *(const uint4*)(gridh + 4 * ((raw.a >> (8 * (2 * h + e))) & 255));
      const uint4 sm = *(const uint4*)(signm + 4 * ((raw.b >> (14 * h + 7 * e)) & 127));
      out[4 * e + 0] = d4m_hmul2(gv.x ^ sm.x, dl2);
      out[4 * e + 1] = d4m_hmul2(gv.y ^ sm.y, dl2);
      out[4 * e + 2] = d4m_hmul2(gv.z ^ sm.z, dl2);
      out[4 * e + 3] = d4m_hmul2(gv.w ^ sm.w, dl2);
    }
  }
}

__device__ __forceinline__ void d4m_mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

template <int F>
__device__ __forceinline__ void d4m_body(int ncols, int nrows, int ntok, int row_bytes, int x_stride, int y_stride,
                                         const unsigned char* __restrict__ w, const u16* __restrict__ wd,
                                         const float* __restrict__ x, float* __restrict__ y, int bx, int by) {
#if __CUDA_ARCH__ >= 800
  // two stages: a step's tiles are written while the other's are read
  __shared__ __align__(16) u16 sa[2][D4M_R * D4M_LD];
  __shared__ __align__(16) u16 sb[2][D4M_T * D4M_LD];
  __shared__ __align__(16) unsigned gridh[F == 3 ? 256 * 4 : 4];
  __shared__ __align__(16) unsigned signm[F == 3 ? 128 * 4 : 4];
  const int r0 = by * D4M_R, t0 = bx * D4M_T;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tg = lane & 3;
  const int wr = (warp & 1) * 32, wt = (warp >> 1) * 32;
  const int lr = tid >> 1, lh = (tid & 1) * 16;      // this thread's loads: row/token LR, columns LH..LH+15
  const int jn = min(4, max(0, (ntok - t0 - wt + 7) >> 3));   // the warp's 8-token tiles with a token in
  const bool rok = r0 + lr < nrows, tok = t0 + lr < ntok;
  const float* xr = x + (long long)(t0 + lr) * x_stride + lh;
  if (F == 3) {
    // the tables in shared memory: from constant memory a warp's different entries are read one by one
    for (int i = tid; i < 256 * 4; i += 128) {
      const unsigned long long gv = d4_iq2xxs_grid[i >> 2];
      const int b = 2 * (i & 3);
      gridh[i] = d4m_h2((float)((gv >> (8 * b)) & 255), (float)((gv >> (8 * b + 8)) & 255));
    }
    for (int i = tid; i < 128 * 4; i += 128) {
      const unsigned char sg = d4_ksigns_iq2xs[i >> 2];
      const int b = 2 * (i & 3);
      signm[i] = (sg & (1 << b) ? 0x8000u : 0u) | (sg & (2 << b) ? 0x80000000u : 0u);
    }
    __syncthreads();
  }
  float c[2][4][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) c[i][j][e] = 0.f;
  d4m_raw raw;
  float4 xv[4];
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  if (rok) d4m_load<F>(w, wd, r0 + lr, row_bytes, ncols, lh, raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) xv[i] = tok ? __ldg((const float4*)xr + i) : z4;
  int st = 0;
  for (int k0 = 0; k0 < ncols; k0 += 32, st ^= 1) {
    {
      unsigned v[8];
      if (rok) d4m_dec<F>(raw, k0 + lh, gridh, signm, v);
      else {
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = 0u;
      }
      uint4* da = (uint4*)(sa[st] + lr * D4M_LD + lh);
      da[0] = make_uint4(v[0], v[1], v[2], v[3]);
      da[1] = make_uint4(v[4], v[5], v[6], v[7]);
      uint4* db = (uint4*)(sb[st] + lr * D4M_LD + lh);
      db[0] = make_uint4(d4m_h2(xv[0].x, xv[0].y), d4m_h2(xv[0].z, xv[0].w), d4m_h2(xv[1].x, xv[1].y), d4m_h2(xv[1].z, xv[1].w));
      db[1] = make_uint4(d4m_h2(xv[2].x, xv[2].y), d4m_h2(xv[2].z, xv[2].w), d4m_h2(xv[3].x, xv[3].y), d4m_h2(xv[3].z, xv[3].w));
    }
    __syncthreads();
    // the next step's bytes on their way while this one's are multiplied
    if (k0 + 32 < ncols) {
      if (rok) d4m_load<F>(w, wd, r0 + lr, row_bytes, ncols, k0 + 32 + lh, raw);
#pragma unroll
      for (int i = 0; i < 4; ++i) xv[i] = tok ? __ldg((const float4*)(xr + k0 + 32) + i) : z4;
    }
#pragma unroll
    for (int ks = 0; ks < 32; ks += 16) {
      unsigned a[2][4], b[4][2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const u16* p = sa[st] + (wr + 16 * i + g) * D4M_LD + ks + 2 * tg;
        a[i][0] = *(const unsigned*)p;
        a[i][1] = *(const unsigned*)(p + 8 * D4M_LD);
        a[i][2] = *(const unsigned*)(p + 8);
        a[i][3] = *(const unsigned*)(p + 8 * D4M_LD + 8);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const u16* p = sb[st] + (wt + 8 * j + g) * D4M_LD + ks + 2 * tg;
        b[j][0] = *(const unsigned*)p;
        b[j][1] = *(const unsigned*)(p + 8);
      }
      // (a chunk's expert has a few tokens: the tiles of none are not multiplied)
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          if (j < jn) d4m_mma(c[i][j], a[i], b[j][0], b[j][1]);
    }
  }
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int r = r0 + wr + 16 * i + g + (e >> 1) * 8, t = t0 + wt + 8 * j + 2 * tg + (e & 1);
        if (r < nrows && t < ntok) y[(long long)t * y_stride + r] = c[i][j][e];
      }
#endif
}

template <int F>
__device__ __forceinline__ void d4m_id(int ncols, int nrows, int row_bytes, int x_stride, int y_stride, long long add,
                                       const unsigned long long* __restrict__ ptrs, const unsigned* __restrict__ counts,
                                       const unsigned* __restrict__ bases, const float* __restrict__ x, float* __restrict__ y) {
  const int z = blockIdx.z, count = counts[z];
  if (blockIdx.x * D4M_T >= count) return;
  const long long base = bases[z];
  d4m_body<F>(ncols, nrows, count, row_bytes, x_stride, y_stride, (const unsigned char*)(ptrs[z] + add), nullptr,
              x + base * x_stride, y + base * y_stride, blockIdx.x, blockIdx.y);
}

#define D4M_MM(name, f)                                                                                   \
  extern "C" __global__ void __launch_bounds__(128)                                                       \
  name(int ncols, int nrows, int ntok, int row_bytes, int x_stride, int y_stride, const unsigned char* w, \
       const u16* wd, const float* x, float* y) {                                                         \
    d4m_body<f>(ncols, nrows, ntok, row_bytes, x_stride, y_stride, w, wd, x, y, blockIdx.x, blockIdx.y);  \
  }
#define D4M_MM_ID(name, f)                                                                                \
  extern "C" __global__ void __launch_bounds__(128)                                                       \
  name(int ncols, int nrows, int row_bytes, int x_stride, int y_stride, long long add,                    \
       const unsigned long long* ptrs, const unsigned* counts, const unsigned* bases, const float* x, float* y) { \
    d4m_id<f>(ncols, nrows, row_bytes, x_stride, y_stride, add, ptrs, counts, bases, x, y);               \
  }

D4M_MM(d4_mm_q8_0, 0)
D4M_MM(d4_mm_f16, 1)
D4M_MM(d4_mm_q2_k, 2)
D4M_MM(d4_mm_iq2_xxs, 3)
D4M_MM_ID(d4_mm_id_q2_k, 2)
D4M_MM_ID(d4_mm_id_iq2_xxs, 3)
