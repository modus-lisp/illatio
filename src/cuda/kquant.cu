// kquant.cu — Q4_K, Q5_K and Q6_K matrices on the GPU (cuda-dense.lisp).
//
// The GPU's activations stay Q8_0's 32-wide blocks (dense.cu), with each
// block's int8 sum alongside for the K-quants' mins: a 32-weight group of
// any of these formats is 32 small integers q with
//
//     weight = A_lo*q (first 16) or A_hi*q (last 16), less B
//
// (Q4_K/Q5_K: A = d*scale, B = dmin*min; Q6_K: two 16-weight scales, its
// q taken as q-32, B = 0), so a group against an activation block (int8 y,
// scale s, sum Y) is s*(A_lo*sum(q*y over 16) + A_hi*sum(q*y over 16)) -
// B*s*Y, the integer sums exact.  ggml's own CUDA kernels do the same
// against its q8_1 blocks.  Q4_K and Q5_K blocks stay as the file has them
// (144 and 176 bytes: 16-byte aligned); Q6_K's 210 are padded to 224.

#define Q6P 224                                 // a padded Q6_K block

struct kgroup { int q[8]; float alo, ahi, b; }; // 32 int8s as words; the scales

// ggml's get_scale_min_k4: group J's 6-bit scale and min from the 12 bytes at P.
__device__ __forceinline__ void scale_min_k4(int j, const u8* p, int& sc, int& m) {
  if (j < 4) { sc = p[j] & 63; m = p[j + 4] & 63; }
  else { sc = (p[j + 4] & 0xF) | ((p[j - 4] >> 6) << 4); m = (p[j + 4] >> 4) | ((p[j] >> 6) << 4); }
}

// Group G (0..7) of the super-block at BLK, in its format's layout.
__device__ __forceinline__ void kgroup_q4k(const u8* __restrict__ blk, int g, kgroup& o) {
  const int4* qs = (const int4*)(blk + 16 + (g >> 1) * 32);
  int4 a = __ldg(qs), b = __ldg(qs + 1);
  int w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  int sh = (g & 1) * 4;
#pragma unroll
  for (int i = 0; i < 8; i++) o.q[i] = (w[i] >> sh) & 0x0F0F0F0F;
  int sc, m; scale_min_k4(g, blk + 4, sc, m);
  float d = h2f(*(const u16*)blk), dmin = h2f(*(const u16*)(blk + 2));
  o.alo = o.ahi = d * (float)sc; o.b = dmin * (float)m;
}

__device__ __forceinline__ void kgroup_q5k(const u8* __restrict__ blk, int g, kgroup& o) {
  const int4* qs = (const int4*)(blk + 48 + (g >> 1) * 32);
  const int4* qh = (const int4*)(blk + 16);
  int4 a = __ldg(qs), b = __ldg(qs + 1), c = __ldg(qh), e = __ldg(qh + 1);
  int w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  int h[8] = {c.x, c.y, c.z, c.w, e.x, e.y, e.z, e.w};
  int sh = (g & 1) * 4;
#pragma unroll
  for (int i = 0; i < 8; i++) o.q[i] = ((w[i] >> sh) & 0x0F0F0F0F) | (((h[i] >> g) & 0x01010101) << 4);
  int sc, m; scale_min_k4(g, blk + 4, sc, m);
  float d = h2f(*(const u16*)blk), dmin = h2f(*(const u16*)(blk + 2));
  o.alo = o.ahi = d * (float)sc; o.b = dmin * (float)m;
}

__device__ __forceinline__ void kgroup_q6k(const u8* __restrict__ blk, int g, kgroup& o) {
  int c = g >> 2, p = g & 3;
  const int4* ql = (const int4*)(blk + c * 64 + (p & 1) * 32);
  const int4* qh = (const int4*)(blk + 128 + c * 32);
  int4 a = __ldg(ql), b = __ldg(ql + 1), x = __ldg(qh), y = __ldg(qh + 1);
  int w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  int h[8] = {x.x, x.y, x.z, x.w, y.x, y.y, y.z, y.w};
  int sl = (p >> 1) * 4, shh = 2 * p;
#pragma unroll
  for (int i = 0; i < 8; i++)
    o.q[i] = __vsub4(((w[i] >> sl) & 0x0F0F0F0F) | (((h[i] >> shh) & 0x03030303) << 4), 0x20202020);
  const i8* sc = (const i8*)(blk + 192);
  float d = h2f(*(const u16*)(blk + 208));
  o.alo = d * (float)sc[2 * g]; o.ahi = d * (float)sc[2 * g + 1]; o.b = 0.0f;
}

// The super-block of format TYPE (0 Q4_K, 1 Q5_K, 2 Q6_K) holding row R's
// group GI (of COLS/32), and the group's place in it.
__device__ __forceinline__ void kgroup_at(int type, const u8* __restrict__ w, long long r, int cols, int gi, kgroup& o) {
  int nsb = cols >> 8;
  long long sb = r * nsb + (gi >> 3);
  if (type == 0) kgroup_q4k(w + sb * 144, gi & 7, o);
  else if (type == 1) kgroup_q5k(w + sb * 176, gi & 7, o);
  else kgroup_q6k(w + sb * Q6P, gi & 7, o);
}

// A group against activation block X (ints XQ, scale S, sum XSUM).
__device__ __forceinline__ float kgroup_dot(const kgroup& g, const int4* xq, float s, int ysum) {
  int4 a = __ldg(xq), b = __ldg(xq + 1);
  int lo = __dp4a(g.q[0], a.x, 0); lo = __dp4a(g.q[1], a.y, lo); lo = __dp4a(g.q[2], a.z, lo); lo = __dp4a(g.q[3], a.w, lo);
  int hi = __dp4a(g.q[4], b.x, 0); hi = __dp4a(g.q[5], b.y, hi); hi = __dp4a(g.q[6], b.z, hi); hi = __dp4a(g.q[7], b.w, hi);
  return s * (g.alo * (float)lo + g.ahi * (float)hi) - g.b * (s * (float)ysum);
}

// Row R against x: a lane a group at a time; the warp's total in every lane.
__device__ __forceinline__ float row_dot_k(int type, const u8* __restrict__ w, long long r, int cols,
                                           const i8* __restrict__ xq, const float* __restrict__ xs,
                                           const int* __restrict__ xsum, int lane) {
  int ng = cols >> 5;
  float sum = 0.0f;
  for (int gi = lane; gi < ng; gi += 32) {
    kgroup g; kgroup_at(type, w, r, cols, gi, g);
    sum += kgroup_dot(g, (const int4*)(xq + gi * 32), __ldg(xs + gi), __ldg(xsum + gi));
  }
  return warp_sum(sum);
}

extern "C" __global__ void mv_k(int type, const u8* __restrict__ w, const i8* __restrict__ xq,
                                const float* __restrict__ xs, const int* __restrict__ xsum,
                                float* __restrict__ out, const float* __restrict__ bias,
                                int rows, int cols, int acc) {
  long long r = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (r >= rows) return;
  float v = row_dot_k(type, w, r, cols, xq, xs, xsum, lane);
  if (lane == 0) {
    if (bias) v += bias[r];
    out[r] = acc ? out[r] + v : v;
  }
}

extern "C" __global__ void mv_k_glu(int gtype, const u8* __restrict__ gw, int utype, const u8* __restrict__ uw,
                                    const i8* __restrict__ xq, const float* __restrict__ xs,
                                    const int* __restrict__ xsum, float* __restrict__ out, int rows, int cols) {
  long long r = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (r >= rows) return;
  float g = row_dot_k(gtype, gw, r, cols, xq, xs, xsum, lane);
  float u = row_dot_k(utype, uw, r, cols, xq, xs, xsum, lane);
  if (lane == 0) out[r] = (g / (1.0f + expf(-g))) * u;
}

// Q6_K's 210-byte blocks to 224, N of them.
extern "C" __global__ void repack_q6k(const u8* __restrict__ raw, u8* __restrict__ out, long long n) {
  long long b = (long long)blockIdx.x * blockDim.y + threadIdx.y;
  if (b >= n) return;
  for (int j = threadIdx.x; j < Q6P; j += blockDim.x) out[b * Q6P + j] = j < 210 ? raw[b * 210 + j] : 0;
}

// Rows TOKENS[t] of a K-quant matrix as floats: a block a row, a thread a group.
extern "C" __global__ void embed_k(int type, const u8* __restrict__ w, const int* __restrict__ tokens,
                                   float* __restrict__ out, int cols) {
  int t = blockIdx.x;
  long long row = tokens[t];
  for (int gi = threadIdx.x; gi < (cols >> 5); gi += blockDim.x) {
    kgroup g; kgroup_at(type, w, row, cols, gi, g);
    for (int i = 0; i < 32; i++) {
      int q = (int)(signed char)((g.q[i >> 2] >> (8 * (i & 3))) & 0xFF);
      out[(long long)t * cols + gi * 32 + i] = (i < 16 ? g.alo : g.ahi) * (float)q - g.b;
    }
  }
}

// --- matrix x matrix (prefill), dp4a ----------------------------------------------
// Y[t][r] = W[r]·X[t] for N tokens, W in a K-quant format.  A block takes 64
// rows x 64 tokens and four groups (128 columns) at a time: each of its 256
// threads unpacks one (row, group) into shared memory as int8 words with
// its scales, then holds 4 rows x 4 tokens of the product.

#define MK_R 64
#define MK_T 64
extern "C" __global__ void __launch_bounds__(256)
mm_k(int type, const u8* __restrict__ w, const i8* __restrict__ xq, const float* __restrict__ xs,
     const int* __restrict__ xsum, float* __restrict__ y, const float* __restrict__ bias,
     int rows, int cols, int n, int acc) {
  __shared__ int sw[4][MK_R][9];                // a group's 8 words, padded
  __shared__ float sa[4][MK_R][3];              // alo, ahi, b
  __shared__ int sx[4][MK_T][9];
  __shared__ float sxs[4][MK_T][2];             // s, s*sum
  int r0 = blockIdx.y * MK_R, t0 = blockIdx.x * MK_T;
  int tid = threadIdx.x, tr = tid >> 4, tt = tid & 15;
  int ng = cols >> 5;
  float c[4][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) c[i][j] = 0.0f;
  for (int g0 = 0; g0 < ng; g0 += 4) {
    {
      int gg = tid >> 6, rr = tid & 63, r = r0 + rr, gi = g0 + gg;
      kgroup g;
      if (r < rows && gi < ng) kgroup_at(type, w, r, cols, gi, g);
      else { for (int i = 0; i < 8; i++) g.q[i] = 0; g.alo = g.ahi = g.b = 0.0f; }
#pragma unroll
      for (int i = 0; i < 8; i++) sw[gg][rr][i] = g.q[i];
      sa[gg][rr][0] = g.alo; sa[gg][rr][1] = g.ahi; sa[gg][rr][2] = g.b;
      int t = t0 + rr;
      bool ok = t < n && gi < ng;
      const int4* xp = (const int4*)(xq + (long long)t * cols + (long long)gi * 32);
      int4 a = ok ? __ldg(xp) : make_int4(0, 0, 0, 0), b = ok ? __ldg(xp + 1) : make_int4(0, 0, 0, 0);
      sx[gg][rr][0] = a.x; sx[gg][rr][1] = a.y; sx[gg][rr][2] = a.z; sx[gg][rr][3] = a.w;
      sx[gg][rr][4] = b.x; sx[gg][rr][5] = b.y; sx[gg][rr][6] = b.z; sx[gg][rr][7] = b.w;
      float sv = ok ? __ldg(xs + (long long)t * ng + gi) : 0.0f;
      sxs[gg][rr][0] = sv; sxs[gg][rr][1] = ok ? sv * (float)__ldg(xsum + (long long)t * ng + gi) : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int gg = 0; gg < 4; gg++) {
      int lo[4][4], hi[4][4];
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) lo[i][j] = hi[i][j] = 0;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        int a[4], b[4], a2[4], b2[4];
#pragma unroll
        for (int i = 0; i < 4; i++) { a[i] = sw[gg][tr + 16 * i][k]; a2[i] = sw[gg][tr + 16 * i][k + 4]; }
#pragma unroll
        for (int j = 0; j < 4; j++) { b[j] = sx[gg][tt + 16 * j][k]; b2[j] = sx[gg][tt + 16 * j][k + 4]; }
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
          for (int j = 0; j < 4; j++) { lo[i][j] = __dp4a(a[i], b[j], lo[i][j]); hi[i][j] = __dp4a(a2[i], b2[j], hi[i][j]); }
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float alo = sa[gg][tr + 16 * i][0], ahi = sa[gg][tr + 16 * i][1], bb = sa[gg][tr + 16 * i][2];
#pragma unroll
        for (int j = 0; j < 4; j++)
          c[i][j] += sxs[gg][tt + 16 * j][0] * (alo * (float)lo[i][j] + ahi * (float)hi[i][j]) - bb * sxs[gg][tt + 16 * j][1];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int r = r0 + tr + 16 * i, t = t0 + tt + 16 * j;
      if (r < rows && t < n) {
        float v = c[i][j] + (bias ? bias[r] : 0.0f);
        long long o = (long long)t * rows + r;
        y[o] = acc ? y[o] + v : v;
      }
    }
}
