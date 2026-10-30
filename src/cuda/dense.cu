// dense.cu — the qwen2 / qwen3 / llama decoder's kernels (cuda-dense.lisp).
//
// Compiled by NVRTC when the device opens (cuda.lisp); every kernel is
// extern "C" so the Lisp side finds it by name.  The arithmetic follows the
// CPU path (model.lisp, tensor.lisp) where it is cheap to: activations are
// quantized as QUANTIZE-Q8! does (scale amax/127 stored rounded to f16, the
// int8s rounded half-even against the unrounded scale), a block's int8
// products are summed exactly before its two scales multiply them, and
// norms sum their squares in double.  What differs is the order partial sums
// meet in, which is the GPU's to choose.

typedef signed char i8;
typedef unsigned char u8;
typedef unsigned short u16;
#define INFINITY __int_as_float(0x7f800000)

__device__ __forceinline__ float h2f(u16 h) {
  float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ u16 f2h(float f) {
  u16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}
__device__ __forceinline__ double warp_sum_d(double v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}

// --- weights ------------------------------------------------------------------

// A Q8_0 tensor's 34-byte blocks (f16 scale, 32 int8) split into its int8s,
// row-major and 16-byte aligned, and its scales: N blocks.
extern "C" __global__ void repack_q8(const u8* __restrict__ raw, i8* __restrict__ qs,
                                     u16* __restrict__ d, long long n) {
  long long b = (long long)blockIdx.x * blockDim.y + threadIdx.y;
  if (b >= n) return;
  const u8* p = raw + b * 34;
  int j = threadIdx.x;                  // 32 threads a block
  qs[b * 32 + j] = (i8)p[2 + j];
  if (j == 0) d[b] = (u16)(p[0] | (p[1] << 8));
}

// Rows TOKENS[t] of a Q8_0 matrix (COLS wide) as floats, one row a block.
extern "C" __global__ void embed_q8(const i8* __restrict__ qs, const u16* __restrict__ d,
                                    const int* __restrict__ tokens, float* __restrict__ out, int cols) {
  int t = blockIdx.x;
  long long row = tokens[t];
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    long long i = row * cols + j;
    out[(long long)t * cols + j] = h2f(d[i >> 5]) * (float)qs[i];
  }
}

// --- activations --------------------------------------------------------------

// One block of 32 (a warp, lane J holding element J) quantized as
// QUANTIZE-Q8! does.
__device__ __forceinline__ void quant_block(float v, i8* q, float* s, int lane) {
  float amax = warp_max(fabsf(v));
  float dd = amax / 127.0f;
  float id = dd == 0.0f ? 0.0f : 1.0f / dd;
  q[lane] = (i8)__float2int_rn(v * id);
  if (lane == 0) *s = h2f(f2h(dd));
}

// X (N rows of COLS floats) quantized to XQ / XS: a warp a block.
extern "C" __global__ void quant_q8(const float* __restrict__ x, i8* __restrict__ xq,
                                    float* __restrict__ xs, long long nblocks) {
  long long b = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (b >= nblocks) return;
  quant_block(x[b * 32 + lane], xq + b * 32, xs + b, lane);
}

// RMSNORM! of each of the N rows of X (D wide), weighted by W, then quantized:
// a block a row.  The squares are summed in double; the scale is a float as
// (/ 1 (sqrt (coerce (+ (/ sum n) eps) 'single-float))).  H gets the floats
// when it isn't null.
extern "C" __global__ void rmsnorm_q8(const float* __restrict__ x, const float* __restrict__ w,
                                      float* __restrict__ h, i8* __restrict__ xq, float* __restrict__ xs,
                                      int dim, float eps) {
  __shared__ double part[32];
  __shared__ float sc;
  const float* xr = x + (long long)blockIdx.x * dim;
  double acc = 0.0;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) { double v = xr[i]; acc += v * v; }
  acc = warp_sum_d(acc);
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  if (lane == 0) part[warp] = acc;
  __syncthreads();
  if (warp == 0) {
    double v = lane < nw ? part[lane] : 0.0;
    v = warp_sum_d(v);
    if (lane == 0) sc = 1.0f / sqrtf((float)(v / (double)dim + (double)eps));
  }
  __syncthreads();
  float s = sc;
  long long base = (long long)blockIdx.x * dim;
  for (int b = warp; b < (dim >> 5); b += nw) {
    int i = b * 32 + lane;
    float v = (xr[i] * s) * w[i];
    if (h) h[base + i] = v;
    quant_block(v, xq + base + b * 32, xs + (base >> 5) + b, lane);
  }
}

// --- matrix x vector ----------------------------------------------------------

// Row R of a Q8_0 matrix against quantized X: each lane takes 16 bytes (half
// a block) at a time, the two halves' int sums meet before the scales.  The
// warp's total, in every lane.
__device__ __forceinline__ float row_dot_q8(const i8* __restrict__ wq, const u16* __restrict__ wd,
                                            long long r, int cols,
                                            const i8* __restrict__ xq, const float* __restrict__ xs,
                                            int lane) {
  int nb = cols >> 5, nh = nb * 2;
  const int4* w4 = (const int4*)(wq + r * cols);
  const int4* x4 = (const int4*)xq;
  const u16* wdr = wd + r * nb;
  float sum = 0.0f;
  for (int base = 0; base < nh; base += 32) {
    int hb = base + lane;
    int acc = 0;
    if (hb < nh) {
      int4 a = __ldg(w4 + hb), b = __ldg(x4 + hb);
      acc = __dp4a(a.x, b.x, acc); acc = __dp4a(a.y, b.y, acc);
      acc = __dp4a(a.z, b.z, acc); acc = __dp4a(a.w, b.w, acc);
    }
    acc += __shfl_xor_sync(0xffffffff, acc, 1);
    if (hb < nh && !(lane & 1)) {
      int bi = hb >> 1;
      sum += (float)acc * (h2f(__ldg(wdr + bi)) * __ldg(xs + bi));
    }
  }
  return warp_sum(sum);
}

// OUT[r] = W[r]·x (+ BIAS[r]), or OUT[r] += it when ACC: a warp a row.
extern "C" __global__ void mv_q8(const i8* __restrict__ wq, const u16* __restrict__ wd,
                                 const i8* __restrict__ xq, const float* __restrict__ xs,
                                 float* __restrict__ out, const float* __restrict__ bias,
                                 int rows, int cols, int acc) {
  long long r = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (r >= rows) return;
  float v = row_dot_q8(wq, wd, r, cols, xq, xs, lane);
  if (lane == 0) {
    if (bias) v += bias[r];
    out[r] = acc ? out[r] + v : v;
  }
}

// The FFN's two products and SWIGLU! at once: OUT[r] = silu(G[r]·x) * (U[r]·x).
extern "C" __global__ void mv_q8_glu(const i8* __restrict__ gq, const u16* __restrict__ gd,
                                     const i8* __restrict__ uq, const u16* __restrict__ ud,
                                     const i8* __restrict__ xq, const float* __restrict__ xs,
                                     float* __restrict__ out, int rows, int cols) {
  long long r = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (r >= rows) return;
  float g = row_dot_q8(gq, gd, r, cols, xq, xs, lane);
  float u = row_dot_q8(uq, ud, r, cols, xq, xs, lane);
  if (lane == 0) out[r] = (g / (1.0f + expf(-g))) * u;
}

// --- matrix x matrix (prefill) ------------------------------------------------
// Y[t][r] = W[r]·X[t] for N tokens.  A block takes 64 rows x 64 tokens and
// walks K four Q8_0 blocks (128 columns) at a time through shared memory;
// its 256 threads each hold 4 rows x 4 tokens.  dp4a: Pascal has it too.

#define MM_R 64
#define MM_T 64
#define MM_KB 4

extern "C" __global__ void __launch_bounds__(256)
mm_q8(const i8* __restrict__ wq, const u16* __restrict__ wd,
      const i8* __restrict__ xq, const float* __restrict__ xs,
      float* __restrict__ y, const float* __restrict__ bias,
      int rows, int cols, int n, int acc) {
  __shared__ int sw[MM_R][MM_KB * 8 + 1];   // a row's 4 blocks, as int32 words
  __shared__ int sx[MM_T][MM_KB * 8 + 1];
  __shared__ float swd[MM_R][MM_KB];
  __shared__ float sxd[MM_T][MM_KB];
  int r0 = blockIdx.y * MM_R, t0 = blockIdx.x * MM_T;
  int tid = threadIdx.x;
  int tr = tid >> 4, tt = tid & 15;          // this thread: rows tr+16i, tokens tt+16j
  int nb = cols >> 5;
  float c[4][4];
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) c[i][j] = 0.0f;
  for (int kb = 0; kb < nb; kb += MM_KB) {
    // load: 64 rows x 32 words, 64 tokens x 32 words; 256 threads, 8 words each
    for (int e = tid; e < MM_R * MM_KB * 8; e += 256) {
      int rr = e / (MM_KB * 8), ww = e % (MM_KB * 8);
      int r = r0 + rr, b = kb + ww / 8;
      sw[rr][ww] = (r < rows && b < nb) ? __ldg((const int*)(wq + (long long)r * cols + (long long)b * 32) + (ww & 7)) : 0;
      int t = t0 + rr;
      sx[rr][ww] = (t < n && b < nb) ? __ldg((const int*)(xq + (long long)t * cols + (long long)b * 32) + (ww & 7)) : 0;
    }
    for (int e = tid; e < MM_R * MM_KB; e += 256) {
      int rr = e / MM_KB, bb = e % MM_KB;
      int r = r0 + rr, b = kb + bb, t = t0 + rr;
      swd[rr][bb] = (r < rows && b < nb) ? h2f(__ldg(wd + (long long)r * nb + b)) : 0.0f;
      sxd[rr][bb] = (t < n && b < nb) ? __ldg(xs + (long long)t * nb + b) : 0.0f;
    }
    __syncthreads();
    for (int bb = 0; bb < MM_KB; bb++) {
      int s[4][4];
      for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) s[i][j] = 0;
      for (int w = 0; w < 8; w++) {
        int a[4], b[4];
        for (int i = 0; i < 4; i++) a[i] = sw[tr + 16 * i][bb * 8 + w];
        for (int j = 0; j < 4; j++) b[j] = sx[tt + 16 * j][bb * 8 + w];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) s[i][j] = __dp4a(a[i], b[j], s[i][j]);
      }
      for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
        c[i][j] += (float)s[i][j] * (swd[tr + 16 * i][bb] * sxd[tt + 16 * j][bb]);
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
    int r = r0 + tr + 16 * i, t = t0 + tt + 16 * j;
    if (r < rows && t < n) {
      float v = c[i][j] + (bias ? bias[r] : 0.0f);
      long long o = (long long)t * rows + r;
      y[o] = acc ? y[o] + v : v;
    }
  }
}

// A = silu(A) * B, elementwise.
extern "C" __global__ void swiglu(float* __restrict__ a, const float* __restrict__ b, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float g = a[i]; a[i] = (g / (1.0f + expf(-g))) * b[i]; }
}

// --- attention ----------------------------------------------------------------

// Q/K head norms (qwen3; when QN isn't null), RoPE from the table, and K, V
// into the caches: a block a head of a token, a thread a pair.  Q, K, V are
// N tokens' projections; the token T is at position POS0+T.  ROPE holds
// each position's cos (HD/2) then sin (HD/2).
extern "C" __global__ void qk_rope(float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
                                   const float* __restrict__ qn, const float* __restrict__ kn,
                                   const float* __restrict__ rope,
                                   float* __restrict__ kc, float* __restrict__ vc,
                                   int nh, int nkv, int hd, int pos0, float eps, int neox) {
  __shared__ double part[32];
  __shared__ float sc;
  int head = blockIdx.x, t = blockIdx.y, i = threadIdx.x, half = hd >> 1;
  int pos = pos0 + t;
  int qd = nh * hd, kvd = nkv * hd;
  bool isq = head < nh;
  float* x = isq ? q + (long long)t * qd + head * hd : k + (long long)t * kvd + (head - nh) * hd;
  const float* nw = isq ? qn : kn;
  if (nw) {
    double a = 0.0;
    for (int j = i; j < hd; j += blockDim.x) { double e = x[j]; a += e * e; }
    a = warp_sum_d(a);
    if ((i & 31) == 0) part[i >> 5] = a;
    __syncthreads();
    if (i == 0) {
      double s = 0.0;
      for (int w = 0; w < (blockDim.x + 31) >> 5; w++) s += part[w];
      sc = 1.0f / sqrtf((float)(s / (double)hd + (double)eps));
    }
    __syncthreads();
    for (int j = i; j < hd; j += blockDim.x) x[j] = (x[j] * sc) * nw[j];
    __syncthreads();
  }
  const float* cs = rope + (long long)pos * hd;
  for (int p = i; p < half; p += blockDim.x) {
    int a = neox ? p : 2 * p, b = neox ? p + half : 2 * p + 1;
    float x0 = x[a], x1 = x[b], c = cs[p], s = cs[half + p];
    x[a] = x0 * c - x1 * s;
    x[b] = x0 * s + x1 * c;
  }
  if (!isq) {
    __syncthreads();
    int kh = head - nh;
    long long o = (long long)pos * kvd + kh * hd;
    for (int j = i; j < hd; j += blockDim.x) {
      kc[o + j] = x[j];
      vc[o + j] = v[(long long)t * kvd + kh * hd + j];
    }
  }
}

// Attention, split along the keys: block (head, split, token) attends the
// token's query to positions [split*CHUNK, ...) up to its own, keeping a
// running max, sum and weighted V per warp; the warps' are merged and the
// split's (max, sum, out[HD]) written to PART.  ATTN_MERGE folds the
// splits.  HD up to 256 (8 elements a lane).
#define ATT_MAXE 8
extern "C" __global__ void attn_split(const float* __restrict__ q, const float* __restrict__ kc,
                                      const float* __restrict__ vc, float* __restrict__ part,
                                      int nh, int nkv, int hd, int pos0, int chunk, int nsplit, float scale) {
  __shared__ float sm[4], sl[4];
  __shared__ float so[4][256];
  int head = blockIdx.x, split = blockIdx.y, t = blockIdx.z;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int ne = hd >> 5;
  int n = pos0 + t + 1;                 // positions this token sees
  int p0 = split * chunk, p1 = min(n, p0 + chunk);
  int kvh = head / (nh / nkv);
  int kvd = nkv * hd;
  const float* qr = q + (long long)t * nh * hd + head * hd;
  float qv[ATT_MAXE], o[ATT_MAXE];
  for (int e = 0; e < ne; e++) { qv[e] = qr[lane + 32 * e]; o[e] = 0.0f; }
  float m = -INFINITY, l = 0.0f;
  for (int p = p0 + warp; p < p1; p += 4) {
    const float* kr = kc + (long long)p * kvd + kvh * hd;
    float s = 0.0f;
    for (int e = 0; e < ne; e++) s += qv[e] * kr[lane + 32 * e];
    s = warp_sum(s) * scale;
    float mn = fmaxf(m, s);
    float f = expf(m - mn), w = expf(s - mn);
    const float* vr = vc + (long long)p * kvd + kvh * hd;
    for (int e = 0; e < ne; e++) o[e] = o[e] * f + w * vr[lane + 32 * e];
    l = l * f + w;
    m = mn;
  }
  if (lane == 0) { sm[warp] = m; sl[warp] = l; }
  for (int e = 0; e < ne; e++) so[warp][lane + 32 * e] = o[e];
  __syncthreads();
  float* pr = part + (((long long)t * nh + head) * nsplit + split) * (hd + 2);
  float mx = fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3]));
  for (int j = threadIdx.x; j < hd; j += blockDim.x) {
    float acc = 0.0f;
    for (int w = 0; w < 4; w++) if (sm[w] != -INFINITY) acc += so[w][j] * expf(sm[w] - mx);
    pr[2 + j] = acc;
  }
  if (threadIdx.x == 0) {
    float ls = 0.0f;
    for (int w = 0; w < 4; w++) if (sm[w] != -INFINITY) ls += sl[w] * expf(sm[w] - mx);
    pr[0] = mx; pr[1] = ls;
  }
}

// The splits of each (token, head) merged into OUT: a block each, a thread
// an element.
extern "C" __global__ void attn_merge(const float* __restrict__ part, float* __restrict__ out,
                                      int nh, int hd, int pos0, int chunk, int nsplit) {
  int head = blockIdx.x, t = blockIdx.y, j = threadIdx.x;
  int n = pos0 + t + 1;
  int used = min(nsplit, (n + chunk - 1) / chunk);
  const float* pr = part + ((long long)t * nh + head) * nsplit * (hd + 2);
  float mx = -INFINITY;
  for (int s = 0; s < used; s++) mx = fmaxf(mx, pr[s * (hd + 2)]);
  float l = 0.0f, acc = 0.0f;
  for (int s = 0; s < used; s++) {
    const float* p = pr + s * (hd + 2);
    if (p[0] == -INFINITY) continue;
    float f = expf(p[0] - mx);
    l += p[1] * f;
    if (j < hd) acc += p[2 + j] * f;
  }
  if (j < hd) out[(long long)t * nh * hd + head * hd + j] = acc / l;
}
