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
#define ATT_MAXE 8                      // attention: HD/32 elements a lane, HD up to 256

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

// Decode attention, split along the keys: block (KV head, split) attends
// the query heads that share the KV head (GQA's group, up to 8) to CHUNK
// positions (at most 64) of [split*CHUNK, POS0+1), so each key and value
// is read from memory once.  Memory-level parallelism is the point: K's
// chunk comes in as float4s, all of a thread's at once, into shared memory
// transposed; a thread scores a key for half the group; a warp takes a
// query's softmax; then V is read as float4s, a thread four columns of a
// part of the keys for every query, the parts summed in shared memory.
// Each split's (max, sum, out[HD]) goes to PART for ATTN_MERGE.  HD a
// multiple of 16, up to 256.
#define AD_CHUNK 64
#define AD_MAXG 8
extern "C" __global__ void __launch_bounds__(128)
attn_decode(const float* __restrict__ q, const float* __restrict__ kc,
            const float* __restrict__ vc, float* __restrict__ part,
            int nh, int nkv, int hd, int pos0, int chunk, int nsplit, float scale) {
  extern __shared__ float dsm[];
  float* sk = dsm;                            // [hd][AD_CHUNK+1]; later the V parts
  __shared__ float sq[AD_MAXG * 256];
  __shared__ float sp[AD_MAXG][AD_CHUNK];
  __shared__ float smx[AD_MAXG], ssum[AD_MAXG];
  int kvh = blockIdx.x, split = blockIdx.y, tid = threadIdx.x;
  int group = nh / nkv;
  int n = pos0 + 1;
  int p0 = split * chunk, cnt = min(chunk, n - p0);
  int kvd = nkv * hd;
  if (cnt <= 0) {
    if (tid < group) {
      float* pr = part + ((long long)(kvh * group + tid) * nsplit + split) * (hd + 2);
      pr[0] = -INFINITY; pr[1] = 0.0f;
    }
    return;
  }
  for (int e = tid; e < group * hd; e += blockDim.x) sq[e] = q[kvh * group * hd + e] * scale;
  int h4 = hd >> 2;                           // float4s a row
  const float* kb = kc + (long long)p0 * kvd + kvh * hd;
  const float* vb = vc + (long long)p0 * kvd + kvh * hd;
  {
    float4 r[16];                             // 64 rows x 64 float4 at most / 128 threads
#pragma unroll
    for (int u = 0; u < 16; u++) {
      int e = tid + u * 128, kk = e / h4, c = e % h4;
      r[u] = (kk < cnt) ? __ldg((const float4*)(kb + (long long)kk * kvd) + c) : make_float4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < 16; u++) {
      int e = tid + u * 128, kk = e / h4, c = e % h4;
      if (kk < AD_CHUNK) {
        sk[(4 * c + 0) * (AD_CHUNK + 1) + kk] = r[u].x; sk[(4 * c + 1) * (AD_CHUNK + 1) + kk] = r[u].y;
        sk[(4 * c + 2) * (AD_CHUNK + 1) + kk] = r[u].z; sk[(4 * c + 3) * (AD_CHUNK + 1) + kk] = r[u].w;
      }
    }
  }
  __syncthreads();
  {
    int kk = tid & 63;
    for (int g = tid >> 6; g < group; g += 2) {
      float s = -INFINITY;
      if (kk < cnt) {
        const float* qg = sq + g * hd;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int j = 0; j < hd; j += 4) {
          a0 += qg[j] * sk[j * (AD_CHUNK + 1) + kk];
          a1 += qg[j + 1] * sk[(j + 1) * (AD_CHUNK + 1) + kk];
          a2 += qg[j + 2] * sk[(j + 2) * (AD_CHUNK + 1) + kk];
          a3 += qg[j + 3] * sk[(j + 3) * (AD_CHUNK + 1) + kk];
        }
        s = (a0 + a1) + (a2 + a3);
      }
      sp[g][kk] = s;
    }
  }
  __syncthreads();
  {
    int warp = tid >> 5, lane = tid & 31;
    for (int g = warp; g < group; g += 4) {
      float s0 = sp[g][lane], s1 = sp[g][lane + 32];
      float mx = warp_max(fmaxf(s0, s1));
      float w0 = lane < cnt ? expf(s0 - mx) : 0.0f, w1 = lane + 32 < cnt ? expf(s1 - mx) : 0.0f;
      sp[g][lane] = w0; sp[g][lane + 32] = w1;
      float ls = warp_sum(w0 + w1);
      if (lane == 0) { smx[g] = mx; ssum[g] = ls; }
    }
  }
  __syncthreads();                            // sp, smx, ssum; sk is free for the V parts
  {
    int c = tid % h4, qt = tid / h4;          // four columns; which part of the keys
    int nq = 128 / h4;                        // parts: 4 for HD 128, 2 for 256, 8 for 64
    float4 acc[AD_MAXG];
#pragma unroll
    for (int g = 0; g < AD_MAXG; g++) acc[g] = make_float4(0, 0, 0, 0);
    for (int kk = qt; kk < cnt; kk += nq) {
      float4 v = __ldg((const float4*)(vb + (long long)kk * kvd) + c);
#pragma unroll
      for (int g = 0; g < AD_MAXG; g++) {
        if (g < group) {
          float pw = sp[g][kk];
          acc[g].x += pw * v.x; acc[g].y += pw * v.y; acc[g].z += pw * v.z; acc[g].w += pw * v.w;
        }
      }
    }
#pragma unroll
    for (int g = 0; g < AD_MAXG; g++)
      if (g < group) *(float4*)(sk + (qt * group + g) * hd + 4 * c) = acc[g];
    __syncthreads();
    for (int e = tid; e < group * hd; e += blockDim.x) {
      int g = e / hd, j = e % hd;
      float o = 0.0f;
      for (int u = 0; u < nq; u++) o += sk[(u * group + g) * hd + j];
      float* pr = part + ((long long)(kvh * group + g) * nsplit + split) * (hd + 2);
      pr[2 + j] = o;
      if (j == 0) { pr[0] = smx[g]; pr[1] = ssum[g]; }
    }
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
  if (used < 1) return;
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

// --- prefill attention ----------------------------------------------------------
// Flash style: a block takes one query head and 4*QPW of the chunk's tokens
// (a warp QPW of them) and walks the keys 32 at a time through shared
// memory -- K transposed and padded, so lane j reading key j's element e is
// conflict-free -- a lane a key for the scores and a lane HD/32 elements for
// the output, keeping each query's running max and sum.  Causal: token T
// sees positions up to POS0+T.  HD up to 256, QPW up to 8.
#define FA_K 32
extern "C" __global__ void __launch_bounds__(128)
attn_prefill(const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
             float* __restrict__ out, int nh, int nkv, int hd, int pos0, int n, float scale, int qpw) {
  extern __shared__ float fsm[];
  float* sk = fsm;                            // [hd][FA_K+1]
  float* sv = sk + hd * (FA_K + 1);           // [FA_K][hd]
  float* sq = sv + FA_K * hd;                 // [4*qpw][hd]
  int nq = 4 * qpw;
  int head = blockIdx.x, t0 = blockIdx.y * nq;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int ne = hd >> 5;
  int kvh = head / (nh / nkv), kvd = nkv * hd, qd = nh * hd;
  for (int e = threadIdx.x; e < nq * hd; e += blockDim.x) {
    int tq = e / hd, j = e % hd;
    sq[e] = (t0 + tq < n) ? q[(long long)(t0 + tq) * qd + head * hd + j] * scale : 0.0f;
  }
  float o[8][ATT_MAXE], m[8], l[8];
  for (int i = 0; i < 8; i++) { m[i] = -INFINITY; l[i] = 0.0f; for (int e = 0; e < ATT_MAXE; e++) o[i][e] = 0.0f; }
  int last = pos0 + min(n, t0 + nq) - 1;      // the furthest position a query here sees
  for (int p0 = 0; p0 <= last; p0 += FA_K) {
    __syncthreads();
    for (int e = threadIdx.x; e < FA_K * hd; e += blockDim.x) {
      int kk = e / hd, j = e % hd, p = p0 + kk;
      bool ok = p <= last;
      sk[j * (FA_K + 1) + kk] = ok ? kc[(long long)p * kvd + kvh * hd + j] : 0.0f;
      sv[kk * hd + j] = ok ? vc[(long long)p * kvd + kvh * hd + j] : 0.0f;
    }
    __syncthreads();
    int p = p0 + lane;
    int tw = t0 + warp * qpw;                 // the warp's first token
    if (tw >= n || p0 > pos0 + min(n - 1, tw + qpw - 1)) continue;
    // scores: a lane a key, the warp's queries at once; Q read four at a time
    float sc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) sc[i] = 0.0f;
    for (int j = 0; j < hd; j += 4) {
      float k0 = sk[(j + 0) * (FA_K + 1) + lane], k1 = sk[(j + 1) * (FA_K + 1) + lane];
      float k2 = sk[(j + 2) * (FA_K + 1) + lane], k3 = sk[(j + 3) * (FA_K + 1) + lane];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        if (i < qpw) {
          float4 qv = *(const float4*)(sq + (warp * qpw + i) * hd + j);
          sc[i] += qv.x * k0 + qv.y * k1 + qv.z * k2 + qv.w * k3;
        }
      }
    }
    float pw[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      pw[i] = 0.0f;
      if (i < qpw && tw + i < n) {
        int qpos = pos0 + tw + i;
        float s = p > qpos ? -INFINITY : sc[i];
        float mn = fmaxf(m[i], warp_max(s));
        float pr = mn == -INFINITY ? 0.0f : expf(s - mn);
        float f = mn == -INFINITY ? 1.0f : expf(m[i] - mn);
        l[i] = l[i] * f + warp_sum(pr);
#pragma unroll
        for (int e = 0; e < ATT_MAXE; e++) o[i][e] *= f;
        m[i] = mn;
        pw[i] = pr;
      }
    }
    // the output: each key's V row once, for every query of the warp
    for (int kk = 0; kk < FA_K; kk++) {
      float v[ATT_MAXE];
      const float* vr = sv + kk * hd;
#pragma unroll
      for (int e = 0; e < ATT_MAXE; e++) v[e] = e < ne ? vr[lane + 32 * e] : 0.0f;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        if (i < qpw) {
          float w = __shfl_sync(0xffffffff, pw[i], kk);
#pragma unroll
          for (int e = 0; e < ATT_MAXE; e++) o[i][e] += w * v[e];
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) {
    if (i >= qpw) break;
    int t = t0 + warp * qpw + i;
    if (t >= n) break;
    float inv = 1.0f / l[i];
#pragma unroll
    for (int e = 0; e < ATT_MAXE; e++)
      if (e < ne) out[(long long)t * qd + head * hd + lane + 32 * e] = o[i][e] * inv;
  }
}

// --- matrix x matrix on the tensor cores (sm_80 and up) --------------------------
// mma.m16n8k32.s8: one instruction is one Q8_0 block of a 16-row x 8-token
// tile, its int32 sums exact; each is then scaled by its row's and token's
// block scales into float accumulators.  A block takes 128 rows x 64 tokens,
// 8 warps of 32 x 32, and K four blocks (128 columns) at a time.

#define MMA_R 128
#define MMA_T 64
#define MMA_W 36                                // words a row in shared: 32 + 4 against bank conflicts

__device__ __forceinline__ void mma_s8(int* c, int a0, int a1, int a2, int a3, int b0, int b1) {
#if __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

extern "C" __global__ void __launch_bounds__(256)
mm_q8_mma(const i8* __restrict__ wq, const u16* __restrict__ wd,
          const i8* __restrict__ xq, const float* __restrict__ xs,
          float* __restrict__ y, const float* __restrict__ bias,
          int rows, int cols, int n, int acc) {
  __shared__ int sw[MMA_R * MMA_W];
  __shared__ int sx[MMA_T * MMA_W];
  __shared__ float swd[MMA_R][4];
  __shared__ float sxd[MMA_T][4];
  int r0 = blockIdx.y * MMA_R, t0 = blockIdx.x * MMA_T;
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int g = lane >> 2, tg = lane & 3;
  int wr = (warp >> 1) * 32, wt = (warp & 1) * 32;   // this warp's 32 rows, 32 tokens
  int nb = cols >> 5;
  float c[2][4][4];
  for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) for (int k = 0; k < 4; k++) c[i][j][k] = 0.0f;
  for (int kb = 0; kb < nb; kb += 4) {
    // 128 rows x 8 int4 and 64 tokens x 8 int4: 256 threads, 4 + 2 each
    for (int e = tid; e < MMA_R * 8; e += 256) {
      int rr = e >> 3, c4 = e & 7, r = r0 + rr, b = kb + (c4 >> 1);
      int4 v = make_int4(0, 0, 0, 0);
      if (r < rows && b < nb) v = __ldg((const int4*)(wq + (long long)r * cols + (long long)kb * 32) + c4);
      *(int4*)(sw + rr * MMA_W + c4 * 4) = v;
    }
    for (int e = tid; e < MMA_T * 8; e += 256) {
      int tt = e >> 3, c4 = e & 7, t = t0 + tt, b = kb + (c4 >> 1);
      int4 v = make_int4(0, 0, 0, 0);
      if (t < n && b < nb) v = __ldg((const int4*)(xq + (long long)t * cols + (long long)kb * 32) + c4);
      *(int4*)(sx + tt * MMA_W + c4 * 4) = v;
    }
    for (int e = tid; e < MMA_R * 4; e += 256) {
      int rr = e >> 2, bb = e & 3, r = r0 + rr, b = kb + bb;
      swd[rr][bb] = (r < rows && b < nb) ? h2f(__ldg(wd + (long long)r * nb + b)) : 0.0f;
    }
    {
      int e = tid; int tt = e >> 2, bb = e & 3, t = t0 + tt, b = kb + bb;
      sxd[tt][bb] = (t < n && b < nb) ? __ldg(xs + (long long)t * nb + b) : 0.0f;
    }
    __syncthreads();
    for (int bb = 0; bb < 4; bb++) {
      int a[2][4], b[4][2];
      for (int i = 0; i < 2; i++) {
        const int* p = sw + (wr + i * 16 + g) * MMA_W + bb * 8 + tg;
        a[i][0] = p[0]; a[i][1] = p[8 * MMA_W]; a[i][2] = p[4]; a[i][3] = p[8 * MMA_W + 4];
      }
      for (int j = 0; j < 4; j++) {
        const int* p = sx + (wt + j * 8 + g) * MMA_W + bb * 8 + tg;
        b[j][0] = p[0]; b[j][1] = p[4];
      }
      for (int i = 0; i < 2; i++) {
        float d0 = swd[wr + i * 16 + g][bb], d1 = swd[wr + i * 16 + g + 8][bb];
        for (int j = 0; j < 4; j++) {
          int s[4] = {0, 0, 0, 0};
          mma_s8(s, a[i][0], a[i][1], a[i][2], a[i][3], b[j][0], b[j][1]);
          float x0 = sxd[wt + j * 8 + tg * 2][bb], x1 = sxd[wt + j * 8 + tg * 2 + 1][bb];
          c[i][j][0] += (float)s[0] * (d0 * x0);
          c[i][j][1] += (float)s[1] * (d0 * x1);
          c[i][j][2] += (float)s[2] * (d1 * x0);
          c[i][j][3] += (float)s[3] * (d1 * x1);
        }
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) for (int k = 0; k < 4; k++) {
    int r = r0 + wr + i * 16 + g + (k >> 1) * 8;
    int t = t0 + wt + j * 8 + tg * 2 + (k & 1);
    if (r < rows && t < n) {
      float v = c[i][j][k] + (bias ? bias[r] : 0.0f);
      long long o = (long long)t * rows + r;
      y[o] = acc ? y[o] + v : v;
    }
  }
}
