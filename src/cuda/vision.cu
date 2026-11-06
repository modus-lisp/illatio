// vision.cu — DeepSeek V4.1's vision encoder (ds4's ds4_deepseek4_vision_gpu.cuh,
// for the names llama.cpp's mmproj gives its tensors): the ViT's own steps.
// Its products are D41_MM_F32's (the patch embedding) and G53_MM_BF16's (the
// rest); everything in floats, no tensor cores (a P40 runs it as well).
// A picture's patches are ROWS x 1024 (ROWS = grid rows x columns, row-major).
//
//   v_bias(count, width, y, bias, res)      Y += BIAS (per column), + RES when not null
//   v_add(count, y, x)                      Y += X
//   v_rms(width, eps, x, w, out)            OUT = X / rms(X) * W, a row a block
//   v_rope(rows, gw, qkv, q, k, v)          QKV's thirds apart, Q and K turned by 2-D RoPE
//   v_attend(rows, q, k, v, out)            softmax(Q K^T / 8) V over all ROWS, a head at a time
//   v_swiglu(count, g, u)                   G = silu(G) * U
//   v_merge3(gh, gw, x, out)                3 x 3 patches' 1024 values each into one row of 9216
//   v_gelu(count, x)                        X = gelu(X) (erf's)

extern "C" __global__ void v_bias(long long count, int width, float* y, const float* bias, const float* res) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  float v = y[i] + bias[i % width];
  if (res) v += res[i];
  y[i] = v;
}

extern "C" __global__ void v_add(long long count, float* y, const float* x) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) y[i] += x[i];
}

// grid rows, block 256.
extern "C" __global__ void v_rms(int width, float eps, const float* x, const float* w, float* out) {
  __shared__ float red[32];
  const float* r = x + (long long)blockIdx.x * width;
  float* o = out + (long long)blockIdx.x * width;
  float ss = 0.f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) ss += r[i] * r[i];
  ss = d4_block_sum(ss, red);
  const float s = rsqrtf(ss / (float)width + eps);
  for (int i = threadIdx.x; i < width; i += blockDim.x) o[i] = r[i] * s * w[i];
}

// A head's 64 dims: lane l (< 32) pairs with l + 32 (rotate-half); lanes
// 0..15 turn by the patch's grid row, 16..31 by its column, frequency
// 10000^-(l%16)/16.  grid (rows, 16), block 32.
extern "C" __global__ void v_rope(int rows, int gw, const float* qkv, float* q, float* k, float* v) {
  const int row = blockIdx.x, h = blockIdx.y, l = threadIdx.x;
  if (row >= rows) return;
  const float* s = qkv + (long long)row * 3072 + h * 64;
  const int y = row / gw, x = row - y * gw;
  const float ang = (float)(l < 16 ? y : x) * powf(10000.f, -(float)(l & 15) / 16.f);
  float sn, cs;
  sincosf(ang, &sn, &cs);
  const long long o = (long long)row * 1024 + h * 64;
  const float q0 = s[l], q1 = s[l + 32], k0 = s[1024 + l], k1 = s[1024 + l + 32];
  q[o + l] = q0 * cs - q1 * sn; q[o + l + 32] = q1 * cs + q0 * sn;
  k[o + l] = k0 * cs - k1 * sn; k[o + l + 32] = k1 * cs + k0 * sn;
  v[o + l] = s[2048 + l]; v[o + l + 32] = s[2048 + l + 32];
}

// Attention over every patch (no mask): a block takes 64 queries of head
// blockIdx.y, the keys 32 a step, softmax kept running.  Thread (ty, tx) =
// (tid/16, tid%16) holds queries 4ty.. : their scores for keys 2tx, 2tx+1,
// their outputs' dims 4tx..; a query's 16 threads are half a warp, so its
// maxima and sums are shuffles.  grid (ceil(rows/64), 16), block 256.
#define VA_Q 64
#define VA_K 32
extern "C" __global__ void __launch_bounds__(256)
v_attend(int rows, const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
         float* __restrict__ out) {
  __shared__ float qs[VA_Q][64 + 1];
  __shared__ float ks[VA_K][64 + 1];
  __shared__ __align__(16) float vs[VA_K][64];
  __shared__ float ps[VA_Q][VA_K + 1];
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int q0 = blockIdx.x * VA_Q, h = blockIdx.y;
  const unsigned F = 0xffffffffu;
  for (int i = tid; i < VA_Q * 64; i += 256) {
    const int r = i >> 6, c = i & 63;
    qs[r][c] = q0 + r < rows ? q[(long long)(q0 + r) * 1024 + h * 64 + c] * 0.125f : 0.f;
  }
  float o[4][4], m[4], l[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    m[i] = -INFINITY; l[i] = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[i][j] = 0.f;
  }
  for (int k0 = 0; k0 < rows; k0 += VA_K) {
    __syncthreads();
    for (int i = tid; i < VA_K * 64; i += 256) {
      const int r = i >> 6, c = i & 63;
      const bool ok = k0 + r < rows;
      const long long a = (long long)(k0 + r) * 1024 + h * 64 + c;
      ks[r][c] = ok ? k[a] : 0.f;
      vs[r][c] = ok ? v[a] : 0.f;
    }
    __syncthreads();
    float s[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = 0.f;
#pragma unroll 8
    for (int d = 0; d < 64; ++d) {
      const float b0 = ks[2 * tx][d], b1 = ks[2 * tx + 1][d];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float a = qs[4 * ty + i][d];
        s[i][0] = fmaf(a, b0, s[i][0]); s[i][1] = fmaf(a, b1, s[i][1]);
      }
    }
    const bool ok0 = k0 + 2 * tx < rows, ok1 = k0 + 2 * tx + 1 < rows;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (!ok0) s[i][0] = -INFINITY;
      if (!ok1) s[i][1] = -INFINITY;
      float mx = fmaxf(s[i][0], s[i][1]);
      for (int off = 8; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(F, mx, off));
      const float mn = fmaxf(m[i], mx);
      const float alpha = expf(m[i] - mn);
      const float p0 = expf(s[i][0] - mn), p1 = expf(s[i][1] - mn);
      float sum = p0 + p1;
      for (int off = 8; off; off >>= 1) sum += __shfl_xor_sync(F, sum, off);
      l[i] = l[i] * alpha + sum; m[i] = mn;
#pragma unroll
      for (int j = 0; j < 4; ++j) o[i][j] *= alpha;
      ps[4 * ty + i][2 * tx] = p0; ps[4 * ty + i][2 * tx + 1] = p1;
    }
    __syncthreads();
#pragma unroll 4
    for (int kk = 0; kk < VA_K; ++kk) {
      const float4 vv = *(const float4*)&vs[kk][4 * tx];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float p = ps[4 * ty + i][kk];
        o[i][0] = fmaf(p, vv.x, o[i][0]); o[i][1] = fmaf(p, vv.y, o[i][1]);
        o[i][2] = fmaf(p, vv.z, o[i][2]); o[i][3] = fmaf(p, vv.w, o[i][3]);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r = q0 + 4 * ty + i;
    if (r >= rows) continue;
    const float inv = 1.f / l[i];
#pragma unroll
    for (int j = 0; j < 4; ++j) out[(long long)r * 1024 + h * 64 + 4 * tx + j] = o[i][j] * inv;
  }
}

extern "C" __global__ void v_swiglu(long long count, float* g, const float* u) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float x = g[i];
  g[i] = x / (1.f + expf(-x)) * u[i];
}

// Output row r: block (r / ow, r % ow) of 3 x 3 patches (ow = ceil(gw/3)),
// value d = channel * 9 + (dy * 3 + dx); patches past the grid are zeros.
// grid (ceil(9216/256), output rows), block 256.
extern "C" __global__ void v_merge3(int gh, int gw, const float* x, float* out) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (d >= 9216) return;
  const int ow = (gw + 2) / 3, by = row / ow, bx = row - by * ow;
  const int c = d / 9, w = d - c * 9;
  const int sy = by * 3 + w / 3, sx = bx * 3 + w % 3;
  out[(long long)row * 9216 + d] = (sy < gh && sx < gw) ? x[((long long)sy * gw + sx) * 1024 + c] : 0.f;
}

extern "C" __global__ void v_gelu(long long count, float* x) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float v = x[i];
  x[i] = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
}
