// ds4f_v41.cu — DeepSeek V4.1 Flash's decode (cuda-ds41f.lisp): what differs
// from V4 (ds4f.cu, whose products, attention and experts it uses).  BF16
// rounding (nearest, ties even, kept as f32) at the boundaries ds4 rounds
// at; the FP8/FP4 round trips of the raw and compressed rows and of the
// indexer; RoPE with every turned element rounded; HC's split kept apart
// from its sum (V4.1 sums by the previous sublayer's pre); Engram's gate.
// Where ds41f.lisp's order of operations matters the products and sums are
// written with _rn intrinsics, so no multiply-add is fused.

__device__ __forceinline__ float d41_bf16(float x) {
  unsigned b = __float_as_uint(x);
  if (((b >> 23) & 0xffu) != 0xffu) { b += 0x7fffu + ((b >> 16) & 1u); b &= 0xffff0000u; }
  return __uint_as_float(b);
}

// X[0..N) to BF16 in place; into Q8_0 blocks XQ / XS too when XQ isn't
// null (N a multiple of 32).
extern "C" __global__ void d41_round(int n, float* x, i8* xq, float* xs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = d41_bf16(x[i]);
  x[i] = v;
  if (xq) d4_quant_block(v, xq + (i & ~31), xs + (i >> 5), threadIdx.x & 31);
}

// RMS norm of row blockIdx.x (N wide) of X times W into OUT, BF16; quantized
// into XQ / XS too when XQ isn't null.  A multiple of 32 threads.
extern "C" __global__ void d41_rms(int n, float eps, const float* x, const float* w, float* out, i8* xq, float* xs) {
  __shared__ float red[32];
  const float* xr = x + (long long)blockIdx.x * n;
  float* orow = out + (long long)blockIdx.x * n;
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) { float v = xr[i]; ss += v * v; }
  ss = d4_block_sum(ss, red);
  float scale = 1.f / sqrtf(ss / (float)n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = d41_bf16(__fmul_rn(__fmul_rn(xr[i], scale), w[i]));
    orow[i] = v;
    if (xq) d4_quant_block(v, xq + (i & ~31), xs + (i >> 5), threadIdx.x & 31);
  }
}

// RoPE on the last NROT dims of each of NH heads of WIDTH at X: CS the
// pairs' (cos, sin), from the CPU; each turned element BF16.
extern "C" __global__ void d41_rope(int nh, int width, int nrot, float* x, const float* cs) {
  int half_rot = nrot / 2, gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= nh * half_rot) return;
  int h = gid / half_rot, i = gid % half_rot;
  float* p = x + (long long)h * width + (width - nrot) + 2 * i;
  float c = cs[2 * i], s = cs[2 * i + 1], x0 = p[0], x1 = p[1];
  p[0] = d41_bf16(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)));
  p[1] = d41_bf16(__fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c)));
}

// The nearest E4M3 value (saturating at 448, ties to the even code) and E2M1 value (at 6).
__device__ __forceinline__ float d41_e4m3(float x) {
  float ax = fminf(fabsf(x), 448.f), q;
  if (ax < 0.015625f) q = rintf(ax * 512.f) * (1.f / 512.f);
  else { int e; frexpf(ax, &e); float step = ldexpf(1.f, e - 4); q = rintf(ax / step) * step; }
  q = fminf(q, 448.f);
  return x < 0.f ? -q : q;
}
__device__ __forceinline__ float d41_e2m1(float x) {
  float ax = fminf(fabsf(x), 6.f), q;
  if (ax < 2.f) q = rintf(ax * 2.f) * 0.5f;
  else if (ax < 4.f) q = rintf(ax);
  else q = rintf(ax * 0.5f) * 2.f;
  return x < 0.f ? -q : q;
}
__device__ __forceinline__ float d41_pow2ceil(float x) {
  unsigned b = __float_as_uint(x);
  if (b & 0x7fffffu) b = (b & 0xff800000u) + 0x800000u;
  return __uint_as_float(b);
}

// ds4_gpu_dsv41_quantize: X[0..N) in place, a warp a block of 32 (KIND 0:
// FP8, E4M3 values with a power-of-two scale; 1: FP4, E2M1 with a power of
// two; 2: FP4, E2M1 with an E4M3 scale a half block of 16).
extern "C" __global__ void d41_quant(int n, int kind, float* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const unsigned F = 0xffffffffu;
  float v = d41_bf16(x[i]), peak = fabsf(v);
  for (int o = 1; o < (kind == 2 ? 16 : 32); o <<= 1) peak = fmaxf(peak, __shfl_xor_sync(F, peak, o));
  float scale = kind == 0 ? d41_pow2ceil(__fmul_rn(fmaxf(peak, 1e-4f), 1.f / 448.f))
              : kind == 1 ? d41_pow2ceil(__fmul_rn(fmaxf(peak, 0x1.8p-124f), 1.f / 6.f))
                          : d41_e4m3(__fdiv_rn(fmaxf(peak, 0.01171875f), 6.f));
  float q = __fdiv_rn(v, scale);
  x[i] = d41_bf16(__fmul_rn(kind == 0 ? d41_e4m3(q) : d41_e2m1(q), scale));
}

// hc_split_sinkhorn_one for 4 streams, kept apart from the sum: MIX's 24
// into PC: POST[0..4), COMB[4..20) (as D4_HC_POST reads them), PRE[24..28).
// A warp.
extern "C" __global__ void d41_hc_split(int iters, float eps, const float* mix, const float* scale,
                                        const float* base, float* pc) {
  const unsigned F = 0xffffffffu;
  int lane = threadIdx.x, src = lane % 4, dst = (lane / 4) % 4, idx = src + dst * 4;
  float c = mix[8 + idx] * scale[2] + base[8 + idx];
  float mx = fmaxf(c, __shfl_xor_sync(F, c, 1)); mx = fmaxf(mx, __shfl_xor_sync(F, mx, 2));
  c = expf(c - mx);
  float sum = c + __shfl_xor_sync(F, c, 1); sum += __shfl_xor_sync(F, sum, 2);
  c = c * (1.f / sum) + eps;
  for (int it = 0; it < iters; ++it) {
    if (it > 0) {
      float r = c + __shfl_xor_sync(F, c, 1); r += __shfl_xor_sync(F, r, 2);
      c *= 1.f / (r + eps);
    }
    float col = c + __shfl_xor_sync(F, c, 4); col += __shfl_xor_sync(F, col, 8);
    c *= 1.f / (col + eps);
  }
  if (lane < 16) pc[4 + idx] = c;
  if (lane < 4) pc[24 + lane] = 1.f / (1.f + expf(-(mix[lane] * scale[0] + base[lane]))) + eps;
  if (lane >= 4 && lane < 8) pc[lane - 4] = 2.f / (1.f + expf(-(mix[lane] * scale[1] + base[lane])));
}

// OUT[j] = bf16(sum_h STREAMS[h][j] PRE[h]), in stream order.
extern "C" __global__ void d41_hc_sum(int d, int n, const float* streams, const float* pre, float* out) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= d) return;
  float acc = 0.f;
  for (int h = 0; h < n; ++h) acc = __fadd_rn(acc, __fmul_rn(streams[h * d + j], pre[h]));
  out[j] = d41_bf16(acc);
}

// The ratio-2 compressor's pair: on an even POS this token's KV and score
// kept (PKV, PSC); on an odd one the per-dim softmax of the pair into
// LATENT, BF16.
extern "C" __global__ void d41_pool2(int hd, int odd, const float* kv, const float* sc, float* pkv, float* psc,
                                     float* latent) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= hd) return;
  if (!odd) { pkv[j] = kv[j]; psc[j] = sc[j]; return; }
  float sa = psc[j], sb = sc[j], p = fmaxf(sa, sb), ea = expf(sa - p), eb = expf(sb - p);
  latent[j] = d41_bf16(__fdiv_rn(__fadd_rn(__fmul_rn(pkv[j], ea), __fmul_rn(kv[j], eb)), __fadd_rn(ea, eb)));
}

// The indexer's scores: key c of KEYS (NC rows of 128) against the NH
// query heads Q (128 each): sum_h relu(q_h . k) W[h], over 64.  A warp a
// key, a lane 4 dims of each head; the query in shared memory.
extern "C" __global__ void d41_ix_scores(int nh, int nc, const float* __restrict__ q, const float* __restrict__ w,
                                         const float* __restrict__ keys, float* __restrict__ scores) {
  extern __shared__ float qs[];
  for (int i = threadIdx.x; i < nh * 128; i += blockDim.x) qs[i] = q[i];
  __syncthreads();
  int c = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (c >= nc) return;
  float4 k = __ldg((const float4*)(keys + (long long)c * 128) + lane);
  float total = 0.f;
  for (int h = 0; h < nh; ++h) {
    float4 a = *(const float4*)(qs + h * 128 + lane * 4);
    float dot = warp_sum(a.x * k.x + a.y * k.y + a.z * k.z + a.w * k.w);
    total += fmaxf(dot, 0.f) * w[h];
  }
  if (lane == 0) scores[c] = total * (1.f / 64.f);
}

// Engram's gate on stream blockIdx.x of STREAMS (D each): its key
// KV[h] and the value KV[4] (BF16), its RMS-normed dot with the key under
// QW*KW, over sqrt(D); the stream += sigmoid(signed sqrt(dot)) * value, BF16.
extern "C" __global__ void d41_engram(int d, int n, float eps, float* streams, const float* kv,
                                      const float* qw, const float* kw) {
  __shared__ float red[32];
  int h = blockIdx.x;
  float* r = streams + (long long)h * d;
  const float* k = kv + (long long)h * d;
  const float* v = kv + (long long)n * d;
  float h2 = 0.f, k2 = 0.f, dot = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float ri = r[i], ki = k[i];
    h2 += ri * ri; k2 += ki * ki; dot += ri * (qw[h * d + i] * kw[h * d + i]) * ki;
  }
  h2 = d4_block_sum(h2, red); k2 = d4_block_sum(k2, red); dot = d4_block_sum(dot, red);
  dot = dot * (1.f / sqrtf(h2 / (float)d + eps)) * (1.f / sqrtf(k2 / (float)d + eps)) * (1.f / sqrtf((float)d));
  float g = d4_sigmoid(copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot));
  for (int i = threadIdx.x; i < d; i += blockDim.x) r[i] = d41_bf16(__fadd_rn(r[i], __fmul_rn(g, v[i])));
}

// The shared expert's middle: bf16(silu(min(bf16 G, CLAMP)) * clamp(bf16 U)),
// quantized into XQ / XS.
extern "C" __global__ void d41_swiglu(int ff, float clamp, const float* g, const float* u, float* mid,
                                      i8* xq, float* xs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ff) return;
  float gv = fminf(d41_bf16(g[i]), clamp), uv = fminf(fmaxf(d41_bf16(u[i]), -clamp), clamp);
  float v = d41_bf16(__fmul_rn(__fmul_rn(gv, d4_sigmoid(gv)), uv));
  mid[i] = v;
  d4_quant_block(v, xq + (i & ~31), xs + (i >> 5), threadIdx.x & 31);
}

// Y = W X, W's rows F32 (the router's): 4 rows a block, K across its 256 threads.
extern "C" __global__ void d41_mv_f32(int ncols, int nrows, const float* __restrict__ w,
                                      const float* __restrict__ x, float* __restrict__ y) {
  __shared__ float red[32];
  int r0 = blockIdx.x * 4, tid = threadIdx.x;
  float s[4] = {0.f, 0.f, 0.f, 0.f};
  const float4* x4 = (const float4*)x;
  int n4 = ncols / 4;
  for (int i = tid; i < n4; i += 256) {
    float4 xv = x4[i];
#pragma unroll
    for (int r = 0; r < 4; ++r)
      if (r0 + r < nrows) {
        float4 wv = __ldg((const float4*)(w + (long long)(r0 + r) * ncols) + i);
        s[r] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
      }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    float t = d4_block_sum(s[r], red);
    if (tid == 0 && r0 + r < nrows) y[r0 + r] = t;
  }
}
