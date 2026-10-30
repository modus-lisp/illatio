// ds4f.cu — DeepSeek V4 Flash's decode on an NVIDIA GPU (cuda-ds4f.lisp).
//
// metal/ds4f.metal's decode kernels, one for one: the same ops, the same
// orders of summation where they matter, a block where Metal has a
// threadgroup (of as many threads).  The Q8_0 products are dense.cu's
// (row_dot_q8, the activations quantized as ds4 does: Q8_0 blocks, their
// scales kept as floats); the routed experts' are ggml's
// kernel_mul_mv_iq2_xxs_f32 and kernel_mul_mv_q2_K_f32 (MIT, Copyright (c)
// 2023-2026 The ggml authors), reading each expert's slot in the pool.
// Every kernel takes its scalars first, then its buffers, as the Metal
// ones take their argument struct then their buffers.

#define D4_QK_K 256

__device__ __forceinline__ float d4_sigmoid(float x) {
  return x >= 0.f ? 1.f / (1.f + expf(-x)) : expf(x) / (1.f + expf(x));
}

// A block's sum of V (every thread gets it): RED, 32 floats of shared memory.
__device__ __forceinline__ float d4_block_sum(float v, float* red) {
  v = warp_sum(v);
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
  if (lane == 0) red[w] = v;
  __syncthreads();
  float t = lane < nw ? red[lane] : 0.f;
  t = warp_sum(t);
  __syncthreads();
  return t;
}

// --- quantized activations, Q8_0 products ------------------------------------------

// X quantized into Q8_0 blocks, each scale amax/127 as a float (ds4's
// quantize_row_q8_0, which keeps it unrounded): a warp a block.
extern "C" __global__ void d4_quant(const float* __restrict__ x, i8* __restrict__ xq,
                                    float* __restrict__ xs, int nblocks) {
  int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (b >= nblocks) return;
  float v = x[b * 32 + lane];
  float amax = warp_max(fabsf(v));
  float dd = amax / 127.0f;
  float id = dd == 0.0f ? 0.0f : 1.0f / dd;
  xq[b * 32 + lane] = (i8)__float2int_rn(v * id);
  if (lane == 0) xs[b] = dd;
}

// Y[r] = W[r]·X for a Q8_0 W (repacked: int8s, then f16 scales), row r
// against X's group r/GROUP_ROWS (COLS wide, its blocks one after another):
// the grouped output projection, or a plain product with GROUP_ROWS = ROWS.
extern "C" __global__ void d4_mvq8(int rows, int cols, int group_rows,
                                   const i8* __restrict__ wq, const u16* __restrict__ wd,
                                   const i8* __restrict__ xq, const float* __restrict__ xs,
                                   float* __restrict__ y) {
  int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  long long g = r / group_rows;
  float v = row_dot_q8(wq, wd, r, cols, xq + g * cols, xs + g * (cols >> 5), lane);
  if (lane == 0) y[r] = v;
}

// Y = W X, W's rows F16 (the file's): 4 rows a block, K across its 256 threads.
extern "C" __global__ void d4_mv_f16(int ncols, int nrows, const u16* __restrict__ w,
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
        uint2 h = __ldg((const uint2*)(w + (long long)(r0 + r) * ncols) + i);
        s[r] += h2f(h.x & 0xffff) * xv.x + h2f(h.x >> 16) * xv.y + h2f(h.y & 0xffff) * xv.z + h2f(h.y >> 16) * xv.w;
      }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    float t = d4_block_sum(s[r], red);
    if (tid == 0 && r0 + r < nrows) y[r0 + r] = t;
  }
}

// --- norms, hyper-connections, RoPE ------------------------------------------------

__device__ void d4_rope_row(float* v, int hd, int nrot, const float* cs, int tid);

// A warp's 32 consecutive values V as a Q8_0 block (D4_QUANT's rounding).
__device__ __forceinline__ void d4_quant_block(float v, i8* q, float* s, int lane) {
  float amax = warp_max(fabsf(v));
  float dd = amax / 127.0f;
  float id = dd == 0.0f ? 0.0f : 1.0f / dd;
  q[lane] = (i8)__float2int_rn(v * id);
  if (lane == 0) *s = dd;
}

// RMS norm of row blockIdx.x (N wide) of X into OUT (times W when HAS_W):
// in place when OUT is X.  XQ, when not null, gets the row quantized too
// (a single row, N a multiple of 32, a block of a multiple of 32 threads).
extern "C" __global__ void d4_rms(int n, float eps, int has_w, const float* x, const float* w, float* out,
                                  i8* xq, float* xs) {
  __shared__ float red[32];
  const float* xr = x + (long long)blockIdx.x * n;
  float* orow = out + (long long)blockIdx.x * n;
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) { float v = xr[i]; ss += v * v; }
  ss = d4_block_sum(ss, red);
  float scale = 1.f / sqrtf(ss / (float)n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = has_w ? (xr[i] * scale) * w[i] : xr[i] * scale;
    orow[i] = v;
    if (xq) d4_quant_block(v, xq + (i & ~31), xs + (i >> 5), threadIdx.x & 31);
  }
}

// RoPE on the last NROT dims of each of NH heads of HD at X (CS as D4_ROPE's),
// then every head quantized into XQ / XS: a block a head, HD threads.
extern "C" __global__ void d4_rope_q(int hd, int nrot, float* x, const float* cs, i8* xq, float* xs) {
  __shared__ float v[1024];
  int h = blockIdx.x, tid = threadIdx.x;
  float* xh = x + (long long)h * hd;
  v[tid] = xh[tid];
  __syncthreads();
  d4_rope_row(v, hd, nrot, cs, tid);
  float y = v[tid];
  xh[tid] = y;
  long long i = (long long)h * hd + tid;
  d4_quant_block(y, xq + (i & ~31LL), xs + (i >> 5), tid & 31);
}

// hc_split_sinkhorn_one and the pre-weighted sum: MIX (the N*(N+2) mixes)
// into PC (POST[N], then COMB[N*N]) and OUT[j] = sum_h STREAMS[h][j] PRE[h].
// Written for 4 streams (the host checks).  Grid (D/256, tokens).
extern "C" __global__ void d4_hc_pre_fin(int d, int n, int iters, float eps,
                                         const float* mix, const float* scale, const float* base,
                                         const float* streams, float* out, float* pc) {
  __shared__ float pre[4];
  int tg = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, gid = blockIdx.x * blockDim.x + tid;
  mix += tok * 32; streams += (long long)tok * n * d; out += (long long)tok * d; pc += tok * 32;
  if (tid < 32) {
    const unsigned F = 0xffffffffu;
    int lane = tid, src = lane % 4, dst = (lane / 4) % 4, idx = src + dst * 4;
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
    if (tg == 0 && lane < 16) pc[4 + idx] = c;
    if (lane < 4) pre[lane] = 1.f / (1.f + expf(-(mix[lane] * scale[0] + base[lane]))) + eps;
    if (tg == 0 && lane >= 4 && lane < 8) {
      int i = lane - 4;
      pc[i] = 2.f / (1.f + expf(-(mix[4 + i] * scale[1] + base[4 + i])));
    }
  }
  __syncthreads();
  if (gid < d) {
    float acc = 0.f;
    for (int h = 0; h < n; ++h) acc += streams[h * d + gid] * pre[h];
    out[gid] = acc;
  }
}

// hc_post_one: stream DST = BLOCK * POST[DST] + sum_src COMB[dst + src*n] * STREAMS[src].
extern "C" __global__ void d4_hc_post(int d, int n, int ntok, float eps,
                                      const float* block, const float* streams, const float* pc, float* outs) {
  long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int hcd = n * d;
  int tok = gid / hcd, r = gid % hcd;
  if (tok >= ntok) return;
  int dst = r / d, j = r % d;
  block += (long long)tok * d; streams += (long long)tok * hcd; pc += tok * 32;
  float v = block[j] * pc[dst];
  for (int src = 0; src < n; ++src) v += pc[n + dst + src * n] * streams[src * d + j];
  outs[(long long)tok * hcd + r] = v;
}

// RoPE on the last NROT dims of each of NH heads of HD at X, CS the pairs'
// (cos, sin) from the CPU (D4-ROPE-CS), a table a token.
extern "C" __global__ void d4_rope(int nh, int hd, int nrot, int ntok, float* x, const float* cs) {
  int half_rot = nrot / 2, per = nh * half_rot;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= ntok * per) return;
  int tok = gid / per, h = (gid % per) / half_rot, i = 2 * (gid % half_rot);
  cs += tok * nrot;
  float* t = x + ((long long)tok * nh + h) * hd + (hd - nrot);
  float c = cs[i], s = cs[i + 1], x0 = t[i], x1 = t[i + 1];
  t[i] = x0 * c - x1 * s;
  t[i + 1] = x0 * s + x1 * c;
}

// --- FP8 / FP4 round trips, the KV row, the compressors ----------------------------

__constant__ float d4_e4m3[127] = {
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
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f};

__device__ float d4_e4m3_round(float x) {
  float sign = x < 0.f ? -1.f : 1.f, ax = fminf(fabsf(x), 448.f);
  int lo = 0, hi = 126;
  while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (d4_e4m3[mid] <= ax) lo = mid; else hi = mid - 1; }
  int best = lo;
  if (best < 126) {
    float bd = fabsf(ax - d4_e4m3[best]), nd = fabsf(ax - d4_e4m3[best + 1]);
    if (nd < bd || (nd == bd && ((best + 1) % 2 == 0) && (best % 2 == 1))) best++;
  }
  return sign * d4_e4m3[best];
}

__constant__ float d4_e2m1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__device__ float d4_e2m1_round(float x) {
  float sign = x < 0.f ? -1.f : 1.f, ax = fminf(fabsf(x), 6.f);
  int best = 0; float bd = fabsf(ax);
  for (int i = 1; i < 8; ++i) {
    float dd = fabsf(ax - d4_e2m1[i]);
    if (dd < bd || (dd == bd && (i % 2 == 0) && (best % 2 == 1))) { best = i; bd = dd; }
  }
  return sign * d4_e2m1[best];
}

// A row of HD in shared memory: FP8 (blocks of 64 over the first HD - NROT
// dims, a power-of-two scale) when KIND is 0, else the indexer's QAT (a
// scaled 128-point Hadamard, then FP4 in blocks of 32).  HD threads.
__device__ void d4_round_row(float* v, int hd, int nrot, int kind, int tid, float* red) {
  if (kind == 0) {
    int nb = (hd - nrot) / 64;
    if (tid < nb) {
      float amax = 0.f;
      for (int i = 0; i < 64; ++i) amax = fmaxf(amax, fabsf(v[tid * 64 + i]));
      red[tid] = fmaxf(amax, 1e-4f);
    }
    __syncthreads();
    if (tid < hd - nrot) {
      float scale = exp2f(ceilf(log2f(red[tid / 64] / 448.f)));
      float x = fminf(fmaxf(v[tid] / scale, -448.f), 448.f);
      v[tid] = d4_e4m3_round(x) * scale;
    }
    __syncthreads();
  } else {
    for (int stride = 1; stride < 128; stride *= 2) {
      if (tid < 64) {
        int base = (tid / stride) * 2 * stride, i = tid % stride;
        float x = v[base + i], y = v[base + stride + i];
        v[base + i] = x + y; v[base + stride + i] = x - y;
      }
      __syncthreads();
    }
    if (tid < 128) v[tid] *= 0.08838834764831845f;
    __syncthreads();
    if (tid < 4) {
      float amax = 0.f;
      for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[tid * 32 + i]));
      red[tid] = fmaxf(amax, 7.052966104933725e-38f);
    }
    __syncthreads();
    if (tid < 128) {
      float scale = exp2f(ceilf(log2f(red[tid / 32] / 6.f)));
      float x = fminf(fmaxf(v[tid] / scale, -6.f), 6.f);
      v[tid] = d4_e2m1_round(x) * scale;
    }
    __syncthreads();
  }
}

__device__ void d4_rope_row(float* v, int hd, int nrot, const float* cs, int tid) {
  if (tid < nrot / 2) {
    int i = 2 * tid;
    float* t = v + (hd - nrot);
    float c = cs[i], s = cs[i + 1], x0 = t[i], x1 = t[i + 1];
    t[i] = x0 * c - x1 * s; t[i + 1] = x0 * s + x1 * c;
  }
  __syncthreads();
}

__device__ __forceinline__ float d4_f16_round(float x) { return h2f(f2h(x)); }

// The KV row: RMS-normed (W), RoPE'd, through FP8, then through F16 into
// ring row SLOT (+ the block's token) of RAW.  HD threads, a block a token.
extern "C" __global__ void d4_kv_fin(int hd, int nrot, float eps, int slot,
                                     const float* kv0, const float* w, const float* cs, float* raw) {
  extern __shared__ float sh[];
  float* v = sh;
  float* red = sh + hd;
  int t = blockIdx.x, tid = threadIdx.x;
  kv0 += (long long)t * hd; cs += t * nrot;
  float x = kv0[tid];
  float ss = d4_block_sum(x * x, red);
  float scale = 1.f / sqrtf(ss / (float)hd + eps);
  v[tid] = (x * scale) * w[tid];
  __syncthreads();
  d4_rope_row(v, hd, nrot, cs, tid);
  d4_round_row(v, hd, nrot, 0, tid, red);
  raw[(long long)(slot + t) * hd + tid] = d4_f16_round(v[tid]);
}

// compressor_decode_one: this token's projected row KV and score SC (plus
// APE's row POS_MOD) into window row ROW (WIDTH each).
extern "C" __global__ void d4_comp_store(int width, int row, int pos_mod, int ape_cols,
                                         const float* kv, const float* sc, const float* ape,
                                         float* st_kv, float* st_score) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= width) return;
  st_kv[row * width + j] = kv[j];
  st_score[row * width + j] = sc[j] + ape[pos_mod * ape_cols + j];
}

// Every RATIO tokens: the window softmax-pooled per dim, RMS-normed (W),
// RoPE'd (CS: its first position), rounded (KIND 0: FP8, 1: the indexer's
// QAT), through F16 as compressed row DST of ROWS; with ratio 4, the
// current half then becomes the previous one.  HD threads.
extern "C" __global__ void d4_comp_pool(int hd, int ratio, int width, float eps, int kind, int nrot, int dst,
                                        float* st_kv, float* st_score, const float* w, const float* cs,
                                        float* rows) {
  extern __shared__ float sh[];
  float* v = sh;
  float* red = sh + hd;
  int j = threadIdx.x, r4 = ratio == 4;
  float mx = -1e30f;
  for (int r = 0; r < ratio; ++r) {
    mx = fmaxf(mx, st_score[r * width + j]);
    if (r4) mx = fmaxf(mx, st_score[(ratio + r) * width + hd + j]);
  }
  float pooled = 0.f;
  if (mx > -5e29f) {
    float denom = 0.f, sum = 0.f;
    for (int r = 0; r < ratio; ++r) {
      if (r4) {
        float wp = expf(st_score[r * width + j] - mx);
        float wc = expf(st_score[(ratio + r) * width + hd + j] - mx);
        denom += wp + wc;
        sum += wp * st_kv[r * width + j];
        sum += wc * st_kv[(ratio + r) * width + hd + j];
      } else {
        float ww = expf(st_score[r * width + j] - mx);
        denom += ww; sum += ww * st_kv[r * width + j];
      }
    }
    pooled = denom > 0.f ? sum / denom : 0.f;
  }
  float ss = d4_block_sum(pooled * pooled, red);
  float scale = 1.f / sqrtf(ss / (float)hd + eps);
  v[j] = (pooled * scale) * w[j];
  __syncthreads();
  d4_rope_row(v, hd, nrot, cs, j);
  d4_round_row(v, hd, nrot, kind, j, red);
  rows[(long long)dst * hd + j] = d4_f16_round(v[j]);
  if (r4) {
    __syncthreads();
    for (int r = 0; r < ratio; ++r)
      for (int k = j; k < width; k += hd) {
        st_kv[r * width + k] = st_kv[(ratio + r) * width + k];
        st_score[r * width + k] = st_score[(ratio + r) * width + k];
      }
  }
}

// --- the indexer ----------------------------------------------------------------

// The indexer's queries: each of NH heads of 128 RoPE'd (CS) then QAT'd; W,
// the head weights, scaled.  Grid (heads, tokens), 128 threads.
extern "C" __global__ void d4_ixq_fin(int nh, int nrot, float wscale, float* q, const float* cs, float* w) {
  __shared__ float v[128], red[32];
  int h = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x;
  cs += tok * nrot; w += tok * nh;
  float* qh = q + ((long long)tok * nh + h) * 128;
  v[tid] = qh[tid];
  __syncthreads();
  d4_rope_row(v, 128, nrot, cs, tid);
  d4_round_row(v, 128, nrot, 1, tid, red);
  qh[tid] = v[tid];
  if (tid == 0) w[h] *= wscale;
}

// One token's indexer scores: SCORES[c] = sum over heads h of relu(Q[h] .
// K[c]) * W[h], for the NCOMP rows of K (128 wide).  A block takes 32 rows
// (staged in shared memory) and 8 warps; warp w the heads w, w+8, ...,
// lane l row l; the warps' sums meet in head order... as the Metal kernel
// adds its heads in order, so do these: each warp's sum of its heads is
// kept apart per head and added in order at the end.
#define D4_IX_MAXH 64
extern "C" __global__ void d4_ix_scores1(int nh, int ncomp, const float* __restrict__ q,
                                         const float* __restrict__ w, const float* __restrict__ k,
                                         float* __restrict__ scores) {
  __shared__ float ks[32][129];
  __shared__ float part[D4_IX_MAXH][33];
  int c0 = blockIdx.x * 32, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
  for (int i = tid; i < 32 * 128; i += 256) {
    int c = c0 + i / 128;
    ks[i / 128][i % 128] = c < ncomp ? k[(long long)c * 128 + i % 128] : 0.f;
  }
  __syncthreads();
  for (int h = wp; h < nh; h += 8) {
    const float* qh = q + h * 128;
    float dot = 0.f;
    for (int d = 0; d < 128; d++) dot += __ldg(qh + d) * ks[lane][d];
    part[h][lane] = fmaxf(dot, 0.f) * w[h];
  }
  __syncthreads();
  if (tid < 32 && c0 + tid < ncomp) {
    float acc = 0.f;
    for (int h = 0; h < nh; h++) acc += part[h][tid];
    scores[c0 + tid] = acc;
  }
}

__device__ __forceinline__ unsigned d4_key(float f) {
  unsigned u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ unsigned warp_sum_u(unsigned v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ unsigned warp_prefix_excl(unsigned v) {
  int lane = threadIdx.x & 31;
  unsigned x = v;
  for (int o = 1; o < 32; o <<= 1) {
    unsigned y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  return x - v;
}

// The indexer's pick: of N scores, the K highest, their rows in order into
// IDX (the earlier of equal scores first, as TOP-K-ROWS).  1024 threads.
extern "C" __global__ void d4_topk(int n, int k, const float* scores, unsigned* idx) {
  __shared__ unsigned cnt[32], pg[32], pe[32];
  int tid = threadIdx.x;
  if (n <= k) { for (int i = tid; i < n; i += 1024) idx[i] = i; return; }
  int per = (n + 1023) / 1024, s0 = tid * per, s1 = min(n, s0 + per);
  unsigned t = 0;
  for (int bit = 31; bit >= 0; --bit) {
    unsigned cand = t | (1u << bit);
    unsigned c = 0;
    for (int i = s0; i < s1; ++i) c += d4_key(scores[i]) >= cand;
    c = warp_sum_u(c);
    if (tid % 32 == 0) cnt[tid / 32] = c;
    __syncthreads();
    unsigned tot = 0;
    for (int s = 0; s < 32; ++s) tot += cnt[s];
    __syncthreads();
    if (tot >= (unsigned)k) t = cand;
  }
  unsigned g = 0, e = 0;
  for (int i = s0; i < s1; ++i) { unsigned kk = d4_key(scores[i]); g += kk > t; e += kk == t; }
  unsigned gx = warp_prefix_excl(g), ex = warp_prefix_excl(e);
  if (tid % 32 == 31) { pg[tid / 32] = gx + g; pe[tid / 32] = ex + e; }
  __syncthreads();
  unsigned gbefore = gx, ebefore = ex, gtot = 0;
  for (int s = 0; s < 32; ++s) { if (s < tid / 32) { gbefore += pg[s]; ebefore += pe[s]; } gtot += pg[s]; }
  unsigned ties = k - gtot;
  unsigned at = gbefore + min(ebefore, ties), et = ebefore;
  for (int i = s0; i < s1; ++i) {
    unsigned kk = d4_key(scores[i]);
    if (kk > t) idx[at++] = i;
    else if (kk == t && et < ties) { idx[at++] = i; et++; }
  }
}

// --- attention ------------------------------------------------------------------

// One token's attention: a block takes 4 heads (128 threads).  Its keys:
// NRAW rows of the raw ring KR (RCAP rows) from R0, then NCOMP compressed
// rows of KC (the first NCOMP, or IDX's).  The scores (a thread a key), the
// softmax from each head's sink (the sink in the denominator), then the sum
// (a thread 4 dims of each head).  All in f32.
extern "C" __global__ void d4_attend1(int hd, float scale, int rcap, int r0, int nraw, int ncomp, int ident,
                                      const float* __restrict__ q, const float* __restrict__ kr,
                                      const float* __restrict__ kc, const unsigned* __restrict__ idx,
                                      const float* __restrict__ sinks, float* __restrict__ out) {
  extern __shared__ float sh[];
  int total = nraw + ncomp, h0 = blockIdx.x * 4, tid = threadIdx.x;
  float* qs = sh;
  float* ps = sh + 4 * hd;
  float* red = ps + 4 * total;
  for (int i = tid; i < 4 * hd; i += 128) qs[i] = q[(long long)h0 * hd + i];
  __syncthreads();
  for (int j = tid; j < total; j += 128) {
    const float* row = j < nraw ? kr + (long long)((r0 + j) % rcap) * hd
                                : kc + (long long)(ident ? j - nraw : idx[j - nraw]) * hd;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    for (int d = 0; d < hd; d += 4) {
      float4 kk = __ldg((const float4*)(row + d));
      float4 a = *(const float4*)(qs + d), b = *(const float4*)(qs + hd + d);
      float4 c = *(const float4*)(qs + 2 * hd + d), e = *(const float4*)(qs + 3 * hd + d);
      s0 += kk.x * a.x + kk.y * a.y + kk.z * a.z + kk.w * a.w;
      s1 += kk.x * b.x + kk.y * b.y + kk.z * b.z + kk.w * b.w;
      s2 += kk.x * c.x + kk.y * c.y + kk.z * c.z + kk.w * c.w;
      s3 += kk.x * e.x + kk.y * e.y + kk.z * e.z + kk.w * e.w;
    }
    ps[j] = s0 * scale; ps[total + j] = s1 * scale; ps[2 * total + j] = s2 * scale; ps[3 * total + j] = s3 * scale;
  }
  __syncthreads();
  {
    int hh = tid / 32, lane = tid % 32;
    float* p = ps + hh * total;
    float mx = sinks[h0 + hh];
    for (int j = lane; j < total; j += 32) mx = fmaxf(mx, p[j]);
    mx = warp_max(mx);
    float den = 0.f;
    for (int j = lane; j < total; j += 32) { float w = expf(p[j] - mx); p[j] = w; den += w; }
    den = warp_sum(den) + expf(sinks[h0 + hh] - mx);
    if (lane == 0) red[hh] = 1.f / den;
  }
  __syncthreads();
  float4 o0 = {0.f, 0.f, 0.f, 0.f}, o1 = o0, o2 = o0, o3 = o0;
  int d = tid * 4;
  for (int j = 0; j < total; ++j) {
    const float* row = j < nraw ? kr + (long long)((r0 + j) % rcap) * hd
                                : kc + (long long)(ident ? j - nraw : idx[j - nraw]) * hd;
    float4 v = __ldg((const float4*)(row + d));
    float p0 = ps[j], p1 = ps[total + j], p2 = ps[2 * total + j], p3 = ps[3 * total + j];
    o0.x += p0 * v.x; o0.y += p0 * v.y; o0.z += p0 * v.z; o0.w += p0 * v.w;
    o1.x += p1 * v.x; o1.y += p1 * v.y; o1.z += p1 * v.z; o1.w += p1 * v.w;
    o2.x += p2 * v.x; o2.y += p2 * v.y; o2.z += p2 * v.z; o2.w += p2 * v.w;
    o3.x += p3 * v.x; o3.y += p3 * v.y; o3.z += p3 * v.z; o3.w += p3 * v.w;
  }
  float* ob = out + (long long)h0 * hd + d;
  float r0s = red[0], r1s = red[1], r2s = red[2], r3s = red[3];
  *(float4*)ob = make_float4(o0.x * r0s, o0.y * r0s, o0.z * r0s, o0.w * r0s);
  *(float4*)(ob + hd) = make_float4(o1.x * r1s, o1.y * r1s, o1.z * r1s, o1.w * r1s);
  *(float4*)(ob + 2 * hd) = make_float4(o2.x * r2s, o2.y * r2s, o2.z * r2s, o2.w * r2s);
  *(float4*)(ob + 3 * hd) = make_float4(o3.x * r3s, o3.y * r3s, o3.z * r3s, o3.w * r3s);
}

// --- the routed experts ------------------------------------------------------------

// MID[r][j] = silu(clamp(G)) * clamp(U) * W[r], each routed row r of FF
// values; quantized into XQ / XS too when XQ isn't null (N * FF a multiple
// of the block, 256).
extern "C" __global__ void d4_swiglu_w(int n, int ff, float clamp, const float* g, const float* u,
                                       const float* w, float* mid, i8* xq, float* xs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * ff) return;
  float gv = g[i], uv = u[i];
  if (clamp > 1e-6f) { gv = fminf(gv, clamp); uv = fminf(fmaxf(uv, -clamp), clamp); }
  float v = (gv * d4_sigmoid(gv)) * uv * w[i / ff];
  mid[i] = v;
  if (xq) d4_quant_block(v, xq + (i & ~31), xs + (i >> 5), threadIdx.x & 31);
}

__constant__ unsigned long long d4_iq2xxs_grid[256] = {
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
__constant__ unsigned char d4_ksigns_iq2xs[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

#define D4_MV_NSG 2                     // warps a block
#define D4_MV_NR0 4                     // rows a warp

// Y[z] = W[z] X[z] for each expert z (grid z): W[z]'s rows at PTRS[z] + ADD
// (a device address: the expert's slot), X[z] at z*X_STRIDE, Y[z] at
// z*Y_STRIDE.  ggml's kernel_mul_mv_iq2_xxs_f32.  64 threads.
extern "C" __global__ void d4_mv_iq2_xxs(int ncols, int nrows, int row_bytes, long long add, int x_stride, int y_stride,
                                         const unsigned long long* __restrict__ ptrs,
                                         const float* __restrict__ src1, float* __restrict__ dst) {
  __shared__ unsigned long long svalues[256];
  __shared__ unsigned char ssigns[128];
  int tiisg = threadIdx.x & 31, sgitg = threadIdx.x >> 5;
  const int nb = ncols / D4_QK_K, nb32 = nb * (D4_QK_K / 32);
  const int first_row = (blockIdx.x * D4_MV_NSG + sgitg) * D4_MV_NR0;
  const int z = blockIdx.z;
  const unsigned char* x = (const unsigned char*)(ptrs[z] + add) + (long long)first_row * row_bytes;
  const float* y = src1 + (long long)z * x_stride;
  {
    int pos = (32 * sgitg + tiisg) * 4;
    for (int i = 0; i < 4; ++i) svalues[pos + i] = d4_iq2xxs_grid[pos + i];
    pos = (32 * sgitg + tiisg) * 2;
    for (int i = 0; i < 2; ++i) ssigns[pos + i] = d4_ksigns_iq2xs[pos + i];
    __syncthreads();
  }
  float sumf[D4_MV_NR0] = {0.f, 0.f, 0.f, 0.f};
  if (first_row < nrows) {
    const float* y4 = y + 32 * tiisg;
    for (int ib32 = tiisg; ib32 < nb32; ib32 += 32) {
      float yl[32];
#pragma unroll
      for (int i = 0; i < 32; i += 4) {
        float4 t = __ldg((const float4*)(y4 + i));
        yl[i] = t.x; yl[i + 1] = t.y; yl[i + 2] = t.z; yl[i + 3] = t.w;
      }
      const int ibl = ib32 / (D4_QK_K / 32), ib = ib32 % (D4_QK_K / 32);
      const unsigned char* xr = x + ibl * 66;          // a block: d, then 32 u16s
#pragma unroll
      for (int row = 0; row < D4_MV_NR0; row++) {
        const unsigned char* br = xr + (long long)row * row_bytes;
        const u16* q2 = (const u16*)(br + 2) + 4 * ib;
        const float db = h2f(__ldg((const u16*)br));
        const unsigned a0 = __ldg(q2), a1 = __ldg(q2 + 1), a2 = __ldg(q2 + 2), a3 = __ldg(q2 + 3);
        const unsigned aux32g = a0 | (a1 << 16);
        const unsigned aux32 = a2 | (a3 << 16);
        const float d = db * (0.5f + (aux32 >> 28));
        float sum = 0.f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
          const unsigned char* grid = (const unsigned char*)(svalues + ((aux32g >> (8 * l)) & 255));
          const unsigned char signs = ssigns[(aux32 >> 7 * l) & 127];
#pragma unroll
          for (int j = 0; j < 8; ++j) sum += yl[8 * l + j] * grid[j] * (signs & (1 << j) ? -1.f : 1.f);
        }
        sumf[row] += d * sum;
      }
      y4 += 32 * 32;
    }
  }
  float* out = dst + (long long)z * y_stride;
#pragma unroll
  for (int row = 0; row < D4_MV_NR0; ++row) {
    const float s = warp_sum(sumf[row]);
    if (tiisg == 0 && first_row + row < nrows) out[first_row + row] = s * 0.25f;
  }
}

// The same for Q2_K: ggml's kernel_mul_mv_q2_K_f32.  A block: 16 scale
// bytes, 64 of quants, then d and dmin (84 bytes).
extern "C" __global__ void d4_mv_q2_k(int ncols, int nrows, int row_bytes, long long add, int x_stride, int y_stride,
                                      const unsigned long long* __restrict__ ptrs,
                                      const float* __restrict__ src1, float* __restrict__ dst) {
  int tiisg = threadIdx.x & 31, sgitg = threadIdx.x >> 5;
  const int nb = ncols / D4_QK_K;
  const int first_row = (blockIdx.x * D4_MV_NSG + sgitg) * D4_MV_NR0;
  const int z = blockIdx.z;
  const unsigned char* x = (const unsigned char*)(ptrs[z] + add) + (long long)first_row * row_bytes;
  const float* y = src1 + (long long)z * x_stride;
  float sumf[D4_MV_NR0] = {0.f, 0.f, 0.f, 0.f};
  const int ix = tiisg / 8, it = tiisg % 8, iq = it / 4, ir = it % 4, is = (8 * ir) / 16;
  if (first_row < nrows) {
    const float* y4 = y + ix * D4_QK_K + 128 * iq + 8 * ir;
    for (int ib = ix; ib < nb; ib += 4) {
      float yl[32];
      float sumy[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        yl[i + 0] = y4[i + 0];   sumy[0] += yl[i + 0];
        yl[i + 8] = y4[i + 32];  sumy[1] += yl[i + 8];
        yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
        yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
      }
#pragma unroll
      for (int row = 0; row < D4_MV_NR0; row++) {
        const unsigned char* blk = x + (long long)row * row_bytes + ib * 84;
        const unsigned char* sc = blk + 8 * iq + is;
        const u16* qs = (const u16*)(blk + 16) + 16 * iq + 4 * ir;
        float acc1[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
          const unsigned q = __ldg(qs + i / 2);
          acc1[0] += yl[i + 0] * (q & 0x0003); acc2[0] += yl[i + 1] * (q & 0x0300);
          acc1[1] += yl[i + 8] * (q & 0x000c); acc2[1] += yl[i + 9] * (q & 0x0c00);
          acc1[2] += yl[i + 16] * (q & 0x0030); acc2[2] += yl[i + 17] * (q & 0x3000);
          acc1[3] += yl[i + 24] * (q & 0x00c0); acc2[3] += yl[i + 25] * (q & 0xc000);
        }
        const float dall = h2f(__ldg((const u16*)(blk + 80))), dmin = h2f(__ldg((const u16*)(blk + 82))) * 1.f / 16.f;
        const unsigned s0 = __ldg(sc), s2 = __ldg(sc + 2), s4 = __ldg(sc + 4), s6 = __ldg(sc + 6);
        sumf[row] += dall * ((acc1[0] + 1.f / 256.f * acc2[0]) * (s0 & 0xF) * 1.f / 1.f +
                             (acc1[1] + 1.f / 256.f * acc2[1]) * (s2 & 0xF) * 1.f / 4.f +
                             (acc1[2] + 1.f / 256.f * acc2[2]) * (s4 & 0xF) * 1.f / 16.f +
                             (acc1[3] + 1.f / 256.f * acc2[3]) * (s6 & 0xF) * 1.f / 64.f) -
                     dmin * (sumy[0] * (s0 & 0xF0) + sumy[1] * (s2 & 0xF0) +
                             sumy[2] * (s4 & 0xF0) + sumy[3] * (s6 & 0xF0));
      }
      y4 += 4 * D4_QK_K;
    }
  }
  float* out = dst + (long long)z * y_stride;
#pragma unroll
  for (int row = 0; row < D4_MV_NR0; ++row) {
    const float s = warp_sum(sumf[row]);
    if (tiisg == 0 && first_row + row < nrows) out[first_row + row] = s;
  }
}

// MOE[r] = sum of the K experts' rows of DOWN, in order, plus SHARED:
// expert i's row PERM[i] (i when PERM is null; the rows are by device).
extern "C" __global__ void d4_moe_sum(int d, int k, const float* down, const float* shared_, float* moe,
                                      const unsigned* perm) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= d) return;
  float acc = 0.f;
  for (int i = 0; i < k; ++i) acc += down[(perm ? perm[i] : i) * d + r];
  moe[r] = acc + shared_[r];
}

// The output head's mix: W[h] = sigmoid(PRE[h] SCALE + BASE[h]) + EPS, OUT[j]
// = sum_h STREAMS[h][j] W[h].  Grid (D/256, tokens).
extern "C" __global__ void d4_head_mix(int d, int n, float eps, const float* pre, const float* scale,
                                       const float* base, const float* streams, float* out) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
  if (j >= d) return;
  pre += t * 16; streams += (long long)t * n * d; out += (long long)t * d;
  float acc = 0.f;
  for (int h = 0; h < n; ++h) {
    float z = pre[h] * scale[0] + base[h];
    acc += streams[h * d + j] * (d4_sigmoid(z) + eps);
  }
  out[j] = acc;
}
