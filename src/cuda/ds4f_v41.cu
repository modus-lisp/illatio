// ds4f_v41.cu — DeepSeek V4.1 Flash on the GPU (cuda-ds41f.lisp's decode,
// cuda-ds41f-prefill.lisp's passes): what differs from V4 (ds4f.cu,
// ds4f_chunk.cu, ds4f_mm.cu, whose products, attention and experts it uses),
// and the passes' own (the chunk kernels, the tensor cores' attention and
// indexer scores, the router's tiled product, Engram's decode).  BF16
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

// RoPE on the last NROT dims of each of NH heads of WIDTH at X, for NTOK
// tokens (NH heads each, one after another): CS the pairs' (cos, sin), from
// the CPU, NROT a token; each turned element BF16.
extern "C" __global__ void d41_rope(int nh, int width, int nrot, int ntok, float* x, const float* cs) {
  int half_rot = nrot / 2;
  long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (long long)ntok * nh * half_rot) return;
  int t = gid / (nh * half_rot), h = (gid / half_rot) % nh, i = gid % half_rot;
  float* p = x + ((long long)t * nh + h) * width + (width - nrot) + 2 * i;
  cs += (long long)t * nrot;
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
// A warp a token (32 apart in MIX and PC).
extern "C" __global__ void d41_hc_split(int iters, float eps, const float* mix, const float* scale,
                                        const float* base, float* pc) {
  const unsigned F = 0xffffffffu;
  mix += blockIdx.x * 32; pc += blockIdx.x * 32;
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

// OUT[j] = bf16(sum_h STREAMS[h][j] PRE[h]), in stream order; grid (D/256,
// tokens), a token's PRE PSTRIDE after the one before (0: the same for all).
extern "C" __global__ void d41_hc_sum(int d, int n, int pstride, const float* streams, const float* pre, float* out) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
  if (j >= d) return;
  streams += (long long)t * n * d; pre += (long long)t * pstride; out += (long long)t * d;
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
  streams += (long long)blockIdx.y * n * d; kv += (long long)blockIdx.y * (n + 1) * d;   // grid (n, tokens)
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
// for FF values (a chunk's tokens' one after another); quantized into XQ /
// XS too when XQ isn't null.
extern "C" __global__ void d41_swiglu(int ff, float clamp, const float* g, const float* u, float* mid,
                                      i8* xq, float* xs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ff) return;
  float gv = fminf(d41_bf16(g[i]), clamp), uv = fminf(fmaxf(d41_bf16(u[i]), -clamp), clamp);
  float v = d41_bf16(__fmul_rn(__fmul_rn(gv, d4_sigmoid(gv)), uv));
  mid[i] = v;
  if (xq) d4_quant_block(v, xq + (i & ~31), xs + (i >> 5), threadIdx.x & 31);
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

// --- a prefill chunk's own ------------------------------------------------------------

// Y[t] = W X[t] for a chunk's tokens (grid (ROWS/4, tokens)), W's rows F32:
// D41_MV_F32's sums, X_STRIDE and Y_STRIDE a token apart.
extern "C" __global__ void d41_mv_f32n(int ncols, int nrows, int x_stride, int y_stride, const float* __restrict__ w,
                                       const float* __restrict__ x, float* __restrict__ y) {
  __shared__ float red[32];
  int r0 = blockIdx.x * 4, tid = threadIdx.x;
  x += (long long)blockIdx.y * x_stride; y += (long long)blockIdx.y * y_stride;
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

// The ratio-2 compressor's pairs that end in a chunk (from POS0): pair B
// ends at the odd position P1 + 2B, its first token the one before -- the
// chunk's row, or the one carried in CKV / CSC when it came before the
// chunk.  D41_POOL2's arithmetic into LATENT row B.  Grid NB, HD threads.
extern "C" __global__ void d41_pool2_chunk(int hd, int pos0, int p1, const float* pkv, const float* psc,
                                           const float* ckv, const float* csc, float* latent) {
  int b = blockIdx.x, j = threadIdx.x;
  int p = p1 + 2 * b, a = p - 1;
  float ka, sa;
  if (a >= pos0) { ka = pkv[(long long)(a - pos0) * hd + j]; sa = psc[(long long)(a - pos0) * hd + j]; }
  else { ka = ckv[j]; sa = csc[j]; }
  float kb = pkv[(long long)(p - pos0) * hd + j], sb = psc[(long long)(p - pos0) * hd + j];
  float m = fmaxf(sa, sb), ea = expf(sa - m), eb = expf(sb - m);
  latent[(long long)b * hd + j] = d41_bf16(__fdiv_rn(__fadd_rn(__fmul_rn(ka, ea), __fmul_rn(kb, eb)), __fadd_rn(ea, eb)));
}

// D41_IX_SCORES for a chunk's tokens: SCORES[t][c] (NC a row) for the keys
// token t sees (VIS[t]), -inf past them.  Grid (NC/8, tokens), 256 threads.
extern "C" __global__ void d41_ix_scores_chunk(int nh, int nc, const float* __restrict__ q, const float* __restrict__ w,
                                               const float* __restrict__ keys, const unsigned* __restrict__ vis,
                                               float* __restrict__ scores) {
  extern __shared__ float qs[];
  int t = blockIdx.y;
  q += (long long)t * nh * 128; w += (long long)t * nh; scores += (long long)t * nc;
  int c = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  unsigned v = vis[t];
  if ((unsigned)(blockIdx.x * (blockDim.x >> 5)) >= v) {
    if (c < nc && lane == 0) scores[c] = -INFINITY;
    return;
  }
  for (int i = threadIdx.x; i < nh * 128; i += blockDim.x) qs[i] = q[i];
  __syncthreads();
  if (c >= nc) return;
  if ((unsigned)c >= v) { if (lane == 0) scores[c] = -INFINITY; return; }
  float4 k = __ldg((const float4*)(keys + (long long)c * 128) + lane);
  float total = 0.f;
  for (int h = 0; h < nh; ++h) {
    float4 a = *(const float4*)(qs + h * 128 + lane * 4);
    float dot = warp_sum(a.x * k.x + a.y * k.y + a.z * k.z + a.w * k.w);
    total += fmaxf(dot, 0.f) * w[h];
  }
  if (lane == 0) scores[c] = total * (1.f / 64.f);
}

// Engram's table rows, decoded: OUT[r][j] = bf16(CODES[RAW[r][j]] *
// 2^(RAW[r][256 + j/32] - 127)), a row 264 bytes (256 E4M3 codes, 8 E8M0
// scales).  grid ceil(NROWS*256/256), block 256.
extern "C" __global__ void d41_eg_decode(int nrows, const unsigned char* __restrict__ raw,
                                         const float* __restrict__ codes, float* __restrict__ out) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)nrows * 256) return;
  const unsigned char* p = raw + (i / 256) * 264;
  const int j = (int)(i % 256);
  out[i] = d41_bf16(ldexpf(codes[p[j]], (int)p[256 + j / 32] - 127));
}

// --- a chunk's attention on the tensor cores ---------------------------------------
//
// D4_ATTEND's (ds4f_chunk.cu) attention, args as its, for V4.1, whose Q
// and keys are BF16 values already (Q rounded and turned through BF16, the
// keys through FP8 or FP4 into BF16): S = Q K^T as BF16 products summed in
// f32 is exact but for the order of its sums.  P goes in as two BF16 parts
// (hi + lo, 16 bits of it), V (the keys) exactly.  A block a token, 8
// warps; keys 16 a step: warp w's S tile heads 16*(w%4).., keys 8*(w/4)..;
// its O columns 64w.. for all 64 heads, in registers (m16n8k16's fragments:
// lane l's rows l/4 and l/4+8, columns 2*(l%4) and the next).  HD 512, NH 64.
// grid N, block 256, shared D41_ATTEND_SHARED (sm_80 and up).

#define D41_AQ (512 + 8)                // a row of Q or of a key, BF16 (+8 against bank conflicts)
#define D41_AK 16                       // keys a step
#define D41_AP (D41_AK + 8)             // a row of P
#define D41_AS (D41_AK + 1)             // a row of S
// 66560 + 16640 + 4352 + 2 * 3072 + 768 = 94464
#define D41_ATTEND_SHARED (2 * 64 * D41_AQ + 2 * D41_AK * D41_AQ + 4 * 64 * D41_AS + 2 * 2 * 64 * D41_AP + 4 * 192)

__device__ __forceinline__ unsigned d41_bf16_bits(float x) {   // the nearest BF16 (ties even): its bits
  unsigned u = __float_as_uint(x);
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ unsigned d41_bf16_pair(float x, float y) {
  return d41_bf16_bits(x) | (d41_bf16_bits(y) << 16);
}
__device__ __forceinline__ void d41_mma_bf16(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

extern "C" __global__ void __launch_bounds__(256, 1)
d41_attend_mma(int nh, int hd, float scale, int maxc, int rcap, int ident,
               const float* __restrict__ q, const float* __restrict__ kr, const float* __restrict__ kc,
               const unsigned* __restrict__ raws, const unsigned* __restrict__ nraw,
               const unsigned* __restrict__ ncomp, const unsigned* __restrict__ idx,
               const float* __restrict__ sinks, float* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char d41_sm[];
  unsigned short* qs = (unsigned short*)d41_sm;                 // [64][D41_AQ]
  unsigned short* ks = qs + 64 * D41_AQ;                        // [D41_AK][D41_AQ]
  float* ss = (float*)(ks + D41_AK * D41_AQ);                   // [64][D41_AS]
  unsigned short* ph = (unsigned short*)(ss + 64 * D41_AS);     // [64][D41_AP]
  unsigned short* pl = ph + 64 * D41_AP;
  float* ml = (float*)(pl + 64 * D41_AP);                       // 64 maxima, then 64 sums
  float* al = ml + 128;                                         // the step's alphas, then 1/L
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tq = lane & 3;
  const unsigned nr = nraw[t], nc = ncomp[t], total = nr + nc, r0 = raws[t];
  const float2* qt = (const float2*)(q + (long long)t * 64 * 512);
  for (int i = tid; i < 64 * 256; i += 256) {
    const float2 v = qt[i];
    *(unsigned*)(qs + (i >> 8) * D41_AQ + 2 * (i & 255)) = d41_bf16_pair(v.x, v.y);
  }
  if (tid < 64) { ml[tid] = sinks[tid]; ml[64 + tid] = 1.0f; }   // max = sink, sum = exp(0)
  float o[4][8][4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int r = 0; r < 4; ++r) o[mt][nt][r] = 0.f;
  __syncthreads();
  for (unsigned kb = 0; kb < total; kb += D41_AK) {
    // this step's keys, as BF16
    for (int i = tid; i < D41_AK * 256; i += 256) {
      const unsigned j = kb + (i >> 8);
      const int c = 2 * (i & 255);
      float2 v = make_float2(0.f, 0.f);
      if (j < nr) v = *(const float2*)(kr + (long long)(rcap ? (r0 + j) % (unsigned)rcap : r0 + j) * 512 + c);
      else if (j < total) v = *(const float2*)(kc + (long long)(ident ? j - nr : idx[(long long)t * maxc + (j - nr)]) * 512 + c);
      *(unsigned*)(ks + (i >> 8) * D41_AQ + c) = d41_bf16_pair(v.x, v.y);
    }
    __syncthreads();
    // S, a tile a warp
    {
      const int mt = w & 3, nt = w >> 2;
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      const unsigned short* qa = qs + (mt * 16 + g) * D41_AQ + 2 * tq;
      const unsigned short* kp = ks + (nt * 8 + g) * D41_AQ + 2 * tq;
#pragma unroll 8
      for (int k = 0; k < 512; k += 16)
        d41_mma_bf16(c, *(const unsigned*)(qa + k), *(const unsigned*)(qa + 8 * D41_AQ + k),
                     *(const unsigned*)(qa + k + 8), *(const unsigned*)(qa + 8 * D41_AQ + k + 8),
                     *(const unsigned*)(kp + k), *(const unsigned*)(kp + k + 8));
      float* s0 = ss + (mt * 16 + g) * D41_AS + nt * 8 + 2 * tq;
      s0[0] = c[0]; s0[1] = c[1]; s0[8 * D41_AS] = c[2]; s0[8 * D41_AS + 1] = c[3];
    }
    __syncthreads();
    // the running softmax: a thread a head's 4 keys; P split hi + lo
    {
      const int h = tid >> 2, j0 = (tid & 3) * 4;
      const unsigned F = 0xffffffffu;
      float v[4], mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        v[j] = (kb + j0 + j < total) ? ss[h * D41_AS + j0 + j] * scale : -INFINITY;
        mx = fmaxf(mx, v[j]);
      }
      mx = fmaxf(mx, __shfl_xor_sync(F, mx, 1)); mx = fmaxf(mx, __shfl_xor_sync(F, mx, 2));
      const float m = ml[h], mn = fmaxf(m, mx);
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float e = expf(v[j] - mn);
        const unsigned hb = d41_bf16_bits(e);
        sum += e;
        ph[h * D41_AP + j0 + j] = (unsigned short)hb;
        pl[h * D41_AP + j0 + j] = (unsigned short)d41_bf16_bits(e - __uint_as_float(hb << 16));
      }
      sum += __shfl_xor_sync(F, sum, 1); sum += __shfl_xor_sync(F, sum, 2);
      if ((tid & 3) == 0) {
        const float alpha = expf(m - mn);
        ml[h] = mn;
        ml[64 + h] = ml[64 + h] * alpha + sum;
        al[h] = alpha;
      }
    }
    __syncthreads();
    // O = diag(alpha) O + P V for this warp's columns
    {
      unsigned b[8][2];
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const unsigned short* vp = ks + 2 * tq * D41_AQ + 64 * w + 8 * nt + g;
        b[nt][0] = vp[0] | ((unsigned)vp[D41_AQ] << 16);
        b[nt][1] = vp[8 * D41_AQ] | ((unsigned)vp[9 * D41_AQ] << 16);
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        const float a0 = al[mt * 16 + g], a1 = al[mt * 16 + g + 8];
        const unsigned short* hp = ph + (mt * 16 + g) * D41_AP + 2 * tq;
        const unsigned short* lp = pl + (mt * 16 + g) * D41_AP + 2 * tq;
        const unsigned h0 = *(const unsigned*)hp, h1 = *(const unsigned*)(hp + 8 * D41_AP),
                       h2 = *(const unsigned*)(hp + 8), h3 = *(const unsigned*)(hp + 8 * D41_AP + 8);
        const unsigned l0 = *(const unsigned*)lp, l1 = *(const unsigned*)(lp + 8 * D41_AP),
                       l2 = *(const unsigned*)(lp + 8), l3 = *(const unsigned*)(lp + 8 * D41_AP + 8);
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          float* c = o[mt][nt];
          c[0] = __fmul_rn(c[0], a0); c[1] = __fmul_rn(c[1], a0);
          c[2] = __fmul_rn(c[2], a1); c[3] = __fmul_rn(c[3], a1);
          d41_mma_bf16(c, l0, l1, l2, l3, b[nt][0], b[nt][1]);
          d41_mma_bf16(c, h0, h1, h2, h3, b[nt][0], b[nt][1]);
        }
      }
    }
    __syncthreads();
  }
  if (tid < 64) al[tid] = 1.0f / ml[64 + tid];
  __syncthreads();
  float* ot = out + (long long)t * 64 * 512;
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) {
    const int h = mt * 16 + g;
    const float i0 = al[h], i1 = al[h + 8];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int col = 64 * w + 8 * nt + 2 * tq;
      *(float2*)(ot + (long long)h * 512 + col) = make_float2(__fmul_rn(o[mt][nt][0], i0), __fmul_rn(o[mt][nt][1], i0));
      *(float2*)(ot + (long long)(h + 8) * 512 + col) = make_float2(__fmul_rn(o[mt][nt][2], i1), __fmul_rn(o[mt][nt][3], i1));
    }
  }
}

// --- candidate blocks (the late indexers past CAND_TOP blocks) ---------------------
//
// On the candidate source layer each token keeps its CAND_TOP best blocks of
// BSZ compressed rows (its newest always); the later index layers pick only
// in them.  D41_BLOCK_MAX: BMAX[t][b] = the best of token t's SCORES (rows of
// NC) in block b (among its VIS[t] rows, VIS1 when VIS is null), its newest
// block +inf; BVIS[t] its blocks.  grid (ceil(NB/256), N), block 256.
extern "C" __global__ void d41_block_max(int nc, int nb, int bsz, const unsigned* vis, int vis1,
                                         const float* __restrict__ scores, float* bmax, unsigned* bvis) {
  const int t = blockIdx.y;
  const unsigned b = blockIdx.x * 256 + threadIdx.x, v = vis ? vis[t] : (unsigned)vis1, ub = bsz;
  const unsigned blocks = (v + ub - 1) / ub;
  if (blockIdx.x == 0 && threadIdx.x == 0) bvis[t] = blocks;
  if (b >= (unsigned)nb) return;
  float m = -INFINITY;
  if (b < blocks) {
    if (b == (v - 1) / ub) m = INFINITY;
    else for (unsigned j = b * ub; j < min(v, (b + 1) * ub); ++j) m = fmaxf(m, scores[(long long)t * nc + j]);
  }
  bmax[(long long)t * nb + b] = m;
}

// MASK[t][b] = 1 for the blocks BIDX lists for token t (the first
// min(BVIS[t], K) of its row of K), else 0.  grid N, block 1024.
extern "C" __global__ void d41_block_mask(int nb, int k, const unsigned* bvis, const unsigned* bidx,
                                          unsigned char* mask) {
  const int t = blockIdx.x;
  unsigned char* mr = mask + (long long)t * nb;
  for (int b = threadIdx.x; b < nb; b += blockDim.x) mr[b] = 0;
  __syncthreads();
  const unsigned m = min(bvis[t], (unsigned)k);
  for (unsigned i = threadIdx.x; i < m; i += blockDim.x) mr[bidx[(long long)t * k + i]] = 1;
}

// SCORES[t][c] = -inf outside token t's candidate blocks.  grid (ceil(NC/256), N), block 256.
extern "C" __global__ void d41_block_apply(int nc, int nb, int bsz, const unsigned char* mask, float* scores) {
  const int t = blockIdx.y;
  const unsigned c = blockIdx.x * 256 + threadIdx.x;
  if (c >= (unsigned)nc) return;
  const unsigned b = c / (unsigned)bsz;
  if (b >= (unsigned)nb || !mask[(long long)t * nb + b]) scores[(long long)t * nc + c] = -INFINITY;
}

// Y[t][r] = W[r] . X[t], all f32, for many tokens (the router): a block
// 64 tokens x 64 rows, a thread 4 x 4 of them, K 16 a step through shared
// memory (each sum in K's order, one FMA a term).  NCOLS a multiple of 16;
// X_STRIDE and Y_STRIDE a token apart.  grid (ceil(N/64), ceil(NROWS/64)), block 256.
extern "C" __global__ void __launch_bounds__(256)
d41_mm_f32(int ncols, int nrows, int n, int x_stride, int y_stride, const float* __restrict__ w,
           const float* __restrict__ x, float* __restrict__ y) {
  __shared__ float xs[16][64 + 4], ws[16][64 + 4];
  const int t0 = blockIdx.x * 64, r0 = blockIdx.y * 64, tid = threadIdx.x;
  const int tx = tid & 15, ty = tid >> 4;          // rows 4tx.., tokens 4ty..
  const int lr = tid >> 2, lk = (tid & 3) * 4;     // loads: row/token lr, columns lk..lk+3
  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
  for (int k0 = 0; k0 < ncols; k0 += 16) {
    float4 xv = make_float4(0.f, 0.f, 0.f, 0.f), wv = make_float4(0.f, 0.f, 0.f, 0.f);
    if (t0 + lr < n) xv = *(const float4*)(x + (long long)(t0 + lr) * x_stride + k0 + lk);
    if (r0 + lr < nrows) wv = __ldg((const float4*)(w + (long long)(r0 + lr) * ncols + k0 + lk));
    xs[lk][lr] = xv.x; xs[lk + 1][lr] = xv.y; xs[lk + 2][lr] = xv.z; xs[lk + 3][lr] = xv.w;
    ws[lk][lr] = wv.x; ws[lk + 1][lr] = wv.y; ws[lk + 2][lr] = wv.z; ws[lk + 3][lr] = wv.w;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const float4 a = *(const float4*)&xs[k][4 * ty], b = *(const float4*)&ws[k][4 * tx];
      const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int t = t0 + 4 * ty + i;
    if (t >= n) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int r = r0 + 4 * tx + j;
      if (r < nrows) y[(long long)t * y_stride + r] = acc[i][j];
    }
  }
}

// D41_IX_SCORES_CHUNK on the tensor cores (NH 32): Q and the keys are FP4
// values with power-of-two scales, exact in BF16, so each head's dot is
// exact products summed in f32; the heads then added in order, as there.
// A block 128 keys x 32 tokens, 8 warps: the keys in shared memory, then
// token by token its Q, warp w's dots keys 16w.. for all 32 heads, the
// heads' sum a thread a key.  grid (ceil(NC/128), ceil(N/32)), block 256,
// shared D41_IXM_SHARED.
#define D41_IXR (128 + 8)               // a BF16 row of 128 (+8 against bank conflicts)
#define D41_IXS2 (128 + 4)              // a row of the dots
#define D41_IXM_SHARED (2 * 128 * D41_IXR + 2 * 32 * D41_IXR + 4 * 32 * D41_IXS2)

extern "C" __global__ void __launch_bounds__(256)
d41_ix_scores_mma(int nc, int n, const float* __restrict__ q, const float* __restrict__ w,
                  const float* __restrict__ keys, const unsigned* __restrict__ vis, float* __restrict__ scores) {
  extern __shared__ __align__(16) unsigned char d41_ixm[];
  unsigned short* ks = (unsigned short*)d41_ixm;               // [128][D41_IXR]
  unsigned short* qs = ks + 128 * D41_IXR;                      // [32][D41_IXR]
  float* ss = (float*)(qs + 32 * D41_IXR);                      // [32][D41_IXS2]
  const int c0 = blockIdx.x * 128, t0 = blockIdx.y * 32, tid = threadIdx.x;
  const int lane = tid & 31, wp = tid >> 5, g = lane >> 2, tq = lane & 3;
  const int tn = min(32, n - t0);
  unsigned maxvis = 0;
  for (int i = 0; i < tn; ++i) maxvis = max(maxvis, vis[t0 + i]);
  if ((unsigned)c0 >= maxvis) {
    for (int i = tid; i < tn * 128; i += 256) {
      const int c = c0 + (i & 127);
      if (c < nc) scores[(long long)(t0 + (i >> 7)) * nc + c] = -INFINITY;
    }
    return;
  }
  for (int i = tid; i < 128 * 64; i += 256) {
    const int r = i >> 6, c = 2 * (i & 63);
    float2 v = make_float2(0.f, 0.f);
    if (c0 + r < nc) v = *(const float2*)(keys + (long long)(c0 + r) * 128 + c);
    *(unsigned*)(ks + r * D41_IXR + c) = d41_bf16_pair(v.x, v.y);
  }
  for (int ti = 0; ti < tn; ++ti) {
    const int t = t0 + ti;
    const unsigned v = vis[t];
    __syncthreads();                    // the keys in; the last token's dots read
    const float2* qt = (const float2*)(q + (long long)t * 32 * 128);
    for (int i = tid; i < 32 * 64; i += 256) {
      const float2 x = qt[i];
      *(unsigned*)(qs + (i >> 6) * D41_IXR + 2 * (i & 63)) = d41_bf16_pair(x.x, x.y);
    }
    __syncthreads();
    if ((unsigned)(c0 + 16 * wp) < v) {
      float c[2][2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int r = 0; r < 4; ++r) c[mt][nt][r] = 0.f;
#pragma unroll
      for (int k = 0; k < 128; k += 16) {
        unsigned b[2][2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const unsigned short* kp = ks + (16 * wp + 8 * nt + g) * D41_IXR + k + 2 * tq;
          b[nt][0] = *(const unsigned*)kp; b[nt][1] = *(const unsigned*)(kp + 8);
        }
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          const unsigned short* qa = qs + (16 * mt + g) * D41_IXR + k + 2 * tq;
          const unsigned a0 = *(const unsigned*)qa, a1 = *(const unsigned*)(qa + 8 * D41_IXR),
                         a2 = *(const unsigned*)(qa + 8), a3 = *(const unsigned*)(qa + 8 * D41_IXR + 8);
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) d41_mma_bf16(c[mt][nt], a0, a1, a2, a3, b[nt][0], b[nt][1]);
        }
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          float* s0 = ss + (16 * mt + g) * D41_IXS2 + 16 * wp + 8 * nt + 2 * tq;
          s0[0] = c[mt][nt][0]; s0[1] = c[mt][nt][1];
          s0[8 * D41_IXS2] = c[mt][nt][2]; s0[8 * D41_IXS2 + 1] = c[mt][nt][3];
        }
    }
    __syncthreads();
    if (tid < 128) {
      const int cc = c0 + tid;
      if (cc < nc) {
        float out = -INFINITY;
        if ((unsigned)cc < v) {
          const float* wt = w + (long long)t * 32;
          float total = 0.f;
          for (int h = 0; h < 32; ++h) total += fmaxf(ss[h * D41_IXS2 + tid], 0.f) * wt[h];
          out = total * (1.f / 64.f);
        }
        scores[(long long)t * nc + cc] = out;
      }
    }
  }
}
