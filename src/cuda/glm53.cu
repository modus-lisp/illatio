// glm53.cu — GLM-5.3 Flash's decode on an NVIDIA GPU (cuda-glm53.lisp):
// what differs from DeepSeek V4 (whose Q8_0 products, HC, norms, shared and
// routed experts it uses).  BF16 products (HC's fn, the indexer); KDA's
// conv and delta rule (a warp a value row, a block a head, ds4's
// glm53_kda_decode); DSA's latent cache (F16) and its pooled indexer
// keys; the absorbed attention over the latent; the streams' mean.

__device__ __forceinline__ float g53_bf(u16 b) { return __uint_as_float(((unsigned)b) << 16); }

// Y = W X for a BF16 matrix W (NROWS rows of NCOLS, a multiple of 4): a block 4 rows.
extern "C" __global__ void g53_mv_bf16(int ncols, int nrows, const u16* __restrict__ w,
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
        s[r] += g53_bf(h.x & 0xffff) * xv.x + g53_bf(h.x >> 16) * xv.y + g53_bf(h.y & 0xffff) * xv.z
              + g53_bf(h.y >> 16) * xv.w;
      }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    float t = d4_block_sum(s[r], red);
    if (tid == 0 && r0 + r < nrows) y[r0 + r] = t;
  }
}

// --- KDA ---------------------------------------------------------------------------

// The 4-tap conv of Q, K, V (QKV: 3 x P raw projections) over the raw
// history HIST (3 x 3 x P, the oldest slot first), then SiLU, in place;
// the history shifted, the raw token last.  W: 3 x P x 4 taps.
extern "C" __global__ void g53_kda_conv(int p, float* qkv, const float* w, float* hist) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * p) return;
  int a = i / p, c = i - a * p;
  float* h = hist + (long long)a * 3 * p + c;
  const float* wc = w + (long long)i * 4;
  float raw = qkv[i];
  float acc = h[0] * wc[0];
  acc = fmaf(h[p], wc[1], acc);
  acc = fmaf(h[2 * p], wc[2], acc);
  acc = fmaf(raw, wc[3], acc);
  h[0] = h[p]; h[p] = h[2 * p]; h[2 * p] = raw;
  qkv[i] = acc / (1.f + expf(-acc));
}

// One token's delta rule for head blockIdx.x (128 threads, HD = 128): Q, K
// (QKV after the conv) L2-normed, Q / sqrt(HD) too; decay per key channel
// exp(LB sigmoid(exp(A_LOG) (F + DT_BIAS))); S[v][:] *= decay, then
// S[v][:] += k (v - S[v].k) beta; o[v] = S[v].q; o RMS-normed per head
// (EPS) times ONORM, times sigmoid(G).  S: heads x HD (value) x HD (key).
extern "C" __global__ void g53_kda_step(int p, float lb, float eps, const float* qkv, const float* f,
                                        const float* braw, const float* g, const float* dt_bias,
                                        const float* a_log, const float* onorm, float* S, float* out) {
  __shared__ float qs[128], ks[128], dec[128], os[128], red[32];
  int h = blockIdx.x, i = threadIdx.x, c = h * 128 + i;
  float q = qkv[c], k = qkv[p + c];
  float ea = expf(a_log[h]);
  float gv = ea * (f[c] + dt_bias[c]);
  dec[i] = expf(lb * (1.f / (1.f + expf(-gv))));
  float qn = d4_block_sum(q * q, red), kn = d4_block_sum(k * k, red);
  qs[i] = q * (rsqrtf(qn + 1e-6f) * 0.08838834764831845f);
  ks[i] = k * rsqrtf(kn + 1e-6f);
  float beta = 1.f / (1.f + expf(-braw[h]));
  __syncthreads();
  int lane = i & 31, wp = i >> 5;
  float4 d4 = ((const float4*)dec)[lane], k4 = ((const float4*)ks)[lane], q4 = ((const float4*)qs)[lane];
  for (int v = wp; v < 128; v += 4) {
    float4* row = (float4*)(S + ((long long)h * 128 + v) * 128) + lane;
    float4 s = *row;
    s.x *= d4.x; s.y *= d4.y; s.z *= d4.z; s.w *= d4.w;
    float hk = warp_sum(s.x * k4.x + s.y * k4.y + s.z * k4.z + s.w * k4.w);
    float delta = (qkv[2 * p + h * 128 + v] - hk) * beta;
    s.x = fmaf(k4.x, delta, s.x); s.y = fmaf(k4.y, delta, s.y);
    s.z = fmaf(k4.z, delta, s.z); s.w = fmaf(k4.w, delta, s.w);
    *row = s;
    float o = warp_sum(s.x * q4.x + s.y * q4.y + s.z * q4.z + s.w * q4.w);
    if (lane == 0) os[v] = o;
  }
  __syncthreads();
  float o = os[i];
  float ss = d4_block_sum(o * o, red);
  float sc = rsqrtf(ss / 128.f + eps);
  out[c] = o * sc * onorm[i] * (1.f / (1.f + expf(-g[c])));
}

// --- DSA ---------------------------------------------------------------------------

// Token blockIdx.x's latent row KV (512) RMS-normed by W (EPS) into the
// cache's row POS + blockIdx.x, as F16.  A block of 512 threads.
extern "C" __global__ void g53_kv_store(int lat, float eps, int pos, const float* kv, const float* w, u16* cache) {
  __shared__ float red[32];
  int i = threadIdx.x;
  pos += blockIdx.x;
  float v = kv[(long long)blockIdx.x * lat + i];
  float ss = d4_block_sum(v * v, red);
  float sc = rsqrtf(ss / lat + eps);
  cache[(long long)pos * lat + i] = f2h(v * sc * w[i]);
}

// The indexer's raw key K and gate G (HD each) into TAIL's row R (keys
// first, 4 rows of HD, then gates); on a pool's last row (R = 3) the pool's
// key into IXK's row P: each raw key LayerNormed (eps 1e-6, W, B), weighted
// per channel by softmax over the 4 of gate + APE; F16.  A block of HD.
extern "C" __global__ void g53_ix_pool(int hd, int r, int p, const float* k, const float* g, float* tail,
                                       const float* w, const float* b, const float* ape, u16* ixk) {
  __shared__ float red[32];
  int i = threadIdx.x;
  tail[r * hd + i] = k[i];
  tail[(4 + r) * hd + i] = g[i];
  if (r != 3) return;
  __syncthreads();
  float kn[4], l[4], mx = -1e30f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float x = tail[j * hd + i];
    float mu = d4_block_sum(x, red) / hd;
    float dv = x - mu;
    float var = d4_block_sum(dv * dv, red) / hd;
    kn[j] = dv * rsqrtf(var + 1e-6f) * w[i] + b[i];
    l[j] = tail[(4 + j) * hd + i] + ape[j * hd + i];
    mx = fmaxf(mx, l[j]);
  }
  float z = 0.f, acc = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) { l[j] = expf(l[j] - mx); z += l[j]; }
#pragma unroll
  for (int j = 0; j < 4; ++j) acc += (l[j] / z) * kn[j];
  ixk[(long long)p * hd + i] = f2h(acc);
}

// The indexer's score of each complete pool (NROWS): SCALE sum_h
// relu(q_h . key) w_h, NH heads of 128.  A warp a pool, 8 a block.
extern "C" __global__ void g53_ix_scores(int nh, int nrows, float scale, const float* __restrict__ q,
                                         const float* __restrict__ w, const u16* __restrict__ ixk, float* scores) {
  int p = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (p >= nrows) return;
  float kv[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) kv[j] = h2f(ixk[(long long)p * 128 + lane * 4 + j]);
  float acc = 0.f;
  for (int h = 0; h < nh; ++h) {
    const float* qh = q + h * 128 + lane * 4;
    float d = warp_sum(qh[0] * kv[0] + qh[1] * kv[1] + qh[2] * kv[2] + qh[3] * kv[3]);
    acc += fmaxf(d, 0.f) * w[h];
  }
  if (lane == 0) scores[p] = acc * scale;
}

// The positions attended: pool IDX[s/4]'s tokens in slots 0..4K, the
// incomplete pool's NTAIL from TAIL0 after.
extern "C" __global__ void g53_expand(int k, int ntail, int tail0, const unsigned* idx, unsigned* sel) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < 4 * k) sel[s] = idx[s >> 2] * 4 + (s & 3);
  else if (s < 4 * k + ntail) sel[s] = tail0 + (s - 4 * k);
}

// Head blockIdx.x's attention over NS cached latents (the first NS, or
// SEL's): scores (QK, the query in the latent, . the row) * SCALE, softmax,
// the latents' weighted sum into OUT.  256 threads, LAT = 512; SC: NS floats
// of shared memory.
extern "C" __global__ void g53_attend(int lat, int ns, float scale, const float* __restrict__ qk,
                                      const u16* __restrict__ cache, const unsigned* __restrict__ sel,
                                      float* __restrict__ out) {
  extern __shared__ float sc[];
  __shared__ float red[32];
  int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
  const float* q = qk + (long long)h * lat;
  float qv[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) qv[j] = q[lane * 16 + j];
  for (int t = wp; t < ns; t += 8) {
    const u16* row = cache + (long long)(sel ? sel[t] : t) * lat + lane * 16;
    uint4 a = *(const uint4*)row, b = *(const uint4*)(row + 8);
    unsigned u[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    float d = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) d += qv[2 * j] * h2f(u[j] & 0xffff) + qv[2 * j + 1] * h2f(u[j] >> 16);
    d = warp_sum(d);
    if (lane == 0) sc[t] = d * scale;
  }
  __syncthreads();
  float mx = -1e30f;
  for (int t = tid; t < ns; t += 256) mx = fmaxf(mx, sc[t]);
  mx = warp_max(mx);
  if (lane == 0) red[wp] = mx;
  __syncthreads();
  mx = red[0];
  for (int j = 1; j < 8; ++j) mx = fmaxf(mx, red[j]);
  __syncthreads();
  float z = 0.f;
  for (int t = tid; t < ns; t += 256) { float e = expf(sc[t] - mx); sc[t] = e; z += e; }
  z = d4_block_sum(z, red);
  float inv = 1.f / fmaxf(z, 1e-20f);
  // a thread 2 dims, the tokens in order
  float a0 = 0.f, a1 = 0.f;
  for (int t = 0; t < ns; ++t) {
    unsigned u = *(const unsigned*)(cache + (long long)(sel ? sel[t] : t) * lat + 2 * tid);
    float pw = sc[t];
    a0 = fmaf(pw, h2f(u & 0xffff), a0);
    a1 = fmaf(pw, h2f(u >> 16), a1);
  }
  out[(long long)h * lat + 2 * tid] = a0 * inv;
  out[(long long)h * lat + 2 * tid + 1] = a1 * inv;
}

// The streams' mean (N of D) into OUT.
extern "C" __global__ void g53_mean(int d, int n, const float* s, float* out) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= d) return;
  float acc = 0.f, w = 1.f / n;
  for (int h = 0; h < n; ++h) acc += w * s[(long long)h * d + j];
  out[j] = acc;
}

// --- a prefill chunk's -------------------------------------------------------------

// Y (N rows, Y_STRIDE apart) = W X (N rows, X_STRIDE apart) for a BF16 W:
// d41_mm_f32's tiles, the weights widened (exact).  NCOLS a multiple of 16.
// grid (ceil(N/64), ceil(NROWS/64)), block 256.
extern "C" __global__ void __launch_bounds__(256)
g53_mm_bf16(int ncols, int nrows, int n, int x_stride, int y_stride, const u16* __restrict__ w,
            const float* __restrict__ x, float* __restrict__ y) {
  __shared__ float xs[16][64 + 4], ws[16][64 + 4];
  const int t0 = blockIdx.x * 64, r0 = blockIdx.y * 64, tid = threadIdx.x;
  const int tx = tid & 15, ty = tid >> 4;
  const int lr = tid >> 2, lk = (tid & 3) * 4;
  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
  for (int k0 = 0; k0 < ncols; k0 += 16) {
    float4 xv = make_float4(0.f, 0.f, 0.f, 0.f), wv = make_float4(0.f, 0.f, 0.f, 0.f);
    if (t0 + lr < n) xv = *(const float4*)(x + (long long)(t0 + lr) * x_stride + k0 + lk);
    if (r0 + lr < nrows) {
      uint2 h = __ldg((const uint2*)(w + (long long)(r0 + lr) * ncols + k0 + lk));
      wv = make_float4(g53_bf(h.x & 0xffff), g53_bf(h.x >> 16), g53_bf(h.y & 0xffff), g53_bf(h.y >> 16));
    }
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
    int t = t0 + 4 * ty + i;
    if (t >= n) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int r = r0 + 4 * tx + j;
      if (r < nrows) y[(long long)t * y_stride + r] = acc[i][j];
    }
  }
}

// G53_KDA_CONV over N tokens: a thread a channel of Q, K or V (each N x P,
// in place), its taps over the history then the chunk, in token order.
extern "C" __global__ void g53_kda_conv_chunk(int p, int n, float* q, float* k, float* v, const float* w, float* hist) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * p) return;
  int a = i / p, c = i - a * p;
  float* x = a == 0 ? q : (a == 1 ? k : v);
  float* h = hist + (long long)a * 3 * p + c;
  const float* wc = w + (long long)i * 4;
  float h0 = h[0], h1 = h[p], h2 = h[2 * p];
  for (int t = 0; t < n; ++t) {
    float raw = x[(long long)t * p + c];
    float acc = h0 * wc[0];
    acc = fmaf(h1, wc[1], acc);
    acc = fmaf(h2, wc[2], acc);
    acc = fmaf(raw, wc[3], acc);
    h0 = h1; h1 = h2; h2 = raw;
    x[(long long)t * p + c] = acc / (1.f + expf(-acc));
  }
  h[0] = h0; h[p] = h1; h[2 * p] = h2;
}

// Token blockIdx.y's head blockIdx.x (128 threads): Q and K L2-normed (Q
// / sqrt(128) too), in place; F's decay in place; BRAW's beta in place.
extern "C" __global__ void g53_kda_prep(int p, float lb, float* q, float* k, float* f, float* braw,
                                        const float* dt_bias, const float* a_log) {
  __shared__ float red[32];
  int h = blockIdx.x, i = threadIdx.x, c = h * 128 + i;
  long long o = (long long)blockIdx.y * p + c;
  float qv = q[o], kv = k[o];
  float ea = expf(a_log[h]);
  float gv = ea * (f[o] + dt_bias[c]);
  f[o] = expf(lb * (1.f / (1.f + expf(-gv))));
  float qn = d4_block_sum(qv * qv, red), kn = d4_block_sum(kv * kv, red);
  q[o] = qv * (rsqrtf(qn + 1e-6f) * 0.08838834764831845f);
  k[o] = kv * rsqrtf(kn + 1e-6f);
  if (i == 0) braw[(long long)blockIdx.y * 64 + h] = 1.f / (1.f + expf(-braw[(long long)blockIdx.y * 64 + h]));
}

// The delta rule over N tokens: head blockIdx.x, value rows 4 blockIdx.y ..
// (a warp each, its state row in registers, 4 keys a lane), G53_KDA_STEP's
// arithmetic a token at a time.  O: N x P.
extern "C" __global__ void g53_kda_rec(int p, int n, const float* q, const float* k, const float* v,
                                       const float* dec, const float* beta, float* S, float* o) {
  int h = blockIdx.x, lane = threadIdx.x & 31, row = blockIdx.y * 4 + (threadIdx.x >> 5);
  float4* sp = (float4*)(S + ((long long)h * 128 + row) * 128) + lane;
  float4 s = *sp;
  for (int t = 0; t < n; ++t) {
    long long base = (long long)t * p + h * 128;
    float4 d4 = ((const float4*)(dec + base))[lane], k4 = ((const float4*)(k + base))[lane];
    float4 q4 = ((const float4*)(q + base))[lane];
    s.x *= d4.x; s.y *= d4.y; s.z *= d4.z; s.w *= d4.w;
    float hk = warp_sum(s.x * k4.x + s.y * k4.y + s.z * k4.z + s.w * k4.w);
    float delta = (v[base + row] - hk) * beta[(long long)t * 64 + h];
    s.x = fmaf(k4.x, delta, s.x); s.y = fmaf(k4.y, delta, s.y);
    s.z = fmaf(k4.z, delta, s.z); s.w = fmaf(k4.w, delta, s.w);
    float ov = warp_sum(s.x * q4.x + s.y * q4.y + s.z * q4.z + s.w * q4.w);
    if (lane == 0) o[base + row] = ov;
  }
  *sp = s;
}

// Token blockIdx.y's head blockIdx.x of O RMS-normed (EPS) times ONORM and
// sigmoid(G), in place.  128 threads.
extern "C" __global__ void g53_kda_out(int p, float eps, float* o, const float* g, const float* onorm) {
  __shared__ float red[32];
  int i = threadIdx.x;
  long long c = (long long)blockIdx.y * p + blockIdx.x * 128 + i;
  float ov = o[c];
  float ss = d4_block_sum(ov * ov, red);
  o[c] = ov * rsqrtf(ss / 128.f + eps) * onorm[i] * (1.f / (1.f + expf(-g[c])));
}

// The pools the chunk completes (from P0, a block each): G53_IX_POOL's key
// from the raw keys and gates of positions 4p..4p+3, those before POS0 in
// TAIL, the others the chunk's (K, G: N x HD).  128 threads.
extern "C" __global__ void g53_ix_pool_chunk(int hd, int pos0, int p0, const float* k, const float* g, const float* tail,
                                             const float* w, const float* b, const float* ape, u16* ixk) {
  __shared__ float red[32];
  int p = p0 + blockIdx.x, i = threadIdx.x;
  float kn[4], l[4], mx = -1e30f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int pos = 4 * p + j;
    float x = pos < pos0 ? tail[j * hd + i] : k[(long long)(pos - pos0) * hd + i];
    float gv = pos < pos0 ? tail[(4 + j) * hd + i] : g[(long long)(pos - pos0) * hd + i];
    float mu = d4_block_sum(x, red) / hd;
    float dv = x - mu;
    float var = d4_block_sum(dv * dv, red) / hd;
    kn[j] = dv * rsqrtf(var + 1e-6f) * w[i] + b[i];
    l[j] = gv + ape[j * hd + i];
    mx = fmaxf(mx, l[j]);
  }
  float z = 0.f, acc = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) { l[j] = expf(l[j] - mx); z += l[j]; }
#pragma unroll
  for (int j = 0; j < 4; ++j) acc += (l[j] / z) * kn[j];
  ixk[(long long)p * hd + i] = f2h(acc);
}

// The incomplete pool's raw keys and gates (positions from 4 * (END/4) to
// END, the chunk's from POS0) into TAIL.  A block of HD.
extern "C" __global__ void g53_ix_tail(int hd, int pos0, int end, const float* k, const float* g, float* tail) {
  int i = threadIdx.x;
  for (int pos = end & ~3; pos < end; ++pos)
    if (pos >= pos0) {
      tail[(pos & 3) * hd + i] = k[(long long)(pos - pos0) * hd + i];
      tail[(4 + (pos & 3)) * hd + i] = g[(long long)(pos - pos0) * hd + i];
    }
}

// Token blockIdx.y's (at POS0 + y) indexer scores of the R pools: those it
// sees complete (p < (pos+1)/4) as G53_IX_SCORES, the rest -inf.  Q: N x
// NH x 128, W: N x NH.
extern "C" __global__ void g53_ix_scores_chunk(int nh, int nrows, int pos0, float scale, const float* __restrict__ q,
                                               const float* __restrict__ w, const u16* __restrict__ ixk, float* scores) {
  int p = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31, t = blockIdx.y;
  if (p >= nrows) return;
  if (p >= (pos0 + t + 1) / 4) { if (lane == 0) scores[(long long)t * nrows + p] = -INFINITY; return; }
  float kv[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) kv[j] = h2f(ixk[(long long)p * 128 + lane * 4 + j]);
  float acc = 0.f;
  const float* qt = q + (long long)t * nh * 128;
  for (int h = 0; h < nh; ++h) {
    const float* qh = qt + h * 128 + lane * 4;
    float d = warp_sum(qh[0] * kv[0] + qh[1] * kv[1] + qh[2] * kv[2] + qh[3] * kv[3]);
    acc += fmaxf(d, 0.f) * w[(long long)t * nh + h];
  }
  if (lane == 0) scores[(long long)t * nrows + p] = acc * scale;
}

// Each sparse token's (visible past DENSE) selection, row t of SEL (MAXC
// wide): the top K of its complete pools' scores, their tokens, then its
// incomplete pool's; NSEL[t] its count.  A block (1024) a token.
extern "C" __global__ void g53_select_chunk(int nrows, int k, int pos0, int dense, int maxc, const float* scores,
                                            unsigned* idx, unsigned* sel, unsigned* nsel) {
  int t = blockIdx.x, vis = pos0 + t + 1;
  if (vis <= dense) return;
  unsigned* it = idx + (long long)t * k;
  d4_topk_row(vis / 4, k, scores + (long long)t * nrows, it);
  __syncthreads();
  int ntail = vis & 3;
  unsigned* st = sel + (long long)t * maxc;
  for (int s = threadIdx.x; s < 4 * k + ntail; s += blockDim.x)
    st[s] = s < 4 * k ? it[s >> 2] * 4 + (s & 3) : (vis - ntail) + (s - 4 * k);
  if (threadIdx.x == 0) nsel[t] = 4 * k + ntail;
}

// The cache's first ROWS latents (F16) as floats.
extern "C" __global__ void g53_cache_f32(long long n, const u16* cache, float* out) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = h2f(cache[i]);
}

// Each token's key counts for D4_ATTEND: a dense token (visible <= DENSE)
// its first VISIBLE rows, a sparse one its NSEL selected.
extern "C" __global__ void g53_attend_counts(int n, int pos0, int dense, const unsigned* nsel, unsigned* raws,
                                             unsigned* nraw, unsigned* ncomp) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  int vis = pos0 + t + 1;
  raws[t] = 0;
  nraw[t] = vis <= dense ? vis : 0;
  ncomp[t] = vis <= dense ? 0 : nsel[t];
}
