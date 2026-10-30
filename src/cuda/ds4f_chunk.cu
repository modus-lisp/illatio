// ds4f_chunk.cu — a DS4F prefill chunk's kernels on an NVIDIA GPU, but for
// the matrix products (ds4f_mm.cu): metal/ds4f.metal's chunk kernels, as
// metal-ds4f-prefill.lisp dispatches them, one for one.  As in ds4f.cu,
// a block where Metal has a threadgroup (of as many threads), a warp a
// simdgroup, the scalars first and then the buffers, in Metal's orders; the
// same orders of summation where they matter.  Metal's 8x8 simdgroup
// matrices (D4_ATTEND, D4_IX_SCORES) become a thread a matrix element, its
// dot taken 8 terms a step as the matrices take them (within a step the
// hardware's order is its own: the sums agree to rounding, not bit for bit).
// Uses ds4f.cu's helpers (d4_block_sum, d4_rope_row, d4_round_row,
// d4_f16_round, d4_topk_row, warp_prefix_excl), concatenated ahead of this.
//
// The kernels and their launches, in metal-ds4f-prefill.lisp's terms
// (grid x [y], block, dynamic shared bytes; 0 where the shared memory is
// static):
//
//   d4_attend(int nh, int hd, float scale, int maxc, int rcap, int ident,
//             const float* q, const float* kr, const float* kc, const unsigned* raws,
//             const unsigned* nraw, const unsigned* ncomp, const unsigned* idx,
//             const float* sinks, float* out)
//     args (nh hd 1/sqrt(hd) top-k 0 (if indexed 0 1)); grid N, block 512, shared 0
//     (static, ~19 KB: Metal's 4*(8*512 + 64*8 + 8*64 + 128) is not needed).
//     Written for HD 512 and NH 64, as Metal's.
//   d4_ix_scores(int n, int ncomp, int nh, const float* q, const float* w, const float* k,
//                const unsigned* vis, float* scores)
//     args (N nix ixh); grid (ceil(nix/32), ceil(N/8)), block 128, shared 0
//     (static, ~20 KB).  128-wide heads.
//   d4_topk_chunk(int n, int k, int stride, const float* scores, unsigned* idx,
//                 const unsigned* vis)
//     Metal's chunk d4_topk (ds4f.cu's d4_topk is decode's, one token, no VIS):
//     args (nix top-k nix); grid N, block 1024, shared 0.  N unused, as Metal's.
//   d4_comp_pool_chunk(int hd, int ratio, int width, float eps, int kind, int nrot,
//                      int dst0, int pos0, int p0, int wstart, int ape_cols,
//                      const float* pkv, const float* psc, const float* ape,
//                      const float* st_kv, const float* st_sc, const float* w,
//                      const float* cs, float* rows)
//     args as Metal's; grid nb, block hd, shared 4*(hd+32).
//   d4_comp_carry(int ratio, int width, int pos0, int n, int wstart, int ape_cols,
//                 int prev_from, int set_prev, const float* pkv, const float* psc,
//                 const float* ape, float* st_kv, float* st_sc)
//     args as Metal's; grid 1, block 1024, shared 0.
//   d4_ring_gather(int hd, int rcap, int r0, int k0, int nrows, const float* ring, float* kr)
//   d4_ring_store(int hd, int rcap, int r0, int k0, int nrows, float* ring, const float* kr)
//     args as Metal's; grid ceil(nrows*hd/256), block 256, shared 0.
//   d4_gather_rows(int d, int nrows, const float* src, const unsigned* map, float* x)
//     args (d rows); grid ceil(rows*d/256), block 256.
//   d4_moe_sum_rows(int d, int nu, int ntok, const float* down, const unsigned* map,
//                   const float* shared_, float* moe)
//     args (d nu N); grid ceil(N*d/256), block 256.
//   d4_route(int ne, int nu, int hash, float scale, int has_bias, const float* logits,
//            const float* bias, const int* tid2eid, const unsigned* tokens,
//            unsigned* sel, float* wout)
//     args as Metal's; grid N, block ne (a multiple of 32, 1024 at most),
//     shared 4*(2*ne + 64 + nu).  BIAS / TID2EID may be null when unused.
//   d4_group_zero(int ne, unsigned* counts)
//     Metal's takes a dummy word; here NE, the bound.  grid ceil(ne/256), block min(ne,256).
//   d4_group_count(int n, int ne, int nu, const unsigned* sel, unsigned* counts, unsigned* rank)
//     args (rows ne nu); grid ceil(rows/256), block 256.
//   d4_group_scan(int n, int ne, int nu, const unsigned* counts, unsigned* bases)
//     args (rows ne nu); grid 1, block ne (a multiple of 32), shared 0 (static).
//   d4_group_scatter(int n, int ne, int nu, const unsigned* sel, const unsigned* rank,
//                    const unsigned* bases, const float* w, unsigned* rmap, unsigned* smap,
//                    float* wrow)
//     args (rows ne nu); grid ceil(rows/256), block 256.
//
// The rest of a chunk's ops are ds4f.cu's, which take a token dimension
// already: d4_kv_fin (grid N, block hd, shared 4*(hd+32)), d4_hc_pre_fin
// (grid (ceil(d/256), N), block 256), d4_ixq_fin (grid (ixh, N), block 128),
// d4_rms (grid rows, block min(1024, n); XQ, XS null), d4_rope (NTOK N),
// d4_hc_post (NTOK N), d4_swiglu_w (N rows; XQ, XS null).

// --- attention for a chunk --------------------------------------------------------

#define D4_HD 512                       // the head width: the host checks
#define D4_KB 8                         // keys a step
#define D4_KS (D4_HD + 1)               // a key's row in shared memory (+1 against bank conflicts)

// One token's attention (a block; 16 warps: warp s takes heads 8*(s/2).. and
// the half s%2 of the 512 dimensions, lane l its dims l, l+32, ...), over its
// keys: NRAW[t] rows of KR from RAWS[t] (a ring of RCAP rows, or a line when
// RCAP is 0), then NCOMP[t] rows of KC, the first NCOMP when IDENT, else the
// ones IDX lists (row t of IDX, MAXC wide).  Keys go D4_KB a step, gathered
// into shared memory: S (a thread a (head, key)), the running softmax from
// each head's sink (the sink in the denominator), then O = alpha O + P V.
// The key is the value (one KV head).  All in f32.
extern "C" __global__ void __launch_bounds__(512)
d4_attend(int nh, int hd, float scale, int maxc, int rcap, int ident,
          const float* __restrict__ q, const float* __restrict__ kr, const float* __restrict__ kc,
          const unsigned* __restrict__ raws, const unsigned* __restrict__ nraw,
          const unsigned* __restrict__ ncomp, const unsigned* __restrict__ idx,
          const float* __restrict__ sinks, float* __restrict__ out) {
  __shared__ float ks[D4_KB * D4_KS];   // the step's keys
  __shared__ float ps[64 * D4_KB];      // P, a head's 8 keys a row
  __shared__ float ml[128];             // 64 maxima, then 64 sums
  __shared__ float al[64];              // the step's alphas, then 1/L
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, sg = tid >> 5;
  const int hg = sg / 2, dh = sg % 2;
  const unsigned nr = nraw[t], nc = ncomp[t], total = nr + nc, r0 = raws[t];
  const float* qt = q + (long long)t * nh * D4_HD;
  if (tid < 64) { ml[tid] = sinks[tid]; ml[64 + tid] = 1.0f; }   // max = sink, sum = exp(0)
  float o[8][8];
#pragma unroll
  for (int hh = 0; hh < 8; ++hh)
#pragma unroll
    for (int c = 0; c < 8; ++c) o[hh][c] = 0.f;
  __syncthreads();
  for (unsigned kb = 0; kb < total; kb += D4_KB) {
    // gather this step's keys
    for (int i = tid; i < D4_KB * D4_HD; i += 512) {
      const unsigned j = kb + i / D4_HD, dd = i % D4_HD;
      float v = 0.f;
      if (j < nr) v = kr[(long long)(rcap ? (r0 + j) % (unsigned)rcap : r0 + j) * D4_HD + dd];
      else if (j < total) v = kc[(long long)(ident ? j - nr : idx[(long long)t * maxc + (j - nr)]) * D4_HD + dd];
      ks[(i / D4_HD) * D4_KS + dd] = v;
    }
    __syncthreads();
    // S, then the running softmax: thread (head, key) for 64 heads x 8 keys
    {
      const int h = tid / D4_KB, j = tid % D4_KB;
      const float* qh = qt + (long long)h * D4_HD;
      const float* kj = ks + j * D4_KS;
      float s = 0.f;
      for (int c = 0; c < D4_HD; c += 8) {
        float d8 = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) d8 += __ldg(qh + c + k) * kj[c + k];
        s += d8;
      }
      const unsigned F = 0xffffffffu;
      const float v = (kb + j < total) ? s * scale : -INFINITY;
      float mx = v;
      mx = fmaxf(mx, __shfl_xor_sync(F, mx, 1)); mx = fmaxf(mx, __shfl_xor_sync(F, mx, 2));
      mx = fmaxf(mx, __shfl_xor_sync(F, mx, 4));
      const float m = ml[h], mn = fmaxf(m, mx);
      const float e = expf(v - mn);
      float sum = e;
      sum += __shfl_xor_sync(F, sum, 1); sum += __shfl_xor_sync(F, sum, 2); sum += __shfl_xor_sync(F, sum, 4);
      ps[h * D4_KB + j] = e;
      __syncthreads();
      if (j == 0) {
        const float alpha = expf(m - mn);
        ml[h] = mn;
        ml[64 + h] = ml[64 + h] * alpha + sum;
        al[h] = alpha;
      }
    }
    __syncthreads();
    // O = diag(alpha) O + P V for this warp's heads and half (the scaling
    // rounded apart, as Metal's separate product)
#pragma unroll
    for (int hh = 0; hh < 8; ++hh) {
      const int h = hg * 8 + hh;
      const float a = al[h];
      const float* p = ps + h * D4_KB;
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const int d = dh * 256 + lane + 32 * c;
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < D4_KB; ++j) acc += p[j] * ks[j * D4_KS + d];
        o[hh][c] = __fmul_rn(o[hh][c], a) + acc;
      }
    }
    __syncthreads();
  }
  if (tid < 64) al[tid] = 1.0f / ml[64 + tid];
  __syncthreads();
  float* ot = out + (long long)t * nh * D4_HD;
#pragma unroll
  for (int hh = 0; hh < 8; ++hh) {
    const int h = hg * 8 + hh;
    const float inv = al[h];
#pragma unroll
    for (int c = 0; c < 8; ++c) ot[(long long)h * D4_HD + dh * 256 + lane + 32 * c] = __fmul_rn(o[hh][c], inv);
  }
}

// --- the indexer for a chunk -------------------------------------------------------

#define D4_IXS (128 + 1)                // a 128-wide row in shared memory (+1 against bank conflicts)

// ds4's kernel_dsv4_indexer_scores_tiled_f32 (MIT, antirez/ds4): SCORES[t][r]
// = sum over heads h of relu(Q[t][h] . K[r]) * W[t][h] (W scaled), for the
// rows token t sees (VIS[t]), -inf past them.  A block (4 warps) takes 8
// tokens x 32 rows; warp s columns 8s.., its lane l cells l and l+32 of the
// warp's 8x8 (row cell/8, column cell%8), as Metal's simdgroup matrix lays
// them out; the heads added in order.
extern "C" __global__ void __launch_bounds__(128)
d4_ix_scores(int n, int ncomp, int nh, const float* __restrict__ q, const float* __restrict__ w,
             const float* __restrict__ k, const unsigned* __restrict__ vis, float* __restrict__ scores) {
  const unsigned TM = 8, TN = 32, D = 128;
  __shared__ float qtg[TM * D4_IXS];
  __shared__ float ktg[TN * D4_IXS];
  const unsigned c0 = blockIdx.x * TN, t0 = blockIdx.y * TM, un = n, uc = ncomp;
  const int tid = threadIdx.x, lane = tid & 31, sg = tid >> 5;
  unsigned maxvis = 0;
  for (unsigned r = 0; r < TM; r++) if (t0 + r < un) maxvis = max(maxvis, vis[t0 + r]);
  if (c0 >= maxvis) {
    for (unsigned i = tid; i < TM * TN; i += 128) {
      const unsigned t = t0 + i / TN, c = c0 + i % TN;
      if (t < un && c < uc) scores[(long long)t * uc + c] = -INFINITY;
    }
    return;
  }
  for (unsigned i = tid; i < TN * D; i += 128) {
    const unsigned c = c0 + i / D;
    ktg[(i / D) * D4_IXS + i % D] = c < uc ? k[(long long)c * D + i % D] : 0.f;
  }
  const unsigned cell0 = lane, cell1 = lane + 32u;
  const unsigned row0 = cell0 >> 3, row1 = cell1 >> 3;
  const unsigned col0 = (unsigned)sg * 8 + (cell0 & 7u), col1 = (unsigned)sg * 8 + (cell1 & 7u);
  const unsigned tok0 = t0 + row0, tok1 = t0 + row1, cmp0 = c0 + col0, cmp1 = c0 + col1;
  const float* q0 = qtg + row0 * D4_IXS;
  const float* q1 = qtg + row1 * D4_IXS;
  const float* k0 = ktg + col0 * D4_IXS;
  const float* k1 = ktg + col1 * D4_IXS;
  float acc0 = 0.f, acc1 = 0.f;
  __syncthreads();
  for (int h = 0; h < nh; h++) {
    for (unsigned i = tid; i < TM * D; i += 128) {
      const unsigned t = t0 + i / D;
      qtg[(i / D) * D4_IXS + i % D] = t < un ? q[((long long)t * nh + h) * D + i % D] : 0.f;
    }
    __syncthreads();
    float d0 = 0.f, d1 = 0.f;
    for (unsigned db = 0; db < D; db += 8) {
      float e0 = 0.f, e1 = 0.f;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) { e0 += q0[db + kk] * k0[db + kk]; e1 += q1[db + kk] * k1[db + kk]; }
      d0 += e0; d1 += e1;
    }
    if (tok0 < un) acc0 += fmaxf(d0, 0.f) * w[(long long)tok0 * nh + h];
    if (tok1 < un) acc1 += fmaxf(d1, 0.f) * w[(long long)tok1 * nh + h];
    __syncthreads();
  }
  if (tok0 < un && cmp0 < uc) scores[(long long)tok0 * uc + cmp0] = cmp0 < vis[tok0] ? acc0 : -INFINITY;
  if (tok1 < un && cmp1 < uc) scores[(long long)tok1 * uc + cmp1] = cmp1 < vis[tok1] ? acc1 : -INFINITY;
}

// The indexer's pick for each token of the chunk (a block a token): of the
// VIS[t] scores of its row (STRIDE apart), the K highest, into row t of IDX
// (K wide); a token that sees K or fewer takes them all.  D4_TOPK's body.
extern "C" __global__ void d4_topk_chunk(int n, int k, int stride, const float* scores, unsigned* idx,
                                         const unsigned* vis) {
  const int tok = blockIdx.x;
  d4_topk_row((int)vis[tok], k, scores + (long long)tok * stride, idx + (long long)tok * k);
}

// --- the compressors over a chunk --------------------------------------------------

// Token Q's projected row (WIDTH: for ratio 4 a previous-window half then a
// current one) and score (plus APE's row Q mod RATIO), column COL: the
// chunk's own (PKV, PSC) from POS0, else the window the context carries
// (ST_*: ratio 4, rows RATIO + Q mod RATIO for the current window, Q mod
// RATIO for the one before; ratio 128, Q mod RATIO).  Q < 0: none.
__device__ __forceinline__ void d4_comp_src(int ratio, int width, unsigned pos0, unsigned wstart, int ape_cols,
                                            int q, int col, const float* pkv, const float* psc,
                                            const float* ape, const float* st_kv, const float* st_sc,
                                            float& kv, float& sc) {
  if (q < 0) { kv = 0.f; sc = -1e30f; return; }
  const unsigned uq = (unsigned)q, ur = (unsigned)ratio;
  if (uq >= pos0) {
    const long long o = (long long)(uq - pos0) * width + col;
    kv = pkv[o]; sc = psc[o] + ape[(uq % ur) * ape_cols + col];
  } else {
    const unsigned row = (ratio == 4 && uq >= wstart) ? ur + uq % ur : uq % ur;
    kv = st_kv[row * width + col]; sc = st_sc[row * width + col];
  }
}

// Every window that ends in the chunk (boundary B at P0 + B * RATIO), as
// D4_COMP_POOL pools it: compressed row DST0 + B.  A block a boundary, HD
// threads; CS its RoPE table (NROT a boundary).
extern "C" __global__ void d4_comp_pool_chunk(int hd, int ratio, int width, float eps, int kind, int nrot,
                                              int dst0, int pos0, int p0, int wstart, int ape_cols,
                                              const float* pkv, const float* psc, const float* ape,
                                              const float* st_kv, const float* st_sc, const float* w,
                                              const float* cs, float* rows) {
  extern __shared__ float sh[];
  float* v = sh;
  float* red = sh + hd;
  const int b = blockIdx.x, j = threadIdx.x, r4 = ratio == 4;
  const unsigned up0 = pos0, uws = wstart;
  const int p = (int)((unsigned)p0 + (unsigned)b * (unsigned)ratio);
  float mx = -1e30f;
  for (int r = 0; r < ratio; ++r) {
    float kv, sc;
    d4_comp_src(ratio, width, up0, uws, ape_cols, p - ratio + 1 + r, (r4 ? hd : 0) + j,
                pkv, psc, ape, st_kv, st_sc, kv, sc);
    mx = fmaxf(mx, sc);
    if (r4) {
      d4_comp_src(ratio, width, up0, uws, ape_cols, p - 2 * ratio + 1 + r, j, pkv, psc, ape, st_kv, st_sc, kv, sc);
      mx = fmaxf(mx, sc);
    }
  }
  float pooled = 0.f;
  if (mx > -5e29f) {
    float denom = 0.f, sum = 0.f;
    for (int r = 0; r < ratio; ++r) {
      float kv, sc;
      if (r4) {
        float kp, sp;
        d4_comp_src(ratio, width, up0, uws, ape_cols, p - 2 * ratio + 1 + r, j, pkv, psc, ape, st_kv, st_sc, kp, sp);
        d4_comp_src(ratio, width, up0, uws, ape_cols, p - ratio + 1 + r, hd + j, pkv, psc, ape, st_kv, st_sc, kv, sc);
        const float wp = expf(sp - mx), wc = expf(sc - mx);
        denom += wp + wc; sum += wp * kp; sum += wc * kv;
      } else {
        d4_comp_src(ratio, width, up0, uws, ape_cols, p - ratio + 1 + r, j, pkv, psc, ape, st_kv, st_sc, kv, sc);
        const float ww = expf(sc - mx);
        denom += ww; sum += ww * kv;
      }
    }
    pooled = denom > 0.f ? sum / denom : 0.f;
  }
  const float ss = d4_block_sum(pooled * pooled, red);
  const float scale = 1.f / sqrtf(ss / (float)hd + eps);
  v[j] = (pooled * scale) * w[j];
  __syncthreads();
  d4_rope_row(v, hd, nrot, cs + (long long)b * nrot, j);
  d4_round_row(v, hd, nrot, kind, j, red);
  rows[(long long)(dst0 + b) * hd + j] = d4_f16_round(v[j]);
}

// The window the context carries after the chunk (ending at E = POS0 + N - 1,
// in window [CW, CW + RATIO)), as the CPU's would hold it: the current
// window's rows for the chunk's tokens in it, and for ratio 4 (SET_PREV) the
// window from PREV_FROM as the previous half.  One block of 1024; the carried
// rows it reads are read before any write.
extern "C" __global__ void __launch_bounds__(1024)
d4_comp_carry(int ratio, int width, int pos0, int n, int wstart, int ape_cols, int prev_from, int set_prev,
              const float* pkv, const float* psc, const float* ape, float* st_kv, float* st_sc) {
  const unsigned tid = threadIdx.x, ur = ratio, uw = width, up0 = pos0, uws = wstart;
  const unsigned e = up0 + (unsigned)n - 1, cw = (e / ur) * ur, r4 = ratio == 4;
  // the previous half's new rows, gathered first (they may read rows being rewritten)
  float pk[4], ps[4];
  const unsigned np = r4 && set_prev ? ur * uw : 0;
#pragma unroll
  for (unsigned k = 0; k < 4; ++k) {
    const unsigned i = tid + k * 1024;
    if (i < np) {
      const unsigned r = i / uw, col = i % uw, q = (unsigned)prev_from + r;
      if (q >= up0) {
        const long long o = (long long)(q - up0) * uw + col;
        pk[k] = pkv[o]; ps[k] = psc[o] + ape[(q % ur) * ape_cols + col];
      } else {
        const unsigned row = q >= uws ? ur + q % ur : q % ur;
        pk[k] = st_kv[row * uw + col]; ps[k] = st_sc[row * uw + col];
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (unsigned k = 0; k < 4; ++k) {
    const unsigned i = tid + k * 1024;
    if (i < np) { st_kv[i] = pk[k]; st_sc[i] = ps[k]; }
  }
  // the current window's rows for the chunk's tokens in it
  const unsigned q0 = max(cw, up0);
  const unsigned nc = (e + 1 - q0) * uw;
  for (unsigned i = tid; i < nc; i += 1024) {
    const unsigned q = q0 + i / uw, col = i % uw;
    const unsigned row = r4 ? ur + q % ur : q % ur;
    const long long o = (long long)(q - up0) * uw + col;
    st_kv[row * uw + col] = pkv[o];
    st_sc[row * uw + col] = psc[o] + ape[(q % ur) * ape_cols + col];
  }
}

// --- the raw window ----------------------------------------------------------------

// KR[K0 + i] = RING[(R0 + i) % RCAP] for NROWS rows of HD: the window's rows,
// oldest first, ahead of the chunk's own.
extern "C" __global__ void d4_ring_gather(int hd, int rcap, int r0, int k0, int nrows,
                                          const float* ring, float* kr) {
  const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (long long)nrows * hd) return;
  const unsigned i = gid / hd, j = gid % hd;
  kr[(long long)(k0 + i) * hd + j] = ring[(long long)(((unsigned)r0 + i) % (unsigned)rcap) * hd + j];
}

// RING[(R0 + i) % RCAP] = KR[K0 + i]: the chunk's last rows into the window.
extern "C" __global__ void d4_ring_store(int hd, int rcap, int r0, int k0, int nrows,
                                         float* ring, const float* kr) {
  const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (long long)nrows * hd) return;
  const unsigned i = gid / hd, j = gid % hd;
  ring[(long long)(((unsigned)r0 + i) % (unsigned)rcap) * hd + j] = kr[(long long)(k0 + i) * hd + j];
}

// --- the routed experts' rows --------------------------------------------------------

// X[r] = SRC[MAP[r]], rows of D: the tokens an expert serves, gathered.
extern "C" __global__ void d4_gather_rows(int d, int nrows, const float* src, const unsigned* map, float* x) {
  const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (long long)nrows * d) return;
  const int r = gid / d, j = gid % d;
  x[gid] = src[(long long)map[r] * d + j];
}

// MOE[t] = its NU slots' rows of DOWN (MAP[t * NU + slot]), in slot order,
// plus SHARED[t].
extern "C" __global__ void d4_moe_sum_rows(int d, int nu, int ntok, const float* down, const unsigned* map,
                                           const float* shared_, float* moe) {
  const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (long long)ntok * d) return;
  const int t = gid / d, j = gid % d;
  float acc = 0.f;
  for (int s = 0; s < nu; ++s) acc += down[(long long)map[t * nu + s] * d + j];
  moe[gid] = acc + shared_[gid];
}

// --- a chunk's routing ---------------------------------------------------------------

// D4-ROUTE for each token (a block, a thread an expert): the hash table's
// experts (HASH), else the top NU of sqrt(softplus(logit)) + bias, the
// earlier expert first of equal ones (the CPU's insertion order, its slot
// order); weights the unbiased probabilities, renormalized, scaled.  Metal's
// precise:: functions are CUDA's expf, logf, sqrtf (none fast-math) and a
// rounded division.
extern "C" __global__ void d4_route(int ne, int nu, int hash, float scale, int has_bias,
                                    const float* logits, const float* bias, const int* tid2eid,
                                    const unsigned* tokens, unsigned* sel, float* wout) {
  extern __shared__ float sh[];
  float* probs = sh;                                // NE
  float* sv = sh + ne;                              // NE: the scores the pick runs over
  float* rv = sh + 2 * ne;                          // 32: warp maxima
  unsigned* ri = (unsigned*)(sh + 2 * ne + 32);     // 32: their experts
  unsigned* picked = (unsigned*)(sh + 2 * ne + 64); // NU
  const int t = blockIdx.x, e = threadIdx.x;
  const float l = logits[(long long)t * ne + e];
  const float sp = l > 20.f ? l : (l < -20.f ? expf(l) : logf(1.f + expf(l)));
  const float p = sqrtf(sp);
  probs[e] = p;
  sv[e] = p + (has_bias ? bias[e] : 0.f);
  __syncthreads();
  if (hash) {
    if (e < nu) picked[e] = (unsigned)tid2eid[(long long)tokens[t] * nu + e];
  } else {
    for (int k = 0; k < nu; ++k) {
      // the largest, the earlier of equal ones: reduce (value, -index)
      float v = sv[e]; unsigned idx = e;
      for (int off = 16; off > 0; off /= 2) {
        const float ov = __shfl_down_sync(0xffffffffu, v, off);
        const unsigned oi = __shfl_down_sync(0xffffffffu, idx, off);
        if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
      }
      if (e % 32 == 0) { rv[e / 32] = v; ri[e / 32] = idx; }
      __syncthreads();
      if (e == 0) {
        float bv = rv[0]; unsigned bi = ri[0];
        for (int s = 1; s < ne / 32; ++s)
          if (rv[s] > bv || (rv[s] == bv && ri[s] < bi)) { bv = rv[s]; bi = ri[s]; }
        picked[k] = bi;
        sv[bi] = -INFINITY;
      }
      __syncthreads();
    }
  }
  __syncthreads();
  if (e == 0) {
    float sum = 0.f;
    for (int i = 0; i < nu; ++i) sum += probs[picked[i]];
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (int i = 0; i < nu; ++i) {
      sel[t * nu + i] = picked[i];
      wout[t * nu + i] = __fdiv_rn(probs[picked[i]], sum) * scale;
    }
  }
}

// The chunk's N (token, slot) rows grouped by expert: each one's rank among
// its expert's (COUNT, atomically), the experts' first rows (SCAN), then each
// row's token and weight, and the way back from (token, slot) (SCATTER).
extern "C" __global__ void d4_group_zero(int ne, unsigned* counts) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < ne) counts[e] = 0u;
}

extern "C" __global__ void d4_group_count(int n, int ne, int nu, const unsigned* sel, unsigned* counts,
                                          unsigned* rank) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  rank[gid] = atomicAdd(counts + sel[gid], 1u);
}

// A block of NE threads (a multiple of 32).
extern "C" __global__ void d4_group_scan(int n, int ne, int nu, const unsigned* counts, unsigned* bases) {
  __shared__ unsigned sh[32];
  const int e = threadIdx.x;
  const unsigned c = e < ne ? counts[e] : 0u;
  const unsigned x = warp_prefix_excl(c);
  if (e % 32 == 31) sh[e / 32] = x + c;
  __syncthreads();
  unsigned before = x;
  for (int s = 0; s < e / 32; ++s) before += sh[s];
  if (e < ne) bases[e] = before;
}

extern "C" __global__ void d4_group_scatter(int n, int ne, int nu, const unsigned* sel, const unsigned* rank,
                                            const unsigned* bases, const float* w, unsigned* rmap,
                                            unsigned* smap, float* wrow) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  const unsigned row = bases[sel[gid]] + rank[gid];
  rmap[row] = gid / nu; smap[gid] = row; wrow[row] = w[gid];
}
