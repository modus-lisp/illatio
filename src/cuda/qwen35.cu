// qwen35.cu — Qwen3.5's gated delta net and gated attention (cuda-dense.lisp).
//
// The pieces of a linear layer between its projections, and a full
// layer's Q/K preparation, as qwen35.lisp computes them; the projections,
// the attention and the FFN are dense.cu's and kquant.cu's.  Every kernel
// takes N tokens laid out one after another, so decode (N = 1) and a
// prefill chunk run the same code; the recurrent parts walk their tokens in
// order inside a channel's or a head's own block.

__device__ __forceinline__ float sigmoid1(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus1(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }
__device__ __forceinline__ float silu1(float x) { return x / (1.0f + expf(-x)); }

// BETA = sigmoid(BETA); ALPHA = softplus(ALPHA + DT) * A: per V head of each token.
extern "C" __global__ void q35_gates(float* __restrict__ alpha, float* __restrict__ beta,
                                     const float* __restrict__ a, const float* __restrict__ dt,
                                     int vh, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vh * n) return;
  int j = i % vh;
  beta[i] = sigmoid1(beta[i]);
  alpha[i] = softplus1(alpha[i] + dt[j]) * a[j];
}

// C = silu(the causal convolution of each channel's last 4 inputs), as
// Q35-CONV!: a thread a channel, its tokens in order; STATE (3 a channel,
// oldest first) moves on by one a token.
extern "C" __global__ void q35_conv(const float* __restrict__ qkv, const float* __restrict__ kern,
                                    float* __restrict__ state, float* __restrict__ c, int ch, int n) {
  int ci = blockIdx.x * blockDim.x + threadIdx.x;
  if (ci >= ch) return;
  float x0 = state[ci * 3], x1 = state[ci * 3 + 1], x2 = state[ci * 3 + 2];
  float w0 = kern[ci * 4], w1 = kern[ci * 4 + 1], w2 = kern[ci * 4 + 2], w3 = kern[ci * 4 + 3];
  for (int t = 0; t < n; t++) {
    float x3 = qkv[(long long)t * ch + ci];
    float sum = 0.0f;
    sum += x0 * w0; sum += x1 * w1; sum += x2 * w2; sum += x3 * w3;
    c[(long long)t * ch + ci] = silu1(sum);
    x0 = x1; x1 = x2; x2 = x3;
  }
  state[ci * 3] = x0; state[ci * 3 + 1] = x1; state[ci * 3 + 2] = x2;
}

// L2NORM! of each of C's first NHEADS heads (KD wide) per token, C's rows CH
// wide: a block a (head, token).
extern "C" __global__ void q35_l2norm(float* __restrict__ c, int kd, int nheads, int ch, float eps) {
  __shared__ double part[32];
  int u = blockIdx.x, t = u / nheads, j = u % nheads;
  float* x = c + (long long)t * ch + j * kd;
  double s = 0.0;
  for (int i = threadIdx.x; i < kd; i += blockDim.x) { float v = x[i]; s += (double)(v * v); }
  s = warp_sum_d(s);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
  __syncthreads();
  double tot = 0.0;
  for (int w = 0; w < (blockDim.x + 31) >> 5; w++) tot += part[w];
  float scale = 1.0f / fmaxf(sqrtf((float)tot), eps);
  for (int i = threadIdx.x; i < kd; i += blockDim.x) x[i] *= scale;
}

// The delta rule, as Q35-DELTA!: block (V head, 32 of its values), a thread
// a value v holding its column of the head's state S (kd x dv, stored k-major:
// S[k][v] at k*dv + v) in registers through the chunk's tokens.  KD 128.
#define DKD 128
extern "C" __global__ void __launch_bounds__(32)
q35_delta(const float* __restrict__ c, const float* __restrict__ beta, const float* __restrict__ g,
          float* __restrict__ S, float* __restrict__ o, int kh, int vh, int vd, int ch, int n, float scale) {
  __shared__ float kv[DKD], qv[DKD];
  int h = blockIdx.x, tid = threadIdx.x, v = blockIdx.y * 32 + tid;
  int kq = (h % kh) * DKD, kk = kh * DKD + (h % kh) * DKD, vv = 2 * kh * DKD + h * vd;
  float* Sh = S + (long long)h * DKD * vd + v;
  float s[DKD];
#pragma unroll
  for (int k = 0; k < DKD; k++) s[k] = Sh[k * vd];
  for (int t = 0; t < n; t++) {
    const float* ct = c + (long long)t * ch;
    __syncwarp();
    for (int i = tid; i < DKD; i += 32) { kv[i] = ct[kk + i]; qv[i] = ct[kq + i] * scale; }
    __syncwarp();
    float decay = expf(g[t * vh + h]), b = beta[t * vh + h];
    float sk = 0.0f;
#pragma unroll
    for (int k = 0; k < DKD; k++) { s[k] *= decay; sk += s[k] * kv[k]; }
    float d = (ct[vv + v] - sk) * b;
    float ov = 0.0f;
#pragma unroll
    for (int k = 0; k < DKD; k++) { s[k] += kv[k] * d; ov += s[k] * qv[k]; }
    o[((long long)t * vh + h) * vd + v] = ov;
  }
#pragma unroll
  for (int k = 0; k < DKD; k++) Sh[k * vd] = s[k];
}

// O = rmsnorm(O)*W * silu(Z) per V head (VD wide) of each token, then the
// head quantized: a block a (token, head), a warp a block of 32.
extern "C" __global__ void q35_outnorm(float* __restrict__ o, const float* __restrict__ w,
                                       const float* __restrict__ z, int vd, float eps,
                                       i8* __restrict__ xq, float* __restrict__ xs, int* __restrict__ xsum) {
  __shared__ double part[32];
  __shared__ float sc;
  long long base = (long long)blockIdx.x * vd;
  double s = 0.0;
  for (int i = threadIdx.x; i < vd; i += blockDim.x) { float v = o[base + i]; s += (double)(v * v); }
  s = warp_sum_d(s);
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  if (lane == 0) part[warp] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    double tot = 0.0;
    for (int k = 0; k < nw; k++) tot += part[k];
    sc = 1.0f / sqrtf((float)(tot / (double)vd + (double)eps));
  }
  __syncthreads();
  for (int b = warp; b < (vd >> 5); b += nw) {
    int i = b * 32 + lane;
    float v = ((o[base + i] * sc) * w[i]) * silu1(z[base + i]);
    o[base + i] = v;
    quant_block(v, xq + base + b * 32, xs + ((base + b * 32) >> 5), lane, xsum ? xsum + ((base + b * 32) >> 5) : nullptr);
  }
}

// A full layer's Q (from QG, [q | gate] per head) and K: each head RMS-normed
// and weighted, its first RD dimensions rotated (NEOX pairs), Q to Q, the
// gate to GATE, K and V into the caches at position POS0+T.  A block a
// (head, token); ROPE holds each position's cos (RD/2) then sin.
extern "C" __global__ void q35_qk(const float* __restrict__ qg, const float* __restrict__ kin,
                                  const float* __restrict__ vin, const float* __restrict__ qn,
                                  const float* __restrict__ kn, const float* __restrict__ rope,
                                  float* __restrict__ q, float* __restrict__ gate,
                                  float* __restrict__ kc, float* __restrict__ vc,
                                  int nh, int nkv, int hd, int rd, int pos0, float eps) {
  __shared__ double part[32];
  __shared__ float sc;
  __shared__ float xs[512];
  int u = blockIdx.x, t = blockIdx.y, pos = pos0 + t, tid = threadIdx.x;
  int kvd = nkv * hd, qd = nh * hd;
  bool isq = u < nh;
  int j = isq ? u : u - nh;
  const float* qgt = qg + (long long)t * 2 * qd;
  const float* src = isq ? qgt + j * 2 * hd : kin + (long long)t * kvd + j * hd;
  const float* nw = isq ? qn : kn;
  float* dst = isq ? q + (long long)t * qd + j * hd : kc + (long long)pos * kvd + j * hd;
  const float* rp = rope + (long long)pos * rd;
  double s = 0.0;
  for (int i = tid; i < hd; i += blockDim.x) { float x = src[i]; xs[i] = x; s += (double)(x * x); }
  s = warp_sum_d(s);
  if ((tid & 31) == 0) part[tid >> 5] = s;
  __syncthreads();
  if (tid == 0) {
    double tot = 0.0;
    for (int w = 0; w < (blockDim.x + 31) >> 5; w++) tot += part[w];
    sc = 1.0f / sqrtf((float)(tot / (double)hd + (double)eps));
  }
  __syncthreads();
  for (int i = tid; i < hd; i += blockDim.x) xs[i] = (xs[i] * sc) * nw[i];
  __syncthreads();
  int half = rd >> 1;
  for (int i = tid; i < hd; i += blockDim.x) {
    float r;
    if (i < half) r = xs[i] * rp[i] - xs[i + half] * rp[half + i];
    else if (i < rd) { int k = i - half; r = xs[k] * rp[half + k] + xs[i] * rp[k]; }
    else r = xs[i];
    dst[i] = r;
    if (isq) gate[(long long)t * qd + j * hd + i] = qgt[j * 2 * hd + hd + i];
    else vc[(long long)pos * kvd + j * hd + i] = vin[(long long)t * kvd + j * hd + i];
  }
}

// ATT *= sigmoid(GATE), then quantized for the output projection: a warp a block of 32.
extern "C" __global__ void q35_gate_q8(float* __restrict__ att, const float* __restrict__ gate,
                                       i8* __restrict__ xq, float* __restrict__ xs, int* __restrict__ xsum,
                                       long long nblocks) {
  long long b = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (b >= nblocks) return;
  long long i = b * 32 + lane;
  float v = att[i] * sigmoid1(gate[i]);
  att[i] = v;
  quant_block(v, xq + b * 32, xs + b, lane, xsum ? xsum + b : nullptr);
}

// OUT[t][r] = W[r]·H[t] for an F32 matrix (ssm_alpha, ssm_beta: a few rows),
// against the unquantized activations, as ggml multiplies F32 by F32: a warp
// a (row, token).
extern "C" __global__ void mm_f32(const float* __restrict__ w, const float* __restrict__ h,
                                  float* __restrict__ out, int rows, int cols, int n) {
  long long u = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (u >= (long long)rows * n) return;
  int r = u % rows, t = u / rows;
  const float4* w4 = (const float4*)(w + (long long)r * cols);
  const float4* h4 = (const float4*)(h + (long long)t * cols);
  float s = 0.0f;
  for (int i = lane; i < (cols >> 2); i += 32) {
    float4 a = __ldg(w4 + i), b = __ldg(h4 + i);
    s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  }
  s = warp_sum(s);
  if (lane == 0) out[(long long)t * rows + r] = s;
}
