// nanogpt.metal -- the speedrun's architecture (modded-nanogpt's records
// 5-11) on the GPU (nanogpt.lisp): RMSNorm without weights, QK-norm and
// rotary embeddings on each 64-wide head, ReLU^2, a softcapped head.  The
// products, the attention softmax, the residual adds and AdamW are gpt2.metal's.

#include <metal_stdlib>
using namespace metal;

// args: n, C
kernel void ng_embed_fwd(constant uint *a [[buffer(0)]], device const int *inp [[buffer(1)]],
                         device const float *wte [[buffer(2)]], device float *out [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    const uint n = a[0], C = a[1];
    if (gid >= n * C) return;
    out[gid] = wte[inp[gid / C] * C + gid % C];
}

// a thread a channel, over every row in order (the table zeroed first, or
// holding the micro-batches' sum so far)
kernel void ng_embed_bwd(constant uint *a [[buffer(0)]], device const int *inp [[buffer(1)]],
                         device float *dwte [[buffer(2)]], device const float *dout [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    const uint n = a[0], C = a[1];
    if (i >= C) return;
    for (uint bt = 0; bt < n; bt++) dwte[inp[bt] * C + i] += dout[bt * C + i];
}

// y = x / rms(x), a simdgroup a row; R keeps 1/rms.  args: n, C
kernel void ng_rms_fwd(constant uint *a [[buffer(0)]], device const float *x [[buffer(1)]],
                       device float *y [[buffer(2)]], device float *r [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]], ushort l [[thread_index_in_simdgroup]]) {
    const uint n = a[0], C = a[1];
    if (row >= n) return;
    device const float *xr = x + row * C;
    float s = 0.0f;
    for (uint i = l; i < C; i += 32) s += xr[i] * xr[i];
    const float inv = 1.0f / precise::sqrt(simd_sum(s) / C + 1e-6f);
    for (uint i = l; i < C; i += 32) y[row * C + i] = xr[i] * inv;
    if (l == 0) r[row] = inv;
}

// dx = r (dy - y mean(dy y)), y = x r: written, or added to DX if ADD.  args: n, C, add
kernel void ng_rms_bwd(constant uint *a [[buffer(0)]], device float *dx [[buffer(1)]],
                       device const float *dy [[buffer(2)]], device const float *x [[buffer(3)]],
                       device const float *r [[buffer(4)]],
                       uint row [[threadgroup_position_in_grid]], ushort l [[thread_index_in_simdgroup]]) {
    const uint n = a[0], C = a[1], add = a[2];
    if (row >= n) return;
    const float inv = r[row];
    device const float *xr = x + row * C;
    device const float *d = dy + row * C;
    float s = 0.0f;
    for (uint i = l; i < C; i += 32) s += d[i] * xr[i] * inv;
    const float mean = simd_sum(s) / C;
    for (uint i = l; i < C; i += 32) {
        float g = inv * (d[i] - xr[i] * inv * mean);
        dx[row * C + i] = add ? dx[row * C + i] + g : g;
    }
}

// Q and K of each head: RMS-normed over its 64 dims, then rotated (the
// halves' pairs (i, i+32), position t mod T, frequency base^(-i/32)).  A
// simdgroup a (row, head, Q or K): lane l holds dims l and l+32, a pair.
// QKV (n x 3C) in, QK (n x 2C: Q's then K's) and R (n x 2 x NH: 1/rms) out.
// args: n, T, C, NH, base (float)
kernel void ng_qk_fwd(constant uint *a [[buffer(0)]], device const float *qkv [[buffer(1)]],
                      device float *qk [[buffer(2)]], device float *r [[buffer(3)]],
                      uint3 g [[threadgroup_position_in_grid]], ushort l [[thread_index_in_simdgroup]]) {
    const uint n = a[0], T = a[1], C = a[2], NH = a[3];
    const float base = as_type<float>(a[4]);
    const uint row = g.x, h = g.y, which = g.z;
    if (row >= n) return;
    device const float *x = qkv + row * 3 * C + which * C + h * 64;
    const float x1 = x[l], x2 = x[l + 32];
    const float inv = 1.0f / precise::sqrt(simd_sum(x1 * x1 + x2 * x2) / 64.0f + 1e-6f);
    const float y1 = x1 * inv, y2 = x2 * inv;
    const float ang = float(row % T) * precise::pow(base, -float(l) / 32.0f);
    const float c = precise::cos(ang), s = precise::sin(ang);
    device float *o = qk + row * 2 * C + which * C + h * 64;
    o[l] = y1 * c + y2 * s;
    o[l + 32] = -y1 * s + y2 * c;
    if (l == 0) r[(row * 2 + which) * NH + h] = inv;
}

// its gradient: the rotation's transpose, then the norm's; written into
// DQKV's Q and K.  args as ng_qk_fwd
kernel void ng_qk_bwd(constant uint *a [[buffer(0)]], device float *dqkv [[buffer(1)]],
                      device const float *dqk [[buffer(2)]], device const float *qkv [[buffer(3)]],
                      device const float *r [[buffer(4)]],
                      uint3 g [[threadgroup_position_in_grid]], ushort l [[thread_index_in_simdgroup]]) {
    const uint n = a[0], T = a[1], C = a[2], NH = a[3];
    const float base = as_type<float>(a[4]);
    const uint row = g.x, h = g.y, which = g.z;
    if (row >= n) return;
    device const float *d = dqk + row * 2 * C + which * C + h * 64;
    const float ang = float(row % T) * precise::pow(base, -float(l) / 32.0f);
    const float c = precise::cos(ang), s = precise::sin(ang);
    const float d1 = d[l], d2 = d[l + 32];
    const float dy1 = d1 * c - d2 * s, dy2 = d1 * s + d2 * c;
    device const float *x = qkv + row * 3 * C + which * C + h * 64;
    const float inv = r[(row * 2 + which) * NH + h];
    const float y1 = x[l] * inv, y2 = x[l + 32] * inv;
    const float mean = simd_sum(dy1 * y1 + dy2 * y2) / 64.0f;
    device float *o = dqkv + row * 3 * C + which * C + h * 64;
    o[l] = inv * (dy1 - y1 * mean);
    o[l + 32] = inv * (dy2 - y2 * mean);
}

kernel void ng_relu2_fwd(constant uint *a [[buffer(0)]], device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= a[0]) return;
    const float v = max(x[i], 0.0f);
    y[i] = v * v;
}

kernel void ng_relu2_bwd(constant uint *a [[buffer(0)]], device float *dx [[buffer(1)]],
                         device const float *x [[buffer(2)]], device const float *dy [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= a[0]) return;
    dx[i] = 2.0f * max(x[i], 0.0f) * dy[i];
}

// The softcapped logits' softmax and each row's loss, a threadgroup of 1024
// a row: logit z becomes cap tanh(z / cap) (cap 0: z).  args: n, V, Vp,
// whether there are targets, cap (float)
static inline float ng_cap(float z, float cap) { return cap > 0.0f ? cap * precise::tanh(z / cap) : z; }

kernel void ng_softcap_ce(constant uint *a [[buffer(0)]], device const float *logits [[buffer(1)]],
                          device float *probs [[buffer(2)]], device const int *targets [[buffer(3)]],
                          device float *losses [[buffer(4)]], threadgroup float *red [[threadgroup(0)]],
                          uint r [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]], ushort l [[thread_index_in_simdgroup]]) {
    const uint V = a[1], Vp = a[2];
    const float cap = as_type<float>(a[4]);
    device const float *x = logits + r * Vp;
    device float *p = probs + r * Vp;
    float mx = -INFINITY;
    for (uint i = tid; i < V; i += 1024) mx = max(mx, ng_cap(x[i], cap));
    mx = simd_max(mx);
    if (l == 0) red[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = simd_max(red[l]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint i = tid; i < V; i += 1024) { float e = precise::exp(ng_cap(x[i], cap) - mx); p[i] = e; s += e; }
    s = simd_sum(s);
    if (l == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = simd_sum(red[l]);
    for (uint i = tid; i < V; i += 1024) p[i] = p[i] / s;
    for (uint i = V + tid; i < Vp; i += 1024) p[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0 && a[3]) losses[r] = -precise::log(p[targets[r]]);
}

// dz = (p - onehot) dl (1 - (capped/cap)^2), the padding 0.  args: n, V, Vp, dl, cap
kernel void ng_softcap_ce_bwd(constant uint *a [[buffer(0)]], device float *dlogits [[buffer(1)]],
                              device const float *probs [[buffer(2)]], device const int *targets [[buffer(3)]],
                              device const float *logits [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
    const uint V = a[1], Vp = a[2];
    const float dl = as_type<float>(a[3]), cap = as_type<float>(a[4]);
    const uint i = g.x, r = g.y;
    if (i >= Vp || r >= a[0]) return;
    const uint k = r * Vp + i;
    if (i >= V) { dlogits[k] = 0.0f; return; }
    float d = (probs[k] - (i == (uint)targets[r] ? 1.0f : 0.0f)) * dl;
    if (cap > 0.0f) { float t = precise::tanh(logits[k] / cap); d *= 1.0f - t * t; }
    dlogits[k] = d;
}

// --- Muon ---------------------------------------------------------------------

// SGD's Nesterov momentum: BUF = mu BUF + G, then X = G + mu BUF.  args: n, mu
kernel void ng_nesterov(constant uint *a [[buffer(0)]], device const float *g [[buffer(1)]],
                        device float *buf [[buffer(2)]], device float *x [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= a[0]) return;
    const float mu = as_type<float>(a[1]);
    const float b = mu * buf[i] + g[i];
    buf[i] = b;
    x[i] = g[i] + mu * b;
}

// OUT[0] = 1 / (|x| + 1e-7) from G2_SUMSQ's partial sums.  args: groups
kernel void ng_inv_norm(constant uint *a [[buffer(0)]], device const float *part [[buffer(1)]],
                        device float *out [[buffer(2)]], uint l [[thread_index_in_simdgroup]]) {
    float s = 0.0f;
    for (uint i = l; i < a[0]; i += 32) s += part[i];
    s = simd_sum(s);
    if (l == 0) out[0] = 1.0f / (precise::sqrt(s) + 1e-7f);
}

kernel void ng_scale(constant uint *a [[buffer(0)]], device float *x [[buffer(1)]],
                     device const float *s [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < a[0]) x[i] *= s[0];
}

// Y = alpha X + beta Y.  args: n, alpha, beta
kernel void ng_axpby(constant uint *a [[buffer(0)]], device const float *x [[buffer(1)]],
                     device float *y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= a[0]) return;
    y[i] = as_type<float>(a[1]) * x[i] + as_type<float>(a[2]) * y[i];
}

// Training's softcapped cross-entropy, forward and backward at once: each
// row's max and sum of exponentials found online in one pass over its
// logits, then a second writing the loss and dz = (p - onehot) dl (1 -
// (capped/cap)^2) -- no probabilities stored, no pass to read them back.
// A threadgroup of 1024 a row.  args: n, V, Vp, dl, cap
kernel void ng_softcap_ce_fused(constant uint *a [[buffer(0)]], device const float *logits [[buffer(1)]],
                                device float *dlogits [[buffer(2)]], device const int *targets [[buffer(3)]],
                                device float *losses [[buffer(4)]], threadgroup float *red [[threadgroup(0)]],
                                uint r [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                                ushort sg [[simdgroup_index_in_threadgroup]], ushort l [[thread_index_in_simdgroup]]) {
    const uint V = a[1], Vp = a[2];
    const float dl = as_type<float>(a[3]), cap = as_type<float>(a[4]);
    device const float *x = logits + r * Vp;
    device float *d = dlogits + r * Vp;
    float m = -INFINITY, s = 0.0f;
    for (uint i = tid; i < V; i += 1024) {
        const float z = ng_cap(x[i], cap);
        if (z > m) { s = s * precise::exp(m - z) + 1.0f; m = z; } else s += precise::exp(z - m);
    }
    float M = simd_max(m);
    s = simd_sum(m == -INFINITY ? 0.0f : s * precise::exp(m - M));
    if (l == 0) { red[sg] = M; red[32 + sg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gm = simd_max(red[l]);
    const float S = simd_sum(red[32 + l] * precise::exp(red[l] - gm));
    const uint y = (uint)targets[r];
    for (uint i = tid; i < Vp; i += 1024) {
        if (i >= V) { d[i] = 0.0f; continue; }
        const float z = ng_cap(x[i], cap);
        const float p = precise::exp(z - gm) / S;
        float g = (p - (i == y ? 1.0f : 0.0f)) * dl;
        if (cap > 0.0f) { const float t = z / cap; g *= 1.0f - t * t; }
        d[i] = g;
    }
    if (tid == 0) losses[r] = -(ng_cap(x[y], cap) - gm - precise::log(S));
}
