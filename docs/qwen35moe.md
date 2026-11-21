# Qwen3.5/3.6 MoE (qwen35moe) on Metal

Qwen3.6-35B-A3B and its family run on the GPU (`--gpu`). The CPU path does
not run them yet. The architecture is qwen35's hybrid stack: a gated delta
net in three layers of four, gated attention in the fourth, and the same
norms and RoPE sections. The dense FFN is replaced by a mixture of experts,
following llama.cpp's graph (src/models/qwen35moe.cpp, MIT):

- **The router.** A softmax over 256 experts, of which the 8 largest are
  kept and renormalized. That is the same as a softmax over the 8.
- **The experts.** Each is a SwiGLU of width 512: silu(gate · x) ⊙
  (up · x), then down.
- **The shared expert.** Also a SwiGLU of width 512, scaled by
  sigmoid(`ffn_gate_inp_shexp` · x), and added to the routed sum.

The code is in src/metal-qwen35.lisp (`q35-moe`) and
src/metal/qwen35moe.metal.

## Decode and prefill

A token's 8 experts are 8 *pairs*, p = 8t + k.

- **Decode, and passes under 16 tokens** (`*q35-group-min*`). An
  expert-indexed matrix-vector product per pair: q35_xmv_q4_k, _q5_k and
  _q6k. These are the dense kernels with the expert's offset and the pair's
  input row added.
- **Longer passes.** Grouped by expert, as ds4f's prefill is:
  1. q35_group orders the pairs by expert;
  2. the tokens' rows are gathered in that order;
  3. each expert's rows go through a simdgroup-matrix product (ggml's
     mul_mm_id, in 32-pair tiles: q35_mm_id32_*);
  4. q35_moe_combine_g sums each token's 8 rows back.

  An expert averages 16 pairs in a 512-token pass, so d4_mm_id's 64-pair
  tiles would be mostly empty. The 32-pair tiles were worth about 10%.

## Checks

Against llama.cpp on the same tokens (Homebrew build, unsloth's
UD-Q4_K_M):

- **Layer by layer.** llama-eval-callback's dump of "The capital of
  France is" agrees with illatio's per-layer outputs (from the steering
  capture) to the printed four places.
- **Perplexity.** llama-perplexity's protocol on illatio's README: 9153
  tokens, tokenized identically, c=512, 17 chunks.

| | illatio | llama.cpp |
|---|---|---|
| PPL | 10.437 | 10.454 ± 0.447 |

## Speed

M4 Pro, llama-bench's tests:

| t/s | illatio | llama.cpp |
|---|---|---|
| pp512 | 609 | 733 |
| tg128 | 35.8 | 49.1 |
| pp512 @ d4096 | 508 | 611 |
| tg128 @ d4096 | 33.3 | 48.0 |

How prefill got here (pp512):

| change | t/s |
|---|---|
| a pair at a time | 177 |
| grouped, as simdgroup-matrix products | 342 |
| 32-pair tiles | 375 |
| F32 matrices batched | 609 |

The F32 change: this file keeps `ssm_alpha` and `ssm_beta` in F32, and
mat-q8 ran F32 one token per dispatch. That was 512 dispatches per matrix
per layer in a prefill. It now runs 8 tokens per simdgroup (q35_mv_f32_n).
Dense qwen35 files with F32 matrices benefit too.

Decode is still about 25% behind. It runs ~15 dispatches per layer, each
small (an expert is 512 × 2048), so it is likely bound by dispatches and
barriers rather than bandwidth.

## Driving operandi

The 18-task suite (`--tools core`, in-image through illatio) on
Qwen3.6-35B-A3B scored 17/18 in 617 s. That was before the prefill work.
The miss was csv-filter-sum: the model read the 13 right values and added
them in its head, 587 for 607. Inkling 2 Mini scored 18/18 in 966 s.
