# Inkling 2 Mini, from its documentation

Inkling 2 Mini (Thinking Machines, 2026-11-20, Apache 2.0) is ported here
from what was published, with no reference implementation read: not
llama.cpp's (PR #32694), not Transformers', vLLM's or SGLang's.

The sources are:

- the model card;
- the blog post (thinkingmachines.ai/news/inkling-2-mini);
- the 4-bit training post;
- `config.json`;
- the checkpoint's tensor names, shapes and dtypes;
- the tokenizer files;
- the chat template;
- papers the blog cites (Attention Residuals, arXiv 2603.15031).

Where these leave a choice open, the weights decide it: the right reading
gives low perplexity on ordinary text, and a wrong one does not.

This log records, for each part, what is **documented**, what is **implied**
by shapes and config names, and what is **guessed** and then settled by
experiment.

## Documented

- **Layers.** 24 layers in four blocks of six (layers 0–5, 6–11, 12–17,
  18–23). The middle twelve (`loop_config`: core 6..18, 3 passes, gate rank
  64) run three times with shared weights, 48 layer applications in all.
- **Loop data flow.** "Every pass starts with the output of the prelude (the
  first block). The second and third passes through the middle blocks also
  read the middle blocks' outputs from the previous pass, which we call the
  message." That message is "merged with the prelude's output via a gate"
  that starts closed.
- **Attention residuals.** "each layer reads the outputs of the earlier
  blocks and learns its own mix of them" (Kimi's Attention Residuals, block
  variant). `n_layers_per_attn_res_block` 6, `attn_res_normalize_values`.
- **MoE.** 128 experts, top 8, no shared expert (`n_shared_experts` 0),
  expert width 1152, `route_scale` 8, `use_gate_bias`.
- **Attention.** Local and global layers. `local_layer_ids` lists every
  layer except 5, 11, 17 and 23. The window is 1024; 24 query and 8 KV heads
  of 128.
- **Numerics.**
  - Experts in layers 0–19 are NVFP4: FP4 E2M1 × an E4M3 scale per 16 × an
    FP32 outer scale per expert.
  - Experts in layers 20–23 are MXFP8: E4M3 × an E8M0 scale per 32.
  - Everything else is BF16.
- **Other config values.**
  - `final_logit_softcapping` 30.
  - `rms_norm_eps` 1e-6.
  - `use_value_norm_no_gamma`.
  - `qk_l2_clip_threshold` 22.627 (√512); `rel_l2_clip_threshold` 8.
  - `log_scaling_n_floor` 8192, `log_scaling_alpha` 0.1.
  - `d_rel` 16, `rel_extent` 1024.
  - `sconv_kernel_size` 4.
  - `mlp_gate_softcap` 4, `mlp_up_softcap` 25.
- **Tokenizer.** Byte-level BPE with o200k's pre-tokenizer pattern: 199998
  tokens plus 60 added, 424413 merges.

### From Goodfire's write-up (goodfire.com/research/interpreting-inkling-2-mini, appendix)

- **Layers.** Each layer is an attention sublayer then an MoE sublayer.
- **Sources.** The block sums are v_i = Σ over the block's layers of
  (o_attn + o_moe), and v0 = RMSNorm(embedding). A sublayer in block i reads
  v0, the completed v1..v(i-1), and the running sum of block i. For an
  attention sublayer that sum stops before its own layer; for an MoE
  sublayer it includes its layer's attention output.
- **Depth-wise read.** x = Σ_j α_j v̂_j, with α = softmax_j(qᵀv̂_j/√d) and
  v̂ = v/RMS(v). The sublayer gets RMSNorm(x).
- **Readers.** There are 56: 31 attention, 24 MoE and 1 output. The 31 are
  layer 0's attention having none (it can only read v0), 19 local layers
  with one, and 4 global layers with three (Q, K, V): `attn_queries.0..22`.
- **Cross-loop, for passes t ≥ 2.** Each looped layer has four readers,
  s ∈ {Q, K, V, MoE}:
  - m_s = Σ_{j∈{2,3}} β_j v̂_j^[t−1], with β = softmax(u_sᵀv̂/√d). The
    sources are the previous pass's sums of blocks 2 and 3.
  - z = P[RMSNorm(x); RMSNorm(m)], with P 64×2d (`controllers.*.down`).
  - κ = 1 − tanh(Fz) (`forget`), γ = tanh(Gz) (`gain`).
  - x' = κ⊙x + γ⊙m.
- **Output.** A depth-wise mix of v0..v4, then RMSNorm, then the
  unembedding.

## Implied by the tensors (shapes in the official checkpoint)

| tensor | shape | reading |
|---|---|---|
| `embed`, `embed_norm` | 201024×2304, 2304 | embedding, then an RMSNorm |
| `attn_res.attn_queries.N` | 2304, or 3×2304 at 5/11/17/23 | a pseudo-query per attention input; global layers get one each for Q, K and V |
| `attn_res.mlp_queries.N`, `output_query` | 2304 | one per MLP input, and one for the final output |
| `middle_cycle_cross_pass_queries`, `..._kv_queries` | 24×2304 | 12 looped layers × (attention, MLP), and × (K, V)?: mixing in the previous pass |
| `attn_norm` (local only), `q/k/v_input_norm` (global only) | 2304 | pre-attention RMSNorm(s) |
| `attn_sconv`, `mlp_sconv` | 2304×1×4 | causal depthwise convolution, width 4, on the sublayer's input |
| `attn.k_sconv`/`v_sconv.weight` (local) | 1024×1×4 | causal depthwise convolution on K and V |
| `attn.k_sconv`/`v_sconv.static_w` + `weight_proj` (global) | 4×1024, 32×2304 | the convolution's taps are static plus per token (8 KV heads × 4 taps) |
| `wq_du`, `wk_dv`, `wv_dv`, `wo_ud` | 3072×2304, 1024×2304, 1024×2304, 2304×3072 | Q, K, V, output |
| `gate_attn` | 3072×2304 | a gate on the attention output, per head dimension |
| `attn_sink_proj` | 24×2304 | a sink logit per head per token |
| `wr_du`, `rel_logits_proj.proj` | 384×2304, 16×1024 | a 16-wide relative query per head; logit bias = r·P[:, i−j] for i−j < 1024 |
| `q_l2_clip.scale`, `rel_l2_clip.scale` | 24×128, 24×16 | L2 clipping of Q (and of r) per head, with a learned scale |
| `mlp.gate.weight`, `.bias` | 128×2304, 128 | router, and a bias used in selection |
| `experts.w13`, `w2` | 128 × (2304 out = gate 1152 + up 1152) × 2304 in; 128 × 2304 × 1152 | experts |
| `controllers.{attn,k,v,mlp}.down/forget/gain` | 64×4608, 2304×64, 2304×64 | per looped layer: rank-64 gates from [x; message] |
| `norm`, `unembed` | 2304, 201024×2304 | final norm, untied head |

## Settled by perplexity

The test text is a 60-token Wikipedia-style paragraph (the Eiffel Tower),
scored with a NumPy prototype on the official checkpoint. Unless a line
says otherwise, each figure is nll per token at one pass through the core.
The first all-guesses run gave nll 13.3 (ppl 575,000), worse than uniform
over the vocabulary.

| choice | verdict | evidence |
|---|---|---|
| NVFP4 outer scale | amax / 2688 (6 × 448) | a dequantized expert's max equals its stored amax exactly; its RMS (0.0208) matches the BF16 weights |
| FP4 nibble order | low nibble is the even element | swapped: 2.03 → 5.81 |
| w13 rows | **interleaved**: gate on even rows, up on odd | halves 6.6, interleaved 2.03. This was the bug that hid all the others: experts off scored better than experts on |
| sublayer short conv (`attn_sconv`, `mlp_sconv`) | **residual, on the output**: o + conv(o) | plain conv on the output 8.5, before the sublayer 17.6, none 18.4 |
| K/V short conv | **residual**: k + conv(k), taps oldest first | none 2.03, residual 1.76, reversed taps +0.8 |
| K | normalized: RMS, or L2-clipped to 22.6 like Q (the two tie so far) | none: +0.6 at 3 passes |
| V | RMS-normalized, no gain (`use_value_norm_no_gamma`) | none: 2.03 → 2.90 |
| Q | L2-clipped to 22.6, then × `q_l2_clip.scale` | no clip: 2.57; scale before the clip: worse |
| relative bias | r_i · P[:, i−j], with column = distance; r clipped like Q | none: +2.8; reversed columns: +5.6 |
| sinks | an extra logit (from `attn_sink_proj`) with zero value | none: +3.5 |
| attention output gate | sigmoid, elementwise over the 3072 | none: +3.5 |
| router | sigmoid scores; top 8 by score + bias; the selected scores normalized to sum 1, × 8 | softmax: 2.53; unnormalized: 2.50 |
| cross-pass queries | rows 2i and 2i+1 are layer 6+i's (Q, MoE) in `cross_pass_queries` and (K, V) in `..._kv_queries` | blocked layout: 0.98 → 1.06 |
| loop | Goodfire's equations as written | 1 pass 1.76 → 3 passes 0.98 (ppl 2.67) |
| embedding norm | RMSNorm × `embed_norm` | without it: much worse |

Still open:

- **The activation's softcaps.** Plain silu(g)·u edges out the softcapped
  form by 0.04 on 60 tokens. A larger set will decide it.
- **The log scaling and the relative bias beyond 1024.** Only long contexts
  can test them.

## The Lisp port (src/inkling2.lisp)

- **The CPU reference.** It reads the official safetensors file in place
  (mapped; the BF16 weights are widened to f32 at load, and the experts
  stay NVFP4/MXFP8 in the mapping). It runs one token at a time: 48 layer
  applications, each with its own K/V cache (a 1024-slot ring when local)
  and the last three inputs of each short conv. Against the NumPy prototype
  on 20 tokens: logits within 2.7e-5, nll 1.2521 vs 1.2522.
- **The tokenizer.** tokenizer.json's BPE, o200k's split written out by
  hand (new Unicode tables for its head and tail letter classes:
  Lu/Lt/Lm/Lo and Ll/Lm/Lo), and ignore_merges. It gives the official
  tokenizer's ids on all ten test texts (capitals, contractions, CJK,
  combining marks, numbers, CRLF, specials), and every one round-trips.
- **The chat template.** It renders byte for byte as the official
  template under Hugging Face's tojson semantics (json.dumps,
  ensure_ascii=False). Checked with tools, reasoning, two tool calls, tool
  results named through their call ids, and both effort settings.
- **End to end.** chat-completion on "What is 17 times 23? Reply with just
  the number." (effort none) gives content "391", with its reasoning in
  reasoning_content: the prototype's tokens exactly. The reasoning says
  "17*20=340, 17*3=51, total 357" before answering 391. Both
  implementations agree on that, so it is the model or an error they share
  (the activation's softcap form and the log scaling remain the open
  choices).
- **Speed.** About 1.2 tok/s on the CPU (M4 Pro). It is a reference, not a
  way to run it: the GPU path is next.

## The GPU (src/metal-inkling2.lisp, src/metal/inkling2.metal)

- **The weights.** The checkpoint is wrapped as one Metal buffer, and every
  weight is read in place. Every BF16, U8 and F8 tensor in the file is
  16-byte aligned.
- **Q8_0 copies.** The BF16 projections and the output head get Q8_0
  copies at load (*INK2-Q8*): half the bytes, and the format llama.cpp's
  GGUFs keep them in. The router stays BF16. The perplexity cost is
  nothing measurable (23.86 BF16 vs 23.90).
- **Chunks.** A chunk of up to 256 tokens is one command buffer. The
  projections are Q8_0 matrix products. The experts are grouped by expert,
  each row's codes read once per 8 (token, expert) pairs. Attention keeps a
  running max and sum, so it holds no scores; a local layer's ring is its
  window plus a chunk. One token takes the matrix-vector kernels instead.
- **Agreement.** BF16 matches the CPU reference to 1e-5. Chunked prefill
  matches token-at-a-time to 4e-3 (the half-precision products).

### Against llama.cpp (black box: built from PR #32704, its source not read)

M4 Pro, llama-bench's tests. llama.cpp runs unsloth's UD-Q4_K_M (the same
NVFP4 experts, Q8_0 projections); illatio runs the official checkpoint.

| t/s | llama.cpp | illatio |
|---|---|---|
| pp512 | 73.1 | 164.8 |
| tg128 | 17.1 | 33.5 |
| pp512 @ d4096 | 68.2 | 118.9 |
| tg128 @ d4096 | 17.0 | 22.6 |

### What the comparison found: a gap in quality

Both tokenize the 9586-token test text (illatio's README) identically, and
both are scored with llama-perplexity's protocol (n_ctx chunks, the second
half of each scored).

| chunk | llama.cpp | illatio as settled above | best variant so far |
|---|---|---|---|
| 32 | 20.01 | 27.02 | 23.86 |
| 128 | 9.17 | 14.63 | 12.53 |
| 512 (first 4 chunks) | 5.93 | 9.56 | 8.20 |

So some reading is still wrong: everywhere, and more so with distance. The
short tests could not show it, because they had no oracle.

Swept against these numbers (GPU knobs, *INK2-VARIANT*):

- **Confirmed.**
  - GQA: head h reads KV head h/3.
  - Residual convs: K/V and the sublayers' outputs.
  - Sinks.
  - The sigmoid gate.
  - The sigmoid router: normalized, × 8, the bias only to choose; its input
    the normed x.
  - Interleaved w13, the MXFP8 layers included.
  - The embedding norm.
  - Goodfire's depth temperature 1/√d.
  - The activation: the softcapped form beats plain silu on long text.
  - The relative bias at scale 1.
- **Better than settled.**
  - Q scaled by q_l2_clip.scale before the clip, not after: 26.96 → 26.02.
  - With that, the logits 1.58× sharper (scale 0.14, not 1/√128):
    → 23.9. A fitted constant, not a reading. It stands in for something
    not yet found (the log scaling's form?).
  - A block's empty running sum not a source of its first attention read:
    27.02 → 26.88.
  - The global layers' convs better with only their static taps (26.34):
    the per-token part's form is wrong.
- **Not the cause.** Q8_0 vs BF16 projections.

The remaining candidates are the long-range ones: the relative bias's
reading, the log scaling, the global layers.
