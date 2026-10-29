# illatio

Language-model inference in **pure Common Lisp**: GGUF weights, quantized CPU
kernels, a byte-level BPE tokenizer, sampling, chat templates, tool calls,
and an OpenAI-compatible server. No FFI and no BLAS. The core system has no
dependencies at all.

*Illatio* is Latin for inference, the step from what is given to what follows.

```
bin/illatio "The capital of France is"
bin/illatio --chat
bin/illatio --serve :9931             # then: operandi --local :9931
operandi --illatio "task"             # or no server at all: the model in operandi's image
```

## Status

Three models run, all checked against llama.cpp: Qwen2.5-0.5B-Instruct,
Qwen3-4B-Instruct-2507 (the `qwen2` and `qwen3` architectures; `llama`
shares the code path), in Q8_0 and, for Qwen3-4B, Q4_K_M (Q4_K and Q6_K
weights), and a 27B Qwen3.5 in Q6_K (`qwen35`: gated delta net and gated
attention layers).  On an Apple M-series laptop, against
llama.cpp's CPU backend on the same file.  Qwen2.5-0.5B:

| tok/s | illatio, NEON | illatio, portable | llama.cpp CPU |
|---|---|---|---|
| decode, 1 thread | 88 | 4.6 | 136 |
| prefill, 1 thread | 178 | 9.6 | 334 |
| decode, 10 threads | 99 | 32 | 280–315 |
| prefill, 10 threads | 511 | 72 | 1,892 |

Qwen3-4B (`llama-bench -ngl 0`; its prefill still uses BLAS), and illatio in
modus, the Lisp it is also written for (`#+modus` kernels on modus's own
SIMD lanes):

| tok/s | illatio, SBCL | illatio, modus | llama.cpp |
|---|---|---|---|
| decode, 1 thread | 10.0 | 4.4 | 17.4 |
| prefill, 1 thread | 18.3 | 6.7 | 36.4 |
| decode, 8–10 threads | 38.7 | 13.0 | 43.2 |
| prefill, 8–10 threads | 149.5 | 16.7 | 294 |

Qwen3-4B in Q4_K_M (2.5 GB):

| tok/s | illatio, SBCL | illatio, modus | llama.cpp |
|---|---|---|---|
| decode, 1 thread | 10.6 | 0.8 | 9.3 |
| prefill, 1 thread | 12.1 | 1.1 | 14.7 |
| decode, 8–10 threads | 65.6 | 3.0 | 61.2 |
| prefill, 8–10 threads | 86.6 | 3.7 | 114 |

The 27B `qwen35` in Q6_K (22 GB; `llama-bench -ngl 0 -p 512`, illatio on
the same 512-token prefill, except its 1-thread one, on 96):

| tok/s | illatio, SBCL | llama.cpp |
|---|---|---|
| decode, 1 thread | 1.10 | 1.21 |
| prefill, 1 thread | 5.0 | 3.15 |
| decode, 10 threads | 9.3 | 6.9 |
| prefill, 10 threads | 35.5 | 22.8 |
| decode, Metal GPU | 11.1 | 10.2 |
| prefill, Metal GPU | 123 | 121 |

On the GPU (`load-model-gpu`; `-ngl 99` for llama.cpp) it passes the same
llama.cpp vectors at the same noise floor as the CPU.

Other checks:

- **Load time.** 2.8 s.
- **Tokenizer.** llama.cpp's ids on every test prompt.
- **Chat template.** Byte-identical to llama-server's `/apply-template`, with
  and without tools.
- **Logits.** At llama.cpp's noise floor (see Testing), for both models; on
  Qwen3-4B every one of the 213 test steps picks llama.cpp's token.
- **First turn under operandi.** `operandi --illatio` answers its first turn
  in 30 s, prefill of its 6,328-token system prompt included.

It follows [ds4](https://github.com/antirez/ds4) in spirit: a small engine
built for a few chosen models, not a general GGUF runner. Correctness comes
before speed, comments sit beside the mechanics they explain, and the
public API is narrow.

## Getting a model

```
mkdir -p ~/models && curl -L -o ~/models/qwen2.5-0.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
```

That path is the default. `-m PATH` or `$ILLATIO_MODEL` picks another, e.g.
Qwen3-4B (4.3 GB):

```
curl -L -o ~/models/qwen3-4b-instruct-2507-q8_0.gguf \
  https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q8_0.gguf
```

or the same model in Q4_K_M (2.5 GB): `.../Qwen3-4B-Instruct-2507-Q4_K_M.gguf`.

## Use

**Command line.** `bin/illatio` keeps a core image and rebuilds it when a
source file changes. Modes are flags, and a bare word is always text:

```
bin/illatio [-m PATH] [-n N] [--temp T] [--seed S] "text"   # continue raw text, streamed
bin/illatio --chat [-s "system prompt"]                     # one line = one user turn
bin/illatio --serve [[HOST]:PORT]                           # default 127.0.0.1:9931
```

**Server.** `POST /v1/chat/completions` (blocking, or `"stream": true` for
SSE, tool calls included), `GET /v1/models` (with `context_length`, which
operandi reads to size its compaction budget), and `GET /health`. One
request runs at a time. A request that shares a prefix with the previous one
(an agent's growing conversation) reuses the KV cache and evaluates only the
new tokens.

**In a Lisp image.**

```lisp
(asdf:load-system "illatio/openai")
(defvar *m* (illatio:load-model "~/models/qwen2.5-0.5b-instruct-q8_0.gguf"))
(illatio:complete *m* "The capital of France is" :max-tokens 20)

(defvar *e* (illatio:make-engine *m*))
(illatio:chat-completion *e* request)                 ; parsed JSON in, response out
(illatio:chat-completion-chunks *e* request #'print)  ; the same, as stream chunks
```

`chat-completion-chunks` yields the objects an SSE stream carries, before
they become text. The server writes them as events; operandi's `--illatio`
folds them straight into its stream state.

## How it works

- **GGUF.** Header, metadata and tensor directory are read through a stream.
  Each tensor is read from disk once, straight into its own arrays
  (`src/gguf.lisp`).
- **K-quants.** Q4_K and Q6_K weights stay 4.5 / 6.6 bits a weight in
  memory and are multiplied against Q8_K activations, as ggml does: the
  integer sums exact, the floats combined in the order of its NEON kernels
  (`src/kquant.lisp`; `%Q4K-DOT` / `%Q6K-DOT` in `src/neon-sbcl.lisp` give
  the portable dots' bits).  Their noise floor against llama.cpp is about
  2.5 times Q8_0's, measured the same way, and vectors-test's bounds for
  them follow it.
- **Q8_0.** Weights stay int8 with per-32-block scales.
  Activations are quantized to Q8_0 the way ggml's ARM path does it
  (amax/127, the scale rounded to fp16, ties-to-even with the unrounded
  1/d), so every product is an integer dot product per block
  (`src/tensor.lisp`).
- **Three tokens per multiply.** In prefill, three tokens' int8 activations
  are packed into one 64-bit word at 21-bit spacing. One multiply by a weight
  does three products, and a 32-block's sums still separate exactly. The
  integers and the float accumulation order are the same as the scalar
  path's, so batched prefill is bit-identical to feeding tokens one at a
  time (`src/batch.lisp`).
- **Q6_K batches through I8MM.** A batch rebuilds each pair of Q6_K rows
  once, as int8 q-32 laid out for SMMLA (2x8 by 8x2 int8 into 2x2 int32),
  and runs it against the batch's tokens four at a time; the per-(row,
  token) floats are the one-row kernel's, so prefill stays bit-identical to
  decoding (`%Q6K-UNPACK-2`, `%Q6K-GEMM-4X4`).
- **qwen35's recurrence** goes a token at a time inside each channel's
  convolution and each head's delta-rule state, with everything around it
  batched (`src/qwen35.lisp`).  A head's state is stored k-major, so sixteen
  values go a NEON lane each (`%DELTA-BLOCK`), every one of them through
  ggml's operations in ggml's order.
- **Weights outside the heap.** `:STATIC` vectors live in mapped memory
  behind a hand-written array header (on SBCL; modus has its own static
  vectors), so the collector never scans or moves them and a 22 GB model
  runs in a 2 GB heap.  A context's caches are static too, freed by
  `FREE-CONTEXT` or when the context is collected.
- **The GPU, from Lisp.** `src/metal.lisp` drives Metal through the
  Objective-C runtime -- `objc_msgSend` called with each message's own
  alien signature, no Objective-C or C of ours -- and compiles the kernels'
  MSL source (`src/metal/`) when the device opens.  The GGUF is mapped and
  handed to Metal as one buffer, so the GPU reads the weights in their own
  blocks where they lie: no repacked copy, and a 22 GB model loads in
  0.3 s.  A pass (a token, or up to 512 of a prompt) is one command buffer.
  The Q6_K kernels follow ggml's simdgroup-matrix ones, the prefill's
  product with 64-token tiles where ggml's are 32 (each weight rebuilt half
  as often): about 7.3 TFLOPS, near this GPU's peak.  The rest are written
  from the Lisp ops, with the
  delta rule's state k-major as on the CPU and each thread's column of it
  kept in registers across a chunk.
- **The forward pass** mirrors llama.cpp's graph op for op: RMSNorm with its
  sum in double, NEOX RoPE with theta advanced by repeated multiplication as
  ggml does, GQA attention with ggml's softmax, SwiGLU (`src/model.lisp`).
- **NEON, as an option beside the portable code.** On arm64 with
  FEAT_DotProd, the three hot loops run as NEON (`src/neon-sbcl.lisp`, SBCL
  VOPs emitted as instruction words). They are the Q8_0 matrix product, four
  rows per pass with SDOT, and attention's dot product and update. Each is
  written to give the portable kernel's bits: never a fused multiply-add,
  the same rounding steps in the same order. A reduction's order is defined
  by the portable code (sixteen strided partial sums and a fixed tree), so
  the narrow and wide versions are one kernel at two widths.
  consistency-test checks them against each other to the bit on whole
  forward passes. Without SDOT (or not on SBCL/arm64), the portable kernels
  run.
- **The tokenizer** is GPT-2 byte-level BPE. The qwen2/llama3
  pre-tokenizer pattern is matched by hand, as llama.cpp's own custom
  splitter does, with no regex engine (`src/tokenizer.lisp`).
- **Chat templates** are written in Lisp per model instead of interpreting
  Jinja, and tested byte for byte against llama.cpp (`src/openai.lisp`).

## Testing

```
sbcl --dynamic-space-size 4096 --non-interactive --load inspect/vectors-test.lisp
sbcl --dynamic-space-size 4096 --non-interactive --load inspect/consistency-test.lisp
sbcl --non-interactive --load inspect/render-test.lisp
```

Each exits 0 if it passes and 2 if the model file is missing.  They test
Qwen2.5-0.5B; for another model set `ILLATIO_TEST_MODEL` (and, for
vectors-test, `ILLATIO_TEST_VECTORS`, e.g.
`inspect/vectors/qwen3-4b-instruct-2507-q8_0.sexp`).  render-test also
checks Qwen3's template when that model is in `~/models`.

- **vectors-test** runs illatio against llama.cpp's own logprobs on five
  prompts, teacher-forced. Exact token-for-token agreement can't be the bar.
  With Q8_0 activations, a 1-ulp difference (NEON fuses multiply-adds and
  sums lanes in its own order) eventually flips some int8 rounding. We
  measured the floor that sets: illatio against a copy of itself that differs
  only in how it breaks rounding ties disagrees by as much as illatio and
  llama.cpp do, at every percentile (top-1 logprob: median 0.006, p95 0.07).
  The bounds are that floor plus headroom, and a real bug lands far outside
  them: a wrong RoPE pairing is 200× over.
- **consistency-test** covers what must not differ at all: batched prefill
  vs one token at a time, one thread vs many, and a reused context vs a fresh
  one.
- **render-test** checks the chat template against recorded llama-server
  renderings, and tool-call parsing.

The vectors come from a CPU-only llama-server with an f32 KV cache, which is
the plainest ggml path. To regenerate them:

```
llama-server -m ~/models/qwen2.5-0.5b-instruct-q8_0.gguf --port 18931 -ngl 0 -fa off -ctk f32 -ctv f32 -c 4096
inspect/collect-vectors.py http://localhost:18931 inspect/vectors/qwen2.5-0.5b-q8_0.sexp
```

## Next

- **Speed.**
  - A GPU prefill is ~90% its Q6_K product, which runs near the GPU's
    peak; attention, written plainly, is what grows with prompt length.
  - Q4_K prefill through the same rebuild-then-SMMLA path as Q6_K (the
    4B Q4_K_M's prefill is still below llama.cpp's), and Q8_0's.
  - Decode is now near what one core can pull from memory.
  - Disk KV checkpoints for a prompt prefix that never changes (ds4 does
    this) would remove most of operandi's first-turn cost.
- **modus.** The three kernels have `#+modus` arms (`src/neon-modus.lisp`)
  with the portable kernels' bits.  What is left is modus's thread scaling
  (8 threads give 3x) and the serial per-layer ops between the products.
- **More formats.** Q5_K (Q5_K_M files) and Q8_0-with-K-quant mixes.
- **Qwen3.6-35B-A3B** (`qwen35moe`): expert routing on top of `qwen35`.
- **ds4f.** DeepSeek V4 Flash, ported from ds4's CPU reference path, with
  SSD streaming of routed experts for machines with less RAM than the model.

## License

MIT
