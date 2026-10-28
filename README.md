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

The first model runs, and it is checked against llama.cpp. Qwen2.5-0.5B-Instruct
in Q8_0 (the `qwen2` architecture; `llama` shares the code path), on an Apple
M-series laptop, against llama.cpp's CPU backend on the same file:

| tok/s | illatio, NEON | illatio, portable | llama.cpp CPU |
|---|---|---|---|
| decode, 1 thread | 88 | 4.6 | 136 |
| prefill, 1 thread | 178 | 9.6 | 334 |
| decode, 10 threads | 99 | 32 | 280–315 |
| prefill, 10 threads | 511 | 72 | 1,892 |

Other checks:

- **Load time.** 2.8 s.
- **Tokenizer.** llama.cpp's ids on every test prompt.
- **Chat template.** Byte-identical to llama-server's `/apply-template`, with
  and without tools.
- **Logits.** At llama.cpp's noise floor (see Testing).
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

That path is the default. `-m PATH` or `$ILLATIO_MODEL` picks another.

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
- **Q8_0 everywhere.** Weights stay int8 with per-32-block scales.
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

Each exits 0 if it passes and 2 if the model file is missing.

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
  - Prefill could use I8MM's SMMLA, which takes 2×2 tiles of rows by
    tokens, with weights repacked at load as ggml does.
  - Decode is now near what one core can pull from memory.
  - Disk KV checkpoints for a prompt prefix that never changes (ds4 does
    this) would remove most of operandi's first-turn cost.
- **modus.** Modus has integer NEON lanes (its Layer 2b, `%vi-*`) but no
  SDOT yet. The kernels are behind three names (`%q8-dot-4rows`,
  `%f32-dot16`, `%f32-axpy4`) with portable twins, so a `#+modus` arm is
  those three.
- **More formats.** K-quants (Q4_K, Q6_K) and more architectures.
- **ds4f.** DeepSeek V4 Flash, ported from ds4's CPU reference path, with
  SSD streaming of routed experts for machines with less RAM than the model.

## License

MIT
