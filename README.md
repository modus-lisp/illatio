# illatio

Language-model inference in **pure Common Lisp**. GGUF weights, quantized
kernels, a byte-level BPE tokenizer, sampling, chat templates, tool calls
and an OpenAI-compatible server -- no C or Objective-C of its own and no
BLAS: it calls only the operating system (a few libc functions and the
Objective-C runtime, through SBCL's alien interface).  On Apple silicon it
drives the GPU through Metal from Lisp, and it runs
**DeepSeek V4 Flash, a 284B-parameter model, from the SSD on a 48 GB
laptop**, at the speed of the C engine it was ported from.

*Illatio*, Latin: "a carrying in; an inference."

```
bin/illatio "The capital of France is"
bin/illatio --chat
bin/illatio-server --port 9931        # then: operandi --local :9931
operandi --illatio "task"             # or no server at all: the model in operandi's image
```

It follows [ds4](https://github.com/antirez/ds4) in spirit: a small engine
built for a few chosen models, not a general GGUF runner.  Correctness comes
before speed, every path is checked against a reference implementation, and
comments sit beside the mechanics they explain.

## Models

| model | architecture | formats | checked against |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | `qwen2` | Q8_0 | llama.cpp |
| Qwen3-4B-Instruct-2507 | `qwen3` (and `llama`) | Q8_0, Q4_K_M | llama.cpp |
| a 27B Qwen3.5 | `qwen35`: gated delta net and gated attention | Q6_K | llama.cpp, CPU and Metal |
| DeepSeek V4 Flash | `deepseek4`: 43 layers, 256 experts, compressed attention | ds4's IQ2_XXS/Q2_K build (86.7 GB) | ds4's CPU reference |
| DeepSeek V4.1 Flash | `deepseek41`: 40 layers, 384 experts, shared compressed caches, Engram | IQ2_XXS/Q2_K (153 GB of experts) | its CPU reference (from ds4's) |
| GLM-5.3 Flash | `glm5-next`: 45 layers (KDA and DSA), 288 experts | antirez's Q2 build (96.5 GB) | its CPU reference (from ds4's graph); Z.AI's replies |

## DeepSeek V4 Flash, from the SSD

The model is 86.7 GB, 78 GB of it routed experts, on a machine with 48 GB.
On a 48 GB M4 Pro (Metal):

| | illatio | ds4 (its own SSD streaming) |
|---|---|---|
| prefill, a 6,352-token prompt | **105 tok/s** (61 s) | 104 tok/s (61 s) |
| decode, warm | **9.8 tok/s** | 4.2 tok/s |
| decode at 6K tokens of context | 7.1 tok/s | |
| the same system prompt again, a new question | only the question is prefilled | the same (its disk KV cache) |

It chats, reasons and calls tools through the CLI and the OpenAI endpoint,
with ds4's tokenizer and chat format (DSML tool calls, thinking).  Get the
GGUF from [ds4](https://github.com/antirez/ds4) (`download_model.sh`), then

```
bin/illatio -m DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf --chat
```

**How it streams.**  Every layer runs on the GPU; the CPU only routes, and
moves experts.

- **Decode** is one command buffer a layer: the previous layer's routed
  experts, then this layer's attention block whole (hyper-connections,
  norms, projections, RoPE, the FP8/FP4 round trips the model was trained
  through, the raw window, the compressors, the indexer's top 512,
  attention), then its FFN up to the router.  The router's logits come back
  to the CPU, which looks the six chosen experts up in the pool, reads the
  missing ones, guesses the next layer's (its router on this layer's input
  finds 82% of them) and starts reading those too.
- **Prefill** takes 4096 tokens at a time, a layer at a time, all on the
  GPU: routing included, each layer's experts one indexed product, and the
  compressors closing every window that ends in the chunk at once.  A
  layer's experts are read while the layer before it computes.
- **One expert pool** for both: slots of a GPU buffer, sized to what the
  GPU's working set leaves (22 GB here) and locked in memory.  Work in
  flight comes first: slots in use are pinned, demand reads go ahead of
  reads ahead, and a prefill passes through a layer's worth of slots it
  gives back, keeping only the experts its last tokens chose, so decode
  starts warm.  Misses are read in 1 MB pieces in parallel (about 5 GB/s);
  the pool starts with ds4's hotlist of the most used experts.
- **Prompts are prefilled once.**  A prefill saves the context's state
  (about 60 MB at 6K tokens: DeepSeek V4's compressed attention keeps
  little) at each message boundary and at its end, under
  `~/.cache/illatio/kv`; a context starting over takes the longest saved
  prefix of its prompt.  A new conversation with the same system prompt
  restores all of it.  A restored context is bit-identical to a prefilled
  one.
- **No weight is held twice.**  The GPU reads the file's own blocks (Q8_0,
  F16) from its buffers; the CPU's copies are loaded only if a CPU path
  wants them, so the process holds 7.4 GB of weights besides the pool.

Decode is now mostly expert reads (about 70 ms of a 100 ms token); the GPU
is no longer the bottleneck.  The CPU reference path, a port of ds4's op
for op, runs it too (decode 3.5 tok/s, prefill 7).


## DeepSeek V4 and V4.1 Flash on CUDA

On Linux the DeepSeek models run on NVIDIA GPUs (tested: an RTX 3090 and a
Tesla P40 on a two-socket EPYC with 1 TB).  The routed experts are read
once into pinned host memory on the GPUs' own NUMA node; the 3090 keeps a
pool of them by least recent use, the P40 a fixed tier of the most used
(computed there), and the rest are read through PCIe as they are wanted.
V4.1's Engram tables want to be in memory (`/dev/shm`, `*ds41f-engram-file*`).
The experts are kept there too, laid out for the copies (`*expert-cache-dir*`):
the first start reads them from the model (an hour from a busy disk), later
ones map and pin them where they lie (71 s to serving), one copy in memory.

| DeepSeek V4.1 Flash | |
|---|---|
| decode | ~20 tok/s (17-21: the experts it misses) |
| prefill, 245 / 1,904 / 3,800 tokens | 4.2 s / 7.2 s / 8.4 s |
| prefill, 20,181 tokens | 52 s (387 tok/s) |
| decode at 20K tokens of context | 13 tok/s |
| prefill, 129,211 tokens (`-c 131072`) | 6 min (~370 tok/s) |
| decode at 129K tokens of context | 12 tok/s |

A prompt goes through layer-major passes (`src/cuda-ds41f-prefill.lisp`):
each layer's dense weights read once for thousands of tokens, its experts
grouped into one product each, attention and the indexer on the tensor
cores.  A short pass copies in only the experts its tokens route to; a long
one copies a layer's experts while the one before computes, and is bound
by PCIe below about 3,500 tokens.  A context goes back exactly (the server
reuses the longest common prefix), and speculative decoding from the
context is on: text that repeats goes about 1.2x as fast.

V4.1 sees pictures with `--mmproj`, its vision projector as llama.cpp
ships it (unsloth's `mmproj-BF16.gguf`): requests' `image_url` parts (data:
URLs: PNG, JPEG, WebP or SVG), and `/props` tells llama.cpp's web UI it may
attach them.  The picture is decoded by [pigment](../pigment) (a JPEG turned
upright by its EXIF orientation), fitted to at most 1,024 tokens as ds4
fits it (`src/image.lisp`), and encoded by the 32-block ViT on the last
GPU (1.6 GB with its scratch; 0.35 s for a small picture, 3.8 s for the
largest, on the P40).  Its tokens stand past the vocabulary, named by the
picture's bytes, so a conversation's picture is encoded once and its
prefix reused; at them the model takes the encoder's rows, routes with its
vision bias, and skips Engram.

## GLM-5.3 Flash on CUDA

GLM-5.3 Flash runs the same way, on the same machinery: DeepSeek V4's
hyper-connections, expert pool, second-device tier and expert kernels (its
experts have V4's shape), with its own attention.  Three layers in four are
KDA, Kimi's delta attention (a 128 x 128 state a head, no cache); every
fourth is DSA, attention through a 512-wide latent cached in F16, past
2,051 tokens over only the 512 pools of 4 tokens an indexer picks
(`src/glm53.lisp`, `src/cuda-glm53.lisp`, `src/cuda/glm53.cu`).  The pools
start from illatio's own profile of its routing (`src/glm53-hotlist.lisp`):
about 81% of the experts a token wants are on a device.

| GLM-5.3 Flash (3090 + P40) | |
|---|---|
| decode | ~22 tok/s |
| prefill, 267 / 1,917 / 3,817 tokens | 3.8 s / 5.2 s / 11.3 s |
| prefill, 12,017 tokens | 38 s (315 tok/s) |
| decode at 12K tokens of context | 18 tok/s |
| a chat's next turn, at 4.4K tokens | first token in 0.8-2.7 s (a fresh prompt: 17-20 s) |

Greedy replies to ds4's GLM-5.3 quality prompts read as Z.AI's do (the
same structure, often their first lines word for word).  KDA's state goes
forward only, so a context saves it (in pinned host memory) before a
prompt's last token and every 512 tokens decoded, and goes back to the
latest save before where its history changes: a chat's next turn evaluates
only what is new since the last prompt (its reply, as rendered, and the
next message).  The MTP layer (a drafted token ahead) is not run yet.

## The other models

On an Apple M-series laptop, against llama.cpp's CPU backend on the same
file.  Qwen2.5-0.5B:

| tok/s | illatio, NEON | illatio, portable | llama.cpp CPU |
|---|---|---|---|
| decode, 1 thread | 88 | 4.6 | 136 |
| prefill, 1 thread | 178 | 9.6 | 334 |
| decode, 10 threads | 99 | 32 | 280–315 |
| prefill, 10 threads | 511 | 72 | 1,892 |

Qwen3-4B (`llama-bench -ngl 0`; its prefill still uses BLAS), and illatio in
[modus](https://github.com/modus-lisp), the Lisp it is also written for
(`#+modus` kernels on modus's own SIMD lanes):

| tok/s | illatio, SBCL | illatio, modus | llama.cpp |
|---|---|---|---|
| decode, 1 thread | 10.0 | 4.4 | 17.4 |
| prefill, 1 thread | 18.3 | 6.7 | 36.4 |
| decode, 8–10 threads | 38.7 | 13.0 | 43.2 |
| prefill, 8–10 threads | 149.5 | 16.7 | 294 |

On the GPU (Metal, `--gpu`; `llama-bench -ngl 99 -p 512 -n 128`): the
products where they lie in the file (half products summed in floats), the
KV cache in halves, a flash attention on the half simdgroup matrices
(`src/metal-dense.lisp`):

| tok/s | illatio, Metal | llama.cpp, Metal |
|---|---|---|
| Qwen3-4B Q8_0, prefill | 848 | 879 |
| Qwen3-4B Q8_0, decode | 49.8 | 53.4 |
| Qwen2.5-0.5B Q8_0, prefill | 6,724 | 7,782 |
| Qwen2.5-0.5B Q8_0, decode | 262 | 265 |
| Qwen3-4B Q4_K_M, prefill | 822 | 843 |
| Qwen3-4B Q4_K_M, decode | 77.3 | 80.6 |
| Qwen3-4B Q4_K_M, decode at 4096 tokens of context | 63.6 | 66.2 |
| Qwen2.5-0.5B Q8_0, decode at 4096 tokens of context | 217 | 243 |

All three give the CPU path's top token after a 591-token prompt and its
first 32 greedy tokens.  Q8_0, Q4_K and Q6_K matrices so far.

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
| decode, Metal GPU | 11.1 | 10.1 |
| prefill, Metal GPU | 127 | 115 |
| decode at 4096 tokens of context, Metal GPU | 10.3 | 9.8 |
| prefill at 4096 tokens of context, Metal GPU | 109 | 106 |

Loading takes 2.8 s on the CPU and 0.3 s on the GPU (the file is mapped and
read in place).  The tokenizer gives llama.cpp's ids on every test prompt,
and the chat templates render byte for byte as llama-server's
`/apply-template`, with and without tools.

## Getting a model

```
mkdir -p ~/models && curl -L -o ~/models/qwen2.5-0.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
```

That path is the default; `-m PATH` or `$ILLATIO_MODEL` picks another, e.g.
Qwen3-4B (4.3 GB), or the same in Q4_K_M (2.5 GB):

```
curl -L -o ~/models/qwen3-4b-instruct-2507-q8_0.gguf \
  https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q8_0.gguf
```

## Use

**Command line.** `bin/illatio` keeps an SBCL core image and rebuilds it
when a source file changes.  Modes are flags, and a bare word is always text:

```
bin/illatio [-m PATH] [-n N] [--temp T] [--seed S] "text"   # continue raw text, streamed
bin/illatio --chat [-s "system prompt"]                     # one line = one user turn
bin/illatio --serve [[HOST]:PORT]                           # default 127.0.0.1:9931
bin/illatio --bench [-p 512] [-n 128] [-d 0,4096] [-r 3]     # llama-bench's tests and table
```

The server is also a command of its own, as llama-server is to llama-cli
(the same core and flags, plus `--host` and `--port`; `-n` is the
`max_tokens` of a request that gives none, 2048 by default):

```
bin/illatio-server -m PATH [--host 127.0.0.1] [--port 9931] [-c 8192] [-n 2048]
```

It answers `POST /v1/chat/completions` (streamed with `"stream": true`),
`GET /v1/models` and `GET /health` (and pictures, with `--images`: below), on loopback unless told otherwise: there
is no authentication in front of it.

A web UI comes from wherever its files are: `--path DIR` serves DIR at `/`
(a missing `FILE` from `FILE.gz`, gzip-encoded).  llama.cpp's SimpleChat,
say, with the thinking left in the text it shows (its settings' BaseURL set
to the server's):

```
bin/illatio-server --gpu -m M --port 8081 \
  --path llama.cpp/tools/server/public_simplechat --reasoning-format none
```

`--gpu` runs a `qwen35` model on the GPU, and on a Mac a dense one (`qwen2`,
`qwen3`, `llama`) in Q8_0, Q4_K or Q6_K too; DeepSeek V4 Flash always uses it on a Mac,
and the DeepSeek models always use CUDA where it is.  Raw text ends with llama-simple's speed line, so a quick
comparison is

```
llama-simple      -m M -n 32 "The capital of France is"   # Metal by default
bin/illatio --gpu -m M -n 32 "The capital of France is"
```

**Server.** `POST /v1/chat/completions` (blocking, or `"stream": true` for
SSE, tool calls and reasoning included), `GET /v1/models` (with
`context_length`, which operandi reads to size its compaction budget), and
`GET /health`.  One request runs at a time.  A request that extends the
previous one (an agent's growing conversation) evaluates only the new tokens.

**Pictures too.** `--images DIR` loads [ex-nihilo](../ex-nihilo), a sibling
checkout, into the same server: Qwen-Image's weights from DIR (`--image-encoder
FILE` for another prompt encoder), answering `POST /v1/images/generations`
(OpenAI's image API) and `GET /images` (a page to try it).  illatio does not
depend on ex-nihilo; the flag loads it, and fails with a message when it
isn't there.  The two take turns on the GPUs: ex-nihilo's models wait in
pinned host memory, and a picture is made between chat requests, the chat
model moved to host memory meanwhile (its weights copied out once, its
caches each time; DeepSeek's expert pools simply given up, and seeded again
after).  With a 4B chat model on the 3090, a 1024² picture took 15-17 s
(12.5 s alone): about 2 s to bring Qwen-Image back, 1.5 s to put it away.

**Steering.** Directional steering, as ds4 does it: one direction per layer,
taken out of each layer's FFN output (and, if asked, its attention output),
`y ← y − scale · u (u·y)`.  Scale 1 removes the direction, 0 leaves the
model as it was (bit for bit), a negative scale amplifies it; past about ±2
models fall into repetition.  A file is f32 rows of the model's width, one
per layer, so ds4's `.f32` files load as they are.

```
bin/illatio --gpu -m M --steer-build dir.f32 --targets a.txt --contrasts b.txt
bin/illatio --gpu -m M --steer dir.f32 --steer-ffn -1 --chat
bin/illatio-server -m M --steer dir.f32          # a request may send "steering": {"ffn": F, "attn": F}
```

`--steer-build` writes the mean last-token FFN output of the target prompts
less the contrast prompts' (`--steer-at attn` for the attention's), each a
user turn in the model's chat template, a unit vector per layer.  It applies
to the dense models (CPU, CUDA and Metal) and DeepSeek V4, V4.1 and GLM-5.3
on CUDA; not yet to DeepSeek on Metal, nor to Qwen3.5.  A change of steering starts a cached
prompt over.  What a direction captures depends on the model: ds4 reports
a clean terse/verbose axis for DeepSeek V4 from its verbosity pairs; for
Qwen3-4B the same pairs give something less clean (at −1 one answer grew
and another shrank).

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
they become text.  The server writes them as events; operandi's `--illatio`
folds them straight into its stream state.

## How it works

- **GGUF.**  Header, metadata and tensor directory are read through a
  stream; each tensor once, straight into its own arrays (`src/gguf.lisp`).
  A file too big for memory is mapped whole as one Lisp byte vector
  (`MAP-FILE-VECTOR`: the array header sits in an anonymous page right
  before the file's first byte), so its tensors are read where they lie.
- **Quantized kernels, with the reference's bits.**  Q8_0, Q4_K and Q6_K
  weights stay quantized and multiply quantized activations as ggml does:
  integer sums exact, floats combined in the order of its kernels
  (`src/tensor.lisp`, `src/kquant.lisp`); IQ2_XXS and Q2_K likewise for
  DeepSeek's experts (`src/iquant.lisp`).  Prefill packs three tokens'
  int8 activations into one 64-bit word, so one multiply does three
  products and batched prefill stays bit-identical to decoding a token at a
  time (`src/batch.lisp`); Q6_K batches go through I8MM.
- **NEON, beside the portable code.**  On arm64 the hot loops run as NEON
  (`src/neon-sbcl.lisp`: SBCL VOPs emitted as instruction words), each
  written to give the portable kernel's bits -- never a fused multiply-add,
  the same rounding steps in the same order.  consistency-test checks them
  against each other to the bit on whole forward passes.
- **Weights outside the heap.**  `:STATIC` vectors live in mapped memory
  behind a hand-written array header, so the collector never scans or moves
  them and a 22 GB model runs in a 2 GB heap.
- **The GPU, from Lisp.**  `src/metal.lisp` drives Metal through the
  Objective-C runtime -- `objc_msgSend` with each message's own alien
  signature, no Objective-C or C of ours -- and compiles the kernels' MSL
  source (`src/metal/`) when the device opens.  Static vectors are handed to
  it as buffers where they lie (unified memory): a context's caches are
  shared by the CPU and the GPU without copies.
- **The forward passes** mirror their references op for op: llama.cpp's
  graph for the Qwen models (`src/model.lisp`, `src/qwen35.lisp`), ds4's
  CPU path for DeepSeek V4 Flash (`src/ds4f.lisp`), whose GPU paths are
  `src/metal-ds4f*.lisp`.
- **The tokenizer** is byte-level BPE with each model's pre-tokenizer split
  matched by hand, with no regex engine (`src/tokenizer.lisp`); the **chat
  templates** are written in Lisp per model instead of interpreting Jinja
  (`src/openai.lisp`).

## Testing

```
sbcl --dynamic-space-size 4096 --non-interactive --load inspect/vectors-test.lisp
sbcl --dynamic-space-size 4096 --non-interactive --load inspect/consistency-test.lisp
sbcl --non-interactive --load inspect/render-test.lisp
```

Each exits 0 if it passes and 2 if the model file is missing.  They test
Qwen2.5-0.5B; `ILLATIO_TEST_MODEL` (and, for vectors-test,
`ILLATIO_TEST_VECTORS`, e.g. `inspect/vectors/qwen3-4b-instruct-2507-q8_0.sexp`)
pick another.

- **vectors-test** runs illatio against llama.cpp's own logprobs on five
  prompts, teacher-forced.  Exact agreement can't be the bar: with
  quantized activations a 1-ulp difference eventually flips an int8
  rounding.  We measured the floor that sets -- illatio against a copy of
  itself that differs only in how it breaks rounding ties disagrees by as
  much as illatio and llama.cpp do, at every percentile -- and the bounds
  are that floor plus headroom.  A real bug lands far outside them: a wrong
  RoPE pairing is 200× over.
- **consistency-test** covers what must not differ at all: batched prefill
  against one token at a time, one thread against many, a reused context
  against a fresh one.
- **render-test** checks the chat templates against recorded llama-server
  renderings, and tool-call parsing.

DeepSeek V4 Flash has its own, against ds4 (they need the GGUF,
`$ILLATIO_DS4F`):

```
sbcl --dynamic-space-size 40000 --non-interactive --load inspect/ds4f-vectors-test.lisp
ILLATIO_DS4F_PREFILL=1 sbcl --dynamic-space-size 40000 --non-interactive --load inspect/ds4f-vectors-test.lisp
sbcl --non-interactive --load inspect/ds4f-tokenizer-test.lisp
sbcl --non-interactive --load inspect/iquant-test.lisp
```

- **ds4f-vectors-test** teacher-forces it through ds4's tokens and compares
  log-probabilities with ds4's CPU path, prompts a token at a time through
  decode or (`ILLATIO_DS4F_PREFILL`) through prefill.  This model's noise
  floor is high -- discrete steps (the top-6 expert choice, the FP8/FP4
  roundings) turn a 3e-4 change into another token now and then -- and the
  bounds are that floor with headroom.  The GPU paths pass where the CPU
  path does (71–72 of 72 argmaxes).
- **ds4f-tokenizer-test**: ds4's ids for texts that exercise its split.
- **iquant-test**: the expert kernels on real tensors against gguf-py's
  dequantizers (identical floats), NEON against portable (identical bits).

The Qwen vectors come from a CPU-only llama-server with an f32 KV cache, the
plainest ggml path:

```
llama-server -m ~/models/qwen2.5-0.5b-instruct-q8_0.gguf --port 18931 -ngl 0 -fa off -ctk f32 -ctv f32 -c 4096
inspect/collect-vectors.py http://localhost:18931 inspect/vectors/qwen2.5-0.5b-q8_0.sexp
```

## Requirements

SBCL 2.5, tested on macOS on Apple silicon (the NEON kernels need arm64
with FEAT_DotProd; without them the portable kernels run, which are plain
Common Lisp).  The Metal paths need macOS on Apple silicon; DeepSeek V4 Flash
wants a 48 GB machine and a fast SSD there.  The CUDA paths need an NVIDIA
driver (kernels compiled by NVRTC at load; the passes' products sm_80 or
newer) and host memory for the experts (78 GB for V4, 153 GB for V4.1, 92 GB for GLM-5.3).  The core system has no dependencies;
the OpenAI layer and the server use `json-simple` and `cl-transport`, from
modus-lisp.

## Next

- **DeepSeek V4 Flash's decode** is mostly expert reads now: fewer misses
  (a better guess at the next layer's experts, a pool policy that keeps
  what decode reuses) or reads overlapped with more of the GPU's work.
  Speculative decoding from the context is in (`src/ds4f-spec.lisp`,
  `*ds4f-spec-k*`), off on Metal (a guessed token's experts are read all
  the same) and on with CUDA.  On CUDA, the P40 could compute more of the
  experts it holds and read through its own link.
- **modus.**  The kernels have `#+modus` arms with the portable kernels'
  bits; what is left is modus's thread scaling.
- **More formats and models.**  Q5_K, Q8_0-with-K-quant mixes, and
  Qwen3.6-35B-A3B (`qwen35moe`).

## Afterword

This began as a Lisp that could run a half-billion-parameter model on its
CPU, and arrived at a 284B mixture of experts running from an SSD on a
laptop, as fast as the C it was ported from, with every step checked against
a reference and the GPU no longer what it waits on.  Most of the work was
not arithmetic but logistics -- what lives where, what moves when, and what
must finish before something new is loaded.

None of it would exist without two people who showed how it is done.
[Georgi Gerganov](https://github.com/ggerganov)'s ggml and llama.cpp made
local inference real for everyone, and their kernels and arithmetic are the
reference illatio measures itself against.  [Salvatore
Sanfilippo](https://github.com/antirez)'s ds4 showed that one careful
engine, built for one model and written to be read, can run DeepSeek V4
from a laptop's SSD; illatio's port follows it op for op.  Thank you both
for building something amazing.

## License

MIT (LICENSE).  Parts are ported from ggml/llama.cpp and from ds4, both
MIT; their notices are in THIRD-PARTY.md.
