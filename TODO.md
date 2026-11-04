# To do

## Models

- **GLM** (glm4moe): GLM-4.7-Flash, `GLM-4.7-Flash-UD-Q4_K_XL.gguf` on the
  Linux box (`/mnt/ollama/models/hf/glm-4.7-flash/`).

## GLM-5.3 Flash on CUDA

Decode ~22 tok/s, prefill ~340 tok/s at 2-4K tokens (glm53.lisp,
cuda-glm53.lisp): DS4F's pool and tier, KDA and DSA of its own.

- Where a token's ~45 ms goes (3090 + P40): the misses ~20 ms (~64
  experts read through at the link's ~21 GB/s, 22 from memory node 0 and
  20 from node 1, where ~14% of the experts landed), the dense weights ~11
  ms (7.6 GB at 600-880 GB/s), the held experts ~2.5 ms, the rest ~10 ms:
  ~1,700 launches, most of them small kernels, and ~4 ms idle while the
  CPU routes each layer (45 syncs).  Routing on the device, no syncs (a
  device-side slot table, misses read through into scratch, the LRU
  updated after), and fused small kernels would take most of that last 10.
- Prefetch (*G53-PREFETCH-MAX*): one guessed expert a layer, copied under
  the next layer's attention, ~60% used (21.3 -> 22.2 tok/s).  More a
  layer spills into the layers' own reads; a better guess is what pays.
- The pool is LRU over a profile's seed; simulated on the profile's
  traces, LRU misses 15% where the profile alone would miss 19%, decayed
  frequencies do no better, and an oracle (Belady) would miss 6%.
- The P40's x8 link (~6 GB/s) could read some misses through itself, but
  an expert takes it ~1.1 ms: only layers with four or more misses gain.

- MTP (G53-SPECULATE, *G53-MTP*, off): the MTP layer's guesses are kept
  75-90% of the time (greedy, code and prose), but a 2-token pass costs
  88 ms against decode's 48 -- its experts 61 ms against 27, the misses
  read through from host memory for each token's -- so it loses ~10%.  It
  pays once misses are cheap (the CPUs computing them).  The pass's tokens
  part from plain decoding's after 40-180 tokens (its attention and KDA
  are the chunk's kernels); its MTP steps could be one 2-row step.
- Rewinds (G53-REWIND) go back to saved KDA states (~150 MB each, pinned
  host memory): before a prompt's last token and every 512 decoded.  A
  history edited far back goes to the latest save before the edit, often
  the start; the passes could save at their ends, or the recurrence
  kernel at given steps.
- Prefill: D4_ATTEND over a float copy of the latents (exact, f32); the
  tensor cores could take it.  The passes' copies as V4.1's: the P40
  could compute its own experts.
- Its dense layers 0-2 get expert slots in the pinned host memory (6 GB
  never used).
- Special tokens in a user's text are parsed as such (as llama.cpp does).

## DS41F (DeepSeek V4.1 Flash) on CUDA

Decode runs (ds41f.lisp, cuda-ds41f.lisp): ~20 t/s on the 3090 + P40
(17-21 by the misses: ~70% of experts found on a device).  The GGUF is audreyt's
(`/mnt/ollama/huggingface/hub/blobs/38/38e0...`); the `/mnt/ollama`
V4.1 file is truncated.  Engram's tables want to be in memory
(`/dev/shm/ds41f-engram.bin`, *ds41f-engram-file*).

- Misses cost the most: ~46 experts a token (10 MB each) read through
  PCIe, ~21 ms of ~77.  The P40 could read through its own link too;
  more slots by moving the head or engram_kv off the 3090.
- Prefill (cuda-ds41f-prefill.lisp) runs layer-major, a pass as long as
  the pool has room for (~4000 tokens): 247 tokens in ~7 s, 1904 in 7.6 s,
  3800 in 8.7 s (440 t/s).  A pass costs ~7 s in copies whatever its length
  (every expert the 3090's pool lacks, ~140 GB at ~25 GB/s, the experts on the cards' own memory node), so prompts up
  to ~3500 tokens are bound by it.  The P40 could compute the experts it
  holds (15% fewer copies) and read through its own link.  Passes under
  1024 tokens take only the experts their tokens route to (64 tokens: 2.9 s
  against 7.5; 256: 4.9 against 7.6), but nothing is copied under their
  compute: the copy stream could fetch likely experts during attention.
  Past that, compute: the experts' D4_MM_ID (~3 s on 3800 tokens) and
  the Q8_0 projections (~2 s).
- Rewinds are exact (D41-REWIND: from an even position, the Engram tail
  from the tokens, a 1024-row ring so the window survives rewinds of up to
  896 tokens, else from the start).  Saved state is still V4 only.
- Speculative decoding (DS41F-CUDA-SPECULATE): a 9-token verifying pass
  costs about two decode steps with its experts held (164 ms against
  9 x 35 ms, decode steps whose experts are all held), but each token's
  expert misses still cost what they do decoding,
  so text that repeats goes ~1.2x as fast and other text ~3% slower.  The
  tokens matched plain decoding's on the prompts tried (the pass is
  decode's products, but its attention and indexer are the chunk's).
- Long prompts are compute-bound: 20181 tokens in 52 s (6 passes), most
  of it the experts' D4_MM_ID (~17 s) and the Q8_0 projections (~11 s),
  both at 20-24 TFLOPS of the 3090's ~70.
- The hotlist (ds41f-hotlist.lisp) is from ~1,500 tokens; more traces.
- ~450 launches a token, many of them BF16 passes that could fold into
  their producers.

## DS4F on CUDA

- Routed experts the GPUs lack computed on the CPUs (AVX2), from pinned
  host memory, while the GPUs compute theirs.
- Speculative decoding (cuda-ds4f-spec.lisp): a verifying pass costs 38 ms
  and 10 ms a token more (decode 31 ms); the extra tokens' cost is mostly
  their experts -- cheaper passes would let more text gain.
- Prefill (layer-major, cuda-ds4f-prefill.lisp, ~420 t/s on a 1904-token
  prompt): its copies (every layer's experts, ~60 GB a pass) and its
  kernels now overlap; next, the second device computing the experts it
  holds (a third fewer copies), the tensor cores in int8 for the experts,
  d4_attend's register spills.
- The Q8_0 products (d4_mvq8, the largest kernel) and IQ2_XXS's closer to
  the memory's speed; fewer, larger launches.

- A 3090 alone (no second device; *ds4f-cuda-tier-device* NIL): decode
  23.9 t/s (28.9 with the P40), prefill the same (4.1 s for 1904 tokens),
  repeating text 25.5 t/s speculating (18.4 not); guesses cost more there
  (each extra token's experts are mostly misses), so speculating loses on
  code and prose (-7 to -11%) -- its guesses want a stricter bar alone.

## CUDA, generally

- K-quant prefill (behind llama.cpp's on Qwen3-4B Q4_K_M); P40 prefill.
- A model split across the two GPUs.
