# To do

## Models

- **GLM** (glm4moe): GLM-4.7-Flash, `GLM-4.7-Flash-UD-Q4_K_XL.gguf` on the
  Linux box (`/mnt/ollama/models/hf/glm-4.7-flash/`).

## DS41F (DeepSeek V4.1 Flash) on CUDA

Decode runs (ds41f.lisp, cuda-ds41f.lisp): 13.0 t/s on the 3090 + P40,
experts found on a device 81% of the time.  The GGUF is audreyt's
(`/mnt/ollama/huggingface/hub/blobs/38/38e0...`); the `/mnt/ollama`
V4.1 file is truncated.  Engram's tables want to be in memory
(`/dev/shm/ds41f-engram.bin`, *ds41f-engram-file*).

- Misses cost the most: ~46 experts a token (10 MB each) read through
  PCIe, ~21 ms of ~77.  The P40 could read through its own link too;
  more slots by moving the head or engram_kv off the 3090.
- Prefill (cuda-ds41f-prefill.lisp) runs layer-major, a pass as long as
  the pool has room for (~4000 tokens): 247 tokens in ~7 s, 1904 in 7.6 s,
  3800 in 8.7 s (440 t/s).  A pass costs ~7 s in copies whatever its length
  (every expert the 3090's pool lacks, ~140 GB at ~20 GB/s), so prompts up
  to ~3500 tokens are bound by it.  The P40 could compute the experts it
  holds (15% fewer copies) and read through its own link; a short pass
  could take only the experts its tokens route to.  Past that, compute:
  the experts' D4_MM_ID (~3 s on 3800 tokens), the Q8_0 projections (~2 s),
  the f32 router (D41_MV_F32N, ~0.65 s: a tiled product would do).
- Speculative decoding and saved state: V4.1 is kept out of both (its
  shared caches, selection and Engram tail need undo/saving).
- Candidate blocks (past 16384 compressed rows at the ratio-1 layers) are
  on the CPU only; the GPU asserts.
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
