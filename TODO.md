# To do

## Models

- **GLM** (glm4moe): GLM-4.7-Flash, `GLM-4.7-Flash-UD-Q4_K_XL.gguf` on the
  Linux box (`/mnt/ollama/models/hf/glm-4.7-flash/`).
- **DeepSeek V4.1 Flash** (`deepseek41`, "ds41f"): a different architecture
  from ds4f -- 40 layers of 5120, 384 experts, engram (hashed n-gram)
  embeddings at layers 1 and 14, KV and indexer rows shared across layers
  (`kv_source_layer_ids`, `index_source_layer_ids`), candidate blocks.
  `/mnt/ollama/DeepSeek-V4.1-Flash-Q2.gguf` (60 GB).  ds4 runs it -- the
  reference to port from, as for ds4f (`ds4_deepseek41_*.h`, `ds4_engram.c`,
  and a CUDA path, `ds4_deepseek41_cuda.cuh`).

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

## CUDA, generally

- K-quant prefill (behind llama.cpp's on Qwen3-4B Q4_K_M); P40 prefill.
- A model split across the two GPUs.
