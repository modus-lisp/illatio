# To do

## Models

- **GLM** (glm4moe): GLM-4.7-Flash, `GLM-4.7-Flash-UD-Q4_K_XL.gguf` on the
  Linux box (`/mnt/ollama/models/hf/glm-4.7-flash/`).
- **DeepSeek V4.1 Flash** (`deepseek41`, "ds41f"): a different architecture
  from ds4f -- 40 layers of 5120, 384 experts, engram (hashed n-gram)
  embeddings at layers 1 and 14, KV and indexer rows shared across layers
  (`kv_source_layer_ids`, `index_source_layer_ids`), candidate blocks.
  `/mnt/ollama/DeepSeek-V4.1-Flash-Q2.gguf` (60 GB).  Needs a reference to
  port from (ds4f had ds4).

## DS4F on CUDA

- Routed experts the GPUs lack computed on the CPUs (AVX2), from pinned
  host memory, while the GPUs compute theirs.
- Speculative decoding (ds4f-spec.lisp, Metal only, off there) on CUDA.
- Prefill: layer-major on the GPU (it is decode a token at a time now).
- The Q8_0 products (d4_mvq8, the largest kernel) and IQ2_XXS's closer to
  the memory's speed; fewer, larger launches.

## CUDA, generally

- K-quant prefill (behind llama.cpp's on Qwen3-4B Q4_K_M); P40 prefill.
- A model split across the two GPUs.
