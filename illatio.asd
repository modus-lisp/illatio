;;;; illatio.asd

(defsystem "illatio"
  :description "Language-model inference in pure Common Lisp: GGUF weights,
                quantized CPU and Metal kernels, BPE tokenizer, sampler,
                DeepSeek V4 Flash streamed from the SSD."
  :version "0.0.1"
  :author "ynniv"
  :license "MIT"
  :depends-on ()
  :serial t
  :components
  ((:module "src"
    :serial t
    :components
    ((:file "packages")
     (:file "half")        ; binary16 <-> single, bit for bit
     (:file "gguf")        ; the container: metadata + tensor directory
     (:file "pool")        ; worker threads, one job at a time
     (:file "neon-sbcl" :if-feature (:and :sbcl :arm64)) ; SDOT kernels (VOPs)
     (:file "neon-modus" :if-feature :modus)             ; the same kernel on modus's %vi lanes
     (:file "tensor")      ; Q8_0 weights and the quantized matvec
     (:file "kquant")      ; Q4_K / Q6_K weights against Q8_K activations
     (:file "iquant")      ; IQ2_XXS / Q2_K (DeepSeek V4 Flash's experts), read in the file
     (:file "unicode-data") ; letter/number/space ranges, generated from SBCL's tables
     (:file "tokenizer")   ; byte-level BPE
     (:file "model")       ; the forward pass
     (:file "batch")       ; prefill: three tokens per multiply
     (:file "qwen35")      ; Qwen3.5's hybrid layers: gated delta net, gated attention
     (:file "metal" :if-feature (:and :sbcl :darwin))        ; the GPU, through the Objective-C runtime
     (:file "metal-qwen35" :if-feature (:and :sbcl :darwin)) ; qwen35 on it (kernels: src/metal/)
     (:file "cuda" :if-feature (:and :sbcl :linux))       ; NVIDIA GPUs: the driver API and NVRTC
     (:file "cuda-dense" :if-feature (:and :sbcl :linux)) ; qwen2/qwen3/llama on them (kernels: src/cuda/)
     (:file "cuda-qwen35" :if-feature (:and :sbcl :linux)) ; qwen35 on them
     (:file "generate")    ; sampling, prompt reuse, streaming text
     (:file "ds4f")        ; DeepSeek V4 Flash (deepseek4), ported from ds4.c
     (:file "ds4f-kv")      ; its state saved: a long prompt prefilled once
     (:file "ds4f-hotlist") ; its most used experts (ds4's profile)
     (:file "avx2" :if-feature (:and :sbcl :x86-64)) ; its experts' row dots as AVX2
     (:file "cuda-ds4f" :if-feature (:and :sbcl :linux)) ; its decode on an NVIDIA GPU, experts in host memory
     (:file "metal-ds4f" :if-feature (:and :sbcl :darwin)) ; it on the GPU: prefill, decode's experts
     (:file "metal-ds4f-decode" :if-feature (:and :sbcl :darwin)) ; a token's every layer on it
     (:file "metal-ds4f-prefill" :if-feature (:and :sbcl :darwin)) ; a prefill chunk's
     (:file "ds4f-spec" :if-feature (:and :sbcl :darwin)))))) ; a few tokens a step, guessed from the context

(defsystem "illatio/openai"
  :description "OpenAI chat completions computed in this image: chat templates
                rendered in Lisp, tool calls parsed out of the reply."
  :depends-on ("illatio" "json-simple")
  :components ((:module "src" :components ((:file "openai")))))

(defsystem "illatio/server"
  :description "The OpenAI-compatible HTTP server: /v1/chat/completions
                (blocking and streamed), /v1/models, /health."
  :depends-on ("illatio/openai" "cl-transport")
  :components ((:module "src" :components ((:file "server")))))

(defsystem "illatio/cli"
  :description "bin/illatio: complete, chat, or serve."
  :depends-on ("illatio/server")
  :components ((:module "bin" :components ((:file "illatio")))))
