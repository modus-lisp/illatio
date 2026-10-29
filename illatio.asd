;;;; illatio.asd

(defsystem "illatio"
  :description "Language-model inference in pure Common Lisp: GGUF weights,
                quantized CPU kernels, BPE tokenizer, sampler.  No FFI."
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
     (:file "unicode-data") ; letter/number/space ranges, generated from SBCL's tables
     (:file "tokenizer")   ; byte-level BPE
     (:file "model")       ; the forward pass
     (:file "batch")       ; prefill: three tokens per multiply
     (:file "qwen35")      ; Qwen3.5's hybrid layers: gated delta net, gated attention
     (:file "generate")))))  ; sampling, prompt reuse, streaming text

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
