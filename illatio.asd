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
     (:file "tensor")      ; Q8_0 weights and the quantized matvec
     (:file "tokenizer")   ; byte-level BPE
     (:file "model")))))   ; the forward pass
