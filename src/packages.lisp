;;;; packages.lisp

(defpackage #:illatio
  (:use #:cl)
  (:export
   ;; gguf
   #:read-gguf #:gguf #:gguf-metadata #:gguf-tensors #:gguf-path
   #:meta #:tensor-info #:tensor-info-name #:tensor-info-type #:tensor-info-dims
   ;; model
   #:load-model #:model #:model-config #:model-vocab #:model-context-length
   #:model-name
   ;; tokenizer
   #:tokenize #:detokenize #:token-piece
   ;; inference
   #:make-context #:context #:context-pos #:eval-tokens #:logits
   #:generate #:complete #:chat #:render-chat
   ;; sampling
   #:make-sampler #:sample))
