;;;; packages.lisp

(defpackage #:illatio
  (:use #:cl)
  (:export
   ;; the file
   #:read-gguf #:gguf #:meta
   ;; a model, and a sequence's state on it
   #:load-model #:model #:model-name #:model-context-length
   #:make-context #:free-context #:context #:context-n-ctx #:context-pos #:context-logits
   ;; text <-> tokens
   #:tokenize #:detokenize
   ;; running it
   #:forward #:eval-tokens #:top-logprobs
   #:make-sampler #:sample #:generate #:complete
   ;; chat completions (system illatio/openai)
   #:make-engine #:engine #:engine-model #:engine-context
   #:chat-completion #:chat-completion-chunks #:render-chat
   ;; the server (system illatio/server)
   #:serve))
