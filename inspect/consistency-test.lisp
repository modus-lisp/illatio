;;;; consistency-test.lisp — illatio agrees with itself, bit for bit.
;;;
;;;   sbcl --dynamic-space-size 4096 --non-interactive --load inspect/consistency-test.lisp
;;;
;;; vectors-test.lisp compares us with llama.cpp within the noise floor.
;;; These are the paths that must not differ at all, because nothing about
;;; them is allowed to change the arithmetic:
;;;   - a batched prefill and feeding the same tokens one at a time;
;;;   - one thread and many (each output is computed whole by one thread);
;;;   - a context reused across prompts that share a prefix (the KV reuse an
;;;     agent's growing conversation relies on) and a fresh context;
;;;   - the NEON kernels and the portable ones they are written to match, on
;;;     a whole prefill and a decoded token (where the machine has NEON).

(load (merge-pathnames "load.lisp" (or *load-truename* *default-pathname-defaults*)))
(in-package #:illatio)

(defvar *failures* 0)
(defun check (ok what)
  (format t "~&  ~:[FAIL~;ok  ~] ~A~%" ok what)
  (unless ok (incf *failures*)))

(defun test-model-path ()
  (or (uiop:getenv "ILLATIO_TEST_MODEL")
      (namestring (merge-pathnames "models/qwen2.5-0.5b-instruct-q8_0.gguf" (user-homedir-pathname)))))

(unless (probe-file (test-model-path))
  (format t "no model at ~A; see README (Testing)~%" (test-model-path))
  (uiop:quit 2))

(defvar *m* (load-model (test-model-path)))
;; 200 tokens: more than two prefill chunks, and not a multiple of three
(defvar *ids* (subseq (tokenize *m* (format nil "~{~R, ~}" (loop for i from 1 to 120 collect i))) 0 200))

(defun logits-after (ids &key (model *m*) (batched t))
  (let ((ctx (make-context model)))
    (if batched
        (eval-tokens ctx ids)
        (loop for (tok . more) on ids do (forward ctx tok :logits (null more))))
    (values (copy-seq (context-logits ctx)) ctx)))

(multiple-value-bind (seq sctx) (logits-after *ids* :batched nil)
  (multiple-value-bind (bat bctx) (logits-after *ids*)
    (check (equalp seq bat) "batched prefill = one token at a time (logits)")
    (check (and (every #'equalp (context-k-cache sctx) (context-k-cache bctx))
                (every #'equalp (context-v-cache sctx) (context-v-cache bctx)))
           "batched prefill = one token at a time (KV cache)"))
  (let ((one (illatio::%make-model)))
    ;; the same weights on a one-thread pool
    (replace-model-pool one *m* 1)
    (check (equalp seq (logits-after *ids* :model one)) "1 thread = many threads"))
  (let ((ctx (make-context *m*)))
    (generate ctx (subseq *ids* 0 120) :max-tokens 5)
    (multiple-value-bind (text out reason reused) (generate ctx *ids* :max-tokens 1)
      (declare (ignore text out reason))
      ;; >= : the 5 generated tokens may themselves continue the text (it counts)
      (check (>= reused 120) (format nil "a shared prefix is reused (~D tokens, prefix 120)" reused)))
    (eval-tokens (make-context *m*) *ids*)
    (let ((fresh (logits-after *ids*)))
      (sync-prompt ctx *ids*)
      (check (equalp fresh (context-logits ctx)) "reused context = fresh context"))))

(if (neon-active-p)
    (let ((neon (logits-after *ids*))
          (portable (let ((*neon-dot* nil)) (logits-after *ids*))))
      (check (equalp neon portable) "NEON kernels = portable kernels (prefill)")
      (flet ((decoded (neon-p)
               (let ((*neon-dot* neon-p) (ctx (make-context *m*)))
                 (eval-tokens ctx *ids*)
                 (forward ctx 13)
                 (copy-seq (context-logits ctx)))))
        (check (equalp (decoded t) (decoded nil)) "NEON kernels = portable kernels (decode)")))
    (format t "~&  --   no NEON here: the portable kernels are the only ones~%"))

(format t "~&~:[ok~;~:*~D failures~]~%" (and (plusp *failures*) *failures*))
(uiop:quit (if (zerop *failures*) 0 1))
