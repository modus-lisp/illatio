;;;; cuda-consistency-test.lisp — the CUDA path against itself on a long prompt.
;;;;
;;;; The vectors tests' prompts are a few tokens, so they say little about a
;;;; prefill's products and attention; this one runs 1000 tokens of real text
;;;; (two chunks, many key tiles) through the batched path and one token at a
;;;; time, and compares the last logits, then four tokens more decoded after
;;;; each.  The GPU sums in its own orders, so the bar is the noise between
;;;; two orderings (vectors-test's floor), not equality.
;;;;
;;;;   sbcl --dynamic-space-size 8192 --non-interactive --load inspect/cuda-consistency-test.lisp
;;;;
;;;; ILLATIO_TEST_MODEL picks the model (default Qwen3-4B Q8_0),
;;;; ILLATIO_TEST_DEVICE the device.  Exits 0 if it passes, 1 if not, 2
;;;; without the model.

(load (merge-pathnames "load.lisp" *load-truename*))
(in-package #:illatio)

(defparameter +worst+ 0.3 "The largest |dlogprob| among the top ten it accepts.")

(let ((path (or (uiop:getenv "ILLATIO_TEST_MODEL") "~/models/qwen3-4b-instruct-2507-q8_0.gguf"))
      (failures 0))
  (unless (probe-file path)
    (format t "no model at ~A~%" path)
    (uiop:quit 2))
  (let* ((m (load-model-cuda path :device (parse-integer (or (uiop:getenv "ILLATIO_TEST_DEVICE") "0"))))
         (text (uiop:read-file-string (merge-pathnames "../README.md" *load-truename*)))
         (ids (subseq (tokenize m text) 0 1000))
         (batched (make-context m :n-ctx 2048))
         (single (make-context m :n-ctx 2048)))
    (flet ((check (what)
             (let* ((a (top-logprobs (context-logits batched) 10))
                    (b (top-logprobs (context-logits single) 10))
                    (worst (loop for (id . x) in a for y = (cdr (assoc id b)) when y maximize (abs (- x y))))
                    (ok (and (= (car (first a)) (car (first b))) (< worst +worst+))))
               (format t "~&~24A argmax ~:[differs~;agrees~], worst |dlogprob| ~,4F~:[  FAIL~;~]~%"
                       what (= (car (first a)) (car (first b))) worst ok)
               (unless ok (incf failures)))))
      (eval-tokens batched ids)
      (dolist (id ids) (forward single id))
      (check "1000 tokens")
      (loop for id in (subseq (tokenize m " and then the model") 0 4) for i from 1
            do (forward batched id) (forward single id)
               (check (format nil "and ~D more" i))))
    (format t "~&~:[ok~;~:*~D failures~]~%" (and (plusp failures) failures))
    (uiop:quit (if (plusp failures) 1 0))))
