;;;; vectors-test.lisp — illatio against llama.cpp's own numbers.
;;;
;;;   sbcl --dynamic-space-size 4096 --non-interactive --load inspect/vectors-test.lisp
;;;
;;; The vectors (inspect/vectors/*.sexp, from collect-vectors.py) hold, per
;;; prompt, llama.cpp's token ids and, per greedy step, its top-10 logprobs.
;;; We teacher-force: each step is fed llama.cpp's token, not ours, so one
;;; near-tie doesn't send the two engines down different texts and turn every
;;; later comparison into noise.
;;;
;;; Exact token-for-token agreement is not the bar, and can't be: with Q8_0
;;; activations a 1-ulp difference (NEON fuses multiply-adds and sums lanes in
;;; its own order) eventually flips some int8 rounding.  We measured the floor
;;; that sets: illatio against a copy of itself that differs only in how it
;;; breaks rounding ties disagrees by as much as illatio and llama.cpp do, at
;;; every percentile (top-1 logprob: median 0.006, p95 0.07; ranks 2-5:
;;; median 0.05, p95 0.22).  So the bounds below are that floor with headroom,
;;; and a real bug -- a wrong rope, norm or mask -- lands far outside them:
;;;   - every one of llama.cpp's top 5 is in our top 10, at every step --
;;;     unless it is so far down the tail that the noise could drop it below
;;;     our tenth: its logprob within the p95 ranks-2-5 bound of our tenth's
;;;     (a 27B model's alternatives sit at -9 or below, and one 0.3 nats away
;;;     from our tenth is noise, not a missing token);
;;;   - |dlogprob| of llama.cpp's top-1: median < 0.02, p95 < 0.15;
;;;   - |dlogprob| of its ranks 2-5:     median < 0.10, p95 < 0.40;
;;;   - our argmax is llama.cpp's, except where llama.cpp's own top two are
;;;     within +NEAR-TIE+ (either is then a faithful answer).
;;; K-quant weights (Q4_K_M) set a floor about 2.5x higher, measured the same
;;; way on Qwen3-4B: illatio against its Q8_K tie variant gives top-1 median
;;; 0.0014, p95 0.12, ranks 2-5 median 0.098, p95 0.34 -- illatio against
;;; llama.cpp gives 0.0020, 0.079, 0.103, 0.366.  Their bounds (+KQUANT-BOUNDS+)
;;; are that floor with the same headroom.
;;; Exits 0 iff all hold.  Needs the model: $ILLATIO_TEST_MODEL, or
;;; ~/models/qwen2.5-0.5b-instruct-q8_0.gguf; and its vectors:
;;; $ILLATIO_TEST_VECTORS, or vectors/qwen2.5-0.5b-q8_0.sexp.  With
;;; $ILLATIO_TEST_GPU set, the model runs on the GPU (metal-qwen35.lisp).

(require :asdf)
(asdf:load-asd (merge-pathnames "../illatio.asd" (or *load-truename* *default-pathname-defaults*)))
(handler-bind ((warning #'muffle-warning)) (asdf:load-system "illatio"))
(in-package #:illatio)

(defparameter +near-tie+ 0.25)
(defparameter +bounds+ '((:top1 0.5 0.02) (:top1 0.95 0.15) (:rest 0.5 0.10) (:rest 0.95 0.40)))
(defparameter +kquant-bounds+ '((:top1 0.5 0.02) (:top1 0.95 0.25) (:rest 0.5 0.20) (:rest 0.95 0.60)))

(defvar *deltas* (list :top1 '() :rest '()))

(defun bounds-for (m) (if (member :q8k (model-act-kinds m)) +kquant-bounds+ +bounds+))

(defun quantile (list p)
  (let ((s (sort (copy-list list) #'<)))
    (nth (floor (* p (1- (length s)))) s)))

(defun test-model-path ()
  (or (uiop:getenv "ILLATIO_TEST_MODEL")
      (namestring (merge-pathnames "models/qwen2.5-0.5b-instruct-q8_0.gguf" (user-homedir-pathname)))))

(defun read-vectors (path)
  (with-open-file (s path) (loop for c = (read s nil) while c collect c)))

(defvar *failures* 0)
(defun check (ok fmt &rest args)
  (unless ok (incf *failures*) (format t "  FAIL ~?~%" fmt args))
  ok)

(defun tail-noise-p (lp ours bounds)
  "True when LP, absent from our top 10 OURS, is within the p95 ranks-2-5
   bound of our tenth logprob: below what this format's noise resolves."
  (let ((tenth (cdr (car (last ours))))
        (b (third (find-if (lambda (e) (and (eq (first e) :rest) (= (second e) 0.95))) bounds))))
    (and tenth b (< lp (+ tenth b)))))

(defun run-case (m c)
  (let* ((name (getf c :name))
         (prompt (getf c :prompt))
         (ctx (make-context m))
         (worst 0) (agree 0) (ties 0))
    (check (equal (tokenize m (getf c :text)) prompt)
           "~A: tokenizer gives ~A, llama.cpp ~A" name (tokenize m (getf c :text)) prompt)
    (eval-tokens ctx prompt)
    (loop for (want . ref) in (getf c :steps)
          for i from 0
          do (let* ((ours (top-logprobs (context-logits ctx) 10))
                    (tie (and (second ref) (< (- (second (first ref)) (second (second ref))) +near-tie+))))
               (loop for (id lp) in (subseq ref 0 (min 5 (length ref)))
                     for mine = (cdr (assoc id ours))
                     for rank from 0
                     do (when (check (or mine (tail-noise-p lp ours (bounds-for m)))
                                     "~A step ~D: llama.cpp's ~D (~,3F) is not in our top 10" name i id lp)
                          (when mine
                            (setf worst (max worst (abs (- mine lp))))
                            (push (abs (- mine lp)) (getf *deltas* (if (zerop rank) :top1 :rest))))))
               (cond ((= (car (first ours)) want) (incf agree))
                     (tie (incf ties))
                     (t (check nil "~A step ~D: argmax ~D, llama.cpp ~D (margin ~,3F)" name i
                               (car (first ours)) want (- (second (first ref)) (second (second ref))))))
               (forward ctx want)))
    (format t "~&~8A ~2D steps: argmax agrees ~2D, near-ties ~D, worst |dlogprob| ~,4F~%"
            name (length (getf c :steps)) agree ties worst)))

(let ((path (test-model-path)))
  (unless (probe-file path)
    (format t "no model at ~A; see README (Testing)~%" path)
    (uiop:quit 2))
  (let ((m (if (uiop:getenv "ILLATIO_TEST_GPU")
               (funcall (intern "LOAD-MODEL-GPU" '#:illatio) path)
               (load-model path))))
    (dolist (c (read-vectors (or (uiop:getenv "ILLATIO_TEST_VECTORS")
                                 (merge-pathnames "vectors/qwen2.5-0.5b-q8_0.sexp"
                                                  (or *load-truename* *default-pathname-defaults*)))))
      (run-case m c))
    (loop for (key p bound) in (bounds-for m)
          for q = (quantile (getf *deltas* key) p)
          do (format t "~&~5A p~2D |dlogprob| ~,4F (bound ~,2F)~%" key (round (* 100 p)) q bound)
             (check (< q bound) "~A p~D ~,4F exceeds ~,2F" key (round (* 100 p)) q bound))
    (format t "~&~:[ok~;~:*~D failures~]~%" (and (plusp *failures*) *failures*))
    (uiop:quit (if (zerop *failures*) 0 1))))
