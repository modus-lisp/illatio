;;;; ds4f-vectors-test.lisp — illatio's DeepSeek V4 Flash against ds4's numbers.
;;;
;;;   sbcl --dynamic-space-size 4096 --non-interactive --load inspect/ds4f-vectors-test.lisp
;;;
;;; As vectors-test.lisp, with ds4 as the reference (collect-ds4f-vectors.py:
;;; its token ids and greedy top logprobs, from its CPU path, the code ds4f
;;; ports): each step is teacher-forced with ds4's token, and our
;;; log-probabilities are compared with its.
;;;
;;; This model's noise floor is high: discrete steps (the top-6 expert
;;; choice, the FP8 and FP4 round trips) turn a 3e-4 change in a matvec into
;;; a different token now and then.  Measured as for vectors-test, illatio
;;; against a copy of itself that differs only in rounding the Q8_0
;;; activation scales to F16: argmax 69/72, top-1 |dlogprob| p50 0.009 p95
;;; 0.18, ranks 2-5 p50 0.19 p95 0.75 -- and illatio against ds4: 70/72,
;;; 0.008 / 0.14, 0.22 / 0.91.  So the bounds are that floor with headroom:
;;; ds4's top token always in our top 10, our argmax ds4's on at least 90%
;;; of steps, and the distribution below.  Needs the GGUF ($ILLATIO_DS4F).

(load (merge-pathnames "load.lisp" (or *load-truename* *default-pathname-defaults*)))
(in-package #:illatio)

(defparameter +near-tie+ 0.25)
(defparameter +ds4f-bounds+ '((:top1 0.5 0.03) (:top1 0.95 0.40) (:rest 0.5 0.35) (:rest 0.95 1.50)))
(defparameter +min-agreement+ 0.9)
(defvar *agree* 0)
(defvar *steps* 0)
(defvar *failures* 0)
(defvar *deltas* (list :top1 '() :rest '()))

(defun quantile (list p) (let ((s (sort (copy-list list) #'<))) (nth (floor (* p (1- (length s)))) s)))

(defun run-case (m c)
  (let ((name (getf c :name)) (ctx (make-d4ctx m :n-ctx 4096)) (agree 0) (ties 0) (worst 0) (t0 (get-internal-real-time)))
    (loop for (tok . rest) on (getf c :prompt) do (ds4f-forward ctx tok :logits (null rest)))
    (loop for (want . ref) in (getf c :steps)
          for i from 0
          do (let ((ours (top-logprobs (d4ctx-logits ctx) 10))
                   (tie (< (- (second (first ref)) (second (second ref))) +near-tie+)))
               (loop for (id lp) in (subseq ref 0 5) for rank from 0
                     for mine = (cdr (assoc id ours))
                     do (if mine
                            (progn (setf worst (max worst (abs (- mine lp))))
                                   (push (abs (- mine lp)) (getf *deltas* (if (zerop rank) :top1 :rest))))
                            (when (zerop rank)
                              (incf *failures*)
                              (format t "  FAIL ~A step ~D: ds4's top ~D (~,3F) not in our top 10~%" name i id lp))))
               (incf *steps*)
               (cond ((= (car (first ours)) want) (incf agree) (incf *agree*))
                     (tie (incf ties) (incf *agree*))
                     (t (format t "  -- ~A step ~D: argmax ~D, ds4 ~D (margin ~,3F)~%" name i (car (first ours)) want
                                (- (second (first ref)) (second (second ref))))))
               (ds4f-forward ctx want)))
    (format t "~&~8A ~3D prompt, ~2D steps: argmax agrees ~2D, near-ties ~D, worst |dlogprob| ~,4F (~,0F s)~%"
            name (length (getf c :prompt)) (length (getf c :steps)) agree ties worst
            (/ (- (get-internal-real-time) t0) internal-time-units-per-second))))

(let* ((path (or (uiop:getenv "ILLATIO_DS4F")
                 (namestring (merge-pathnames "workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
                                              (user-homedir-pathname)))))
       (m (load-ds4f path)))
  (dolist (c (with-open-file (s (merge-pathnames "vectors/ds4f-q2.sexp" (or *load-truename* *default-pathname-defaults*)))
               (loop for c = (read s nil) while c collect c)))
    (run-case m c))
  (loop for (key p bound) in +ds4f-bounds+
        for q = (quantile (getf *deltas* key) p)
        do (format t "~&~5A p~2D |dlogprob| ~,4F (bound ~,2F)~%" key (round (* 100 p)) q bound)
           (unless (< q bound) (incf *failures*)))
  (format t "~&argmax agrees on ~D of ~D steps (at least ~D% required)~%" *agree* *steps* (round (* 100 +min-agreement+)))
  (when (< *agree* (* +min-agreement+ *steps*)) (incf *failures*))
  (format t "~&~:[ok~;~:*~D failures~]~%" (and (plusp *failures*) *failures*))
  (uiop:quit (if (zerop *failures*) 0 1)))
