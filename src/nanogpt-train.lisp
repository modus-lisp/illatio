;;;; nanogpt-train.lisp — Muon, and the speedrun's training loop.
;;;
;;; Muon (Keller Jordan, Jeremy Bernstein et al.; modded-nanogpt's record
;;; 11): each hidden matrix's update is SGD's Nesterov momentum, orthogonalized
;;; by five Newton-Schulz steps (in bf16, the quintic a, b, c = 3.4445,
;;; -4.7750, 2.0315) and scaled by sqrt(max(1, rows/cols)).  A wide X (rows <=
;;; cols) steps X <- aX + (bA + cA^2) X with A = X X^T; a tall one the same
;;; transposed, X <- aX + X (bA + cA^2) with A = X^T X, so nothing is
;;; transposed.  QKV's three matrices are orthogonalized each on its own.  The
;;; embedding and the head are Adam's, as there.

(in-package #:illatio)

(defstruct (muon-scratch (:constructor %make-muon-scratch)) x xn a a2 s)

(defvar *ngpt-scratch* (make-hash-table :test 'eq :weakness :key))
(defun ngpt-scratch (m)
  (or (gethash m *ngpt-scratch*)
      (setf (gethash m *ngpt-scratch*)
            (let* ((gpu (ngpt-gpu m)) (c (ngpt-c m)) (bytes (* 4 4 c c)))
              (%make-muon-scratch :x (gpu-buffer gpu bytes) :xn (gpu-buffer gpu bytes)
                                  :a (gpu-buffer gpu bytes) :a2 (gpu-buffer gpu bytes)
                                  :s (gpu-buffer gpu (* 4 (+ 4 +g2-norm-groups+))))))))

(defparameter *ns-coefficients* '(3.4445f0 -4.7750f0 2.0315f0))

(defun encode-muon-matrix (m e r c w g buf lr mu &key (steps 5))
  "Muon's step for the R x C matrix at W (parameters), G (its gradient),
   BUF (its momentum): (buffer . byte offset)s."
  (let* ((gpu (ngpt-gpu m)) (sc (ngpt-scratch m)) (n (* r c))
         (x (muon-scratch-x sc)) (xn (muon-scratch-xn sc)) (aa (muon-scratch-a sc)) (a2 (muon-scratch-a2 sc))
         (s (muon-scratch-s sc)) (wide (<= r c)) (k (if wide r c)))
    (destructuring-bind (ca cb cc) *ns-coefficients*
      (g2-1d gpu e "ng_nesterov" n (list mu) (list g buf x))
      (dispatch gpu e "g2_sumsq" (list n +g2-norm-groups+) (list x (cons s 16)) +g2-norm-groups+ 1024 :shared 128)
      (dispatch gpu e "ng_inv_norm" (list +g2-norm-groups+) (list (cons s 16) s) 1 32)
      (g2-1d gpu e "ng_scale" n '() (list x s))
      (dotimes (i steps)
        (if wide
            (g2-gemm gpu e nil t r r c x c x c aa r)            ; A = X X^T
            (g2-gemm gpu e t nil c c r x c x c aa c))           ; A = X^T X
        (g2-gemm gpu e nil nil k k k aa k aa k a2 k :alpha cc)  ; A2 = c A A
        (g2-1d gpu e "ng_axpby" (* k k) (list cb 1f0) (list aa a2)) ; A2 = b A + A2
        (if wide
            (g2-gemm gpu e nil nil r c r a2 r x c xn c)         ; XN = B X
            (g2-gemm gpu e nil nil r c c x c a2 c xn c))        ; XN = X B
        (g2-1d gpu e "ng_axpby" n (list ca 1f0) (list x xn))   ; XN = a X + XN
        (rotatef x xn))
      (g2-1d gpu e "ng_axpby" n (list (- (* lr (sqrt (max 1f0 (/ r c))))) 1f0) (list x w)))))

(defun ngpt-encode-update (m e step &key (muon-lr 0.04f0) (mu 0.95f0) (embed-lr 0.6f0) (head-lr 0.008f0)
                                      (beta1 0.9f0) (beta2 0.95f0) (eps 1f-8))
  "One optimizer step: Muon for the hidden matrices, Adam for the embedding and the head."
  (with-ngpt (m)
    (let ((c (ngpt-c m)) (c1 (- 1f0 (expt beta1 step))) (c2 (- 1f0 (expt beta2 step))))
      (flet ((m1 (name &optional (li 0) (plus 0)) (cons (ngpt-m1buf m) (* 4 (+ plus (lof pl name li)))))
             (at (pair plus) (cons (car pair) (+ (cdr pair) (* 4 plus)))))
        (dotimes (li (ngpt-l m))
          (dotimes (k 3)
            (encode-muon-matrix m e c c (at (p :qkvw li) (* k c c)) (at (g :qkvw li) (* k c c))
                                (m1 :qkvw li (* k c c)) muon-lr mu))
          (encode-muon-matrix m e c c (p :projw li) (g :projw li) (m1 :projw li) muon-lr mu)
          (encode-muon-matrix m e (* 4 c) c (p :fcw li) (g :fcw li) (m1 :fcw li) muon-lr mu)
          (encode-muon-matrix m e c (* 4 c) (p :fcprojw li) (g :fcprojw li) (m1 :fcprojw li) muon-lr mu))
        (loop for (name lr) in (list (list :wte embed-lr) (list :head head-lr))
              do (g2-1d gpu e "g2_adamw" (lsize pl name)
                        (list lr beta1 beta2 eps 0f0 c1 c2)
                        (list (p name) (g name) (m1 name) (cons (ngpt-m2buf m) (* 4 (lof pl name)))
                              (ngpt-norm m))))))))

(defun trapezoid (step steps warmup warmdown-frac)
  "The learning rate's multiplier: a linear warmup, 1, then a linear warmdown
   over the last WARMDOWN-FRAC of the steps."
  (let ((down (* warmdown-frac steps)))
    (cond ((< step warmup) (/ (1+ step) warmup))
          ((< step (- steps down)) 1)
          (t (/ (- steps step) down)))))

(defun ngpt-val-loss (m files batches)
  (let ((ld (make-shard-loader files (ngpt-b m) (ngpt-tt m))) (s 0d0))
    (dotimes (i batches (/ s batches))
      (multiple-value-bind (x y) (shard-next ld) (incf s (ngpt-loss m x y))))))

(defun nanogpt-train (&key (data (merge-pathnames "workspace/modded-nanogpt/data/fineweb10B/" (user-homedir-pathname)))
                           (l 6) (c 384) (b 8) (tt 1024) (accum 8) (steps 1000) (warmup 0) (warmdown 0.3)
                           (muon-lr 0.04f0) (embed-lr 0.6f0) (head-lr 0.008f0) (momentum-warmup 500)
                           (cap 30f0) (val-every 50) (val-batches 16) (target nil) (seed 1337)
                           (model nil) (out *standard-output*))
  "Train the speedrun's model on FineWeb: STEPS steps of ACCUM micro-batches
   of B x TT tokens.  Stops early at the validation loss TARGET.  The model."
  (let* ((*g2-precision* '((:head . "_b") (:attention . "_b") (:linear . "_b")))
         (m (or model (make-nanogpt :l l :c c :b b :tt tt :cap cap :seed seed)))
         (train (make-shard-loader (fineweb-shards data "train") b tt))
         (val (fineweb-shards data "val"))
         (tokens (* b tt accum)) (t-start (get-internal-real-time)) (train-ms 0))
    (format out "~:D parameters, ~D layers x ~D channels; ~:D tokens a step (~D x ~D x ~D)~%"
            (ngpt-n-params m) (ngpt-l m) (ngpt-c m) tokens accum b tt)
    (block run
      (dotimes (step (1+ steps))
        (when (or (= step steps) (and (plusp val-every) (zerop (mod step val-every))))
          (let ((vl (ngpt-val-loss m val val-batches)))
            (format out "step ~D: val loss ~,4F  train time ~,1F s  tokens ~:D~%"
                    step vl (/ train-ms 1000.0) (* step tokens))
            (finish-output out)
            (when (and target (<= vl target))
              (format out "target ~,3F reached at step ~D, ~,1F s of training~%" target step (/ train-ms 1000.0))
              (return-from run))))
        (when (< step steps)
          (let ((t0 (get-internal-real-time)) (loss 0d0)
                (k (coerce (trapezoid step steps warmup warmdown) 'single-float))
                (frac (min 1 (/ step (max 1 momentum-warmup)))))
            (dotimes (i accum)
              (multiple-value-bind (x y) (shard-next train) (ngpt-set-batch m x y))
              (with-gpu-work (e (ngpt-gpu m))
                (ngpt-encode-forward m e :dloss (/ 1f0 tokens))
                (ngpt-encode-backward m e :accumulate (plusp i) :fused t))
              (incf loss (/ (ngpt-mean-loss m) accum)))
            (with-gpu-work (e (ngpt-gpu m))
              (ngpt-encode-update m e (1+ step) :muon-lr (* k muon-lr) :embed-lr (* k embed-lr)
                                                :head-lr (* k head-lr)
                                                :mu (coerce (+ (* (- 1 frac) 0.85) (* frac 0.95)) 'single-float)))
            (let ((ms (/ (- (get-internal-real-time) t0) (/ internal-time-units-per-second 1000))))
              (incf train-ms ms)
              (when (or (< step 10) (zerop (mod (1+ step) 10)))
                (format out "step ~D/~D: loss ~,4F  ~,0F ms  ~,0F tok/s~%" (1+ step) steps loss ms
                        (/ (* tokens 1000) ms))
                (finish-output out)))))))
    (format out "wall ~,1F s~%" (/ (- (get-internal-real-time) t-start) internal-time-units-per-second))
    m))
