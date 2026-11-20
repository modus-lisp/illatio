;;;; metal-kquant-test.lisp — the GPU's matrix products for a 4-bit GGUF's
;;;; mixed formats (Q4_K, Q5_K, Q8_0, and F32 for qwen35's small matrices)
;;;; against a CPU dequantization written from ggml's dequantize_row_*.
;;;
;;;   sbcl --non-interactive --load inspect/metal-kquant-test.lisp
;;;
;;; Real tensors from the files' first layers, random X; one token (the
;;; matrix-vector kernels, X in floats: ~1e-7) and 5 and 70 (the tiled
;;; products, X and the weights in halves: ~3e-4).  Models: unsloth's
;;; Qwen3.5-4B Q4_K_M (Q5_K) and bartowski's Agents-A1-4B Q4_K_M (F32) in
;;; ~/models, or the paths in $ILLATIO_TEST_KQUANT (colon separated).

(load (merge-pathnames "load.lisp" (or *load-truename* *default-pathname-defaults*)))
(in-package #:illatio)

(defun f16->f (lo hi)
  (let* ((h (logior lo (ash hi 8))) (s (if (logbitp 15 h) -1 1)) (e (ldb (byte 5 10) h)) (m (ldb (byte 10 0) h)))
    (* s (cond ((= e 0) (* m (expt 2d0 -24))) ((= e 31) 0d0) (t (* (+ 1 (/ m 1024d0)) (expt 2d0 (- e 15))))))))
(defun scale-min (j q)
  (if (< j 4)
      (values (logand (aref q j) 63) (logand (aref q (+ j 4)) 63))
      (values (logior (logand (aref q (+ j 4)) 15) (ash (ash (aref q (- j 4)) -6) 4))
              (logior (ash (aref q (+ j 4)) -4) (ash (ash (aref q j) -6) 4)))))
(defun deq-kblock (type bytes o out base)
  (let* ((d (f16->f (aref bytes o) (aref bytes (+ o 1))))
         (dmin (f16->f (aref bytes (+ o 2)) (aref bytes (+ o 3))))
         (sc (subseq bytes (+ o 4) (+ o 16)))
         (q5 (eq type :q5_k))
         (qs (+ o (if q5 48 16))))
    (dotimes (c 4)
      (multiple-value-bind (s1 m1) (scale-min (* 2 c) sc)
        (multiple-value-bind (s2 m2) (scale-min (1+ (* 2 c)) sc)
          (dotimes (l 32)
            (let ((q (aref bytes (+ qs (* c 32) l)))
                  (h (if q5 (aref bytes (+ o 16 l)) 0)))
              (setf (aref out (+ base (* c 64) l))
                    (- (* d s1 (+ (logand q 15) (if (logbitp (* 2 c) h) 16 0))) (* dmin m1)))
              (setf (aref out (+ base (* c 64) 32 l))
                    (- (* d s2 (+ (ash q -4) (if (logbitp (1+ (* 2 c)) h) 16 0))) (* dmin m2))))))))))

(defun deq-row (type bytes ncols)
  "ggml's dequantize_row_* for one row."
  (let ((out (make-array ncols :element-type 'double-float)))
    (ecase type
      (:f32 (dotimes (i ncols)
              (setf (aref out i) (coerce (sb-kernel:make-single-float
                                          (let ((u (logior (aref bytes (* 4 i)) (ash (aref bytes (+ 1 (* 4 i))) 8)
                                                           (ash (aref bytes (+ 2 (* 4 i))) 16) (ash (aref bytes (+ 3 (* 4 i))) 24))))
                                            (if (logbitp 31 u) (- u (ash 1 32)) u)))
                                         'double-float))))
      (:q8_0
       (dotimes (b (/ ncols 32))
         (let* ((o (* b 34)) (d (f16->f (aref bytes o) (aref bytes (1+ o)))))
           (dotimes (i 32)
             (let ((q (aref bytes (+ o 2 i))))
               (setf (aref out (+ (* b 32) i)) (* d (if (> q 127) (- q 256) q))))))))
      ((:q4_k :q5_k)
       (dotimes (b (/ ncols 256))
         (deq-kblock type bytes (* b (if (eq type :q5_k) 176 144)) out (* b 256)))))
    out))

(defun check-model (path)
 (let* (
       (m (load-model-gpu path)) (gm (model-gpu m)) (gpu (gmodel-gpu gm))
       (l0 (aref (model-layers m) 0))
       (state (sb-ext:seed-random-state 7))
       (ok t))
  (with-open-file (f path :element-type '(unsigned-byte 8))
    (dolist (w (let ((seen '()))
                 (loop for l across (subseq (model-layers m) 0 4)
                       do (dolist (w (list (layer-wqkv l) (layer-ssm-out l) (layer-wz l) (layer-walpha l) (layer-wbeta l)
                                           (layer-wq l) (layer-wgate l) (layer-wdown l)))
                            (when (and w (member (gmat-type w) '(:f32 :q8_0 :q4_k :q5_k))
                                       (not (find (gmat-type w) seen :key #'gmat-type)))
                              (push w seen))))
                 (reverse seen)))
      (dolist (n '(1 5 70))
        (let* ((rows (gmat-rows w)) (cols (gmat-cols w))
               (x (make-array (* n cols) :element-type 'single-float))
               (xb (gpu-buffer gpu (* 4 n cols))) (yb (gpu-buffer gpu (* 4 n rows)))
               (maxerr 0d0) (maxref 0d0))
          (dotimes (i (length x)) (setf (aref x i) (- (random 2f0 state) 1f0)))
          (gbuf-set-floats xb x)
          (with-gpu-work (e gpu) (mat gm e w xb yb n))
          (let ((y (gbuf-floats yb (* n rows)))
                (bytes (make-array (gmat-row-bytes w) :element-type '(unsigned-byte 8))))
            (loop for r from 0 below rows by (max 1 (floor rows 64))
                  do (file-position f (+ (gmat-offset w) (* r (gmat-row-bytes w))))
                     (read-sequence bytes f)
                     (let ((wr (deq-row (gmat-type w) bytes cols)))
                       (dotimes (tk n)
                         (let ((ref (loop for c below cols sum (* (aref wr c) (aref x (+ (* tk cols) c))))))
                           (setf maxref (max maxref (abs ref))
                                 maxerr (max maxerr (abs (- ref (aref y (+ (* tk rows) r)))))))))))
          (let ((rel (/ maxerr maxref)))
            ;; the mm path rounds weights and X to halves
            (when (> rel (if (= n 1) 1d-5 5d-3)) (setf ok nil))
            (format t "~&~24A ~5A ~5Dx~5D n=~2D  max|err|/max|ref| = ~,2E~%"
                    (gmat-name w) (gmat-type w) rows cols n rel))))))
  ok))

(let* ((paths (let ((e (uiop:getenv "ILLATIO_TEST_KQUANT")))
                (if e (uiop:split-string e :separator ":")
                    (mapcar (lambda (f) (namestring (merge-pathnames f (user-homedir-pathname))))
                            '("models/qwen3.5-4b-q4_k_m.gguf" "models/agents-a1-4b-q4_k_m.gguf")))))
       (ok t) (found nil))
  (dolist (p paths)
    (if (probe-file p)
        (progn (format t "~&~A~%" (file-namestring p)) (setf found t)
               (unless (check-model p) (setf ok nil)))
        (format t "~&skip: no model at ~A~%" p)))
  (unless found (uiop:quit 2))
  (format t "~&~:[FAIL~;ok~]~%" ok)
  (uiop:quit (if ok 0 1)))
