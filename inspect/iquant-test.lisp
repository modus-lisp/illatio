;;;; iquant-test.lisp — the DeepSeek V4 Flash expert kernels (src/iquant.lisp)
;;;; against ggml's own dequantizers.
;;;
;;;   sbcl --non-interactive --load inspect/iquant-test.lisp
;;;
;;; Rows of real IQ2_XXS and Q2_K expert tensors are dequantized by gguf-py
;;; (llama.cpp's Python package; $GGUF_PY, default ~/workspace/llama.cpp/
;;; gguf-py; needs numpy) and by illatio: the floats must be identical.  Each
;;; kernel's dot against Q8_K activations must equal the exact dot of ggml's
;;; floats with them to rounding (1e-6 of the sum of |terms|).  Needs the
;;; ds4f GGUF ($ILLATIO_DS4F).

(load (merge-pathnames "load.lisp" (or *load-truename* *default-pathname-defaults*)))
(in-package #:illatio)
(defparameter *path*
  (or (uiop:getenv "ILLATIO_DS4F")
      (namestring (merge-pathnames "workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
                                   (user-homedir-pathname)))))
(defparameter *dequant* (namestring (merge-pathnames "tools/dequant-rows.py" (or *load-truename* *default-pathname-defaults*))))
(defvar *failures* 0)
(defvar *g* (read-gguf *path*))
(defvar *w* (map-file-vector *path*))
(format t "~&mapped ~:D bytes; last byte ~D~%" (length *w*) (aref *w* (1- (length *w*))))
(defun read-floats (file n)
  (with-open-file (s file :element-type '(unsigned-byte 32))
    (let ((v (f32vec n))) (dotimes (i n v) (setf (aref v i) (sb-kernel:make-single-float (let ((u (read-byte s))) (if (>= u (ash 1 31)) (- u (ash 1 32)) u))))))))
(defun check (name type row0 nrows)
  (let* ((ti (gguf-tensor *g* name)) (ncols (first (tensor-info-dims ti)))
         (off (+ (gguf-data-offset *g*) (tensor-info-offset ti)))
         (row-bytes (* (floor ncols 256) (if (eq type :iq2_xxs) 66 84)))
         (out (format nil "/tmp/illatio-iquant-~D.f32" (random 1000000 (make-random-state t)))))
    (uiop:run-program (list "python3" "-I" *dequant* *path* (princ-to-string off)
                            (if (eq type :iq2_xxs) "IQ2_XXS" "Q2_K") (princ-to-string ncols)
                            (princ-to-string row0) (princ-to-string nrows) out) :output nil)
    (let ((ref (read-floats out (* nrows ncols))) (row (f32vec ncols))
          (x (f32vec ncols)) (xv (make-q8kvec ncols)) (worst-deq 0f0) (worst-rel 0d0) (neon-diff 0)
          (state (sb-ext:seed-random-state 7)))
      (dotimes (i ncols) (setf (aref x i) (- (random 2f0 state) 1f0)))
      (quantize-q8k! x xv)
      (dotimes (r nrows)
        (let ((ro (+ off (* (+ row0 r) row-bytes))))
          (if (eq type :iq2_xxs) (iq2xxs-row-floats! *w* ro ncols row) (q2k-row-floats! *w* ro ncols row))
          (dotimes (i ncols) (setf worst-deq (max worst-deq (abs (- (aref row i) (aref ref (+ (* r ncols) i)))))))
          ;; the kernel against ggml's floats times the quantized activations, in double
          (let ((k (if (eq type :iq2_xxs) (iq2xxs-row-dot *w* ro (floor ncols 256) xv 0)
                       (q2k-row-dot *w* ro (floor ncols 256) xv 0)))
                (exact (loop for i below ncols
                             sum (* (coerce (aref ref (+ (* r ncols) i)) 'double-float)
                                    (aref (q8kvec-d xv) (floor i 256)) (i8-ref (q8kvec-qs xv) i)))))
            (let ((l1 (loop for i below ncols
                            sum (abs (* (coerce (aref ref (+ (* r ncols) i)) 'double-float)
                                        (aref (q8kvec-d xv) (floor i 256)) (i8-ref (q8kvec-qs xv) i))))))
              (setf worst-rel (max worst-rel (/ (abs (- k exact)) l1)))
            ;; the NEON kernel gives the portable one's bits
            (let ((n (if (eq type :iq2_xxs) (iq2xxs-row-dot* (make-row-desc) *w* ro (floor ncols 256) xv 0)
                         (q2k-row-dot* (make-row-desc) *w* ro (floor ncols 256) xv 0))))
              (unless (eql n k) (incf neon-diff)))))))
      (delete-file out)
      (format t "~&~A rows ~D..~D: dequant vs ggml max |diff| ~G; kernel vs exact, max |diff|/sum|terms| ~,2E; NEON /= portable on ~D rows~%"
              name row0 (+ row0 nrows -1) worst-deq worst-rel neon-diff)
      (unless (and (zerop worst-deq) (< worst-rel 1d-6) (zerop neon-diff)) (incf *failures*)))))
(check "blk.3.ffn_gate_exps.weight" :iq2_xxs (* 117 2048) 64)
(check "blk.40.ffn_up_exps.weight" :iq2_xxs (* 3 2048) 64)
(check "blk.3.ffn_down_exps.weight" :q2_k (* 117 4096) 64)
(check "blk.20.ffn_down_exps.weight" :q2_k (* 255 4096 ) 64)
(format t "~&~:[ok~;~:*~D failures~]~%" (and (plusp *failures*) *failures*))
(uiop:quit (if (zerop *failures*) 0 1))
