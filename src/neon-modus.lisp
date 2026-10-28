;;;; neon-modus.lisp — the Q8_0 matrix kernel on modus's integer lanes.
;;;
;;; The same four-rows-per-pass shape as %Q8-DOT-4ROWS in neon-sbcl.lisp,
;;; written in modus's own SIMD primitives instead of instruction words:
;;; %VI-SDOT puts each row's 32 int8 x int8 products into s32 lanes, two
;;; %VI-ADDP32 fold four rows into one vector (lane r = row r's exact block
;;; sum), %VI-SCVTF32 makes it floats, and the f32x4 accumulator -- which the
;;; compiler keeps in a register because the scope is call-free -- adds
;;; float(sum) * (dw*dx).  Lane r is Q8-ROW-DOT-AT's arithmetic for its row,
;;; in its order, so the result is its bits (modus tests/simd/q8-kernel.lisp
;;; checks the same kernel against a scalar one).
;;;
;;; The indices are (unsigned-byte 40), byte offsets into weights of up to a
;;; terabyte: an honest width lets modus compile them unchecked, which is
;;; also what keeps the scope call-free.  Weights are u8 (two's complement):
;;; see I8VEC in tensor.lisp.

(in-package #:illatio)

(defun %modus-q8-dot-4rows (wq ws4 xq xs cols nb g xo xso out oo)
  "OUT[oo..oo+3] = rows 4G..4G+3 of the Q8_0 matrix (quants WQ, four-row
   scales WS4) against the activations at XO in XQ (scales at XSO in XS)."
  (declare (type (simple-array (unsigned-byte 8) (*)) wq xq)
           (type (simple-array single-float (*)) ws4 xs out)
           (type (unsigned-byte 40) cols nb g xo xso oo) (optimize (speed 3) (safety 0)))
  (let ((acc (%f32v-broadcast 0.0)) (p0 (* 4 g cols)) (s0 (* g nb)))
    (declare (type f32v-pack acc) (type (unsigned-byte 40) p0 s0))
    (dotimes (b nb)
      (let* ((w0 (+ p0 (* b 32))) (w1 (+ w0 cols)) (w2 (+ w1 cols)) (w3 (+ w2 cols))
             (xb (+ xo (* b 32))))
        (declare (type (unsigned-byte 40) w0 w1 w2 w3 xb))
        (let ((xa (%vi-ld16 xq xb)) (xc (%vi-ld16 xq (+ xb 16))))
          (declare (type vi-pack xa xc))
          (setq acc
                (%f32v+ acc
                        (%f32v* (%vi-scvtf32
                                 (%vi-addp32
                                  (%vi-addp32 (%vi-sdot (%vi-sdot (%vi-movi8 0) (%vi-ld16 wq w0) xa) (%vi-ld16 wq (+ w0 16)) xc)
                                              (%vi-sdot (%vi-sdot (%vi-movi8 0) (%vi-ld16 wq w1) xa) (%vi-ld16 wq (+ w1 16)) xc))
                                  (%vi-addp32 (%vi-sdot (%vi-sdot (%vi-movi8 0) (%vi-ld16 wq w2) xa) (%vi-ld16 wq (+ w2 16)) xc)
                                              (%vi-sdot (%vi-sdot (%vi-movi8 0) (%vi-ld16 wq w3) xa) (%vi-ld16 wq (+ w3 16)) xc))))
                                (%f32v* (%f32v-ref ws4 (* 4 (+ s0 b))) (%f32v-broadcast-ref xs (+ xso b)))))))))
    (%f32v-set out oo acc)
    nil))

;;; Attention's two kernels, F32-DOT16 and F32-AXPY's bits on f32x4 lanes.
;;; The dot's sixteen strided partial sums are four f32x4 accumulators --
;;; lane L of accumulator J is partial sum 4J+L, fed the same products in the
;;; same order -- stored into the sixteen-float array the portable kernel
;;; sums with the same tree.  There is no fused multiply-add on these lanes,
;;; so a product is rounded before it is added, as in the scalar code.

(defun %modus-f32-dot16 (a ao b bo n)
  "F32-DOT16 for N a multiple of 16."
  (declare (type (simple-array single-float (*)) a b)
           (type (unsigned-byte 40) ao bo n) (optimize (speed 3) (safety 0)))
  (let ((s (make-array 16 :element-type 'single-float :initial-element 0f0)))
    (declare (type (simple-array single-float (*)) s))
    (let ((a0 (%f32v-broadcast 0.0)) (a1 (%f32v-broadcast 0.0))
          (a2 (%f32v-broadcast 0.0)) (a3 (%f32v-broadcast 0.0)))
      (declare (type f32v-pack a0 a1 a2 a3))
      (dotimes (k (ash n -4))
        (let ((p (+ ao (* k 16))) (q (+ bo (* k 16))))
          (declare (type (unsigned-byte 40) p q))
          (setq a0 (%f32v+ a0 (%f32v* (%f32v-ref a p) (%f32v-ref b q))))
          (setq a1 (%f32v+ a1 (%f32v* (%f32v-ref a (+ p 4)) (%f32v-ref b (+ q 4)))))
          (setq a2 (%f32v+ a2 (%f32v* (%f32v-ref a (+ p 8)) (%f32v-ref b (+ q 8)))))
          (setq a3 (%f32v+ a3 (%f32v* (%f32v-ref a (+ p 12)) (%f32v-ref b (+ q 12)))))))
      (%f32v-set s 0 a0) (%f32v-set s 4 a1) (%f32v-set s 8 a2) (%f32v-set s 12 a3))
    (+ (+ (+ (+ (aref s 0) (aref s 4)) (+ (aref s 8) (aref s 12)))
          (+ (+ (aref s 1) (aref s 5)) (+ (aref s 9) (aref s 13))))
       (+ (+ (+ (aref s 2) (aref s 6)) (+ (aref s 10) (aref s 14)))
          (+ (+ (aref s 3) (aref s 7)) (+ (aref s 11) (aref s 15)))))))

(defun %modus-f32-axpy (y yo x xo w n)
  "F32-AXPY for N a multiple of 4: Y[yo..] += W * X[xo..]."
  (declare (type (simple-array single-float (*)) y x) (type single-float w)
           (type (unsigned-byte 40) yo xo n) (optimize (speed 3) (safety 0)))
  (let ((wv (%f32v-broadcast w)))
    (declare (type f32v-pack wv))
    (dotimes (k (ash n -2))
      (let ((p (+ yo (* k 4))) (q (+ xo (* k 4))))
        (declare (type (unsigned-byte 40) p q))
        (%f32v-set y p (%f32v+ (%f32v-ref y p) (%f32v* wv (%f32v-ref x q)))))))
  y)
