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
