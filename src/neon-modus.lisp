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

;;; Q4_K and Q6_K against Q8_K (kquant.lisp), one row, on the integer lanes:
;;; each group's quants SDOTted with its activations into s32 lanes, times
;;; its scale (DUP32) into one accumulator, the lanes folded once a block
;;; (two ADDP32s, lane 0).  The mins / -32 sums and the float combine are
;;; Q4K-ROW-DOT's and Q6K-ROW-DOT's scalar code, in their order, so the
;;; result is their bits.  A high nibble is (sshr8 q 4) & 15: the lanes are
;;; signed, the mask takes the sign back off.

(defmacro %s32 (u)
  "The u32 lane U read as signed.  A macro: a call would end the call-free
   scope the vector locals live in."
  (let ((v (gensym)))
    `(let ((,v ,u)) (declare (type (unsigned-byte 32) ,v)) (if (> ,v 2147483647) (- ,v 4294967296) ,v))))

(defun %modus-q4k-row-dot (qs sc dd yq yd b32 nb row xo)
  (declare (type (simple-array (unsigned-byte 8) (*)) qs sc yq)
           (type (simple-array single-float (*)) dd yd)
           (type (simple-array (signed-byte 32) (*)) b32)
           (type (unsigned-byte 40) nb row xo) (optimize (speed 3) (safety 0)))
  (let ((sum 0.0))
    (declare (type single-float sum))
    (dotimes (i nb sum)
      (let* ((blk (+ (* row nb) i)) (yb (+ (ash xo -8) i))
             (ydi (aref yd yb))
             (d (* ydi (aref dd (* 2 blk))))
             (dmin (* ydi (aref dd (+ 1 (* 2 blk)))))
             (imin 0))
        (declare (type (unsigned-byte 40) blk yb) (type single-float ydi d dmin)
                 (type (signed-byte 40) imin))
        (dotimes (g 8)
          (setq imin (+ imin (the (signed-byte 32) (* (aref sc (+ (* blk 16) 8 g))
                                                      (aref b32 (+ (* yb 8) g)))))))
        (setq sum (- sum (* dmin (float imin 1.0))))
        (let ((acc (%vi-movi8 0)) (mask (%vi-movi8 15)))
          (declare (type vi-pack acc mask))
          (dotimes (k 4)
            (let* ((qo (+ (* blk 128) (* k 32)))
                   (yo (+ xo (* i 256) (* k 64)))
                   (qa (%vi-ld16 qs qo)) (qb (%vi-ld16 qs (+ qo 16)))
                   ;; DUP32 takes a variable, not an AREF
                   (s0 (aref sc (+ (* blk 16) (* 2 k))))
                   (s1 (aref sc (+ (* blk 16) (* 2 k) 1))))
              (declare (type (unsigned-byte 40) qo yo) (type vi-pack qa qb) (type (unsigned-byte 8) s0 s1))
              (setq acc (%vi-add32 acc (%vi-mul32 (%vi-sdot (%vi-sdot (%vi-movi8 0) (%vi-and qa mask) (%vi-ld16 yq yo))
                                                            (%vi-and qb mask) (%vi-ld16 yq (+ yo 16)))
                                                  (%vi-dup32 s0))))
              (setq acc (%vi-add32 acc (%vi-mul32 (%vi-sdot (%vi-sdot (%vi-movi8 0) (%vi-and (%vi-sshr8 qa 4) mask) (%vi-ld16 yq (+ yo 32)))
                                                            (%vi-and (%vi-sshr8 qb 4) mask) (%vi-ld16 yq (+ yo 48)))
                                                  (%vi-dup32 s1))))))
          (let ((isum (%s32 (%vi-umov32 (%vi-addp32 (%vi-addp32 acc acc) (%vi-movi8 0)) 0))))
            (declare (type (signed-byte 32) isum))
            (setq sum (+ sum (* d (float isum 1.0))))))))))

(defmacro %q6-grp (lo hi lowhalf shift yoff sidx)
  "One Q6_K group in %MODUS-Q6K-ROW-DOT: 16 weights, a low nibble of the
   bytes at HB+LO and two high bits of those at HB+HI, dotted with the
   activations at YO+YOFF, times scale SIDX of half C, into ACC.  A global
   macro, not a MACROLET: the call-free walker does not see into MACROLET.
   It names the kernel's variables (HB YO BLK C SC QHL YQ ACC)."

                           ;; 16 weights: a low nibble of the bytes at LO, two high
                           ;; bits of those at HI, dotted, times their scale
                           `(let* ((u (aref sc (+ (* blk 16) (* c 8) ,sidx)))
                                   (s (if (> u 127) (- u 256) u)))
                              (declare (type (unsigned-byte 8) u) (type (signed-byte 8) s))
                              (setq acc (%vi-add32 acc (%vi-mul32
                                                        (%vi-sdot (%vi-movi8 0)
                                                                  (%vi-orr ,(if lowhalf
                                                                                `(%vi-and (%vi-ld16 qhl (+ hb ,lo)) (%vi-movi8 15))
                                                                                `(%vi-and (%vi-sshr8 (%vi-ld16 qhl (+ hb ,lo)) 4) (%vi-movi8 15)))
                                                                           (%vi-shl16 (%vi-and ,(if (zerop shift)
                                                                                                    `(%vi-ld16 qhl (+ hb ,hi))
                                                                                                    `(%vi-sshr8 (%vi-ld16 qhl (+ hb ,hi)) ,shift))
                                                                                               (%vi-movi8 3))
                                                                                      4))
                                                                  (%vi-ld16 yq (+ yo ,yoff)))
                                                        (%vi-dup32 s))))))

(defun %modus-q6k-row-dot (qhl sc dd yq yd bs nb row xo)
  (declare (type (simple-array (unsigned-byte 8) (*)) qhl sc yq)
           (type (simple-array single-float (*)) dd yd)
           (type (simple-array (signed-byte 32) (*)) bs)
           (type (unsigned-byte 40) nb row xo) (optimize (speed 3) (safety 0)))
  (let ((sum 0.0))
    (declare (type single-float sum))
    (dotimes (i nb sum)
      (let* ((blk (+ (* row nb) i)) (yb (+ (ash xo -8) i)) (imins 0))
        (declare (type (unsigned-byte 40) blk yb) (type (signed-byte 40) imins))
        (dotimes (g 16)
          (let ((s (aref sc (+ (* blk 16) g))))
            (declare (type (unsigned-byte 8) s))
            (setq imins (+ imins (the (signed-byte 32) (* (aref bs (+ (* yb 16) g))
                                                          (if (> s 127) (- s 256) s)))))))
        ;; the masks are MOVIs in place: held, they and the accumulator left
        ;; too few registers for a group's tree
        (let ((acc (%vi-movi8 0)))
          (declare (type vi-pack acc))
          (dotimes (c 2)
            (let ((hb (+ (* blk 192) (* c 96)))
                  (yo (+ xo (* i 256) (* c 128))))
              (declare (type (unsigned-byte 40) hb yo))
              ;; each group loads its own bytes: nine vectors held across a
              ;; half would not fit the FP registers beside the accumulator
              (progn
                ;; per half: qh at 0 (16 + 16 bytes), ql at 32..95
                (%q6-grp 32 0 t 0 0 0)
                (%q6-grp 48 16 t 0 16 1)
                (%q6-grp 64 0 t 2 32 2)
                (%q6-grp 80 16 t 2 48 3)
                (%q6-grp 32 0 nil 4 64 4)
                (%q6-grp 48 16 nil 4 80 5)
                (%q6-grp 64 0 nil 6 96 6)
                (%q6-grp 80 16 nil 6 112 7))))
          (let ((isum (%s32 (%vi-umov32 (%vi-addp32 (%vi-addp32 acc acc) (%vi-movi8 0)) 0))))
            (declare (type (signed-byte 32) isum))
            (setq sum (+ sum (* (* (aref dd blk) (aref yd yb)) (float (- isum (* 32 imins)) 1.0))))))))))
