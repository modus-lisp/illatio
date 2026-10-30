;;;; avx2.lisp — the routed experts' row dots as AVX2, for SBCL on x86-64.
;;;
;;; DS4F's experts (IQ2_XXS gate and up, Q2_K down) against Q8_K
;;; activations, as ggml's x86 kernels compute them (ggml_vec_dot_iq2_xxs_q8_K
;;; and ggml_vec_dot_q2_K_q8_K, MIT, Copyright (c) 2023-2026 The ggml
;;; authors), with sb-simd: a block's products in int8 x int8 -> int16 pairs
;;; (VPMADDUBSW: the weights' magnitudes unsigned, their signs moved onto the
;;; activations by VPSIGNB), widened to int32 by each sub-block's scale
;;; (VPMADDWD), the block's integer sum then scaled in float.  The weights
;;; are read where they lie -- a system-area pointer into the pinned copy of
;;; the experts (cuda-ds4f.lisp) -- so a CPU computes an expert the GPUs
;;; don't hold without its crossing the bus.
;;;
;;; The integer sums are exact; the float steps are ggml's x86 order, not
;;; the portable kernels' (iquant.lisp, kquant.lisp), which follow ds4's
;;; NEON order: the two differ by a float rounding or so a row, under the
;;; GPU's own difference from either, which the vectors test bounds.

(in-package #:illatio)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :sb-simd)
  (sb-ext:add-package-local-nickname "AVX" :sb-simd-avx2 :illatio)
  (sb-ext:add-package-local-nickname "FMA" :sb-simd-fma :illatio))

(defparameter +iq2xxs-grid-u64+
  (coerce +iq2xxs-grid+ '(simple-array (unsigned-byte 64) (256)))
  "The grid, an entry's eight magnitudes a word.")

(defparameter +iq2xxs-sign-u64+
  (let ((v (make-array 128 :element-type '(unsigned-byte 64))))
    (dotimes (s 128 v)
      (setf (aref v s)
            (loop for j below 8
                  sum (ash (if (logbitp j (aref +ksigns-iq2xs+ s)) #xff #x01) (* 8 j))))))
  "Sign pattern S as eight bytes, 1 or -1 each: VPSIGNB's second operand.")

(defparameter +q2k-scale-shuffle+
  (let ((v (make-array 128 :element-type '(unsigned-byte 8))))
    (dotimes (i 128 v)
      ;; byte i of shuffle i/32: the 16-bit scale for its 16 weights, low then high lane
      (setf (aref v i) (+ (* 4 (floor i 32)) (if (< (mod i 32) 16) 0 2) (mod i 2)))))
  "ggml's k_shuffle: VPSHUFB indices spreading a 128 weights' eight scales
   (as 16-bit words) over the four products of 32.")

(defparameter +f16-floats+
  (let ((v (make-array 65536 :element-type 'single-float)))
    (dotimes (h 65536 v) (setf (aref v h) (half-to-single h))))
  "Every binary16 as a single-float, by its bits.")

(declaim (inline f16-sap))
(defun f16-sap (sap off)
  (declare (type sb-sys:system-area-pointer sap) (type fixnum off))
  (aref (the (simple-array single-float (65536)) +f16-floats+) (sb-sys:sap-ref-16 sap off)))

(defparameter +odd-s16+
  (let ((v (make-array 256 :element-type '(signed-byte 16))))
    (dotimes (i 256 v) (setf (aref v i) (1+ (* 2 (floor i 16))))))
  "Sixteen 16-bit lanes of 2L+1 at 16L: IQ2_XXS's sub-block scale L
   broadcast from memory (see the note on SSE below).")

;;; SBCL moves scalars into vector registers, and does its float arithmetic,
;;; with legacy (non-VEX) SSE instructions, and these run slowly while a
;;; 256-bit register's upper half is dirty -- about 100 ns each on this
;;; machine.  So the loops below have none: what a vector is built from goes
;;; through memory (scalar stores, one 256-bit load), the integer sums go out
;;; to memory, and the float arithmetic on them comes after VZEROUPPER.  (And
;;; sb-simd's U16.16-SHIFTR gives 0 for any count but 0 -- the 32-bit shift
;;; does here, the bytes being masked to their two bits after.)

(declaim (inline %sum8))
(defun %sum8 (v o)
  (declare (type (simple-array (signed-byte 32) (*)) v) (type fixnum o))
  (+ (aref v o) (aref v (+ o 1)) (aref v (+ o 2)) (aref v (+ o 3))
     (aref v (+ o 4)) (aref v (+ o 5)) (aref v (+ o 6)) (aref v (+ o 7))))

(defun iq2xxs-dot-avx2 (w nb qs qo yd)
  "The IQ2_XXS row of NB (at most 16) blocks at system-area pointer W
   against Q8_K activations QS (from QO) and their block scales YD (from
   QO/256)."
  (declare (type sb-sys:system-area-pointer w) (type (integer 0 16) nb) (type fixnum qo)
           (type i8vec qs) (type f32vec yd) (optimize speed (safety 0)))
  (let ((grid +iq2xxs-grid-u64+) (signs +iq2xxs-sign-u64+) (odd +odd-s16+)
        (scr (make-array 8 :element-type '(unsigned-byte 64)))
        (sums (make-array 128 :element-type '(signed-byte 32))))
    (declare (type (simple-array (unsigned-byte 64) (256)) grid)
             (type (simple-array (unsigned-byte 64) (128)) signs)
             (type (simple-array (signed-byte 16) (256)) odd)
             (dynamic-extent scr sums))
    (dotimes (b nb)
      (let ((sumi (avx:s32.8 0)) (bo (* b 66)) (q8 (+ qo (* b 256))))
        (declare (type fixnum bo q8) (type avx:s32.8 sumi))
        (dotimes (ib 8)
          (let* ((so (+ bo 2 (* ib 8)))
                 (g (sb-sys:sap-ref-32 w so))
                 (aux (sb-sys:sap-ref-32 w (+ so 4))))
            (declare (type fixnum so) (type (unsigned-byte 32) g aux))
            (setf (aref scr 0) (aref grid (ldb (byte 8 0) g)) (aref scr 1) (aref grid (ldb (byte 8 8) g))
                  (aref scr 2) (aref grid (ldb (byte 8 16) g)) (aref scr 3) (aref grid (ldb (byte 8 24) g))
                  (aref scr 4) (aref signs (ldb (byte 7 0) aux)) (aref scr 5) (aref signs (ldb (byte 7 7) aux))
                  (aref scr 6) (aref signs (ldb (byte 7 14) aux)) (aref scr 7) (aref signs (ldb (byte 7 21) aux)))
            (let* ((q8s (avx:s8.32-sign (avx:s8.32-aref qs (+ q8 (* ib 32))) (avx:s8.32! (avx:u64.4-aref scr 4))))
                   (dot (avx:s16.16-maddubs (avx:u8.32! (avx:u64.4-aref scr 0)) q8s)))
              (setf sumi (avx::two-arg-s32.8+ sumi (avx:s32.8! (avx:s16.16-madd dot (avx:s16.16-aref odd (* 16 (ash aux -28))))))))))
        (setf (avx:s32.8-aref sums (* b 8)) sumi)))
    (avx:vzeroupper)
    (let ((total 0f0) (y0 (floor qo 256)))
      (declare (type single-float total) (type fixnum y0))
      (dotimes (b nb (* 0.125f0 total))
        (incf total (* (f16-sap w (* b 66)) (aref yd (+ y0 b)) (float (the fixnum (%sum8 sums (* b 8))) 1f0)))))))

(defun q2k-dot-avx2 (w nb qs qo yd bsums)
  "The Q2_K row of NB (at most 16) blocks at system-area pointer W against
   Q8_K activations QS (from QO), their block scales YD and their sums of 16
   BSUMS (from QO/256 and QO/16)."
  (declare (type sb-sys:system-area-pointer w) (type (integer 0 16) nb) (type fixnum qo)
           (type i8vec qs) (type f32vec yd) (type i32vec bsums) (optimize speed (safety 0)))
  (let ((shuf +q2k-scale-shuffle+) (m3 (avx:u8.32 3)) (m4 (avx:u8.32 15))
        (sums (make-array 128 :element-type '(signed-byte 32)))
        (mins (make-array 16 :element-type 'fixnum)))
    (declare (type (simple-array (unsigned-byte 8) (128)) shuf) (dynamic-extent sums mins))
    (dotimes (b nb)
      (let* ((bo (* b 84)) (q8 (+ qo (* b 256))) (sumi (avx:s32.8 0)) (msum 0) (s0 (+ (floor qo 16) (* b 16))))
        (declare (type fixnum bo q8 msum s0) (type avx:s32.8 sumi))
        ;; the mins against the activations' sums of 16, in the integers
        (dotimes (j 16)
          (incf msum (the fixnum (* (ash (sb-sys:sap-ref-8 w (+ bo j)) -4) (aref bsums (+ s0 j))))))
        (setf (aref mins b) msum)
        (let* ((sc (avx:u8.32-and (avx:u8.32-sap-ref w bo) m4)) ; the scales' low nibbles (16 of 32 bytes)
               (all (avx:s16.16-from-s8.16 (avx:s8.16-from-s8.32 (avx:s8.32! sc) 0))))
          (dotimes (j 2)
            (let* ((half (avx:s16.16! (if (zerop j) (avx:u64.4-permute128 (avx:u64.4! all) (avx:u64.4! all) #x00) (avx:u64.4-permute128 (avx:u64.4! all) (avx:u64.4! all) #x11))))
                   (q2bits (avx:u8.32-sap-ref w (+ bo 16 (* j 32))))
                   (qj (+ q8 (* j 128))))
              (declare (type fixnum qj))
              (macrolet ((part (k)
                           `(avx:s32.8! (avx:s16.16-madd
                             (avx:s16.16! (avx:u8.32-shuffle (avx:u8.32! half) (avx:u8.32-aref shuf (* ,k 32))))
                             (avx:s16.16-maddubs (avx:u8.32-and (avx:u8.32! (avx:u32.8-shiftr (avx:u32.8! q2bits) (* 2 ,k))) m3)
                                               (avx:s8.32-aref qs (+ qj (* ,k 32))))))))
                (setf sumi (avx::two-arg-s32.8+ sumi (avx::two-arg-s32.8+ (avx::two-arg-s32.8+ (part 0) (part 1))
                                                                          (avx::two-arg-s32.8+ (part 2) (part 3)))))))))
        (setf (avx:s32.8-aref sums (* b 8)) sumi)))
    (avx:vzeroupper)
    (let ((total 0f0) (y0 (floor qo 256)))
      (declare (type single-float total) (type fixnum y0))
      (dotimes (b nb total)
        (let ((yb (aref yd (+ y0 b))) (bo (* b 84)))
          (incf total (- (* yb (f16-sap w (+ bo 80)) (float (the fixnum (%sum8 sums (* b 8))) 1f0))
                         (* yb (f16-sap w (+ bo 82)) (float (the fixnum (aref mins b)) 1f0)))))))))
