;;;; kquant.lisp — Q4_K and Q6_K weights against Q8_K activations.
;;;
;;; The two K-quant formats a Q4_K_M file is made of.  A super-block is 256
;;; weights:
;;;
;;;   Q4_K  144 bytes: fp16 d, fp16 dmin, 12 bytes of 6-bit scales and mins
;;;         for eight 32-weight groups, 128 bytes of 4-bit quants.  A weight
;;;         is d*scale_g * q - dmin*min_g.
;;;   Q6_K  210 bytes: 128 bytes of low 4 bits, 64 of high 2 bits, sixteen
;;;         int8 scales for 16-weight groups, fp16 d.  A weight is
;;;         d*scale_g * (q - 32).
;;;
;;; Their products are taken, as ggml takes them, against activations in
;;; Q8_K: 256 int8s per block with a float scale d = max/-127 (the sign of
;;; the largest-magnitude value folded in) and the sums of each 16 quants.
;;; Every integer in a super-block product is exact, and the floats combine
;;; in the order of ggml's NEON kernels (ggml-cpu/arch/arm/quants.c,
;;; ggml_vec_dot_q{4,6}_K_q8_K):
;;;
;;;   Q4_K  sum -= (y.d*dmin) * sum_g(min_g * bsum32_g)
;;;         sum += (y.d*d)    * sum_g(scale_g * dot32_g)
;;;   Q6_K  sum += (d*y.d) * (sum_g(scale_g * dot16_g) - 32*sum_g(scale_g * bsum16_g))
;;;
;;; with the 6-bit quants taken as 0..63 and the -32 folded in at the end, as
;;; the NEON kernel does.  At load a matrix is unpacked into flat arrays of
;;; the same size: the quant bytes as they are, the 6-bit scales and mins as
;;; one byte each, the fp16 scales widened to floats (exactly).  A Q6_K
;;; block's bits are stored in the order the NEON kernel reads them, half by
;;; half: 32 bytes of high bits then the 64 of low bits for weights 0-127,
;;; then the same for 128-255 (one stream: the kernel has no register left
;;; for a second).

(in-package #:illatio)

(defconstant +qk-k+ 256)

(deftype i32vec () '(simple-array (signed-byte 32) (*)))
(deftype u8vec () '(simple-array (unsigned-byte 8) (*)))

(defun make-u8vec (n &key static)
  #+modus (if static (modus:make-static-vector n) (make-array n :element-type '(unsigned-byte 8) :initial-element 0))
  #-modus (progn static (make-array n :element-type '(unsigned-byte 8) :initial-element 0)))

(declaim (inline s8))
(defun s8 (b)
  "The byte B read as two's complement."
  (declare (type (unsigned-byte 8) b))
  (if (> b 127) (- b 256) b))

;;; --- weights ------------------------------------------------------------------

(defstruct (kmat (:constructor %make-kmat))
  (type :q4_k)                          ; :q4_k or :q6_k
  (rows 0 :type fixnum)
  (cols 0 :type fixnum)
  (nb 0 :type fixnum)                   ; super-blocks per row
  (qs (make-u8vec 0) :type u8vec)       ; Q4_K: 128 quant bytes a block; Q6_K: 192 (see above)
  (sc (make-u8vec 0) :type u8vec)       ; Q4_K: 8 scales then 8 mins; Q6_K: 16 int8 scales
  (dd (f32vec 0) :type f32vec))         ; Q4_K: d, dmin per block; Q6_K: d

(defun kquant-type-p (type) (member type '(:q4_k :q6_k)))

(defun scale-min-k4 (j bytes o)
  "ggml's get_scale_min_k4: group J's 6-bit scale and min from the 12
   bytes at O in BYTES."
  (declare (type (simple-array (unsigned-byte 8) (*)) bytes) (type fixnum j o))
  (flet ((q (i) (aref bytes (+ o i))))
    (if (< j 4)
        (values (logand (q j) 63) (logand (q (+ j 4)) 63))
        (values (logior (logand (q (+ j 4)) #xF) (ash (ash (q (- j 4)) -6) 4))
                (logior (ash (q (+ j 4)) -4) (ash (ash (q j) -6) 4))))))

(defun load-kmat (g name)
  "Tensor NAME, Q4_K or Q6_K on disk, unpacked as a KMAT."
  (let* ((ti (gguf-tensor g name))
         (type (tensor-info-type ti))
         (dims (tensor-info-dims ti))
         (cols (first dims))
         (rows (reduce #'* (rest dims)))
         (nb (floor cols +qk-k+))
         (blocks (* rows nb))
         (bytes (ecase type (:q4_k 144) (:q6_k 210)))
         (qs (make-u8vec (* blocks (if (eq type :q6_k) 192 128)) :static t))
         (sc (make-u8vec (* blocks 16) :static t))
         (dd (f32vec (* blocks (if (eq type :q4_k) 2 1)) :static t)))
    (unless (zerop (mod cols +qk-k+))
      (error "~A: ~D columns is not a whole number of ~D-weight super-blocks" name cols +qk-k+))
    (call-with-tensor-bytes
     g ti
     (lambda (buf start end)
       (declare (type (simple-array (unsigned-byte 8) (*)) buf) (type fixnum start end))
       (loop for off of-type fixnum from 0 below end by bytes
             for b of-type fixnum from (floor start bytes)
             do (flet ((half (o) (half-to-single (logior (aref buf (+ off o)) (ash (aref buf (+ off o 1)) 8)))))
                  (ecase type
                    (:q4_k
                     (setf (aref dd (* 2 b)) (half 0)
                           (aref dd (1+ (* 2 b))) (half 2))
                     (dotimes (j 8)
                       (multiple-value-bind (s m) (scale-min-k4 j buf (+ off 4))
                         (setf (aref sc (+ (* b 16) j)) s
                               (aref sc (+ (* b 16) 8 j)) m)))
                     (replace qs buf :start1 (* b 128) :start2 (+ off 16) :end2 (+ off 144)))
                    (:q6_k
                     ;; on disk: ql[128] qh[64]; here, per half c: qh[32c..] ql[64c..]
                     (dotimes (c 2)
                       (replace qs buf :start1 (+ (* b 192) (* c 96)) :start2 (+ off 128 (* c 32)) :end2 (+ off 160 (* c 32)))
                       (replace qs buf :start1 (+ (* b 192) (* c 96) 32) :start2 (+ off (* c 64)) :end2 (+ off 64 (* c 64))))
                     (replace sc buf :start1 (* b 16) :start2 (+ off 192) :end2 (+ off 208))
                     (setf (aref dd b) (half 208)))))))
     :chunk (* bytes 4096))
    (%make-kmat :type type :rows rows :cols cols :nb nb :qs qs :sc sc :dd dd)))

;;; --- activations ----------------------------------------------------------------

(defstruct (q8kvec (:constructor %make-q8kvec))
  (qs (make-i8vec 0) :type i8vec)                   ; int8 quants, as bytes
  (d (f32vec 0) :type f32vec)                       ; per 256
  (bsums (make-array 0 :element-type '(signed-byte 32)) :type i32vec)  ; per 16
  (bsum32 (make-array 0 :element-type '(signed-byte 32)) :type i32vec)) ; per 32: pairs of the above

(defun make-q8kvec (n)
  "Room for N activations (rounded up to whole blocks)."
  (let ((n (* +qk-k+ (ceiling n +qk-k+))))
    (%make-q8kvec :qs (make-i8vec n) :d (f32vec (floor n +qk-k+))
                  :bsums (make-array (floor n 16) :element-type '(signed-byte 32))
                  :bsum32 (make-array (floor n 32) :element-type '(signed-byte 32)))))

(defun quantize-q8k! (x out &optional (xo 0) (n (length x)) (qo 0))
  "X[xo..xo+n) into OUT's quants at QO (blocks at QO/256, sums at QO/16), as
   ggml's quantize_row_q8_K_ref: the largest-magnitude value MAX sets
   iscale = -127/max; q = min(127, round-half-even(iscale*x)); d = 1/iscale.
   A block of zeros has d = 0."
  (declare (type f32vec x) (type fixnum xo n qo) (optimize speed (safety 0)))
  (let ((q (q8kvec-qs out)) (ds (q8kvec-d out)) (bs (q8kvec-bsums out)) (b32 (q8kvec-bsum32 out)))
    (declare (type i8vec q) (type f32vec ds) (type i32vec bs b32))
    (dotimes (b (floor n +qk-k+) out)
      (let* ((base (+ xo (* b +qk-k+)))
             (qbase (+ qo (* b +qk-k+)))
             (sb (floor qbase +qk-k+))
             (mx 0f0) (amax 0f0))
        (declare (type fixnum base qbase sb) (type single-float mx amax))
        (dotimes (j +qk-k+)
          (let* ((v (aref x (+ base j))) (a (abs v)))
            (when (> a amax) (setf amax a mx v))))
        (if (zerop amax)
            (progn (setf (aref ds sb) 0f0)
                   (fill q 0 :start qbase :end (+ qbase +qk-k+))
                   (fill bs 0 :start (floor qbase 16) :end (+ (floor qbase 16) 16)))
            (let ((iscale (/ -127f0 mx)))
              (declare (type single-float iscale))
              (dotimes (g 16)
                (let ((sum 0))
                  (declare (type fixnum sum))
                  (dotimes (k 16)
                    (let* ((j (+ (* g 16) k))
                           (v (min 127 (the (signed-byte 32) (round (* iscale (aref x (+ base j))))))))
                      (declare (type fixnum j v))
                      (setf (aref q (+ qbase j)) (i8-byte v))
                      (incf sum v)))
                  (setf (aref bs (+ (floor qbase 16) g)) sum)))
              (setf (aref ds sb) (/ 1f0 iscale))))
        (let ((g0 (floor qbase 16)))
          (dotimes (g 8)
            (setf (aref b32 (+ (floor qbase 32) g))
                  (+ (aref bs (+ g0 (* 2 g))) (aref bs (+ g0 (* 2 g) 1))))))))))

;;; --- the products, portable ----------------------------------------------------

(defun q4k-row-dot (w row xv xo)
  "Row ROW of the Q4_K matrix W against the Q8_K activations in XV at
   element offset XO."
  (declare (type kmat w) (type q8kvec xv) (type fixnum row xo) (optimize speed (safety 0)))
  (let* ((nb (kmat-nb w)) (qs (kmat-qs w)) (sc (kmat-sc w)) (dd (kmat-dd w))
         (yq (q8kvec-qs xv)) (yd (q8kvec-d xv)) (ybs (q8kvec-bsums xv))
         (sum 0f0))
    (declare (type fixnum nb) (type u8vec qs sc) (type i8vec yq) (type f32vec dd yd) (type i32vec ybs)
             (type single-float sum))
    (dotimes (i nb sum)
      (let* ((blk (+ (* row nb) i))
             (yb (+ (floor xo +qk-k+) i))
             (ydi (aref yd yb))
             (d (* ydi (aref dd (* 2 blk))))
             (dmin (* ydi (aref dd (1+ (* 2 blk)))))
             (imin 0) (isum 0))
        (declare (type fixnum blk yb imin isum) (type single-float ydi d dmin))
        (dotimes (g 8)
          (incf imin (* (aref sc (+ (* blk 16) 8 g))
                        (+ (aref ybs (+ (* yb 16) (* 2 g))) (aref ybs (+ (* yb 16) (* 2 g) 1))))))
        (setf sum (- sum (* dmin (float imin 1f0))))
        (dotimes (k 4)                  ; 64 weights: groups 2k (low nibbles), 2k+1 (high)
          (let ((lo 0) (hi 0) (qb (+ (* blk 128) (* k 32))) (yo (+ (* yb +qk-k+) (* k 64))))
            (declare (type fixnum lo hi qb yo))
            (dotimes (l 32)
              (let ((byte (aref qs (+ qb l))))
                (incf lo (* (logand byte #xF) (i8-ref yq (+ yo l))))
                (incf hi (* (ash byte -4) (i8-ref yq (+ yo 32 l))))))
            (incf isum (+ (* lo (aref sc (+ (* blk 16) (* 2 k))))
                          (* hi (aref sc (+ (* blk 16) (* 2 k) 1)))))))
        (setf sum (+ sum (* d (float isum 1f0))))))))

(defun q6k-row-dot (w row xv xo)
  "Row ROW of the Q6_K matrix W against the Q8_K activations in XV at
   element offset XO."
  (declare (type kmat w) (type q8kvec xv) (type fixnum row xo) (optimize speed (safety 0)))
  (let* ((nb (kmat-nb w)) (qhl (kmat-qs w)) (sc (kmat-sc w)) (dd (kmat-dd w))
         (yq (q8kvec-qs xv)) (yd (q8kvec-d xv)) (ybs (q8kvec-bsums xv))
         (sum 0f0))
    (declare (type fixnum nb) (type u8vec qhl sc) (type i8vec yq) (type f32vec dd yd) (type i32vec ybs)
             (type single-float sum))
    (dotimes (i nb sum)
      (let* ((blk (+ (* row nb) i))
             (yb (+ (floor xo +qk-k+) i))
             (imins 0) (isum 0))
        (declare (type fixnum blk yb imins isum))
        (dotimes (g 16)
          (incf imins (* (aref ybs (+ (* yb 16) g)) (s8 (aref sc (+ (* blk 16) g))))))
        (dotimes (c 2)                  ; two 128-weight halves
          (dotimes (part 4)             ; 32 weights each: low/high nibble, qh shift 2*part
            (let ((lb (+ (* blk 192) (* c 96) 32 (if (oddp part) 32 0)))
                  (hb (+ (* blk 192) (* c 96)))
                  (shift (* 2 part))
                  (yo (+ (* yb +qk-k+) (* c 128) (* part 32))))
              (declare (type fixnum lb hb yo) (type (integer 0 6) shift))
              (dotimes (half 2)         ; the two 16-weight groups, each with its scale
                (let ((dot 0))
                  (declare (type fixnum dot))
                  (dotimes (l16 16)
                    (let* ((l (+ (* half 16) l16))
                           (low (if (< part 2) (logand (aref qhl (+ lb l)) #xF) (ash (aref qhl (+ lb l)) -4)))
                           (q6 (logior low (ash (logand (ash (aref qhl (+ hb l)) (- shift)) 3) 4))))
                      (declare (type fixnum l low q6))
                      (incf dot (* q6 (i8-ref yq (+ yo l))))))
                  (incf isum (* dot (s8 (aref sc (+ (* blk 16) (* c 8) (* part 2) half))))))))))
        (setf sum (+ sum (* (* (aref dd blk) (aref yd yb)) (float (- isum (* 32 imins)) 1f0))))))))

#+(and sbcl arm64)
(defun kmat-row-dot-neon (w row xv xo)
  "KMAT-ROW-DOT by %Q4K-DOT / %Q6K-DOT (neon-sbcl.lisp): the same bits."
  (declare (type kmat w) (type q8kvec xv) (type fixnum row xo) (optimize speed (safety 0)))
  (let* ((nb (kmat-nb w)) (b0 (* row nb)) (y0 (floor xo +qk-k+))
         (qs (kmat-qs w)) (sc (kmat-sc w)) (dd (kmat-dd w))
         (yq (q8kvec-qs xv)) (yd (q8kvec-d xv)) (bs (q8kvec-bsums xv)) (b32 (q8kvec-bsum32 xv)))
    (declare (type fixnum nb b0 y0))
    (let ((desc (make-array 6 :element-type '(unsigned-byte 64))))
      (declare (dynamic-extent desc))
      (sb-sys:with-pinned-objects (qs sc dd yq yd bs b32 desc)
        (macrolet ((at (v off) `(sb-sys:sap-int (sb-sys:sap+ (sb-sys:vector-sap ,v) ,off))))
          (let ((q4 (eq (kmat-type w) :q4_k)))
            (setf (aref desc 0) (at qs (* b0 (if q4 128 192)))
                  (aref desc 1) (at sc (* b0 16))
                  (aref desc 2) (at dd (* b0 (if q4 8 4)))
                  (aref desc 3) (at yq xo)
                  (aref desc 4) (at yd (* y0 4))
                  (aref desc 5) (if q4 (at b32 (* y0 32)) (at bs (* y0 64))))
            (if q4
                (%q4k-dot (sb-sys:vector-sap desc) nb)
                (%q6k-dot (sb-sys:vector-sap desc) nb))))))))

(defun kmat-row-dot (w row xv xo)
  "Row ROW of W against XV at XO: the NEON kernel when it can run (see
   NEON-ACTIVE-P), the portable one -- which defines the bits -- otherwise."
  #+(and sbcl arm64) (when (neon-active-p) (return-from kmat-row-dot (kmat-row-dot-neon w row xv xo)))
  #+modus (return-from kmat-row-dot
            (if (eq (kmat-type w) :q4_k)
                (%modus-q4k-row-dot (kmat-qs w) (kmat-sc w) (kmat-dd w) (q8kvec-qs xv) (q8kvec-d xv)
                                    (q8kvec-bsum32 xv) (kmat-nb w) row xo)
                (%modus-q6k-row-dot (kmat-qs w) (kmat-sc w) (kmat-dd w) (q8kvec-qs xv) (q8kvec-d xv)
                                    (q8kvec-bsums xv) (kmat-nb w) row xo)))
  (if (eq (kmat-type w) :q4_k) (q4k-row-dot w row xv xo) (q6k-row-dot w row xv xo)))

;;; --- rows as floats (the embedding) ---------------------------------------------

(defun kmat-row! (w row out)
  "Row ROW of W dequantized into OUT, as ggml's dequantize_row_q{4,6}_K."
  (declare (type kmat w) (type fixnum row) (type f32vec out))
  (let ((nb (kmat-nb w)) (qs (kmat-qs w)) (sc (kmat-sc w)) (dd (kmat-dd w)))
    (dotimes (i nb out)
      (let ((blk (+ (* row nb) i)) (y (* i +qk-k+)))
        (ecase (kmat-type w)
          (:q4_k
           (let ((d (aref dd (* 2 blk))) (mn (aref dd (1+ (* 2 blk)))))
             (dotimes (k 4)
               (let ((d1 (* d (aref sc (+ (* blk 16) (* 2 k)))))
                     (m1 (* mn (aref sc (+ (* blk 16) 8 (* 2 k)))))
                     (d2 (* d (aref sc (+ (* blk 16) (* 2 k) 1))))
                     (m2 (* mn (aref sc (+ (* blk 16) 8 (* 2 k) 1)))))
                 (dotimes (l 32)
                   (let ((byte (aref qs (+ (* blk 128) (* k 32) l))))
                     (setf (aref out (+ y (* k 64) l)) (- (* d1 (logand byte #xF)) m1)
                           (aref out (+ y (* k 64) 32 l)) (- (* d2 (ash byte -4)) m2))))))))
          (:q6_k
           (let ((d (aref dd blk)))
             (dotimes (c 2)
               (dotimes (l 32)
                 (let* ((is (floor l 16))
                        (lo0 (aref qs (+ (* blk 192) (* c 96) 32 l)))
                        (lo1 (aref qs (+ (* blk 192) (* c 96) 64 l)))
                        (h (aref qs (+ (* blk 192) (* c 96) l)))
                        (s (lambda (k) (s8 (aref sc (+ (* blk 16) (* c 8) is k))))))
                   (flet ((put (o k q) (setf (aref out (+ y (* c 128) o l)) (* (* d (funcall s k)) (- q 32)))))
                     (put 0 0 (logior (logand lo0 #xF) (ash (logand h 3) 4)))
                     (put 32 2 (logior (logand lo1 #xF) (ash (logand (ash h -2) 3) 4)))
                     (put 64 4 (logior (ash lo0 -4) (ash (logand (ash h -4) 3) 4)))
                     (put 96 6 (logior (ash lo1 -4) (ash (logand (ash h -6) 3) 4))))))))))))))

;;; --- either format ----------------------------------------------------------------
;;; A model's matrices are Q8_0 (against Q8_0 activations) or K-quants (against
;;; Q8_K).  A QACT holds a vector quantized in whichever formats the model's
;;; matrices need, and QMATVEC! picks the one its matrix wants.

(defstruct (qact (:constructor %make-qact)) q8 q8k)

(defun make-qact (n kinds)
  "Room for N activations in each of KINDS (:q8, :q8k)."
  (%make-qact :q8 (and (member :q8 kinds) (make-q8vec n))
              :q8k (and (member :q8k kinds) (make-q8kvec n))))

(defun quantize-act! (x act)
  (when (qact-q8 act) (quantize-q8! x (qact-q8 act)))
  (when (qact-q8k act) (quantize-q8k! x (qact-q8k act)))
  act)

(defun quantize-act-par! (pool x act)
  "QUANTIZE-ACT! across POOL, a range of blocks each: a block is quantized
   on its own, so the bits are the serial ones."
  (let ((n (length x)))
    (pool-run pool
              (lambda (i c)
                (when (qact-q8 act)
                  (multiple-value-bind (s e) (share (floor n 32) i c)
                    (when (< s e)
                      (quantize-q8! x (qact-q8 act) (* s 32) (* (- e s) 32) (* s 32) s))))
                (when (qact-q8k act)
                  (multiple-value-bind (s e) (share (floor n +qk-k+) i c)
                    (when (< s e)
                      (quantize-q8k! x (qact-q8k act) (* s +qk-k+) (* (- e s) +qk-k+) (* s +qk-k+)))))))
    act))

(defun matrix-act-kind (w)
  (if (kmat-p w) :q8k :q8))

(defun matrix-rows (w) (if (kmat-p w) (kmat-rows w) (q8mat-rows w)))

(defun kmatvec! (pool w xv out &optional bias (xo 0))
  "OUT = W·x (+ BIAS) for a K-quant W, x quantized in the Q8K-VEC XV at XO."
  (declare (type kmat w) (type q8kvec xv) (type f32vec out) (type fixnum xo))
  (let ((rows (kmat-rows w)))
    (parallel-for (r rows pool)
      (setf (aref out r) (kmat-row-dot w r xv xo)))
    (when bias
      (let ((bias bias))
        (declare (type f32vec bias))
        (dotimes (r rows) (setf (aref out r) (+ (aref out r) (aref bias r))))))
    out))

(defun qmatvec! (pool w act out &optional bias)
  "OUT = W·x (+ BIAS), x quantized in the QACT ACT."
  (if (kmat-p w)
      (kmatvec! pool w (qact-q8k act) out bias)
      (matvec! pool w (qact-q8 act) out bias)))

(defun embed-row! (w row out)
  "Row ROW of the embedding matrix W as floats."
  (if (kmat-p w) (kmat-row! w row out) (q8mat-row! w row out)))

(defun load-qmat (g name)
  "Tensor NAME as the matrix type its format calls for."
  (if (kquant-type-p (tensor-info-type (gguf-tensor g name)))
      (load-kmat g name)
      (load-q8mat g name)))
