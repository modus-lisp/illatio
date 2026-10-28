;;;; tensor.lisp — weights in memory, and the one product that matters.
;;;
;;; Almost all of a forward pass is W·x for a quantized W.  We do it the way
;;; ggml's CPU backend does: quantize x to Q8_0 too, take integer dot products
;;; per 32-element block, and scale each by d_w·d_x.  That is the fast path
;;; (fixnum arithmetic, no per-element dequantize) and it is also what keeps
;;; our logits next to llama.cpp's: the rounding of x into int8 and of its
;;; scale into fp16 is part of the result, not noise around it.

(in-package #:illatio)

(deftype f32vec () '(simple-array single-float (*)))
(deftype i8vec () '(simple-array (signed-byte 8) (*)))

(defun f32vec (n) (make-array n :element-type 'single-float :initial-element 0f0))

;;; A Q8_0 matrix: ROWS rows of COLS weights, row-major, each row COLS/32
;;; blocks.  QUANTS holds the int8s, SCALES the per-block fp16 scales already
;;; widened to single-float (widening is exact, so nothing is lost).
(defstruct (q8mat (:constructor %make-q8mat))
  (rows 0 :type fixnum)
  (cols 0 :type fixnum)
  (quants (make-array 0 :element-type '(signed-byte 8)) :type i8vec)
  (scales (f32vec 0) :type f32vec)
  ;; the same scales regrouped four rows at a time, for the NEON kernel:
  ;; row 4g+k's scale for block b is at (g*blocks + b)*4 + k
  (scales4 (f32vec 0) :type f32vec))

(defun regroup-scales (d rows nb)
  "D's per-row block scales, regrouped so a block's four row scales are adjacent."
  (declare (type f32vec d) (type fixnum rows nb))
  (let ((out (f32vec (* rows nb))))
    (dotimes (g (floor rows 4) out)
      (dotimes (b nb)
        (dotimes (k 4)
          (setf (aref out (+ (* 4 (+ (* g nb) b)) k)) (aref d (+ (* (+ (* 4 g) k) nb) b))))))))

;;; --- loading ----------------------------------------------------------------

(defun load-f32 (g name)
  "Tensor NAME as a flat single-float vector (F32 or F16 on disk)."
  (let* ((ti (gguf-tensor g name))
         (n (tensor-elements ti))
         (v (f32vec n)))
    (ecase (tensor-info-type ti)
      (:f32 (call-with-tensor-bytes
             g ti (lambda (buf start end)
                    (loop for i from 0 below end by 4
                          for k from (floor start 4)
                          do (setf (aref v k)
                                   (bits-single (logior (aref buf i) (ash (aref buf (+ i 1)) 8)
                                                        (ash (aref buf (+ i 2)) 16)
                                                        (ash (aref buf (+ i 3)) 24))))))))
      (:f16 (call-with-tensor-bytes
             g ti (lambda (buf start end)
                    (loop for i from 0 below end by 2
                          for k from (floor start 2)
                          do (setf (aref v k)
                                   (half-to-single (logior (aref buf i) (ash (aref buf (+ i 1)) 8))))))))
      (:q8_0 (let ((m (load-q8mat g name)))
               (dotimes (i n)
                 (setf (aref v i) (* (aref (q8mat-scales m) (floor i 32))
                                     (aref (q8mat-quants m) i)))))))
    v))

(defun load-q8mat (g name)
  (let* ((ti (gguf-tensor g name))
         (dims (tensor-info-dims ti))
         (cols (first dims))
         (rows (reduce #'* (rest dims)))
         (n (* rows cols))
         (q (make-array n :element-type '(signed-byte 8)))
         (d (f32vec (floor n 32))))
    (unless (eq (tensor-info-type ti) :q8_0)
      (error "~A is ~A; only Q8_0 matrices are supported so far" name (tensor-info-type ti)))
    ;; 34-byte blocks: fp16 scale, then 32 int8.  The chunk size is a whole
    ;; number of blocks so no block straddles two reads.
    (call-with-tensor-bytes
     g ti
     (lambda (buf start end)
       (declare (type (simple-array (unsigned-byte 8) (*)) buf) (type fixnum start end)
                (optimize speed (safety 0)))
       (loop for off of-type fixnum from 0 below end by 34
             for b of-type fixnum from (floor start 34)
             do (setf (aref d b) (half-to-single (logior (aref buf off) (ash (aref buf (1+ off)) 8))))
                (loop for j of-type fixnum from 0 below 32
                      for x = (aref buf (+ off 2 j))
                      do (setf (aref q (+ (* b 32) j)) (if (> x 127) (- x 256) x)))))
     :chunk (* 34 32768))
    (%make-q8mat :rows rows :cols cols :quants q :scales d
                 :scales4 (if (zerop (mod rows 4)) (regroup-scales d rows (floor cols 32)) (f32vec 0)))))

;;; --- activations ------------------------------------------------------------

;;; A vector quantized for a product: what ggml's quantize_row_q8_0 makes.
(defstruct (q8vec (:constructor %make-q8vec))
  (quants (make-array 0 :element-type '(signed-byte 8)) :type i8vec)
  (scales (f32vec 0) :type f32vec))

(defun make-q8vec (n)
  (%make-q8vec :quants (make-array n :element-type '(signed-byte 8))
               :scales (f32vec (floor n 32))))

(defun quantize-q8! (x out &optional (xo 0) (n (length x)) (qo 0) (so 0))
  "Quantize X[xo..xo+n) into OUT's quants at QO and scales at SO, as ggml does
   on ARM: d = amax/127, rounded to fp16 for storage; q = round-half-even(x/d)
   using the *unrounded* d, as vcvtnq does."
  (declare (type f32vec x) (type fixnum xo n qo so) (optimize speed (safety 0)))
  (let ((q (q8vec-quants out)) (s (q8vec-scales out)))
    (dotimes (b (floor n 32))
      (let* ((base (+ xo (* b 32)))
             (qbase (+ qo (* b 32)))
             (amax (loop for j of-type fixnum from base below (+ base 32)
                         maximize (abs (aref x j)) of-type single-float))
             (d (/ amax 127f0))
             (id (if (zerop d) 0f0 (/ 1f0 d))))
        (declare (type single-float amax d id) (type fixnum base qbase))
        (setf (aref s (+ so b)) (half-to-single (single-to-half d)))
        (loop for j of-type fixnum from 0 below 32
              do (setf (aref q (+ qbase j)) (the (signed-byte 8) (round (* (aref x (+ base j)) id)))))))
    out))

;;; --- the product ------------------------------------------------------------

;;; Q8-ROW-DOT-AT is the portable kernel and the oracle.  With SDOT
;;; (neon-sbcl.lisp), four rows at a time go through one NEON loop that gives
;;; each row the same bits.  DOT-ROWS fills OUT[oo..oo+3] with rows 4g..4g+3
;;; against the activations at XO (scales at XSO), by whichever kernel this
;;; machine has; binding *NEON-DOT* to NIL forces the portable one.

(declaim (inline neon-active-p))
(defun neon-active-p ()
  #+(and sbcl arm64) (neon-dot-p)
  #-(and sbcl arm64) nil)

(defmacro with-dot-rows ((dot-rows w xq xs out) &body body)
  "Run BODY with (DOT-ROWS g xo xso oo) bound, for matrix W against the
   quantized activations XQ/XS, writing OUT."
  (let ((portable
          `(let ((wq (q8mat-quants ,w)) (ws (q8mat-scales ,w)) (cols (q8mat-cols ,w)))
             (declare (type i8vec wq) (type f32vec ws) (type fixnum cols))
             (flet ((,dot-rows (g xo xso oo)
                      (declare (type fixnum g xo xso oo))
                      (dotimes (k 4)
                        (setf (aref ,out (+ oo k))
                              (q8-row-dot-at wq ws (+ (* 4 g) k) cols ,xq ,xs xo xso)))))
               (declare (inline ,dot-rows))
               ,@body))))
    #-(and sbcl arm64) portable
    #+(and sbcl arm64)
    `(if (neon-active-p)
         (let ((wq (q8mat-quants ,w)) (ws4 (q8mat-scales4 ,w)) (cols (q8mat-cols ,w)))
           (declare (type i8vec wq) (type f32vec ws4) (type fixnum cols))
           (sb-sys:with-pinned-objects (wq ws4 ,xq ,xs ,out)
             (let ((wsap (sb-sys:vector-sap wq)) (wssap (sb-sys:vector-sap ws4))
                   (xsap (sb-sys:vector-sap ,xq)) (xssap (sb-sys:vector-sap ,xs))
                   (osap (sb-sys:vector-sap ,out))
                   (nb (floor cols 32)))
               (declare (type (unsigned-byte 32) nb))
               (flet ((,dot-rows (g xo xso oo)
                        (declare (type fixnum g xo xso oo))
                        (%q8-dot-4rows (sb-sys:sap+ wsap (* 4 g cols)) cols
                                       (sb-sys:sap+ xsap xo)
                                       (sb-sys:sap+ wssap (* 16 g nb))
                                       (sb-sys:sap+ xssap (* 4 xso))
                                       nb
                                       (sb-sys:sap+ osap (* 4 oo)))))
                 (declare (inline ,dot-rows))
                 ,@body))))
         ,portable)))

(declaim (inline q8-row-dot-at))
(defun q8-row-dot-at (wq ws row cols xq xs xo xso)
  "Q8-ROW-DOT against the activations at XO (scales at XSO) in XQ/XS."
  (declare (type i8vec wq xq) (type f32vec ws xs) (type fixnum row cols xo xso)
           (optimize speed (safety 0)))
  (let ((sum 0f0)
        (wbase (* row cols))
        (sbase (* row (floor cols 32))))
    (declare (type single-float sum) (type fixnum wbase sbase))
    (dotimes (b (floor cols 32) sum)
      (let ((acc 0)
            (wo (+ wbase (* b 32)))
            (xb (+ xo (* b 32))))
        (declare (type (signed-byte 32) acc) (type fixnum wo xb))
        (dotimes (j 32)
          (setf acc (+ acc (the (signed-byte 16) (* (aref wq (+ wo j)) (aref xq (+ xb j)))))))
        (incf sum (* (float acc 1f0) (* (aref ws (+ sbase b)) (aref xs (+ xso b)))))))))

(defun matvec! (pool w xv out &optional bias)
  "OUT = W·x (+ BIAS), x already quantized in XV; groups of four rows split
   across POOL."
  (declare (type q8mat w) (type q8vec xv) (type f32vec out))
  (let ((xq (q8vec-quants xv)) (xs (q8vec-scales xv)) (rows (q8mat-rows w)))
    (declare (type i8vec xq) (type f32vec xs) (type fixnum rows))
    (assert (zerop (mod rows 4)))
    (with-dot-rows (dot-rows w xq xs out)
      (parallel-for (g (floor rows 4) pool)
        (dot-rows g 0 0 (* 4 g))))
    (when bias
      (let ((bias bias))
        (declare (type f32vec bias) (optimize speed (safety 0)))
        (dotimes (r rows) (setf (aref out r) (+ (aref out r) (aref bias r))))))
    out))

(defun q8mat-row! (w row out)
  "Dequantize row ROW of W into OUT: ggml's get_rows, for embeddings."
  (declare (type q8mat w) (type f32vec out) (type fixnum row))
  (let* ((cols (q8mat-cols w))
         (q (q8mat-quants w)) (s (q8mat-scales w))
         (base (* row cols)))
    (dotimes (j cols out)
      (setf (aref out j) (* (aref s (floor (+ base j) 32)) (aref q (+ base j)))))))

;;; --- float kernels for attention --------------------------------------------
;;; F32-DOT16 defines the order a dot product is summed in: sixteen strided
;;; partial sums (element i goes to sum i mod 16), combined as
;;; ((s_j + s_4+j) + (s_8+j + s_12+j)) for j = 0..3, then (c0+c1)+(c2+c3).  It
;;; is the order four NEON accumulators produce, written out, so %F32-DOT16
;;; (neon-sbcl.lisp) and this give the same bits.  N is a multiple of 16.

(defun f32-dot16 (a ao b bo n)
  (declare (type f32vec a b) (type fixnum ao bo n) (optimize speed (safety 0)))
  (let ((s (make-array 16 :element-type 'single-float :initial-element 0f0)))
    (declare (dynamic-extent s))
    (loop for i of-type fixnum from 0 below n
          do (let ((l (logand i 15)))
               (setf (aref s l) (+ (aref s l) (* (aref a (+ ao i)) (aref b (+ bo i)))))))
    (flet ((c (j) (+ (+ (aref s j) (aref s (+ 4 j))) (+ (aref s (+ 8 j)) (aref s (+ 12 j))))))
      (+ (+ (c 0) (c 1)) (+ (c 2) (c 3))))))

(defun f32-axpy (y yo x xo w n)
  "Y[yo..] += W * X[xo..] over N elements: lane-parallel, so any width gives
   these bits."
  (declare (type f32vec y x) (type fixnum yo xo n) (type single-float w)
           (optimize speed (safety 0)))
  (dotimes (i n y)
    (setf (aref y (+ yo i)) (+ (aref y (+ yo i)) (* w (aref x (+ xo i)))))))
