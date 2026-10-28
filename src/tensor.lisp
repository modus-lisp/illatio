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
  (scales (f32vec 0) :type f32vec))

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
    (%make-q8mat :rows rows :cols cols :quants q :scales d)))

;;; --- activations ------------------------------------------------------------

;;; A vector quantized for a product: what ggml's quantize_row_q8_0 makes.
(defstruct (q8vec (:constructor %make-q8vec))
  (quants (make-array 0 :element-type '(signed-byte 8)) :type i8vec)
  (scales (f32vec 0) :type f32vec))

(defun make-q8vec (n)
  (%make-q8vec :quants (make-array n :element-type '(signed-byte 8))
               :scales (f32vec (floor n 32))))

(defun quantize-q8! (x out)
  "Quantize X into OUT as ggml does on ARM: d = amax/127, rounded to fp16 for
   storage; q = round-half-even(x/d) using the *unrounded* d, as vcvtnq does."
  (declare (type f32vec x) (optimize speed (safety 0)))
  (let ((q (q8vec-quants out)) (s (q8vec-scales out)))
    (dotimes (b (floor (length x) 32))
      (let* ((base (* b 32))
             (amax (loop for j of-type fixnum from base below (+ base 32)
                         maximize (abs (aref x j)) of-type single-float))
             (d (/ amax 127f0))
             (id (if (zerop d) 0f0 (/ 1f0 d))))
        (declare (type single-float amax d id))
        (setf (aref s b) (half-to-single (single-to-half d)))
        (loop for j of-type fixnum from base below (+ base 32)
              do (setf (aref q j) (the (signed-byte 8) (round (* (aref x j) id)))))))
    out))

;;; --- the product ------------------------------------------------------------

(declaim (inline q8-row-dot))
(defun q8-row-dot (wq ws row cols xq xs)
  (declare (type i8vec wq xq) (type f32vec ws xs) (type fixnum row cols)
           (optimize speed (safety 0)))
  (let ((sum 0f0)
        (wbase (* row cols))
        (sbase (* row (floor cols 32))))
    (declare (type single-float sum) (type fixnum wbase sbase))
    (dotimes (b (floor cols 32) sum)
      (let ((acc 0)
            (wo (+ wbase (* b 32)))
            (xo (* b 32)))
        (declare (type (signed-byte 32) acc) (type fixnum wo xo))
        (dotimes (j 32)
          (setf acc (+ acc (the (signed-byte 16) (* (aref wq (+ wo j)) (aref xq (+ xo j)))))))
        (incf sum (* (float acc 1f0) (* (aref ws (+ sbase b)) (aref xs b))))))))

(defun matvec! (pool w xv out &optional bias)
  "OUT = W·x (+ BIAS), x already quantized in XV, rows split across POOL."
  (declare (type q8mat w) (type q8vec xv) (type f32vec out))
  (let ((wq (q8mat-quants w)) (ws (q8mat-scales w)) (cols (q8mat-cols w))
        (xq (q8vec-quants xv)) (xs (q8vec-scales xv)))
    (declare (type fixnum cols))
    (if bias
        (let ((bias bias))
          (declare (type f32vec bias))
          (parallel-for (r (q8mat-rows w) pool)
            (setf (aref out r) (+ (q8-row-dot wq ws r cols xq xs) (aref bias r)))))
        (parallel-for (r (q8mat-rows w) pool)
          (setf (aref out r) (q8-row-dot wq ws r cols xq xs)))))
  out)

(defun q8mat-row! (w row out)
  "Dequantize row ROW of W into OUT: ggml's get_rows, for embeddings."
  (declare (type q8mat w) (type f32vec out) (type fixnum row))
  (let* ((cols (q8mat-cols w))
         (q (q8mat-quants w)) (s (q8mat-scales w))
         (base (* row cols)))
    (dotimes (j cols out)
      (setf (aref out j) (* (aref s (floor (+ base j) 32)) (aref q (+ base j)))))))
