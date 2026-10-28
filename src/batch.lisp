;;;; batch.lisp — prefill: many tokens per pass, three per multiply.
;;;
;;; Decoding is bound by the int8 multiply-add loop, not by memory, so a
;;; batch only pays if it does fewer instructions per product.  Portable Lisp
;;; has no SIMD, but a 64-bit word has room for three lanes: a block of 32
;;; int8 x int8 products sums to at most 32*128*128 = 2^19 in magnitude, which
;;; a 21-bit signed lane holds.  So the activations of three tokens are packed
;;; as x0 + x1*2^21 + x2*2^42, one multiply by a weight yields all three
;;; products, and 32 of them summed still separate exactly into the three
;;; block sums.  The integers are the scalar path's integers and the float
;;; accumulation runs in the same order, so a batched prefill is bit-identical
;;; to feeding the tokens one at a time.

(in-package #:illatio)

(deftype i64vec () '(simple-array (signed-byte 64) (*)))

(defconstant +lane+ 21)

;;; Activations of N tokens, quantized, packed three to a word.
;;; WORDS[g*cols + j] packs column j of tokens 3g, 3g+1, 3g+2 (absent tokens
;;; are zero); SCALES[t*(cols/32) + b] is token t's scale for block b.
(defstruct (q8batch (:constructor %make-q8batch))
  (n 0 :type fixnum)
  (cols 0 :type fixnum)
  (words (make-array 0 :element-type '(signed-byte 64)) :type i64vec)
  (scales (f32vec 0) :type f32vec)
  (row (make-q8vec 32)))                ; scratch: one token's quantization

(defun make-q8batch (max-n cols)
  (%make-q8batch :cols cols
                 :words (make-array (* (ceiling max-n 3) cols) :element-type '(signed-byte 64))
                 :scales (f32vec (* max-n (floor cols 32)))
                 :row (make-q8vec cols)))

(defun quantize-batch! (xs n out)
  "Quantize rows 0..N-1 of XS (N rows of COLS floats, row-major) into OUT,
   each row exactly as QUANTIZE-Q8! would."
  (declare (type f32vec xs) (type fixnum n) (optimize speed (safety 0)))
  (let* ((cols (q8batch-cols out))
         (nb (floor cols 32))
         (words (q8batch-words out))
         (scales (q8batch-scales out))
         (row (q8batch-row out))
         (tmp (f32vec cols)))
    (declare (type fixnum cols nb) (type i64vec words) (type f32vec scales tmp))
    (setf (q8batch-n out) n)
    (fill words 0 :end (* (ceiling n 3) cols))
    (dotimes (tk n out)
      (replace tmp xs :start2 (* tk cols) :end2 (* (1+ tk) cols))
      (quantize-q8! tmp row)
      (let ((q (q8vec-quants row))
            (base (* (floor tk 3) cols))
            (shift (* +lane+ (mod tk 3))))
        (declare (type i8vec q) (type fixnum base) (type (integer 0 42) shift))
        (replace scales (q8vec-scales row) :start1 (* tk nb))
        (dotimes (j cols)
          (setf (aref words (+ base j))
                (the (signed-byte 64) (+ (aref words (+ base j))
                                         (the (signed-byte 50) (ash (aref q j) shift))))))))))

(defmacro sext21 (v)
  "The low 21 bits of V as a signed number: offset by 2^20, mask, unoffset."
  `(the (signed-byte 21) (- (ldb (byte +lane+ 0) (+ ,v #.(ash 1 (1- +lane+)))) #.(ash 1 (1- +lane+)))))

(defmacro with-lanes (((a0 a1 a2) s) &body body)
  "Bind the three signed 21-bit lane values packed in S.  A macro, not a
   function: SBCL would box a (signed-byte 64) passed out of line."
  (let ((w (gensym)) (s1 (gensym)))
    `(let* ((,w ,s)
            (,a0 (sext21 ,w))
            (,s1 (the (signed-byte 43) (ash (- ,w ,a0) (- +lane+))))
            (,a1 (sext21 ,s1))
            (,a2 (the (signed-byte 22) (ash (- ,s1 ,a1) (- +lane+)))))
       (declare (type (signed-byte 64) ,w))
       ,@body)))

(defun matmul! (pool w xb out &optional bias)
  "OUT[t*rows + r] = (W·x_t)[r] (+ BIAS[r]) for the N tokens in XB."
  (declare (type q8mat w) (type q8batch xb) (type f32vec out))
  (let* ((rows (q8mat-rows w)) (cols (q8mat-cols w)) (nb (floor cols 32))
         (n (q8batch-n xb)) (groups (ceiling n 3))
         (wq (q8mat-quants w)) (ws (q8mat-scales w))
         (words (q8batch-words xb)) (xs (q8batch-scales xb))
         (bias (or bias (f32vec 0))) (has-bias (plusp (length bias))))
    (declare (type fixnum rows cols nb n groups) (type i8vec wq) (type f32vec ws xs bias)
             (type i64vec words))
    (parallel-for (r rows pool)
      (locally (declare (optimize speed (safety 0)))
        (let ((wbase (* r cols)) (sbase (* r nb)))
          (declare (type fixnum wbase sbase))
          (dotimes (g groups)
            (let ((s0 0f0) (s1 0f0) (s2 0f0)
                  (xbase (* g cols))
                  (t0 (* 3 g)))
              (declare (type single-float s0 s1 s2) (type fixnum xbase t0))
              (dotimes (b nb)
                (let ((acc 0) (wo (+ wbase (* b 32))) (xo (+ xbase (* b 32))))
                  (declare (type (signed-byte 64) acc) (type fixnum wo xo))
                  (dotimes (j 32)
                    (setf acc (the (signed-byte 64)
                                   (+ acc (the (signed-byte 58)
                                               (* (aref wq (+ wo j))
                                                  (the (signed-byte 50) (aref words (+ xo j)))))))))
                  (with-lanes ((a0 a1 a2) acc)
                    (let ((wd (aref ws (+ sbase b))))
                      ;; same expression, same order, as Q8-ROW-DOT
                      (incf s0 (* (float a0 1f0) (* wd (aref xs (+ (* t0 nb) b)))))
                      (when (< (+ t0 1) n)
                        (incf s1 (* (float a1 1f0) (* wd (aref xs (+ (* (+ t0 1) nb) b))))))
                      (when (< (+ t0 2) n)
                        (incf s2 (* (float a2 1f0) (* wd (aref xs (+ (* (+ t0 2) nb) b))))))))))
              (let ((bb (if has-bias (aref bias r) 0f0)))
                (setf (aref out (+ (* t0 rows) r)) (if has-bias (+ s0 bb) s0))
                (when (< (+ t0 1) n) (setf (aref out (+ (* (+ t0 1) rows) r)) (if has-bias (+ s1 bb) s1)))
                (when (< (+ t0 2) n) (setf (aref out (+ (* (+ t0 2) rows) r)) (if has-bias (+ s2 bb) s2)))))))))
    out))

;;; --- a batch of tokens through the model ------------------------------------

(defparameter *prefill-chunk* 96
  "Tokens per batched pass.  A multiple of three, so every packed word is full.")

(defstruct (batch-scratch (:constructor %make-batch-scratch))
  x h q k v att fa fb xb fb-q)

(defun batch-scratch (ctx)
  (or (getf (context-plist ctx) :batch)
      (setf (getf (context-plist ctx) :batch)
            (let* ((m (context-model ctx)) (n *prefill-chunk*)
                   (d (model-d m)) (f (model-ffn m))
                   (kv (* (model-n-kv-head m) (model-head-dim m))))
              (%make-batch-scratch
               :x (f32vec (* n d)) :h (f32vec (* n d)) :q (f32vec (* n d))
               :k (f32vec (* n kv)) :v (f32vec (* n kv)) :att (f32vec (* n d))
               :fa (f32vec (* n f)) :fb (f32vec (* n f))
               :xb (make-q8batch n d) :fb-q (make-q8batch n f))))))

(defun map-rows! (fn n width &rest vecs)
  "Call FN on row-sized copies of row T of each of VECS (N rows of WIDTH), and
   write the copies back: the single-token ops, applied row by row."
  (let ((rows (mapcar (lambda (v) (declare (ignore v)) (f32vec width)) vecs)))
    (dotimes (tk n)
      (loop for v in vecs for r in rows
            do (replace r v :start2 (* tk width) :end2 (* (1+ tk) width)))
      (apply fn tk rows)
      (loop for v in vecs for r in rows
            do (replace v r :start1 (* tk width))))))

(defun forward-batch (ctx tokens &key (logits t))
  "Run TOKENS (at most *PREFILL-CHUNK*) at the next positions; bit-identical
   to FORWARD on each in turn.  The logits, if LOGITS, are the last token's."
  (let* ((m (context-model ctx))
         (pool (model-pool m))
         (n (length tokens))
         (pos0 (context-pos ctx))
         (d (model-d m)) (hd (model-head-dim m)) (f (model-ffn m))
         (kvd (* (model-n-kv-head m) hd))
         (eps (model-eps m))
         (sc (batch-scratch ctx))
         (x (batch-scratch-x sc)) (h (batch-scratch-h sc))
         (q (batch-scratch-q sc)) (k (batch-scratch-k sc)) (v (batch-scratch-v sc))
         (att (batch-scratch-att sc))
         (fa (batch-scratch-fa sc)) (fb (batch-scratch-fb sc))
         (xb (batch-scratch-xb sc)) (fbq (batch-scratch-fb-q sc)))
    (assert (<= n *prefill-chunk*))
    (when (> (+ pos0 n) (context-n-ctx ctx))
      (error "context is full (~D tokens)" (context-n-ctx ctx)))
    (loop for tok in tokens for tk from 0
          do (let ((row (f32vec d)))
               (q8mat-row! (model-token-embd m) tok row)
               (replace x row :start1 (* tk d))))
    (loop for li below (model-n-layer m)
          for l = (svref (model-layers m) li)
          do (map-rows! (lambda (tk xr hr) (declare (ignore tk)) (rmsnorm! xr (layer-attn-norm l) hr eps))
                        n d x h)
             (quantize-batch! h n xb)
             (matmul! pool (layer-wq l) xb q (layer-bq l))
             (matmul! pool (layer-wk l) xb k (layer-bk l))
             (matmul! pool (layer-wv l) xb v (layer-bv l))
             (map-rows! (lambda (tk qr) (rope! qr (model-n-head m) hd (+ pos0 tk) (model-rope-base m) (model-rope-neox m)))
                        n d q)
             (map-rows! (lambda (tk kr) (rope! kr (model-n-kv-head m) hd (+ pos0 tk) (model-rope-base m) (model-rope-neox m)))
                        n kvd k)
             (replace (svref (context-k-cache ctx) li) k :start1 (* pos0 kvd) :end2 (* n kvd))
             (replace (svref (context-v-cache ctx) li) v :start1 (* pos0 kvd) :end2 (* n kvd))
             ;; token tk sees positions 0..pos0+tk: causality is the length
             (let ((nh (model-n-head m)))
               (parallel-for (job (* n nh) pool)
                 (multiple-value-bind (tk hh) (floor job nh)
                   (attend-head! ctx li q (+ (* tk d) (* hh hd)) att (+ pos0 tk 1) hh))))
             (quantize-batch! att n xb)
             (matmul! pool (layer-wo l) xb h)
             (add! x h (* n d))          ; elementwise, so whole rows at once
             (map-rows! (lambda (tk xr hr) (declare (ignore tk)) (rmsnorm! xr (layer-ffn-norm l) hr eps))
                        n d x h)
             (quantize-batch! h n xb)
             (matmul! pool (layer-wgate l) xb fa)
             (matmul! pool (layer-wup l) xb fb)
             (swiglu! fa fb (* n f))
             (quantize-batch! fa n fbq)
             (matmul! pool (layer-wdown l) fbq h)
             (add! x h (* n d)))
    (incf (context-pos ctx) n)
    (when logits
      (let ((last (f32vec d)) (hr (f32vec d)) (xq (context-xq ctx)))
        (replace last x :start2 (* (1- n) d))
        (rmsnorm! last (model-output-norm m) hr eps)
        (quantize-q8! hr xq)
        (matvec! pool (model-output m) xq (context-logits ctx))))
    ctx))

(defun eval-tokens (ctx tokens)
  "Feed TOKENS; only the last one's logits are computed.  More than one goes
   through FORWARD-BATCH, a chunk at a time."
  (if (null (rest tokens))
      (when tokens (forward ctx (first tokens)))
      (loop for rest = tokens then (nthcdr *prefill-chunk* rest)
            while rest
            do (let ((chunk (subseq rest 0 (min *prefill-chunk* (length rest)))))
                 (forward-batch ctx chunk :logits (<= (length rest) *prefill-chunk*)))))
  (context-logits ctx))
