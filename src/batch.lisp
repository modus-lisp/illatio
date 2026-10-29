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

;;; Activations of N tokens, quantized.  QV holds them as N rows of int8
;;; (token t's at t*cols) with their scales (token t's block b at t*(cols/32)+b).
;;; WORDS packs them three to a word for the portable kernel: WORDS[g*cols + j]
;;; is column j of tokens 3g, 3g+1, 3g+2 (absent tokens are zero).  The NEON
;;; kernel reads QV directly, so the packing is only done when it is needed.
(defstruct (q8batch (:constructor %make-q8batch))
  (n 0 :type fixnum)
  (cols 0 :type fixnum)
  (qv nil)                              ; a Q8VEC, when the model takes Q8_0
  (qk nil)                              ; a Q8KVEC, when it takes Q8_K
  (words (make-array 0 :element-type '(signed-byte 64)) :type i64vec)
  (quads nil)                           ; QK packed by PACK-QUADS!, once a batch,
  (quads-d nil)                         ; when a Q6_K matrix wants them
  (quads-ok nil))

(defun q8batch-quants (xb) (q8vec-quants (q8batch-qv xb)))
(defun q8batch-scales (xb) (q8vec-scales (q8batch-qv xb)))

(defun make-q8batch (max-n cols &optional (kinds '(:q8)))
  (%make-q8batch :cols cols
                 :qv (and (member :q8 kinds)
                          (%make-q8vec :quants (make-i8vec (* max-n cols))
                                       :scales (f32vec (* max-n (floor cols 32)))))
                 :qk (and (member :q8k kinds) (make-q8kvec (* max-n cols)))
                 :words (if (member :q8 kinds)
                            (make-array (* (ceiling max-n 3) cols) :element-type '(signed-byte 64))
                            (make-array 0 :element-type '(signed-byte 64)))))

(defun each-token (pool n fn)
  "(FN tk) for tk below N, a range of tokens to each of POOL's threads (all
   on this one without a POOL)."
  (declare (type function fn) (type fixnum n))
  (if pool
      (pool-run pool (lambda (i c)
                       (multiple-value-bind (s e) (share n i c)
                         (loop for tk from s below e do (funcall fn tk)))))
      (dotimes (tk n) (funcall fn tk))))

(defun quantize-batch! (xs n out &optional pool)
  "Quantize rows 0..N-1 of XS (N rows of COLS floats, row-major) into OUT,
   each row exactly as QUANTIZE-Q8! would (a row is quantized on its own, so
   POOL's threads give the same bits)."
  (declare (type f32vec xs) (type fixnum n) (optimize speed (safety 0)))
  (let* ((cols (q8batch-cols out))
         (nb (floor cols 32))
         (qv (q8batch-qv out)))
    (declare (type fixnum cols nb))
    (setf (q8batch-n out) n (q8batch-quads-ok out) nil)
    (when (q8batch-qk out)
      (let ((qk (q8batch-qk out)))
        (each-token pool n (lambda (tk) (quantize-q8k! xs qk (* tk cols) cols (* tk cols))))))
    (unless qv (return-from quantize-batch! out))
    (each-token pool n (lambda (tk) (quantize-q8! xs qv (* tk cols) cols (* tk cols) (* tk nb))))
    (unless (neon-active-p)
      (let ((words (q8batch-words out)) (q (q8vec-quants qv)))
        (declare (type i64vec words) (type i8vec q))
        (fill words 0 :end (* (ceiling n 3) cols))
        (dotimes (tk n)
          (let ((base (* (floor tk 3) cols))
                (shift (* +lane+ (mod tk 3)))
                (qo (* tk cols)))
            (declare (type fixnum base qo) (type (integer 0 42) shift))
            (dotimes (j cols)
              (setf (aref words (+ base j))
                    (the (signed-byte 64) (+ (aref words (+ base j))
                                             (the (signed-byte 50) (ash (i8-ref q (+ qo j)) shift))))))))))
    out))

(defun q6k-quads-p ()
  "Whether a Q6_K multiply takes four tokens a call (SMMLA)."
  #+(and sbcl arm64) (neon-mmla-p)
  #-(and sbcl arm64) nil)

(defun batch-quads (xb pool)
  "XB's Q8_K activations packed for the four-token Q6_K kernel, made the
   first time a batch asks."
  (let* ((cols (q8batch-cols xb)) (n (q8batch-n xb)) (nq (ceiling n 4))
         (bytes (* nq 4 cols)) (floats (* nq (floor cols +qk-k+) 8)))
    (unless (and (q8batch-quads xb) (<= bytes (length (q8batch-quads xb))))
      (setf (q8batch-quads xb) (make-u8vec bytes)
            (q8batch-quads-d xb) (f32vec floats)))
    (unless (q8batch-quads-ok xb)
      #+(and sbcl arm64)
      (let ((qk (q8batch-qk xb)) (quads (q8batch-quads xb)) (qd (q8batch-quads-d xb)))
        (each-token pool nq (lambda (q) (pack-quads! qk n cols quads qd q))))
      (setf (q8batch-quads-ok xb) t))
    (q8batch-quads xb)))

(defun kmat-quads-dot! (w quads qd n out start end rows)
  #+(and sbcl arm64) (kmat-quads-dot-neon! w quads qd n out start end rows)
  #-(and sbcl arm64) (error "no four-token kernel here ~S" (list w quads qd n out start end rows)))

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
  "OUT[t*rows + r] = (W·x_t)[r] (+ BIAS[r]) for the N tokens in XB.  With SDOT,
   each (row, token) is one NEON row product; otherwise three tokens share
   each multiply.  Both give the scalar path's bits."
  (declare (type q8batch xb) (type f32vec out))
  (when (kmat-p w)
    (let ((rows (kmat-rows w)) (cols (kmat-cols w)) (n (q8batch-n xb)) (qk (q8batch-qk xb)))
      (declare (type fixnum rows cols n))
      (if (and (>= n 2) (eq (kmat-type w) :q6_k) (q6k-quads-p))
          (let ((quads (batch-quads xb pool)) (qd (q8batch-quads-d xb)) (pairs (floor rows 2)))
            ;; two rows by four tokens a kernel call; an odd last row on its own
            (pool-run pool (lambda (i c)
                             (multiple-value-bind (s e) (share pairs i c)
                               (kmat-quads-dot! w quads qd n out (* 2 s) (* 2 e) rows))))
            (when (oddp rows)
              (kmat-range-dot! w qk 0 out (1- rows) rows 0 n cols rows)))
          (pool-run pool (lambda (i c)
                           (multiple-value-bind (s e) (share rows i c)
                             (kmat-range-dot! w qk 0 out s e 0 n cols rows)))))
      (when bias
        (let ((bias bias))
          (declare (type f32vec bias))
          (dotimes (tk n)
            (dotimes (r rows)
              (incf (aref out (+ (* tk rows) r)) (aref bias r))))))
      (return-from matmul! out)))
  (when (neon-active-p)
    (let* ((rows (q8mat-rows w)) (cols (q8mat-cols w)) (nb (floor cols 32))
           (n (q8batch-n xb)) (xq (q8batch-quants xb)) (xs (q8batch-scales xb)))
      (declare (type fixnum rows cols nb n))
      (with-dot-rows (dot-rows w xq xs out)
        (parallel-for (g (floor rows 4) pool)
          (dotimes (tk n)
            (dot-rows g (* tk cols) (* tk nb) (+ (* tk rows) (* 4 g))))))
      (when bias
        (let ((bias bias))
          (declare (type f32vec bias) (optimize speed (safety 0)))
          (dotimes (tk n)
            (dotimes (r rows)
              (incf (aref out (+ (* tk rows) r)) (aref bias r))))))
      (return-from matmul! out)))
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
                                               (* (i8-ref wq (+ wo j))
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
  x h q k v att fa fb xb ab fb-q)

(defun batch-scratch (ctx)
  (or (getf (context-plist ctx) :batch)
      (setf (getf (context-plist ctx) :batch)
            (let* ((m (context-model ctx)) (n *prefill-chunk*)
                   (d (model-d m)) (qd (model-q-dim m)) (f (model-ffn m))
                   (kv (* (model-n-kv-head m) (model-head-dim m))))
              (%make-batch-scratch
               :x (f32vec (* n d)) :h (f32vec (* n d)) :q (f32vec (* n qd))
               :k (f32vec (* n kv)) :v (f32vec (* n kv)) :att (f32vec (* n qd))
               :fa (f32vec (* n f)) :fb (f32vec (* n f))
               :xb (make-q8batch n d (model-act-kinds m)) :ab (make-q8batch n qd (model-act-kinds m))
               :fb-q (make-q8batch n f (model-act-kinds m)))))))

(defun forward-batch (ctx tokens &key (logits t))
  "Run TOKENS (at most *PREFILL-CHUNK*) at the next positions; bit-identical
   to FORWARD on each in turn.  The logits, if LOGITS, are the last token's."
  (let* ((m (context-model ctx))
         (pool (model-pool m))
         (n (length tokens))
         (pos0 (context-pos ctx))
         (d (model-d m)) (qd (model-q-dim m)) (hd (model-head-dim m)) (f (model-ffn m))
         (kvd (* (model-n-kv-head m) hd))
         (eps (model-eps m))
         (sc (batch-scratch ctx))
         (x (batch-scratch-x sc)) (h (batch-scratch-h sc))
         (q (batch-scratch-q sc)) (k (batch-scratch-k sc)) (v (batch-scratch-v sc))
         (att (batch-scratch-att sc))
         (fa (batch-scratch-fa sc)) (fb (batch-scratch-fb sc))
         (xb (batch-scratch-xb sc)) (ab (batch-scratch-ab sc)) (fbq (batch-scratch-fb-q sc)))
    (assert (<= n *prefill-chunk*))
    (when (> (+ pos0 n) (context-n-ctx ctx))
      (error "context is full (~D tokens)" (context-n-ctx ctx)))
    (loop for tok in tokens for tk from 0
          do (let ((row (f32vec d)))
               (embed-row! (model-token-embd m) tok row)
               (replace x row :start1 (* tk d))))
    (loop for li below (model-n-layer m)
          for l = (svref (model-layers m) li)
          do (dotimes (tk n) (rmsnorm! x (layer-attn-norm l) h eps (* tk d)))
             (quantize-batch! h n xb pool)
             (matmul! pool (layer-wq l) xb q (layer-bq l))
             (matmul! pool (layer-wk l) xb k (layer-bk l))
             (matmul! pool (layer-wv l) xb v (layer-bv l))
             (dotimes (tk n)
               (qk-prep! pool m l q k (+ pos0 tk) (* tk qd) (* tk kvd)))
             (replace (svref (context-k-cache ctx) li) k :start1 (* pos0 kvd) :end2 (* n kvd))
             (replace (svref (context-v-cache ctx) li) v :start1 (* pos0 kvd) :end2 (* n kvd))
             ;; token tk sees positions 0..pos0+tk: causality is the length
             (let ((nh (model-n-head m)))
               (parallel-for (job (* n nh) pool)
                 (multiple-value-bind (tk hh) (floor job nh)
                   (attend-head! ctx li q (+ (* tk qd) (* hh hd)) att (+ pos0 tk 1) hh))))
             (quantize-batch! att n ab pool)
             (matmul! pool (layer-wo l) ab h)
             (add! x h (* n d))          ; elementwise, so whole rows at once
             (dotimes (tk n) (rmsnorm! x (layer-ffn-norm l) h eps (* tk d)))
             (quantize-batch! h n xb pool)
             (matmul! pool (layer-wgate l) xb fa)
             (matmul! pool (layer-wup l) xb fb)
             (swiglu-par! pool fa fb (* n f))
             (quantize-batch! fa n fbq pool)
             (matmul! pool (layer-wdown l) fbq h)
             (add! x h (* n d)))
    (incf (context-pos ctx) n)
    (when logits
      (let ((last (f32vec d)) (hr (f32vec d)) (xq (context-xq ctx)))
        (replace last x :start2 (* (1- n) d))
        (rmsnorm! last (model-output-norm m) hr eps)
        (quantize-act! hr xq)
        (qmatvec! pool (model-output m) xq (context-logits ctx))))
    ctx))

(defun eval-tokens (ctx tokens)
  "Feed TOKENS; only the last one's logits are computed.  More than one goes
   through FORWARD-BATCH, a chunk at a time."
  (when (model-impl (context-model ctx))
    (return-from eval-tokens (ds4f-context-eval ctx tokens)))
  #+(and sbcl darwin)
  (when (model-gpu (context-model ctx))
    (return-from eval-tokens (gpu-eval ctx tokens)))
  (if (null (rest tokens))
      (when tokens (forward ctx (first tokens)))
      (loop with batch = (if (model-recurrent-p (context-model ctx)) #'forward-batch-qwen35 #'forward-batch)
            for rest = tokens then (nthcdr *prefill-chunk* rest)
            while rest
            do (let ((chunk (subseq rest 0 (min *prefill-chunk* (length rest)))))
                 (funcall batch ctx chunk :logits (<= (length rest) *prefill-chunk*)))))
  (context-logits ctx))
