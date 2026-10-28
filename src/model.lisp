;;;; model.lisp — the qwen2 / llama decoder, one token at a time.
;;;
;;; Each step mirrors one ggml op of llama.cpp's graph for the architecture,
;;; in the same order and the same precision, because greedy decoding only
;;; agrees with llama.cpp for as long as the logits do:
;;;
;;;   x = embd[token]
;;;   per layer:  h = rmsnorm(x)·attn_norm
;;;               q,k,v = Wq h + bq, Wk h + bk, Wv h + bv   (biases: qwen2 only)
;;;               rope(q), rope(k)                            (NEOX pairing for qwen2)
;;;               x += Wo · attention(q, K[0..pos], V[0..pos])
;;;               h = rmsnorm(x)·ffn_norm
;;;               x += Wdown · (silu(Wgate h) * Wup h)
;;;   logits = Wout · rmsnorm(x)·output_norm

(in-package #:illatio)

(defstruct layer
  attn-norm wq wk wv wo bq bk bv
  ffn-norm wgate wup wdown)

(defstruct (model (:constructor %make-model))
  gguf vocab arch name
  (n-layer 0 :type fixnum)
  (d 0 :type fixnum)             ; embedding width
  (n-head 0 :type fixnum)
  (n-kv-head 0 :type fixnum)
  (head-dim 0 :type fixnum)
  (ffn 0 :type fixnum)
  (n-vocab 0 :type fixnum)
  (context-length 0 :type fixnum)
  (eps 1f-6 :type single-float)
  (rope-base 10000f0 :type single-float)
  (rope-neox nil)
  token-embd output-norm output
  (layers #() :type simple-vector)
  pool)

(defun arch-key (g arch key) (meta g (format nil "~A.~A" arch key)))

(defun load-model (path &key (threads (default-thread-count)))
  "Read the GGUF model at PATH: weights, tokenizer, and a pool of THREADS."
  (let* ((g (read-gguf path))
         (arch (meta g "general.architecture"))
         (n-layer (arch-key g arch "block_count"))
         (d (arch-key g arch "embedding_length"))
         (n-head (arch-key g arch "attention.head_count"))
         (vocab (make-vocab-from-gguf g)))
    (unless (member arch '("qwen2" "llama") :test #'equal)
      (error "architecture ~S is not supported (qwen2 and llama are)" arch))
    (flet ((q8 (name) (load-q8mat g name))
           (f32 (name) (load-f32 g name))
           (opt (name f) (and (gguf-tensor g name nil) (funcall f name))))
      (%make-model
       :gguf g :vocab vocab :arch arch :name (meta g "general.name" arch)
       :n-layer n-layer :d d :n-head n-head
       :n-kv-head (arch-key g arch "attention.head_count_kv")
       :head-dim (floor d n-head)
       :ffn (arch-key g arch "feed_forward_length")
       :n-vocab (length (vocab-pieces vocab))
       :context-length (arch-key g arch "context_length")
       :eps (coerce (arch-key g arch "attention.layer_norm_rms_epsilon") 'single-float)
       :rope-base (coerce (or (arch-key g arch "rope.freq_base") 10000) 'single-float)
       ;; llama.cpp's rope type per architecture: qwen2 rotates (i, i+d/2)
       ;; pairs (NEOX), llama rotates adjacent (2i, 2i+1) pairs.
       :rope-neox (equal arch "qwen2")
       :token-embd (q8 "token_embd.weight")
       :output-norm (f32 "output_norm.weight")
       ;; Without an output matrix the head is tied to the embedding.
       :output (or (opt "output.weight" #'q8) (q8 "token_embd.weight"))
       :layers (coerce
                (loop for i below n-layer
                      collect (flet ((n (s) (format nil "blk.~D.~A" i s)))
                                (make-layer
                                 :attn-norm (f32 (n "attn_norm.weight"))
                                 :wq (q8 (n "attn_q.weight")) :wk (q8 (n "attn_k.weight"))
                                 :wv (q8 (n "attn_v.weight")) :wo (q8 (n "attn_output.weight"))
                                 :bq (opt (n "attn_q.bias") #'f32)
                                 :bk (opt (n "attn_k.bias") #'f32)
                                 :bv (opt (n "attn_v.bias") #'f32)
                                 :ffn-norm (f32 (n "ffn_norm.weight"))
                                 :wgate (q8 (n "ffn_gate.weight"))
                                 :wup (q8 (n "ffn_up.weight"))
                                 :wdown (q8 (n "ffn_down.weight")))))
                'simple-vector)
       :pool (make-pool threads)))))

;;; --- per-sequence state -----------------------------------------------------

(defstruct (context (:constructor %make-context))
  model
  (pos 0 :type fixnum)                 ; tokens in the cache
  (n-ctx 0 :type fixnum)
  (plist (list :tokens (list)))        ; :tokens -- what the cache holds
  k-cache v-cache                      ; per layer: f32 [n-ctx * kv-dim]
  ;; scratch, allocated once
  x h q k v att ffn-a ffn-b xq hq logits)

(defun make-context (model &key (n-ctx (min 4096 (model-context-length model))))
  (let* ((d (model-d model))
         (kv (* (model-n-kv-head model) (model-head-dim model)))
         (f (model-ffn model)))
    (%make-context
     :model model :n-ctx n-ctx
     :k-cache (coerce (loop repeat (model-n-layer model) collect (f32vec (* n-ctx kv))) 'simple-vector)
     :v-cache (coerce (loop repeat (model-n-layer model) collect (f32vec (* n-ctx kv))) 'simple-vector)
     :x (f32vec d) :h (f32vec d) :q (f32vec d) :k (f32vec kv) :v (f32vec kv)
     :att (f32vec d) :ffn-a (f32vec f) :ffn-b (f32vec f)
     :xq (make-q8vec d) :hq (make-q8vec f)
     :logits (f32vec (model-n-vocab model)))))

;;; --- ops --------------------------------------------------------------------

(defun rmsnorm! (x w out eps &optional (xo 0) (n (length w)))
  "ggml_rms_norm then ggml_mul on X[xo..xo+n): the sum of squares is taken in
   double, the scale is a float, and x is scaled before it is weighted.  The
   result goes to OUT at the same offset."
  (declare (type f32vec x w out) (type single-float eps) (type fixnum xo n)
           (optimize speed (safety 0)))
  (let* ((sum (loop for i of-type fixnum from xo below (+ xo n)
                    sum (coerce (* (aref x i) (aref x i)) 'double-float) of-type double-float))
         (scale (/ 1f0 (sqrt (coerce (+ (/ sum n) eps) 'single-float)))))
    (declare (type single-float scale))
    (dotimes (i n out)
      (setf (aref out (+ xo i)) (* (* (aref x (+ xo i)) scale) (aref w i))))))

(defun rope! (x n-heads head-dim pos base neox &optional (xo 0))
  "Rotate each head of X for position POS.  theta is advanced by repeated
   multiplication, as ggml_rope_cache_init does, not by a fresh pow per
   pair: the two differ in the last bits and those bits reach the logits."
  (declare (type f32vec x) (type fixnum n-heads head-dim pos xo) (type single-float base)
           (optimize speed (safety 0)))
  (let* ((half (floor head-dim 2))
         (cos (f32vec half)) (sin (f32vec half))
         (theta-scale (expt base (/ -2f0 head-dim)))
         (theta (coerce pos 'single-float)))
    (declare (type single-float theta theta-scale))
    (dotimes (i half)
      (setf (aref cos i) (cos theta) (aref sin i) (sin theta)
            theta (* theta theta-scale)))
    (dotimes (h n-heads x)
      (let ((o (+ xo (* h head-dim))))
        (dotimes (i half)
          (multiple-value-bind (a b) (if neox
                                         (values (+ o i) (+ o i half))
                                         (values (+ o (* 2 i)) (+ o (* 2 i) 1)))
            (let ((x0 (aref x a)) (x1 (aref x b)) (c (aref cos i)) (s (aref sin i)))
              (setf (aref x a) (- (* x0 c) (* x1 s))
                    (aref x b) (+ (* x0 s) (* x1 c))))))))))

(defmacro with-attention-kernels ((dot axpy) (kc vc q out) &body body)
  "Run BODY with (DOT k-offset q-offset hd) and (AXPY w v-offset out-offset hd)
   bound to the NEON kernels when this machine has them, else the portable
   twins that define their bits."
  (let ((portable `(flet ((,dot (ko qo hd) (f32-dot16 ,kc ko ,q qo hd))
                          (,axpy (w vo oo hd) (f32-axpy ,out oo ,vc vo w hd)))
                     (declare (inline ,dot ,axpy) (ignorable #',dot #',axpy))
                     ,@body)))
    #-(or modus (and sbcl arm64)) portable
    ;; modus: the f32x4 twins (neon-modus.lisp); head dims are multiples of 16
    #+modus
    `(flet ((,dot (ko qo hd) (%modus-f32-dot16 ,kc ko ,q qo hd))
            (,axpy (w vo oo hd) (%modus-f32-axpy ,out oo ,vc vo w hd)))
       (declare (inline ,dot ,axpy) (ignorable #',dot #',axpy))
       ,@body)
    #+(and sbcl arm64)
    `(if (neon-active-p)
         (sb-sys:with-pinned-objects (,kc ,vc ,q ,out)
           (let ((ks (sb-sys:vector-sap ,kc)) (vs (sb-sys:vector-sap ,vc))
                 (qs (sb-sys:vector-sap ,q)) (os (sb-sys:vector-sap ,out)))
             (flet ((,dot (ko qo hd)
                      (declare (type fixnum ko qo hd))
                      (%f32-dot16 (sb-sys:sap+ ks (* 4 ko)) (sb-sys:sap+ qs (* 4 qo)) (ash hd -4)))
                    (,axpy (w vo oo hd)
                      (declare (type fixnum vo oo hd) (type single-float w))
                      (%f32-axpy4 (sb-sys:sap+ os (* 4 oo)) (sb-sys:sap+ vs (* 4 vo)) w (ash hd -2))))
               (declare (inline ,dot ,axpy))
               ,@body)))
         ,portable)))

(defun attend-head! (ctx layer-index q qo out n h)
  "One query head: OUT[qo..] = softmax(q·K / sqrt(hd)) · V over positions
   0..N-1, Q's head H starting at QO.  Query heads share KV heads in groups
   (GQA).  The dot products are summed in F32-DOT16's order."
  (declare (type f32vec q out) (type fixnum layer-index qo n h))
  (let* ((m (context-model ctx))
         (hd (model-head-dim m))
         (group (floor (model-n-head m) (model-n-kv-head m)))
         (kvd (* (model-n-kv-head m) hd))
         (kc (svref (context-k-cache ctx) layer-index))
         (vc (svref (context-v-cache ctx) layer-index))
         (scale (/ 1f0 (sqrt (coerce hd 'single-float))))
         (scores (f32vec n))
         (ko (* (floor h group) hd))
         (mx most-negative-single-float))
    (declare (type fixnum hd group kvd ko) (type f32vec kc vc scores)
             (type single-float scale mx)
             (optimize speed (safety 0)))
    (assert (zerop (mod hd 16)))
    (with-attention-kernels (dot axpy) (kc vc q out)
      (dotimes (p n)
        (let ((s (* (the single-float (dot (+ (* p kvd) ko) qo hd)) scale)))
          (setf (aref scores p) s)
          (when (> s mx) (setf mx s))))
      ;; ggml_vec_soft_max_f32: exp in float, the sum in double.
      (let ((sum 0d0))
        (declare (type double-float sum))
        (dotimes (p n)
          (let ((e (exp (- (aref scores p) mx))))
            (setf (aref scores p) e)
            (incf sum e)))
        (let ((inv (coerce (/ 1d0 sum) 'single-float)))
          (dotimes (p n) (setf (aref scores p) (* (aref scores p) inv)))))
      (dotimes (i hd) (setf (aref out (+ qo i)) 0f0))
      (dotimes (p n)
        (axpy (aref scores p) (+ (* p kvd) ko) qo hd)))))

(defun attention! (ctx layer-index)
  "Every query head of the current token, in parallel."
  (let* ((m (context-model ctx))
         (hd (model-head-dim m))
         (n (1+ (context-pos ctx))))
    (parallel-for (h (model-n-head m) (model-pool m))
      (attend-head! ctx layer-index (context-q ctx) (* h hd) (context-att ctx) n h))))

(defun add! (x y &optional (end (length x)))
  (declare (type f32vec x y) (type fixnum end) (optimize speed (safety 0)))
  (dotimes (i end x) (incf (aref x i) (aref y i))))

(defun swiglu! (a b &optional (end (length a)))
  "A = silu(A) * B, silu(x) = x/(1+e^-x)."
  (declare (type f32vec a b) (type fixnum end) (optimize speed (safety 0)))
  (dotimes (i end a)
    (let ((x (aref a i)))
      (setf (aref a i) (* (/ x (+ 1f0 (exp (- x)))) (aref b i))))))

;;; --- one token --------------------------------------------------------------

(defvar *trace* nil
  "When non-NIL, a function (NAME VECTOR) called on each intermediate of a
   forward pass, named as llama.cpp's graph names them, to diff against
   llama-eval-callback.")

(defmacro traced (name vec)
  `(when *trace* (funcall *trace* ,name ,vec)))

(defun forward (ctx token &key (logits t))
  "Run TOKEN at the next position; the logits (if LOGITS) land in the context."
  (let* ((m (context-model ctx))
         (pool (model-pool m))
         (pos (context-pos ctx))
         (hd (model-head-dim m))
         (kvd (* (model-n-kv-head m) hd))
         (x (context-x ctx)) (h (context-h ctx))
         (xq (context-xq ctx)) (hq (context-hq ctx)))
    (when (>= pos (context-n-ctx ctx))
      (error "context is full (~D tokens)" (context-n-ctx ctx)))
    (q8mat-row! (model-token-embd m) token x)
    (traced "embd" x)
    (loop for li below (model-n-layer m)
          for l = (svref (model-layers m) li)
          do (rmsnorm! x (layer-attn-norm l) h (model-eps m))
             (traced (format nil "attn_norm-~D" li) h)
             (quantize-q8! h xq)
             (matvec! pool (layer-wq l) xq (context-q ctx) (layer-bq l))
             (matvec! pool (layer-wk l) xq (context-k ctx) (layer-bk l))
             (matvec! pool (layer-wv l) xq (context-v ctx) (layer-bv l))
             (traced (format nil "Qcur-~D" li) (context-q ctx))
             (traced (format nil "Kcur-~D" li) (context-k ctx))
             (traced (format nil "Vcur-~D" li) (context-v ctx))
             (rope! (context-q ctx) (model-n-head m) hd pos (model-rope-base m) (model-rope-neox m))
             (rope! (context-k ctx) (model-n-kv-head m) hd pos (model-rope-base m) (model-rope-neox m))
             (replace (svref (context-k-cache ctx) li) (context-k ctx) :start1 (* pos kvd))
             (replace (svref (context-v-cache ctx) li) (context-v ctx) :start1 (* pos kvd))
             (traced (format nil "Qrope-~D" li) (context-q ctx))
             (traced (format nil "Krope-~D" li) (context-k ctx))
             (attention! ctx li)
             (traced (format nil "kqv_out-~D" li) (context-att ctx))
             (quantize-q8! (context-att ctx) xq)
             (matvec! pool (layer-wo l) xq h)
             (traced (format nil "attn_out-~D" li) h)
             (add! x h)
             (traced (format nil "ffn_inp-~D" li) x)
             (rmsnorm! x (layer-ffn-norm l) h (model-eps m))
             (traced (format nil "ffn_norm-~D" li) h)
             (quantize-q8! h xq)
             (matvec! pool (layer-wgate l) xq (context-ffn-a ctx))
             (matvec! pool (layer-wup l) xq (context-ffn-b ctx))
             (traced (format nil "ffn_gate-~D" li) (context-ffn-a ctx))
             (traced (format nil "ffn_up-~D" li) (context-ffn-b ctx))
             (swiglu! (context-ffn-a ctx) (context-ffn-b ctx))
             (traced (format nil "ffn_swiglu-~D" li) (context-ffn-a ctx))
             (quantize-q8! (context-ffn-a ctx) hq)
             (matvec! pool (layer-wdown l) hq h)
             (traced (format nil "ffn_out-~D" li) h)
             (add! x h)
             (traced (format nil "l_out-~D" li) x))
    (incf (context-pos ctx))
    (when logits
      (rmsnorm! x (model-output-norm m) h (model-eps m))
      (quantize-q8! h xq)
      (matvec! pool (model-output m) xq (context-logits ctx)))
    ctx))

(defun argmax (v)
  (declare (type f32vec v))
  (let ((best 0))
    (dotimes (i (length v) best)
      (when (> (aref v i) (aref v best)) (setf best i)))))

(defun tokenize (model text &rest keys)
  (apply #'tokenize-with (model-vocab model) text keys))

(defun detokenize (model ids)
  (detokenize-with (model-vocab model) ids))

(defun greedy (model prompt n &key (ctx (make-context model)))
  "The N tokens greedy decoding produces after PROMPT (a string or ids)."
  (let ((ids (if (stringp prompt) (tokenize model prompt) prompt)))
    (eval-tokens ctx ids)
    (loop repeat n
          for tok = (argmax (context-logits ctx))
          collect tok
          do (forward ctx tok))))

(defun top-logprobs (logits k)
  "The K most likely tokens as (id . logprob), most likely first.  The
   normaliser is summed in double, like everything else that reduces a
   whole vocabulary.  One pass over the logits with the best K kept sorted in
   a small array: a candidate costs one compare unless it beats the K-th."
  (declare (type f32vec logits) (type fixnum k))
  (let* ((n (length logits))
         (k (min k n))
         (ids (make-array k :element-type 'fixnum :initial-element 0))
         (vals (make-array k :element-type 'single-float :initial-element most-negative-single-float))
         (mx (reduce #'max logits))
         (z 0d0))
    (declare (type double-float z))
    (dotimes (i n)
      (let ((x (aref logits i)))
        (setq z (+ z (exp (coerce (- x mx) 'double-float))))
        (when (> x (aref vals (1- k)))
          ;; insert, shifting the smaller ones down
          (let ((j (1- k)))
            (loop while (and (> j 0) (> x (aref vals (1- j))))
                  do (setf (aref vals j) (aref vals (1- j)) (aref ids j) (aref ids (1- j)))
                     (decf j))
            (setf (aref vals j) x (aref ids j) i)))))
    (let ((lz (log z)))
      (loop for j below k collect (cons (aref ids j) (- (aref vals j) mx lz))))))

(defun replace-model-pool (into from threads)
  "Make INTO share FROM's weights but run on a fresh pool of THREADS."
  (loop for slot in (sb-mop:class-slots (class-of from))
        for name = (sb-mop:slot-definition-name slot)
        do (setf (slot-value into name) (slot-value from name)))
  (setf (model-pool into) (make-pool threads))
  into)
