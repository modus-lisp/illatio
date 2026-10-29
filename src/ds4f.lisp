;;;; ds4f.lisp — DeepSeek V4 Flash (`deepseek4`), one token at a time on the CPU.
;;;
;;; A port of ds4.c's CPU decode path (layer_forward_raw_swa_one and what it
;;; calls), op for op.  ds4.c is the reference: when in doubt, it is right.
;;; The shape, per token:
;;;
;;;   x = embd[token] (F16), copied into N_HC = 4 residual streams ("hyper-
;;;   connections"); per layer, twice (attention, then FFN):
;;;     HC pre:  mix = Wfn rmsnorm(all 4 streams); a Sinkhorn split of mix
;;;              gives 4 pre weights, 4 post weights and a 4x4 combine; the
;;;              block's input is the pre-weighted sum of the streams
;;;     block:   attention or FFN on rmsnorm(input) * w
;;;     HC post: stream d = out * post[d] + sum_s comb[d,s] * stream s
;;;   logits = Wout rmsnorm(sigmoid-weighted sum of the streams) * w
;;;
;;; Attention: 64 query heads of 512 share one KV head (the key is the value).
;;; Q comes through a rank-1024 LoRA, normed per head; KV is normed, RoPE
;;; turns the last 64 dims of each head (YaRN on the compressed layers), and
;;; the KV row goes through an FP8 round trip then F16 into the cache.  Every
;;; layer attends to its last 128 rows (sliding window); a layer with a
;;; compress ratio (4 or 128) also attends to compressed rows -- each a
;;; softmax-pooling of RATIO tokens' projected rows (the compressor) -- and
;;; with ratio 4 only to the 512 an indexer scores highest.  Scores carry a
;;; per-head sink in the denominator.  The output is RoPE'd back, projected
;;; in 8 groups to rank 1024, then to the embedding.
;;;
;;; FFN: a shared expert plus 6 of 256 routed experts, chosen by
;;; sqrt(softplus(router)) + bias (or, in the first 3 layers, by a token-id
;;; table), weighted, renormalized and scaled by 1.5; experts are IQ2_XXS
;;; gate/up and Q2_K down, read in the mapped file (iquant.lisp).

(in-package #:illatio)

;;; --- the model ----------------------------------------------------------------------

(defstruct (ds4f (:constructor %make-ds4f))
  gguf file pool vocab
  (n-layer 0 :type fixnum) (d 0 :type fixnum) (n-head 0 :type fixnum) (head-dim 0 :type fixnum)
  (n-rot 0 :type fixnum) (lora-q 0 :type fixnum) (lora-o 0 :type fixnum) (out-groups 0 :type fixnum)
  (n-expert 0 :type fixnum) (n-used 0 :type fixnum) (ff-exp 0 :type fixnum)
  (n-hc 4 :type fixnum) (sinkhorn 20 :type fixnum)
  (swa 128 :type fixnum) (ix-heads 0 :type fixnum) (ix-dim 0 :type fixnum) (ix-top-k 0 :type fixnum)
  (eps 1f-6 :type single-float) (hc-eps 1f-6 :type single-float)
  (rope-base 10000f0 :type single-float) (comp-rope-base 160000f0 :type single-float)
  (rope-factor 16f0 :type single-float) (rope-orig 65536 :type fixnum)
  (beta-fast 32f0 :type single-float) (beta-slow 1f0 :type single-float)
  (expert-scale 1.5f0 :type single-float)
  ratios clamps                         ; per layer: compress ratio, swiglu clamp
  (n-vocab 0 :type fixnum)
  embd-off output output-norm out-hc-fn out-hc-scale out-hc-base
  (layers #() :type simple-vector))

(defstruct (d4layer (:conc-name dl-))
  ratio
  attn-norm q-a q-a-norm q-b kv kv-norm sinks out-a out-b
  comp-kv comp-gate comp-ape comp-norm comp-width
  ix-q-b ix-proj ixc-kv ixc-gate ixc-ape ixc-norm
  ffn-norm gate-inp probs-b tid2eid
  gate-exps up-exps down-exps                ; (offset row-bytes rows cols) in the file
  sh-gate sh-up sh-down
  hc-attn-fn hc-attn-scale hc-attn-base hc-ffn-fn hc-ffn-scale hc-ffn-base)

(defstruct (f32mat (:constructor %make-f32mat)) (rows 0 :type fixnum) (cols 0 :type fixnum) (data nil :type f32vec))

(defun load-f32mat (g name)
  "A 2-D F16/F32 tensor as an F32MAT (rows of dims[0])."
  (let* ((ti (gguf-tensor g name)) (dims (tensor-info-dims ti)))
    (%make-f32mat :cols (first dims) :rows (reduce #'* (rest dims))
                  :data (let ((v (load-f32 g name)) (s (f32vec (tensor-elements ti) :static t)))
                          (replace s v)))))

(defun expert-tensor (g name)
  "(byte offset, bytes a row, rows an expert, cols) of a 3-D expert tensor."
  (let* ((ti (gguf-tensor g name)) (dims (tensor-info-dims ti)))
    (list (+ (gguf-data-offset g) (tensor-info-offset ti))
          (floor (tensor-bytes ti) (* (second dims) (third dims)))
          (second dims) (first dims))))

(defun load-ds4f (path &key (threads (default-thread-count)))
  "DeepSeek V4 Flash at PATH: the non-routed weights in memory (Q8_0 as
   Q8MATs, F16 widened to F32), the routed experts left in the file, which is
   mapped whole (MAP-FILE-VECTOR) so they stream."
  (let* ((g (read-gguf path)) (a "deepseek4"))
    (flet ((k (key &optional default) (or (arch-key g a key) default))
           (q8 (name) (load-q8mat g name))
           (f32 (name) (load-f32 g name))
           (fm (name) (load-f32mat g name))
           (opt (name f) (and (gguf-tensor g name nil) (funcall f name))))
      (let* ((n-layer (k "block_count"))
             (m (%make-ds4f
                 :gguf g :file (map-file-vector path) :pool (make-pool threads)
                 :n-layer n-layer :d (k "embedding_length") :n-head (k "attention.head_count")
                 :head-dim (k "attention.key_length") :n-rot (k "rope.dimension_count")
                 :lora-q (k "attention.q_lora_rank") :lora-o (k "attention.output_lora_rank")
                 :out-groups (k "attention.output_group_count")
                 :n-expert (k "expert_count") :n-used (k "expert_used_count")
                 :ff-exp (k "expert_feed_forward_length")
                 :n-hc (k "hyper_connection.count") :sinkhorn (k "hyper_connection.sinkhorn_iterations")
                 :hc-eps (coerce (k "hyper_connection.epsilon") 'single-float)
                 :swa (k "attention.sliding_window") :ix-heads (k "attention.indexer.head_count")
                 :ix-dim (k "attention.indexer.key_length") :ix-top-k (k "attention.indexer.top_k")
                 :eps (coerce (k "attention.layer_norm_rms_epsilon") 'single-float)
                 :rope-base (coerce (k "rope.freq_base") 'single-float)
                 :comp-rope-base (coerce (k "attention.compress_rope_freq_base") 'single-float)
                 :rope-factor (coerce (k "rope.scaling.factor") 'single-float)
                 :rope-orig (k "rope.scaling.original_context_length")
                 :beta-fast (coerce (k "rope.scaling.yarn_beta_fast") 'single-float)
                 :beta-slow (coerce (k "rope.scaling.yarn_beta_slow") 'single-float)
                 :expert-scale (coerce (k "expert_weights_scale") 'single-float)
                 :ratios (coerce (k "attention.compress_ratios") 'simple-vector)
                 :clamps (map 'simple-vector (lambda (x) (coerce x 'single-float)) (k "swiglu_clamp_exp"))
                 :n-vocab (k "vocab_size")
                 :embd-off (+ (gguf-data-offset g) (tensor-info-offset (gguf-tensor g "token_embd.weight")))
                 :output (q8 "output.weight") :output-norm (f32 "output_norm.weight")
                 :out-hc-fn (fm "output_hc_fn.weight") :out-hc-scale (f32 "output_hc_scale.weight")
                 :out-hc-base (f32 "output_hc_base.weight"))))
        (setf (ds4f-layers m)
              (coerce
               (loop for i below n-layer
                     collect (flet ((n (s) (format nil "blk.~D.~A" i s)))
                               (let ((ratio (svref (ds4f-ratios m) i)))
                                 (make-d4layer
                                  :ratio ratio
                                  :attn-norm (f32 (n "attn_norm.weight"))
                                  :q-a (q8 (n "attn_q_a.weight")) :q-a-norm (f32 (n "attn_q_a_norm.weight"))
                                  :q-b (q8 (n "attn_q_b.weight"))
                                  :kv (q8 (n "attn_kv.weight")) :kv-norm (f32 (n "attn_kv_a_norm.weight"))
                                  :sinks (f32 (n "attn_sinks.weight"))
                                  :out-a (q8 (n "attn_output_a.weight")) :out-b (q8 (n "attn_output_b.weight"))
                                  :comp-kv (opt (n "attn_compressor_kv.weight") #'fm)
                                  :comp-gate (opt (n "attn_compressor_gate.weight") #'fm)
                                  :comp-ape (opt (n "attn_compressor_ape.weight") #'fm)
                                  :comp-norm (opt (n "attn_compressor_norm.weight") #'f32)
                                  :ix-q-b (opt (n "indexer.attn_q_b.weight") #'fm)
                                  :ix-proj (opt (n "indexer.proj.weight") #'fm)
                                  :ixc-kv (opt (n "indexer_compressor_kv.weight") #'fm)
                                  :ixc-gate (opt (n "indexer_compressor_gate.weight") #'fm)
                                  :ixc-ape (opt (n "indexer_compressor_ape.weight") #'fm)
                                  :ixc-norm (opt (n "indexer_compressor_norm.weight") #'f32)
                                  :ffn-norm (f32 (n "ffn_norm.weight"))
                                  :gate-inp (fm (n "ffn_gate_inp.weight"))
                                  :probs-b (opt (n "exp_probs_b.bias") #'f32)
                                  :tid2eid (opt (n "ffn_gate_tid2eid.weight") (lambda (nm) (load-i32-tensor g nm)))
                                  :gate-exps (expert-tensor g (n "ffn_gate_exps.weight"))
                                  :up-exps (expert-tensor g (n "ffn_up_exps.weight"))
                                  :down-exps (expert-tensor g (n "ffn_down_exps.weight"))
                                  :sh-gate (q8 (n "ffn_gate_shexp.weight")) :sh-up (q8 (n "ffn_up_shexp.weight"))
                                  :sh-down (q8 (n "ffn_down_shexp.weight"))
                                  :hc-attn-fn (fm (n "hc_attn_fn.weight")) :hc-attn-scale (f32 (n "hc_attn_scale.weight"))
                                  :hc-attn-base (f32 (n "hc_attn_base.weight"))
                                  :hc-ffn-fn (fm (n "hc_ffn_fn.weight")) :hc-ffn-scale (f32 (n "hc_ffn_scale.weight"))
                                  :hc-ffn-base (f32 (n "hc_ffn_base.weight"))))))
               'simple-vector))
        m))))

(defun load-i32-tensor (g name)
  (let* ((ti (gguf-tensor g name)) (v (make-array (tensor-elements ti) :element-type '(signed-byte 32))))
    (call-with-tensor-bytes g ti (lambda (buf start end)
                                   (loop for i from 0 below end by 4 for k from (floor start 4)
                                         do (setf (aref v k)
                                                  (let ((u (logior (aref buf i) (ash (aref buf (+ i 1)) 8)
                                                                   (ash (aref buf (+ i 2)) 16) (ash (aref buf (+ i 3)) 24))))
                                                    (if (>= u (ash 1 31)) (- u (ash 1 32)) u))))))
    v))

;;; --- small ops, ds4.c's ------------------------------------------------------------

(defun d4-rms! (x out n eps &optional w (xo 0) (oo 0))
  "OUT = X * 1/sqrt(mean(X^2) + eps) (* W): the sum in double, the scale a
   float, as rms_norm_weight / rms_norm_no_weight."
  (declare (type f32vec x out) (type fixnum n xo oo) (type single-float eps) (optimize speed (safety 0)))
  (let ((ss 0d0))
    (declare (type double-float ss))
    (dotimes (i n) (let ((v (coerce (aref x (+ xo i)) 'double-float))) (incf ss (* v v))))
    (let ((scale (/ 1f0 (sqrt (+ (coerce (/ ss n) 'single-float) eps)))))
      (declare (type single-float scale))
      (if w
          (let ((w w)) (declare (type f32vec w))
            (dotimes (i n out) (setf (aref out (+ oo i)) (* (* (aref x (+ xo i)) scale) (aref w i)))))
          (dotimes (i n out) (setf (aref out (+ oo i)) (* (aref x (+ xo i)) scale)))))))

(declaim (inline d4-sigmoid d4-silu d4-softplus))
(defun d4-sigmoid (x)
  (declare (type single-float x))
  (if (>= x 0f0) (/ 1f0 (+ 1f0 (exp (- x)))) (let ((e (exp x))) (/ e (+ 1f0 e)))))
(defun d4-silu (x) (declare (type single-float x)) (* x (d4-sigmoid x)))
(defun d4-softplus (x)
  (declare (type single-float x))
  (cond ((> x 20f0) x) ((< x -20f0) (exp x)) (t (log (+ 1f0 (exp x))))))

(defun f32mv! (pool w x out &optional (xo 0))
  "OUT = W X for an F32MAT, rows split across POOL."
  (declare (type f32mat w) (type f32vec x out) (type fixnum xo))
  (let ((rows (f32mat-rows w)) (cols (f32mat-cols w)) (data (f32mat-data w)))
    (declare (type fixnum rows cols) (type f32vec data))
    (pool-run pool (lambda (i c)
                     (multiple-value-bind (s e) (share rows i c)
                       (locally (declare (optimize speed (safety 0)) (type fixnum s e))
                         (loop for r of-type fixnum from s below e
                               do (let ((acc 0f0) (base (* r cols)))
                                    (declare (type single-float acc) (type fixnum base))
                                    (dotimes (j cols) (incf acc (* (aref data (+ base j)) (aref x (+ xo j)))))
                                    (setf (aref out r) acc)))))))
    out))

(defun q8mv! (pool w x out &optional (xq (make-q8vec (q8mat-cols w))))
  "OUT = W X for a Q8MAT, X quantized to Q8_0 first."
  (quantize-q8! x xq 0 (q8mat-cols w))
  (matvec! pool w xq out))

;;; --- FP8 and FP4 round trips (the model was trained through them) ---------------------

(defparameter +e4m3-values+
  (let ((v (make-array 127 :element-type 'single-float)))
    (dotimes (i 127 v)
      (let ((e (ldb (byte 4 3) i)) (mant (ldb (byte 3 0) i)))
        (setf (aref v i) (if (zerop e)
                             (* (float mant 1f0) 0.001953125f0)
                             (* (+ 1f0 (* (float mant 1f0) 0.125f0)) (expt 2f0 (- e 7)))))))))

(defun e4m3-round (x)
  "X to the nearest E4M3 value (ties to the even code), as dsv4_e4m3fn_dequant_cpu."
  (declare (type single-float x))
  (let* ((sign (if (< x 0f0) -1f0 1f0)) (ax (min (abs x) 448f0)) (lo 0) (hi 126) (tab +e4m3-values+))
    (declare (type (simple-array single-float (127)) tab) (type fixnum lo hi))
    (loop while (< lo hi)
          do (let ((mid (ash (+ lo hi 1) -1)))
               (if (<= (aref tab mid) ax) (setf lo mid) (setf hi (1- mid)))))
    (let ((best lo))
      (when (< best 126)
        (let ((bd (abs (- ax (aref tab best)))) (nd (abs (- ax (aref tab (1+ best))))))
          (when (or (< nd bd) (and (= nd bd) (evenp (1+ best)) (oddp best)))
            (incf best))))
      (* sign (aref tab best)))))

(defun fp8-kv-round! (x o head-dim n-rot)
  "dsv4_fp8_kv_quantize_row_inplace_cpu: X's first HEAD-DIM - N-ROT dims at O,
   64 at a time, through E4M3 with a power-of-two scale."
  (declare (type f32vec x) (type fixnum o head-dim n-rot))
  (loop for off from 0 below (- head-dim n-rot) by 64
        do (let ((amax 0f0))
             (declare (type single-float amax))
             (dotimes (i 64) (setf amax (max amax (abs (aref x (+ o off i))))))
             (when (< amax 1f-4) (setf amax 1f-4))
             (let ((scale (scale-float 1f0 (ceiling (log (/ amax 448f0) 2)))))
               (dotimes (i 64)
                 (let ((v (max -448f0 (min 448f0 (/ (aref x (+ o off i)) scale)))))
                   (setf (aref x (+ o off i)) (* (e4m3-round v) scale)))))))
  x)

(defparameter +e2m1-values+ (coerce '(0f0 0.5f0 1f0 1.5f0 2f0 3f0 4f0 6f0) '(simple-array single-float (8))))

(defun e2m1-round (x)
  (declare (type single-float x))
  (let* ((sign (if (< x 0f0) -1f0 1f0)) (ax (min (abs x) 6f0)) (best 0) (bd (abs ax)))
    (loop for i from 1 below 8
          do (let ((d (abs (- ax (aref +e2m1-values+ i)))))
               (when (or (< d bd) (and (= d bd) (evenp i) (oddp best)))
                 (setf best i bd d))))
    (* sign (aref +e2m1-values+ best))))

(defun indexer-qat! (x o)
  "dsv4_indexer_qat_row_inplace_cpu: a 128-wide row at O through a scaled
   Hadamard transform, then FP4 (E2M1) in blocks of 32."
  (declare (type f32vec x) (type fixnum o))
  (loop for stride = 1 then (* 2 stride) while (< stride 128)
        do (loop for base from 0 below 128 by (* 2 stride)
                 do (dotimes (i stride)
                      (let ((a (aref x (+ o base i))) (b (aref x (+ o base stride i))))
                        (setf (aref x (+ o base i)) (+ a b) (aref x (+ o base stride i)) (- a b))))))
  (dotimes (i 128) (setf (aref x (+ o i)) (* (aref x (+ o i)) 0.08838834764831845f0)))
  (loop for off from 0 below 128 by 32
        do (let ((amax 0f0))
             (dotimes (i 32) (setf amax (max amax (abs (aref x (+ o off i))))))
             (when (< amax 7.052966104933725f-38) (setf amax 7.052966104933725f-38))
             (let ((scale (scale-float 1f0 (ceiling (log (/ amax 6f0) 2)))))
               (dotimes (i 32)
                 (let ((v (max -6f0 (min 6f0 (/ (aref x (+ o off i)) scale)))))
                   (setf (aref x (+ o off i)) (* (e2m1-round v) scale)))))))
  x)

(defun f16-round (x) (half-to-single (single-to-half x)))

;;; --- RoPE: the last N-ROT dims of each head ---------------------------------------

(defun d4-rope! (m x n-heads head-dim pos il &key inverse (xo 0))
  "rope_tail_layer_inplace: on a compressed layer, base COMP-ROPE-BASE with
   YaRN interpolation (its magnitude factor cancelled, as DeepSeek V4's
   reference does), else plain RoPE at ROPE-BASE."
  (let* ((n-rot (ds4f-n-rot m)) (compressed (plusp (svref (ds4f-ratios m) il)))
         (base (if compressed (ds4f-comp-rope-base m) (ds4f-rope-base m)))
         (fscale (if (and compressed (plusp (ds4f-rope-factor m))) (/ 1f0 (ds4f-rope-factor m)) 1f0))
         (ext (if (and compressed (> (ds4f-rope-factor m) 1f0)) 1f0 0f0))
         (attn (if (and (/= ext 0f0) (> fscale 0f0)) (/ 1f0 (+ 1f0 (* 0.1f0 (log (/ 1f0 fscale))))) 1f0))
         (theta-scale (expt base (/ -2f0 n-rot)))
         (sin-sign (if inverse -1f0 1f0))
         (corr0 0f0) (corr1 0f0))
    (declare (type single-float base fscale ext attn theta-scale sin-sign corr0 corr1))
    (when (/= ext 0f0)
      (flet ((corr (beta) (/ (* (float n-rot 1f0) (log (/ (float (ds4f-rope-orig m) 1f0)
                                                          (* beta 2f0 (coerce pi 'single-float)))))
                             (* 2f0 (log base)))))
        (setf corr0 (max 0f0 (ffloor (corr (ds4f-beta-fast m))))
              corr1 (min (float (1- n-rot) 1f0) (fceiling (corr (ds4f-beta-slow m)))))))
    (dotimes (h n-heads x)
      (let ((tail (+ xo (* h head-dim) (- head-dim n-rot))) (theta-extrap (float pos 1f0)))
        (declare (type single-float theta-extrap))
        (loop for i from 0 below n-rot by 2
              do (let* ((interp (* fscale theta-extrap)) (theta interp) (mscale attn))
                   (declare (type single-float interp theta mscale))
                   (when (/= ext 0f0)
                     (let* ((y (/ (- (float (floor i 2) 1f0) corr0) (max 0.001f0 (- corr1 corr0))))
                            (ramp (* (- 1f0 (min 1f0 (max 0f0 y))) ext)))
                       (setf theta (+ (* interp (- 1f0 ramp)) (* theta-extrap ramp))
                             mscale (* mscale (+ 1f0 (* 0.1f0 (log (/ 1f0 fscale))))))))
                   (let ((c (* (cos theta) mscale)) (s (* sin-sign (sin theta) mscale))
                         (x0 (aref x (+ tail i))) (x1 (aref x (+ tail i 1))))
                     (setf (aref x (+ tail i)) (- (* x0 c) (* x1 s))
                           (aref x (+ tail i 1)) (+ (* x0 s) (* x1 c))))
                   (setf theta-extrap (* theta-extrap theta-scale))))))))

;;; --- hyper-connections -------------------------------------------------------------

(defun hc-split (m mix scale base)
  "hc_split_sinkhorn_one: (values pre post comb) from the 24 mixes."
  (let* ((n (ds4f-n-hc m)) (eps 1f-6)
         (pre (f32vec n)) (post (f32vec n)) (c (f32vec (* n n))))
    (declare (type f32vec mix scale base pre post c))
    (dotimes (i n) (setf (aref pre i) (+ (/ 1f0 (+ 1f0 (exp (- (+ (* (aref mix i) (aref scale 0)) (aref base i)))))) eps)))
    (dotimes (i n)
      (let ((o (+ n i)))
        (setf (aref post i) (/ 2f0 (+ 1f0 (exp (- (+ (* (aref mix o) (aref scale 1)) (aref base o)))))))))
    (dotimes (dst n)
      (let ((mx -1f30))
        (dotimes (src n)
          (let* ((idx (+ src (* dst n))) (o (+ (* 2 n) idx))
                 (v (+ (* (aref mix o) (aref scale 2)) (aref base o))))
            (setf (aref c idx) v mx (max mx v))))
        (let ((sum 0f0))
          (dotimes (src n)
            (let ((idx (+ src (* dst n))))
              (setf (aref c idx) (exp (- (aref c idx) mx)))
              (incf sum (aref c idx))))
          (let ((inv (/ 1f0 sum)))
            (dotimes (src n) (let ((idx (+ src (* dst n)))) (setf (aref c idx) (+ (* (aref c idx) inv) eps))))))))
    (flet ((cols () (dotimes (src n)
                      (let ((sum 0f0))
                        (dotimes (dst n) (incf sum (aref c (+ src (* dst n)))))
                        (let ((inv (/ 1f0 (+ sum eps))))
                          (dotimes (dst n) (setf (aref c (+ src (* dst n))) (* (aref c (+ src (* dst n))) inv)))))))
           (rows () (dotimes (dst n)
                      (let ((sum 0f0))
                        (dotimes (src n) (incf sum (aref c (+ src (* dst n)))))
                        (let ((inv (/ 1f0 (+ sum eps))))
                          (dotimes (src n) (setf (aref c (+ src (* dst n))) (* (aref c (+ src (* dst n))) inv))))))))
      (cols)
      (loop repeat (1- (ds4f-sinkhorn m)) do (rows) (cols)))
    (values pre post c)))

(defun hc-pre (m fn scale base streams out flat)
  "hc_pre_from_state_one_scratch: OUT = the pre-weighted sum of STREAMS;
   (values post comb)."
  (let* ((n (ds4f-n-hc m)) (d (ds4f-d m)) (mix (f32vec (f32mat-rows fn))))
    (d4-rms! streams flat (* n d) (ds4f-eps m))
    (f32mv! (ds4f-pool m) fn flat mix)
    (multiple-value-bind (pre post comb) (hc-split m mix scale base)
      (dotimes (j d)
        (let ((acc 0f0))
          (dotimes (h n) (incf acc (* (aref streams (+ (* h d) j)) (aref pre h))))
          (setf (aref out j) acc)))
      (values post comb))))

(defun hc-post! (m out-streams block-out streams post comb)
  "hc_post_one: stream DST = BLOCK-OUT * POST[DST] + sum_src COMB[dst + src*n] * stream SRC."
  (let ((n (ds4f-n-hc m)) (d (ds4f-d m)))
    (dotimes (dst n out-streams)
      (dotimes (j d)
        (let ((acc (* (aref block-out j) (aref post dst))))
          (dotimes (src n) (incf acc (* (aref comb (+ dst (* src n))) (aref streams (+ (* src d) j)))))
          (setf (aref out-streams (+ (* dst d) j)) acc))))))

;;; --- per-context state -------------------------------------------------------------

(defstruct (d4cache (:conc-name dc-))
  raw (n-raw 0)                         ; the last SWA rows, oldest first
  comp (n-comp 0) st-kv st-score        ; compressed rows; the compressor's window
  ix-comp (n-ix 0) ix-kv ix-score)      ; the indexer's

(defstruct (d4ctx (:constructor %make-d4ctx)) model (pos 0) caches logits n-ctx)

(defun make-d4ctx (m &key (n-ctx 4096))
  (let ((hd (ds4f-head-dim m)) (ixd (ds4f-ix-dim m)))
    (%make-d4ctx
     :model m :n-ctx n-ctx :logits (f32vec (ds4f-n-vocab m))
     :caches (map 'simple-vector
                  (lambda (l)
                    (let* ((ratio (dl-ratio l)) (coff (if (= ratio 4) 2 1)) (cap (+ (floor n-ctx (max ratio 1)) 2)))
                      (flet ((neg (n) (make-array n :element-type 'single-float :initial-element -1f30)))
                        (make-d4cache
                         :raw (f32vec (* (ds4f-swa m) hd))
                         :comp (and (plusp ratio) (f32vec (* cap hd)))
                         :st-kv (and (plusp ratio) (f32vec (* coff ratio coff hd)))
                         :st-score (and (plusp ratio) (neg (* coff ratio coff hd)))
                         :ix-comp (and (= ratio 4) (f32vec (* cap ixd)))
                         :ix-kv (and (= ratio 4) (f32vec (* coff ratio coff ixd)))
                         :ix-score (and (= ratio 4) (neg (* coff ratio coff ixd)))))))
                  (ds4f-layers m)))))

(defun push-raw! (m c kv)
  "kv_cache_push_raw: KV, through F16, at the end of the window (the oldest
   row dropped when it is full)."
  (let* ((hd (ds4f-head-dim m)) (cap (ds4f-swa m)) (raw (dc-raw c)))
    (declare (type f32vec raw kv))
    (when (= (dc-n-raw c) cap)
      (replace raw raw :start2 hd)
      (decf (dc-n-raw c)))
    (let ((o (* (dc-n-raw c) hd)))
      (dotimes (i hd) (setf (aref raw (+ o i)) (f16-round (aref kv i)))))
    (incf (dc-n-raw c))))

;;; --- the compressor ----------------------------------------------------------------

(defun compress-step (m wkv wgate ape norm x st-kv st-score head-dim ratio il pos)
  "compressor_decode_one_decode_scratch: this token's projected row and
   score (plus the learned position term APE) into the window; every RATIO
   tokens, the window softmax-pooled per dim, normed, RoPE'd at its first
   position and rounded (FP8, or the indexer's FP4): the compressed row, or
   NIL.  With ratio 4 the window overlaps the previous one by half."
  (let* ((pool (ds4f-pool m)) (coff (if (= ratio 4) 2 1)) (width (* coff head-dim))
         (pos-mod (mod pos ratio)) (row (if (= ratio 4) (+ ratio pos-mod) pos-mod))
         (kv-cur (f32vec width)) (sc-cur (f32vec width)) (ape-d (f32mat-data ape)))
    (declare (type f32vec st-kv st-score kv-cur sc-cur ape-d))
    (f32mv! pool wkv x kv-cur)
    (f32mv! pool wgate x sc-cur)
    (dotimes (j width) (incf (aref sc-cur j) (aref ape-d (+ (* pos-mod (f32mat-cols ape)) j))))
    (replace st-kv kv-cur :start1 (* row width))
    (replace st-score sc-cur :start1 (* row width))
    (when (zerop (mod (1+ pos) ratio))
      (let ((pooled (f32vec head-dim)) (out (f32vec head-dim)))
        (dotimes (j head-dim)
          (let ((mx -1f30))
            (if (= ratio 4)
                (dotimes (r ratio)
                  (setf mx (max mx (aref st-score (+ (* r width) j))
                                (aref st-score (+ (* (+ ratio r) width) head-dim j)))))
                (dotimes (r ratio) (setf mx (max mx (aref st-score (+ (* r width) j))))))
            (if (<= mx -5f29)
                (setf (aref pooled j) 0f0)
                (let ((denom 0f0) (sum 0f0))
                  (if (= ratio 4)
                      (dotimes (r ratio)
                        (let ((wp (exp (- (aref st-score (+ (* r width) j)) mx)))
                              (wc (exp (- (aref st-score (+ (* (+ ratio r) width) head-dim j)) mx))))
                          (incf denom (+ wp wc))
                          (incf sum (* wp (aref st-kv (+ (* r width) j))))
                          (incf sum (* wc (aref st-kv (+ (* (+ ratio r) width) head-dim j))))))
                      (dotimes (r ratio)
                        (let ((w (exp (- (aref st-score (+ (* r width) j)) mx))))
                          (incf denom w)
                          (incf sum (* w (aref st-kv (+ (* r width) j)))))))
                  (setf (aref pooled j) (if (> denom 0f0) (/ sum denom) 0f0))))))
        (d4-rms! pooled out head-dim (ds4f-eps m) norm)
        (d4-rope! m out 1 head-dim (- (1+ pos) ratio) il)
        (if (= head-dim (ds4f-head-dim m))
            (fp8-kv-round! out 0 head-dim (ds4f-n-rot m))
            (indexer-qat! out 0))
        (when (= ratio 4)
          ;; the current half becomes the previous one
          (dotimes (r ratio)
            (replace st-kv st-kv :start1 (* r width) :start2 (* (+ ratio r) width) :end2 (* (+ ratio r 1) width))
            (replace st-score st-score :start1 (* r width) :start2 (* (+ ratio r) width) :end2 (* (+ ratio r 1) width))))
        out))))

(defun push-comp! (rows n-slot row-dim kv c)
  "kv_cache_push_comp: KV through F16 as the next compressed row."
  (let ((o (* (funcall n-slot c) row-dim)))
    (dotimes (i row-dim) (setf (aref rows (+ o i)) (f16-round (aref kv i))))))

;;; --- the indexer -------------------------------------------------------------------

(defun indexer-allowed (m l c attn-norm qr-norm il pos)
  "indexer_allowed_decode_one_decode_scratch: which compressed rows a
   ratio-4 layer attends to -- all of them while there are at most TOP-K,
   else the TOP-K whose ReLU'd per-head dots with the indexer's queries,
   summed under learned head weights, are largest.  NIL means all."
  (let* ((n-comp (dc-n-ix c)) (top-k (min (ds4f-ix-top-k m) n-comp)))
    (when (or (zerop n-comp) (= top-k n-comp)) (return-from indexer-allowed nil))
    (let* ((hd (ds4f-ix-dim m)) (nh (ds4f-ix-heads m)) (pool (ds4f-pool m))
           (q (f32vec (* nh hd))) (w (f32vec nh)) (scores (f32vec n-comp))
           (allowed (make-array n-comp :element-type 'bit :initial-element 0))
           (rows (dc-ix-comp c)))
      (declare (type f32vec q w scores rows))
      (f32mv! pool (dl-ix-q-b l) qr-norm q)
      (d4-rope! m q nh hd pos il)
      (dotimes (h nh) (indexer-qat! q (* h hd)))
      (f32mv! pool (dl-ix-proj l) attn-norm w)
      (let ((scale (/ 1f0 (sqrt (float (* hd nh) 1f0)))))
        (dotimes (h nh) (setf (aref w h) (* (aref w h) scale))))
      (parallel-for (r n-comp pool)
        (let ((s 0f0))
          (declare (type single-float s))
          (dotimes (h nh)
            (let ((dot 0f0))
              (declare (type single-float dot))
              (dotimes (i hd) (incf dot (* (aref rows (+ (* r hd) i)) (aref q (+ (* h hd) i)))))
              (incf s (* (max dot 0f0) (aref w h)))))
          (setf (aref scores r) s)))
      (dotimes (k top-k allowed)
        (let ((best 0) (bs -1f30))
          (dotimes (r n-comp)
            (when (and (zerop (aref allowed r)) (> (aref scores r) bs)) (setf best r bs (aref scores r))))
          (setf (aref allowed best) 1))))))

;;; --- attention ---------------------------------------------------------------------

(defun d4-attend! (m l q c allowed heads)
  "layer_attention_mixed_one: each head over the raw window and the allowed
   compressed rows, the sink in the denominator."
  (let* ((hd (ds4f-head-dim m)) (nh (ds4f-n-head m)) (sinks (dl-sinks l))
         (raw (dc-raw c)) (n-raw (dc-n-raw c))
         (comp (dc-comp c)) (n-comp (if comp (dc-n-comp c) 0))
         (scale (/ 1f0 (sqrt (float hd 1f0)))))
    (declare (type f32vec q heads sinks raw))
    (parallel-for (h nh (ds4f-pool m))
      (let ((score (make-array (+ n-raw n-comp) :element-type 'single-float))
            (mx (aref sinks h)) (qo (* h hd)))
        (declare (type single-float mx))
        (flet ((dot (rows r) (let ((acc 0f0)) (declare (type single-float acc) (type f32vec rows))
                               (dotimes (i hd acc) (incf acc (* (aref q (+ qo i)) (aref rows (+ (* r hd) i))))))))
          (dotimes (r n-raw) (let ((s (* (dot raw r) scale))) (setf (aref score r) s mx (max mx s))))
          (dotimes (r n-comp)
            (if (and allowed (zerop (aref allowed r)))
                (setf (aref score (+ n-raw r)) -1f30)
                (let ((s (* (dot comp r) scale))) (setf (aref score (+ n-raw r)) s mx (max mx s))))))
        (dotimes (i hd) (setf (aref heads (+ qo i)) 0f0))
        (let ((denom (exp (- (aref sinks h) mx))))
          (declare (type single-float denom))
          (flet ((axpy (rows r w) (declare (type f32vec rows))
                   (dotimes (i hd) (incf (aref heads (+ qo i)) (* w (aref rows (+ (* r hd) i)))))))
            (dotimes (r n-raw) (let ((w (exp (- (aref score r) mx)))) (incf denom w) (axpy raw r w)))
            (dotimes (r n-comp)
              (let ((s (aref score (+ n-raw r))))
                (unless (<= s -5f29) (let ((w (exp (- s mx)))) (incf denom w) (axpy comp r w))))))
          (let ((inv (/ 1f0 denom)))
            (dotimes (i hd) (setf (aref heads (+ qo i)) (* (aref heads (+ qo i)) inv)))))))
    heads))

(defun d4-attention (m l li c streams pos)
  "One layer's attention block for the token at POS: (values out post comb)
   with OUT the block's output and POST/COMB its HC post."
  (let* ((d (ds4f-d m)) (hd (ds4f-head-dim m)) (nh (ds4f-n-head m)) (pool (ds4f-pool m))
         (ratio (dl-ratio l)) (eps (ds4f-eps m))
         (cur (f32vec d)) (norm (f32vec d)) (flat (f32vec (* (ds4f-n-hc m) d))))
    (multiple-value-bind (post comb)
        (hc-pre m (dl-hc-attn-fn l) (dl-hc-attn-scale l) (dl-hc-attn-base l) streams cur flat)
      (d4-rms! cur norm d eps (dl-attn-norm l))
      ;; Q through its LoRA, normed per head; KV normed
      (let ((qr (f32vec (ds4f-lora-q m))) (qr-norm (f32vec (ds4f-lora-q m)))
            (q (f32vec (* nh hd))) (kv0 (f32vec hd)) (kv (f32vec hd)))
        (q8mv! pool (dl-q-a l) norm qr)
        (d4-rms! qr qr-norm (ds4f-lora-q m) eps (dl-q-a-norm l))
        (q8mv! pool (dl-q-b l) qr-norm q)
        (dotimes (h nh) (d4-rms! q q hd eps nil (* h hd) (* h hd)))
        (q8mv! pool (dl-kv l) norm kv0)
        (d4-rms! kv0 kv hd eps (dl-kv-norm l))
        (d4-rope! m q nh hd pos li)
        (d4-rope! m kv 1 hd pos li)
        (fp8-kv-round! kv 0 hd (ds4f-n-rot m))
        (push-raw! m c kv)
        (when (plusp ratio)
          (let ((row (compress-step m (dl-comp-kv l) (dl-comp-gate l) (dl-comp-ape l) (dl-comp-norm l)
                                    norm (dc-st-kv c) (dc-st-score c) hd ratio li pos)))
            (when row
              (push-comp! (dc-comp c) #'dc-n-comp hd row c)
              (incf (dc-n-comp c))))
          (when (= ratio 4)
            (let ((row (compress-step m (dl-ixc-kv l) (dl-ixc-gate l) (dl-ixc-ape l) (dl-ixc-norm l)
                                      norm (dc-ix-kv c) (dc-ix-score c) (ds4f-ix-dim m) ratio li pos)))
              (when row
                (push-comp! (dc-ix-comp c) #'dc-n-ix (ds4f-ix-dim m) row c)
                (incf (dc-n-ix c))))))
        (let* ((allowed (and (= ratio 4) (indexer-allowed m l c norm qr-norm li pos)))
               (heads (d4-attend! m l q c allowed (f32vec (* nh hd)))))
          (d4-rope! m heads nh hd pos li :inverse t)
          ;; out: 8 groups of 8 heads, each to rank LORA-O, then to the embedding
          (let* ((groups (ds4f-out-groups m)) (rank (ds4f-lora-o m)) (gdim (* hd (floor nh groups)))
                 (low (f32vec (* groups rank))) (out (f32vec d)))
            (grouped-q8mv! pool (dl-out-a l) heads groups gdim rank low)
            (q8mv! pool (dl-out-b l) low out)
            (values out post comb)))))))

(defun grouped-q8mv! (pool w x groups gdim rank out)
  "matvec_q8_0_grouped_rows: row r of W (GROUPS * RANK rows of GDIM) against
   group r/RANK of X, each group quantized on its own."
  (let ((xq (make-q8vec (* groups gdim))))
    (dotimes (g groups) (quantize-q8! x xq (* g gdim) gdim (* g gdim) (* g (floor gdim 32))))
    (let ((xqq (q8vec-quants xq)) (xs (q8vec-scales xq)))
      ;; four rows at a time, as MATVEC!; a group's RANK rows are a multiple of four
      (with-dot-rows (dot-rows w xqq xs out)
        (parallel-for (g4 (floor (q8mat-rows w) 4) pool)
          (let ((xo (* (floor (* 4 g4) rank) gdim)))
            (dot-rows g4 xo (floor xo 32) (* 4 g4))))))
    out))

;;; --- the FFN -----------------------------------------------------------------------

(defun d4-route (m l li x token)
  "(values experts weights): the hash table's experts in the first layers,
   else the top N-USED of sqrt(softplus(router)) + bias; weights from the
   unbiased probabilities, renormalized and scaled."
  (let* ((ne (ds4f-n-expert m)) (nu (ds4f-n-used m))
         (logits (f32vec ne)) (probs (f32vec ne)) (sel (make-array nu)) (w (f32vec nu)))
    (f32mv! (ds4f-pool m) (dl-gate-inp l) x logits)
    (dotimes (i ne) (setf (aref probs i) (sqrt (d4-softplus (aref logits i)))))
    (if (dl-tid2eid l)
        (dotimes (i nu) (setf (aref sel i) (aref (dl-tid2eid l) (+ (* token nu) i))))
        (let ((s (copy-seq probs)) (idx (make-array nu :initial-element -1)))
          (when (dl-probs-b l) (dotimes (i ne) (incf (aref s i) (aref (dl-probs-b l) i))))
          ;; topk_desc: insertion into a sorted list of K, earlier wins ties
          (dotimes (i ne)
            (loop for j below nu
                  when (or (minusp (aref idx j)) (> (aref s i) (aref s (aref idx j))))
                    do (loop for k from (1- nu) above j do (setf (aref idx k) (aref idx (1- k))))
                       (setf (aref idx j) i)
                       (return)))
          (replace sel idx)))
    (let ((sum 0f0))
      (dotimes (i nu) (setf (aref w i) (aref probs (aref sel i))) (incf sum (aref w i)))
      (when (< sum 6.103515625f-5) (setf sum 6.103515625f-5))
      (dotimes (i nu) (setf (aref w i) (* (/ (aref w i) sum) (ds4f-expert-scale m)))))
    (values sel w)))

(defun d4-ffn (m l li streams token)
  "One layer's FFN block: (values out post comb)."
  (let* ((d (ds4f-d m)) (pool (ds4f-pool m)) (eps (ds4f-eps m)) (ff (ds4f-ff-exp m)) (nu (ds4f-n-used m))
         (clamp (svref (ds4f-clamps m) li)) (wf (ds4f-file m))
         (cur (f32vec d)) (norm (f32vec d)) (flat (f32vec (* (ds4f-n-hc m) d))))
    (multiple-value-bind (post comb)
        (hc-pre m (dl-hc-ffn-fn l) (dl-hc-ffn-scale l) (dl-hc-ffn-base l) streams cur flat)
      (d4-rms! cur norm d eps (dl-ffn-norm l))
      (multiple-value-bind (sel w) (d4-route m l li norm token)
        (let ((xq (make-q8kvec d)) (mid (f32vec (* nu ff))) (moe (f32vec d)) (shared (f32vec d)))
          (quantize-q8k! norm xq)
          ;; gate and up: silu(clamp(g)) * clamp(u) * weight, per expert
          (destructuring-bind (goff grb grows gcols) (dl-gate-exps l)
            (destructuring-bind (uoff urb urows ucols) (dl-up-exps l)
              (declare (ignore ucols urows))
              (let ((nb (floor gcols +qk-k+)))
                (parallel-for (idx (* nu ff) pool)
                  (multiple-value-bind (slot r) (floor idx ff)
                    (let* ((e (aref sel slot))
                           (g (iq2xxs-row-dot wf (+ goff (* (+ (* e grows) r) grb)) nb xq 0))
                           (u (iq2xxs-row-dot wf (+ uoff (* (+ (* e grows) r) urb)) nb xq 0)))
                      (declare (type single-float g u))
                      (when (> clamp 1f-6)
                        (setf g (min g clamp) u (max (- clamp) (min u clamp))))
                      (setf (aref mid idx) (* (* (d4-silu g) u) (aref w slot)))))))))
          ;; down, summed over the experts in order
          (let ((midq (make-q8kvec (* nu ff))))
            (dotimes (i nu) (quantize-q8k! mid midq (* i ff) ff (* i ff)))
            (destructuring-bind (doff drb drows dcols) (dl-down-exps l)
              (let ((nb (floor dcols +qk-k+)))
                (parallel-for (r d pool)
                  (let ((acc 0f0))
                    (declare (type single-float acc))
                    (dotimes (i nu)
                      (incf acc (q2k-row-dot wf (+ doff (* (+ (* (aref sel i) drows) r) drb)) nb midq (* i ff))))
                    (setf (aref moe r) acc))))))
          ;; the shared expert
          (let ((g (f32vec ff)) (u (f32vec ff)))
            (q8mv! pool (dl-sh-gate l) norm g)
            (q8mv! pool (dl-sh-up l) norm u)
            (dotimes (i ff)
              (let ((gi (aref g i)) (ui (aref u i)))
                (when (> clamp 1f-6) (setf gi (min gi clamp) ui (max (- clamp) (min ui clamp))))
                (setf (aref g i) (* (d4-silu gi) ui))))
            (q8mv! pool (dl-sh-down l) g shared))
          (dotimes (i d) (setf (aref moe i) (+ (aref moe i) (aref shared i))))
          (values moe post comb))))))

;;; --- a token -----------------------------------------------------------------------

(defun ds4f-forward (ctx token &key (logits t))
  "TOKEN at the next position; the logits, if LOGITS, in the context."
  #+sbcl (sb-int:with-float-traps-masked (:overflow :inexact :underflow) (ds4f-forward-1 ctx token logits))
  #-sbcl (ds4f-forward-1 ctx token logits))

(defun ds4f-forward-1 (ctx token logits)
  (let* ((m (d4ctx-model ctx)) (d (ds4f-d m)) (n (ds4f-n-hc m)) (pos (d4ctx-pos ctx))
         (streams (f32vec (* n d))) (next (f32vec (* n d))) (wf (ds4f-file m)))
    (declare (type u8vec wf))
    (let ((o (+ (ds4f-embd-off m) (* token d 2))))
      (dotimes (j d)
        (let ((v (f16-at wf (+ o (* 2 j)))))
          (dotimes (h n) (setf (aref streams (+ (* h d) j)) v)))))
    (dotimes (li (ds4f-n-layer m))
      (let ((l (svref (ds4f-layers m) li)) (c (svref (d4ctx-caches ctx) li)))
        (multiple-value-bind (out post comb) (d4-attention m l li c streams pos)
          (hc-post! m next out streams post comb))
        (rotatef streams next)
        (multiple-value-bind (out post comb) (d4-ffn m l li streams token)
          (hc-post! m next out streams post comb))
        (rotatef streams next)))
    (incf (d4ctx-pos ctx))
    (when logits
      ;; output_logits_one: the streams' sigmoid-weighted sum, normed, to logits
      (let* ((flat (f32vec (* n d))) (pre (f32vec n)) (w (f32vec n)) (embd (f32vec d)) (norm (f32vec d))
             (scale (ds4f-out-hc-scale m)) (base (ds4f-out-hc-base m)))
        (d4-rms! streams flat (* n d) (ds4f-eps m))
        (f32mv! (ds4f-pool m) (ds4f-out-hc-fn m) flat pre)
        (dotimes (i n) (setf (aref w i) (+ (d4-sigmoid (+ (* (aref pre i) (aref scale 0)) (aref base i))) (ds4f-hc-eps m))))
        (dotimes (j d)
          (let ((acc 0f0)) (dotimes (h n) (incf acc (* (aref streams (+ (* h d) j)) (aref w h)))) (setf (aref embd j) acc)))
        (d4-rms! embd norm d (ds4f-eps m) (ds4f-output-norm m))
        (q8mv! (ds4f-pool m) (ds4f-output m) norm (d4ctx-logits ctx))))
    ctx))
