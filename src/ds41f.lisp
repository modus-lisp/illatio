;;;; ds41f.lisp — DeepSeek V4.1 Flash (`deepseek41`), one token at a time on the CPU.
;;;
;;; ds4's V4.1 decode (ds41_graph_step and what it calls), as differences
;;; from V4 Flash (ds4f.lisp, whose model struct, matvecs, Sinkhorn split,
;;; router and expert dots this reuses).  The shape is V4's with:
;;;
;;;   - BF16 rounding (round to nearest even, kept as f32) at nearly every
;;;     boundary: norms, projections, HC sums and expands, the shared expert.
;;;   - HC's pre weights one sublayer late: a layer's attention sums the
;;;     streams by the previous FFN mix's pre ({1,0,0,0} at layer 0), its FFN
;;;     by the attention mix's; the head by the last FFN's (no head mix).
;;;   - Compressed KV shared: only layers 2, 8, 14 and 20 compress (ratio 2,
;;;     a softmax over each pair of tokens; ratio 1 at 20: every token), each
;;;     into a cache that it and the layers up to the next such layer read.
;;;     Every compressed layer attends to 512 of the rows, picked by an
;;;     indexer on layers 2, 8, 14, 20, 24, 28, 32, 36 and reused by the
;;;     layers after each.  The indexer's keys are a projection of the
;;;     compressed row; no Hadamard, FP4 with power-of-two scales.
;;;   - Raw KV through FP8 in blocks of 32 over the whole row; compressed
;;;     rows through FP4 with E4M3 scales per 16.
;;;   - RoPE's angle a float product, its cos/sin in double, each turned
;;;     element rounded to BF16; no per-head norm of Q.
;;;   - Engram at layers 1 and 14: rows of a table (94 GiB each, read in the
;;;     file) chosen by hashing the last 2, 3 and 4 tokens, projected to a key
;;;     per stream and a value, gated into the streams before the layer.
;;;   - No hash-routed layers; 384 experts; RMS eps 1e-20.

(in-package #:illatio)

;;; --- BF16 and the V4.1 quantizers ----------------------------------------------------

(declaim (inline bf16 pow2ceil))
(defun bf16 (x)
  "X rounded to BF16 (to nearest, ties even), as a single-float; Inf/NaN as is."
  (declare (type single-float x))
  (let ((b (single-bits x)))
    (if (= (ldb (byte 8 23) b) 255)
        x
        (bits-single (logand (+ b #x7fff (ldb (byte 1 16) b)) #xffff0000)))))

(defun bf16! (v &optional (o 0) (n (- (length v) o)))
  (declare (type f32vec v) (type fixnum o n) (optimize speed (safety 0)))
  (loop for i of-type fixnum from o below (+ o n) do (setf (aref v i) (bf16 (aref v i))))
  v)

(defun pow2ceil (x)
  "The least power of two >= X (X normal and positive)."
  (declare (type single-float x))
  (let ((b (single-bits x)))
    (if (zerop (ldb (byte 23 0) b)) x (bits-single (+ (logand b #xff800000) #x800000)))))

(defun d41-quant! (x o n kind)
  "ds4_gpu_dsv41_quantize: X[O..O+N) in place, in blocks along the row, each
   element first BF16, the result BF16.  KIND :FP8 (E4M3, power-of-two scale
   per 32), :FP4 (E2M1, power-of-two scale per 32), :FP4-E4M3 (E2M1, E4M3
   scale per 16)."
  (declare (type f32vec x) (type fixnum o n))
  (let ((block (if (eq kind :fp4-e4m3) 16 32)))
    (loop for b from o below (+ o n) by block
          do (let ((peak 0f0))
               (declare (type single-float peak))
               (loop for i from b below (+ b block)
                     do (let ((v (bf16 (aref x i)))) (setf (aref x i) v peak (max peak (abs v)))))
               (let ((scale (ecase kind
                              (:fp8 (pow2ceil (* (max peak 1f-4) (/ 1f0 448f0))))
                              (:fp4 (pow2ceil (* (max peak (scale-float 1.5f0 -124)) (/ 1f0 6f0))))
                              (:fp4-e4m3 (e4m3-round (/ (max peak 0.01171875f0) 6f0))))))
                 (declare (type single-float scale))
                 (loop for i from b below (+ b block)
                       do (let ((q (/ (aref x i) scale)))
                            (setf (aref x i)
                                  (bf16 (* (if (eq kind :fp8) (e4m3-round q) (e2m1-round q)) scale))))))))
    x))

;;; --- V4.1's own: the layers' roles, Engram's tables, RoPE's frequencies ------------

(defstruct d41
  kv-src ix-src (cand-src -1) (cand-top 2048) (cand-block 8)
  eg-layers token-map (pad 0) primes mults eg-offs
  rope-plain rope-comp)                 ; 32 frequencies each: layers 0-1, the compressed ones

(defun d41-rope-freqs (m base compressed)
  "ds4's static table: f_i = 1/base^(i/32) in float; on a compressed layer
   YaRN's blend toward f_i/16 over i in [low, high]."
  (let* ((half (floor (ds4f-n-rot m) 2)) (f (f32vec half))
         (dbase (coerce base 'double-float)) (n-rot (ds4f-n-rot m)))
    (dotimes (i half f)
      (setf (aref f i) (/ 1f0 (coerce (expt dbase (/ i (float half 1d0))) 'single-float))))
    (when compressed
      (flet ((corr (beta) (/ (* n-rot (log (/ (float (ds4f-rope-orig m) 1d0) (* beta 2 pi)))) (* 2 (log dbase)))))
        (let ((low (coerce (ffloor (corr (coerce (ds4f-beta-fast m) 'double-float))) 'single-float))
              (high (coerce (fceiling (corr (coerce (ds4f-beta-slow m) 'double-float))) 'single-float))
              (factor (ds4f-rope-factor m)))
          (dotimes (i half)
            (let* ((ramp (min 1f0 (max 0f0 (/ (- (float i 1f0) low) (- high low)))))
                   (smooth (- 1f0 ramp)) (fi (aref f i)))
              (setf (aref f i) (+ (* (/ fi factor) (- 1f0 smooth)) (* fi smooth))))))))
    f))

(defun d41-rope-cs (m pos li &optional inverse)
  "ds4_gpu_dsv41_rope_stride's pairs' (cos, sin): angle pos * f_i (a float
   product), cos and sin in double; sin negated when INVERSE."
  (let* ((v (ds4f-v41 m)) (half (floor (ds4f-n-rot m) 2))
         (f (if (plusp (svref (ds4f-ratios m) li)) (d41-rope-comp v) (d41-rope-plain v)))
         (fpos (float pos 1f0)) (cs (f32vec (* 2 half))))
    (declare (type f32vec f cs) (type single-float fpos))
    (dotimes (i half cs)
      (let ((theta (coerce (* fpos (aref f i)) 'double-float)))
        (setf (aref cs (* 2 i)) (coerce (cos theta) 'single-float)
              (aref cs (1+ (* 2 i))) (let ((s (coerce (sin theta) 'single-float))) (if inverse (- s) s)))))))

(defun d41-rope! (m x n-heads width pos li &key inverse (xo 0))
  "The last N-ROT dims of each WIDTH-wide head at XO, adjacent pairs turned
   by D41-ROPE-CS's, each turned element rounded to BF16."
  (declare (type f32vec x) (type fixnum n-heads width pos li xo))
  (let ((n-rot (ds4f-n-rot m)) (cs (d41-rope-cs m pos li inverse)))
    (declare (type f32vec cs))
    (dotimes (i (floor n-rot 2) x)
      (let ((c (aref cs (* 2 i))) (s (aref cs (1+ (* 2 i)))))
        (dotimes (h n-heads)
          (let* ((j (+ xo (* h width) (- width n-rot) (* 2 i))) (x0 (aref x j)) (x1 (aref x (1+ j))))
            (setf (aref x j) (bf16 (- (* x0 c) (* x1 s)))
                  (aref x (1+ j)) (bf16 (+ (* x0 s) (* x1 c))))))))))

;;; --- loading -------------------------------------------------------------------------

(defun load-ds41f (path &key (threads (default-thread-count)))
  "DeepSeek V4.1 Flash at PATH, as LOAD-DS4F loads V4: the routed experts and
   Engram's tables left in the mapped file, the rest in memory."
  (let* ((g (read-gguf path)) (a "deepseek41") (cpu-q8 (ds4f-cpu-q8-p)))
    (flet ((k (key &optional default) (or (arch-key g a key) default))
           (f (key) (coerce (arch-key g a key) 'single-float))
           (q8-cpu (name) (and cpu-q8 (load-q8mat g name)))
           (fm-cpu (name) (and cpu-q8 (load-f32mat g name)))
           (f32 (name) (load-f32 g name)))
      (flet ((opt (name fn) (and (gguf-tensor g name nil) (funcall fn name))))
        (let* ((n-layer (k "num_hidden_layers"))
               (m (%make-ds4f
                   :gguf g :file (map-file-vector path) :pool (make-pool threads)
                   :n-layer n-layer :d (k "hidden_size") :n-head (k "num_attention_heads")
                   :head-dim (k "head_dim") :n-rot (k "qk_rope_head_dim")
                   :lora-q (k "q_lora_rank") :lora-o (k "o_lora_rank") :out-groups (k "o_groups")
                   :n-expert (k "n_routed_experts") :n-used (k "num_experts_per_tok")
                   :ff-exp (k "moe_intermediate_size")
                   :n-hc (k "hc_mult") :sinkhorn (k "hc_sinkhorn_iters") :hc-eps (f "hc_eps")
                   :swa (k "sliding_window") :ix-heads (k "index_n_heads")
                   :ix-dim (k "index_head_dim") :ix-top-k (k "index_topk")
                   :eps (f "rms_norm_eps") :rope-base (f "rope_theta") :comp-rope-base (f "compress_rope_theta")
                   :rope-factor (f "rope_scaling.factor")
                   :rope-orig (round (k "rope_scaling.original_max_position_embeddings"))
                   :beta-fast (f "rope_scaling.beta_fast") :beta-slow (f "rope_scaling.beta_slow")
                   :expert-scale (f "routed_scaling_factor")
                   :ratios (coerce (k "compress_ratios") 'simple-vector)
                   :clamps (make-array n-layer :initial-element (f "swiglu_limit"))
                   :n-vocab (k "vocab_size")
                   :embd-off (+ (gguf-data-offset g) (tensor-info-offset (gguf-tensor g "token_embd.weight")))
                   :output (q8-cpu "output.weight") :output-norm (f32 "output_norm.weight")))
               (eg-layers (coerce (k "engram.layer_ids") 'list)))
          (setf (ds4f-v41 m)
                (make-d41 :kv-src (coerce (k "kv_source_layer_ids") 'list)
                          :ix-src (coerce (k "index_source_layer_ids") 'list)
                          :cand-src (k "candidate_source_layer_id" -1)
                          :cand-top (k "candidate_topk_blocks" 2048) :cand-block (k "candidate_block_size" 8)
                          :eg-layers eg-layers
                          :token-map (coerce (k "engram.token_map") '(simple-array (unsigned-byte 32) (*)))
                          :pad (k "engram.pad_id")
                          :primes (coerce (k "engram.primes") 'simple-vector)
                          :mults (coerce (k "engram.multipliers") 'simple-vector)
                          :eg-offs (mapcar (lambda (li)
                                             (+ (gguf-data-offset g)
                                                (tensor-info-offset (gguf-tensor g (format nil "blk.~D.engram_embd.weight" li)))))
                                           eg-layers)))
          (setf (d41-rope-plain (ds4f-v41 m)) (d41-rope-freqs m (ds4f-rope-base m) nil)
                (d41-rope-comp (ds4f-v41 m)) (d41-rope-freqs m (ds4f-comp-rope-base m) t))
          (setf (ds4f-layers m)
                (coerce
                 (loop for i below n-layer
                       collect (flet ((n (s) (format nil "blk.~D.~A" i s)))
                                 (make-d4layer
                                  :ratio (svref (ds4f-ratios m) i)
                                  :attn-norm (f32 (n "attn_norm.weight"))
                                  :q-a (q8-cpu (n "attn_q_a.weight")) :q-a-norm (f32 (n "attn_q_a_norm.weight"))
                                  :q-b (q8-cpu (n "attn_q_b.weight"))
                                  :kv (q8-cpu (n "attn_kv.weight")) :kv-norm (f32 (n "attn_kv_a_norm.weight"))
                                  :sinks (f32 (n "attn_sinks.weight"))
                                  :out-a (q8-cpu (n "attn_output_a.weight")) :out-b (q8-cpu (n "attn_output_b.weight"))
                                  :comp-kv (opt (n "attn_compressor_kv.weight") #'fm-cpu)
                                  :comp-gate (opt (n "attn_compressor_gate.weight") #'fm-cpu)
                                  :comp-norm (opt (n "attn_compressor_norm.weight") #'f32)
                                  :ix-q-b (opt (n "indexer.attn_q_b.weight") #'fm-cpu)
                                  :ix-proj (opt (n "indexer.proj.weight") #'fm-cpu)
                                  :ix-k (opt (n "indexer.attn_k.weight") #'fm-cpu)
                                  :ix-k-norm (opt (n "indexer.k_norm.weight") #'f32)
                                  :eg-kv (opt (n "engram_kv.weight") #'fm-cpu)
                                  :eg-q-norm (opt (n "engram_q_norm.weight") #'f32)
                                  :eg-k-norm (opt (n "engram_k_norm.weight") #'f32)
                                  :ffn-norm (f32 (n "ffn_norm.weight"))
                                  :gate-inp (fm-cpu (n "ffn_gate_inp.weight"))
                                  :probs-b (opt (n "exp_probs_b.bias") #'f32)
                                  :gate-exps (expert-tensor g (n "ffn_gate_exps.weight"))
                                  :up-exps (expert-tensor g (n "ffn_up_exps.weight"))
                                  :down-exps (expert-tensor g (n "ffn_down_exps.weight"))
                                  :sh-gate (q8-cpu (n "ffn_gate_shexp.weight")) :sh-up (q8-cpu (n "ffn_up_shexp.weight"))
                                  :sh-down (q8-cpu (n "ffn_down_shexp.weight"))
                                  :hc-attn-fn (fm-cpu (n "hc_attn_fn.weight")) :hc-attn-scale (f32 (n "hc_attn_scale.weight"))
                                  :hc-attn-base (f32 (n "hc_attn_base.weight"))
                                  :hc-ffn-fn (fm-cpu (n "hc_ffn_fn.weight")) :hc-ffn-scale (f32 (n "hc_ffn_scale.weight"))
                                  :hc-ffn-base (f32 (n "hc_ffn_base.weight")))))
                 'simple-vector))
          m)))))

;;; --- per-context state ---------------------------------------------------------------
;;; Each layer's raw window is its own; the compressed rows and the indexer's
;;; keys belong to the four kv-source layers ("owners"), each read by the
;;; layers up to the next owner.  A compressed layer attends to the rows the
;;; latest index-source layer picked for this token (SELECTED).

(defstruct (d41ctx (:include d4ctx (caches #())) (:constructor %make-d41ctx))
  raw                                   ; per layer: SWA x HEAD-DIM, position p at row p mod SWA
  comp ix-comp prev-kv prev-score       ; per owner
  selected (n-sel 0) block-mask
  (tail (list -1 -1 -1)))               ; Engram: the last 3 tokens' compressed ids, newest first

(defun make-d41ctx (m &key (n-ctx 4096))
  (let* ((v (ds4f-v41 m)) (hd (ds4f-head-dim m)) (ixd (ds4f-ix-dim m)))
    (flet ((per-owner (fn) (map 'simple-vector fn (d41-kv-src v)))
           (cap (src) (1+ (floor n-ctx (max 1 (svref (ds4f-ratios m) src))))))
      (%make-d41ctx
       :model m :n-ctx n-ctx :logits (f32vec (ds4f-n-vocab m))
       :raw (map 'simple-vector (lambda (l) (declare (ignore l)) (f32vec (* (ds4f-swa m) hd))) (ds4f-layers m))
       :comp (per-owner (lambda (src) (f32vec (* (cap src) hd))))
       :ix-comp (per-owner (lambda (src) (f32vec (* (cap src) ixd))))
       :prev-kv (per-owner (lambda (src) (declare (ignore src)) (f32vec hd)))
       :prev-score (per-owner (lambda (src) (declare (ignore src)) (f32vec hd)))
       :selected (make-array (ds4f-ix-top-k m) :element-type '(unsigned-byte 32))
       :block-mask (f32vec (1+ (ceiling n-ctx (d41-cand-block v))))))))

(defun free-d41ctx (ctx)
  "Forget the device's copies of CTX's caches."
  (when (fboundp 'forget-gpu-vector)
    (dolist (vs (list (d41ctx-raw ctx) (d41ctx-comp ctx) (d41ctx-ix-comp ctx) (d41ctx-prev-kv ctx) (d41ctx-prev-score ctx)))
      (loop for v across vs do (funcall 'forget-gpu-vector (d41ctx-model ctx) v)))))

(defun d41-owner (m li)
  "The index of the kv-source layer whose caches layer LI reads."
  (position-if (lambda (s) (<= s li)) (d41-kv-src (ds4f-v41 m)) :from-end t))

;;; --- small ops with V4.1's rounding ----------------------------------------------------

(defun d41-rms! (x out n eps w)
  (d4-rms! x out n eps w) (bf16! out 0 n))

(defun d41-hc-mix (m fn scale base streams)
  "ds41_hc_mix: (values pre post comb) of the streams' plain RMS norm through FN."
  (let* ((n (ds4f-n-hc m)) (d (ds4f-d m)) (flat (f32vec (* n d))) (mix (f32vec (f32mat-rows fn))))
    (d4-rms! streams flat (* n d) (ds4f-eps m))
    (f32mv! (ds4f-pool m) fn flat mix)
    (hc-split m mix scale base)))

(defun d41-hc-sum (m streams pre out)
  "OUT = bf16(sum_h PRE[h] * stream h), the sum in stream order."
  (let ((n (ds4f-n-hc m)) (d (ds4f-d m)))
    (declare (type f32vec streams pre out) (type fixnum n d))
    (dotimes (j d out)
      (let ((acc 0f0))
        (declare (type single-float acc))
        (dotimes (h n) (incf acc (* (aref streams (+ (* h d) j)) (aref pre h))))
        (setf (aref out j) (bf16 acc))))))

(defun d41-hc-expand! (m out-streams block-out streams post comb)
  (hc-post! m out-streams block-out streams post comb)
  (bf16! out-streams))

(defun d41-q8mv (m w x rows)
  "bf16(W X) for a Q8MAT, a fresh vector of ROWS."
  (bf16! (q8mv! (ds4f-pool m) w x (f32vec rows))))

(defun d41-f32mv (m w x)
  (bf16! (f32mv! (ds4f-pool m) w x (f32vec (f32mat-rows w)))))

;;; --- Engram ------------------------------------------------------------------------------

(defun d41-engram-hash (m tail token)
  "ds4_engram_hash: (values rows new-tail), ROWS a list per table of the
   24 table rows: for n-grams of 2, 3 and 4 tokens ending at TOKEN, a
   running xor of id * multiplier, 8 primes each."
  (let* ((v (ds4f-v41 m)) (cur (aref (d41-token-map v) token)) (pad (d41-pad v))
         (ids (let ((blocked nil))
                (loop for id in (cons cur tail)
                      do (when (minusp id) (setf blocked t))
                      collect (if blocked pad id))))
         (primes (d41-primes v)) (mults (d41-mults v)) (ng (1- (length ids))) (heads 8)
         (rows (loop for tbl below (length (d41-eg-layers v))
                     collect (let ((h (* (first ids) (svref mults (* tbl (1+ ng))))) (offset 0) (out '()))
                               (loop for j from 1 to ng
                                     do (setf h (logxor h (* (nth j ids) (svref mults (+ (* tbl (1+ ng)) j)))))
                                        (dotimes (hd heads)
                                          (let ((p (svref primes (+ (* tbl ng heads) (* (1- j) heads) hd))))
                                            (push (+ (mod h p) offset) out)
                                            (incf offset p))))
                               (nreverse out)))))
    (values rows (list cur (first tail) (second tail)))))

(defparameter +d41-e4m3-codes+
  (let ((v (f32vec 256)))
    (dotimes (c 256 v)
      (let* ((e (ldb (byte 4 3) c)) (mant (ldb (byte 3 0) c))
             (x (if (zerop e) (* mant (expt 2f0 -9)) (* (+ 8 mant) (expt 2f0 (- e 10))))))
        (setf (aref v c) (if (logbitp 7 c) (- x) x))))))

(defun d41-engram-input (m tbl rows)
  "The 24 rows ROWS of Engram table TBL, decoded ([col][256]): E4M3 codes
   times their E8M0 scale per 32, BF16."
  (let* ((wf (ds4f-file m)) (base (nth tbl (d41-eg-offs (ds4f-v41 m)))) (x (f32vec (* 256 (length rows)))))
    (declare (type u8vec wf))
    (loop for r in rows for col from 0
          do (let ((o (+ base (* r 264))))
               (dotimes (j 256)
                 (setf (aref x (+ (* col 256) j))
                       (bf16 (scale-float (aref +d41-e4m3-codes+ (aref wf (+ o j)))
                                          (- (aref wf (+ o 256 (floor j 32))) 127)))))))
    x))

(defun d41-engram! (m l streams x)
  "ds41 Engram block: a key per stream and a shared value from the table
   rows X, each stream gated by sigmoid(signed sqrt(its normed dot with its key))."
  (let* ((d (ds4f-d m)) (n (ds4f-n-hc m)) (eps (ds4f-eps m))
         (kv (d41-f32mv m (dl-eg-kv l) x)) (qw (dl-eg-q-norm l)) (kw (dl-eg-k-norm l)))
    (declare (type f32vec kv qw kw streams))
    (dotimes (h n streams)
      (let ((h2 0f0) (k2 0f0) (dot 0f0) (ro (* h d)) (ko (* h d)) (vo (* n d)))
        (declare (type single-float h2 k2 dot))
        (dotimes (i d)
          (let ((r (aref streams (+ ro i))) (k (aref kv (+ ko i))))
            (incf h2 (* r r)) (incf k2 (* k k))
            (incf dot (* (* r (* (aref qw (+ ko i)) (aref kw (+ ko i)))) k))))
        (setf dot (* dot (/ 1f0 (sqrt (+ (/ h2 d) eps))) (/ 1f0 (sqrt (+ (/ k2 d) eps))) (/ 1f0 (sqrt (float d 1f0)))))
        (let ((gate (d4-sigmoid (* (float-sign dot 1f0) (sqrt (max (abs dot) 1f-6))))))
          (dotimes (i d)
            (setf (aref streams (+ ro i)) (bf16 (+ (aref streams (+ ro i)) (* gate (aref kv (+ vo i))))))))))))

;;; --- attention ---------------------------------------------------------------------------

(defun d41-publish! (m l li ctx norm pos k)
  "A kv-source layer's compressor: this token's row into the pair (ratio 2)
   or as is (ratio 1); a finished row normed, its indexer key made, both
   RoPE'd at the group's first position and rounded into owner K's caches."
  (let* ((hd (ds4f-head-dim m)) (ixd (ds4f-ix-dim m)) (ratio (dl-ratio l)) (eps (ds4f-eps m))
         (kv (f32mv! (ds4f-pool m) (dl-comp-kv l) norm (f32vec hd))) (latent (f32vec hd)))
    (if (= ratio 1)
        (replace latent (bf16! kv))
        (let ((sc (f32mv! (ds4f-pool m) (dl-comp-gate l) norm (f32vec hd)))
              (pkv (svref (d41ctx-prev-kv ctx) k)) (psc (svref (d41ctx-prev-score ctx) k)))
          (if (evenp pos)
              (progn (replace pkv kv) (replace psc sc))
              (dotimes (j hd)
                (let* ((sa (aref psc j)) (sb (aref sc j)) (p (max sa sb)) (ea (exp (- sa p))) (eb (exp (- sb p))))
                  (setf (aref latent j) (bf16 (/ (+ (* (aref pkv j) ea) (* (aref kv j) eb)) (+ ea eb)))))))))
    (when (zerop (mod (1+ pos) ratio))
      (let* ((n (floor (1+ pos) ratio)) (rpos (- (1+ pos) ratio)) (row (f32vec hd)))
        (d41-rms! latent row hd eps (dl-comp-norm l))
        (let ((ik (d41-f32mv m (dl-ix-k l) row)) (ikn (f32vec ixd)))
          (d41-rms! ik ikn ixd eps (dl-ix-k-norm l))
          (d41-rope! m ikn 1 ixd rpos li)
          (d41-quant! ikn 0 ixd :fp4)
          (replace (svref (d41ctx-ix-comp ctx) k) ikn :start1 (* (1- n) ixd)))
        (d41-rope! m row 1 hd rpos li)
        (d41-quant! row 0 hd :fp4-e4m3)
        (replace (svref (d41ctx-comp ctx) k) row :start1 (* (1- n) hd))))))

(defun d41-top (scores n k)
  "The K best of SCORES[0..N), best first, the lower index first on ties."
  (let ((idx (make-array n)))
    (dotimes (i n) (setf (aref idx i) i))
    (subseq (stable-sort idx #'> :key (lambda (i) (aref scores i))) 0 k)))

(defun d41-index! (m l li ctx norm qr pos k nc)
  "An index-source layer's pick of owner K's NC compressed rows: 32 FP4
   query heads against the FP4 keys, relu'd, weighted, summed, the best 512
   into SELECTED (on the late layers, among the candidate blocks)."
  (let* ((v (ds4f-v41 m)) (ixh (ds4f-ix-heads m)) (ixd (ds4f-ix-dim m))
         (iq (d41-f32mv m (dl-ix-q-b l) qr)) (iw (d41-f32mv m (dl-ix-proj l) norm))
         (keys (svref (d41ctx-ix-comp ctx) k)) (scores (f32vec nc)))
    (declare (type f32vec iq iw keys scores))
    (d41-rope! m iq ixh ixd pos li)
    (d41-quant! iq 0 (* ixh ixd) :fp4)
    (parallel-for (c nc (ds4f-pool m))
      (let ((total 0f0))
        (declare (type single-float total))
        (dotimes (h ixh)
          (let ((dot 0f0))
            (declare (type single-float dot))
            (dotimes (j ixd) (incf dot (* (aref iq (+ (* h ixd) j)) (aref keys (+ (* c ixd) j)))))
            (incf total (* (max dot 0f0) (aref iw h)))))
        (setf (aref scores c) (* total (/ 1f0 64f0)))))
    ;; candidate blocks: the source layer keeps the best blocks; the later index layers pick only in them
    (let* ((bsz (d41-cand-block v)) (mask (d41ctx-block-mask ctx)) (blocks (ceiling nc bsz)))
      (cond ((= li (d41-cand-src v))
             (if (<= blocks (d41-cand-top v))
                 (fill mask 0f0 :end blocks)
                 (let ((bs (f32vec blocks)))
                   (dotimes (b blocks)
                     (setf (aref bs b) (loop for j from (* b bsz) below (min nc (* (1+ b) bsz)) maximize (aref scores j))))
                   (setf (aref bs (floor (1- nc) bsz)) sb-ext:single-float-positive-infinity)
                   (fill mask sb-ext:single-float-negative-infinity :end blocks)
                   (loop for b across (d41-top bs blocks (d41-cand-top v)) do (setf (aref mask b) 0f0)))))
            ((and (>= (d41-cand-src v) 0) (> li (d41-cand-src v)))
             (dotimes (c nc)
               (unless (zerop (aref mask (floor c bsz)))
                 (setf (aref scores c) sb-ext:single-float-negative-infinity))))))
    (let ((top (min nc (ds4f-ix-top-k m))))
      (replace (d41ctx-selected ctx) (d41-top scores nc top))
      (setf (d41ctx-n-sel ctx) top))))

(defun d41-attend! (m l li ctx q k attended pos heads)
  "Each head over the raw window (oldest first) then the ATTENDED selected
   compressed rows of owner K, the sink in the denominator."
  (let* ((hd (ds4f-head-dim m)) (nh (ds4f-n-head m)) (cap (ds4f-swa m)) (sinks (dl-sinks l))
         (raw (svref (d41ctx-raw ctx) li)) (n-raw (min (1+ pos) cap)) (first-raw (- (1+ pos) n-raw))
         (comp (if k (svref (d41ctx-comp ctx) k) raw)) (sel (d41ctx-selected ctx))
         (scale (/ 1f0 (sqrt (float hd 1f0)))))
    (declare (type f32vec q heads sinks raw comp) (type fixnum n-raw first-raw attended))
    (parallel-for (h nh (ds4f-pool m))
      (let ((score (f32vec (+ n-raw attended))) (mx (aref sinks h)) (qo (* h hd)))
        (declare (type single-float mx))
        (with-attention-kernels (dot axpy) (raw raw q heads)
          (dotimes (r n-raw)
            (let ((s (* (the single-float (dot (* (mod (+ first-raw r) cap) hd) qo hd)) scale)))
              (setf (aref score r) s mx (max mx s)))))
        (with-attention-kernels (dot axpy) (comp comp q heads)
          (dotimes (i attended)
            (let ((s (* (the single-float (dot (* (aref sel i) hd) qo hd)) scale)))
              (setf (aref score (+ n-raw i)) s mx (max mx s)))))
        (dotimes (i hd) (setf (aref heads (+ qo i)) 0f0))
        (let ((denom (exp (- (aref sinks h) mx))))
          (declare (type single-float denom))
          (with-attention-kernels (dot axpy) (raw raw q heads)
            (dotimes (r n-raw)
              (let ((w (exp (- (aref score r) mx)))) (incf denom w) (axpy w (* (mod (+ first-raw r) cap) hd) qo hd))))
          (with-attention-kernels (dot axpy) (comp comp q heads)
            (dotimes (i attended)
              (let ((w (exp (- (aref score (+ n-raw i)) mx)))) (incf denom w) (axpy w (* (aref sel i) hd) qo hd))))
          (let ((inv (/ 1f0 denom)))
            (dotimes (i hd) (setf (aref heads (+ qo i)) (* (aref heads (+ qo i)) inv)))))))
    heads))

(defun d41-attention (m l li ctx x pos)
  "ds41_attention: layer LI's attention block for X (the HC sum) at POS."
  (let* ((v (ds4f-v41 m)) (d (ds4f-d m)) (hd (ds4f-head-dim m)) (nh (ds4f-n-head m)) (eps (ds4f-eps m))
         (ratio (dl-ratio l)) (norm (f32vec d)) (lq (ds4f-lora-q m)))
    (d41-rms! x norm d eps (dl-attn-norm l))
    (let* ((qr (d41-q8mv m (dl-q-a l) norm lq)) (qrn (f32vec lq))
           (q (progn (d41-rms! qr qrn lq eps (dl-q-a-norm l)) (d41-q8mv m (dl-q-b l) qrn (* nh hd))))
           (kv0 (d41-q8mv m (dl-kv l) norm hd)) (kv (f32vec hd))
           (k nil) (attended 0))
      (d41-rms! kv0 kv hd eps (dl-kv-norm l))
      (d41-rope! m q nh hd pos li)
      (d41-rope! m kv 1 hd pos li)
      (d41-quant! kv 0 hd :fp8)
      (replace (svref (d41ctx-raw ctx) li) kv :start1 (* (mod pos (ds4f-swa m)) hd))
      (when (plusp ratio)
        (setf k (d41-owner m li))
        (when (member li (d41-kv-src v)) (d41-publish! m l li ctx norm pos k))
        (let ((nc (floor (1+ pos) ratio)))
          (when (and (member li (d41-ix-src v)) (plusp nc))
            (d41-index! m l li ctx norm qrn pos k nc))
          (setf attended (min nc (ds4f-ix-top-k m)))))
      (let ((heads (d41-attend! m l li ctx q k attended pos (f32vec (* nh hd)))))
        (bf16! heads)
        (d41-rope! m heads nh hd pos li :inverse t)
        (let* ((groups (ds4f-out-groups m)) (rank (ds4f-lora-o m)) (gdim (* hd (floor nh groups)))
               (low (bf16! (grouped-q8mv! (ds4f-pool m) (dl-out-a l) heads groups gdim rank (f32vec (* groups rank))))))
          (d41-q8mv m (dl-out-b l) low d))))))

;;; --- FFN ---------------------------------------------------------------------------------

(defun d41-ffn (m l li x)
  "ds41_moe: the routed experts (V4's arithmetic) plus the shared expert
   (BF16 at its gate, up, product and down), their sum BF16."
  (let* ((d (ds4f-d m)) (ff (ds4f-ff-exp m)) (clamp (svref (ds4f-clamps m) li)) (norm (f32vec d)))
    (d41-rms! x norm d (ds4f-eps m) (dl-ffn-norm l))
    (multiple-value-bind (sel w) (d4-route m l li norm 0)
      (prefetch-experts m l sel)
      (let ((moe (d4-routed! m l norm sel w clamp (f32vec d)))
            (g (d41-q8mv m (dl-sh-gate l) norm ff)) (u (d41-q8mv m (dl-sh-up l) norm ff)))
        (declare (type f32vec moe g u))
        (dotimes (i ff)
          (setf (aref g i) (bf16 (* (d4-silu (min (aref g i) clamp)) (max (- clamp) (min (aref u i) clamp))))))
        (let ((sh (d41-q8mv m (dl-sh-down l) g d)))
          (declare (type f32vec sh))
          (dotimes (i d moe) (setf (aref moe i) (bf16 (+ (aref moe i) (aref sh i))))))))))

;;; --- a token -------------------------------------------------------------------------------

(defun ds41f-forward-1 (ctx token logits)
  #+(and sbcl linux)
  (when (ds4f-cuda-p)
    (return-from ds41f-forward-1 (ds41f-cuda-forward ctx token logits)))
  (let* ((m (d41ctx-model ctx)) (v (ds4f-v41 m)) (d (ds4f-d m)) (n (ds4f-n-hc m)) (pos (d41ctx-pos ctx))
         (streams (f32vec (* n d))) (next (f32vec (* n d))) (x (f32vec d)) (wf (ds4f-file m))
         (pre (let ((p (f32vec n))) (setf (aref p 0) 1f0) p)))
    (declare (type u8vec wf))
    (assert (ds4f-output m) () "V4.1 on the CPU wants its weights loaded: bind *DS4F-CPU-Q8* to T")
    (let ((o (+ (ds4f-embd-off m) (* token d 2))))
      (dotimes (j d)
        (let ((e (f16-at wf (+ o (* 2 j))))) (dotimes (h n) (setf (aref streams (+ (* h d) j)) e)))))
    (multiple-value-bind (rows tail) (d41-engram-hash m (d41ctx-tail ctx) token)
      (dotimes (li (ds4f-n-layer m))
        (let ((l (svref (ds4f-layers m) li)) (tbl (position li (d41-eg-layers v))))
          (when tbl (d41-engram! m l streams (d41-engram-input m tbl (nth tbl rows))))
          (multiple-value-bind (pre-a post-a comb-a)
              (d41-hc-mix m (dl-hc-attn-fn l) (dl-hc-attn-scale l) (dl-hc-attn-base l) streams)
            (d41-hc-sum m streams pre x)
            (d41-hc-expand! m next (d41-attention m l li ctx x pos) streams post-a comb-a)
            (rotatef streams next)
            (multiple-value-bind (pre-f post-f comb-f)
                (d41-hc-mix m (dl-hc-ffn-fn l) (dl-hc-ffn-scale l) (dl-hc-ffn-base l) streams)
              (d41-hc-sum m streams pre-a x)
              (d41-hc-expand! m next (d41-ffn m l li x) streams post-f comb-f)
              (rotatef streams next)
              (setf pre pre-f)))))
      (setf (d41ctx-tail ctx) tail))
    (incf (d41ctx-pos ctx))
    (when logits
      (let ((norm (f32vec d)))
        (d41-hc-sum m streams pre x)
        (d41-rms! x norm d (ds4f-eps m) (ds4f-output-norm m))
        (q8mv! (ds4f-pool m) (ds4f-output m) norm (d41ctx-logits ctx))))
    ctx))

;;; --- as an illatio model ---------------------------------------------------------------

(defun load-ds41f-model (path threads)
  (let* ((d (load-ds41f path :threads threads)) (g (ds4f-gguf d)) (v (make-vocab-from-gguf g)))
    (setf (ds4f-vocab d) v)
    (%make-model :gguf g :vocab v :arch "deepseek41" :name (meta g "general.name" "DeepSeek V4.1 Flash")
                 :n-layer (ds4f-n-layer d) :d 0 :n-head 0 :n-kv-head 1 :head-dim 1 :q-dim 0 :ffn 0
                 :n-vocab (ds4f-n-vocab d)
                 :context-length (or (arch-key g "deepseek41" "max_position_embeddings") 4096)
                 :pool (ds4f-pool d) :impl d)))
