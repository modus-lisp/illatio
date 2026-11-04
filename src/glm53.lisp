;;;; glm53.lisp — GLM-5.3 Flash (`glm5-next`), one token at a time on the CPU.
;;;
;;; A port of ds4's GLM 5.3 graph (ds4.c's glm_graph_forward_token and the
;;; kernels it calls), which is the reference.  The shape, per token:
;;;
;;;   x = embd[token] (Q8_0) in N_HC = 4 streams, as DeepSeek V4's; per trunk
;;;   layer (45), HC pre / block / HC post twice, the blocks' outputs not
;;;   added to anything (the combine carries the residual).  HC's fn is BF16,
;;;   its flat norm's eps the model's (1e-5).  The head reads the streams'
;;;   plain mean.
;;;
;;;   Attention: 3 layers of 4 are KDA (Kimi delta attention: a 4-tap conv,
;;;   then a gated delta rule over a 128 x 128 state per head); the fourth
;;;   (3, 7, ..., 43) is DSA: MLA through a 512-wide latent (F16 in the
;;;   cache), keys and values per head absorbed into the query and the
;;;   output, no RoPE.  Past 2051 tokens a DSA layer attends only the 512
;;;   pools of 4 tokens an indexer scores highest, and the incomplete pool.
;;;
;;;   FFN: SwiGLU (clamp 10) in the first 3 layers; then a shared expert and
;;;   8 of 288 routed ones, chosen by sigmoid(router) + bias, weighted by the
;;;   unbiased sigmoids renormalized x 2.5.  The experts are IQ2_XXS gate/up,
;;;   Q2_K down, read in the mapped file as DeepSeek's are (D4-ROUTED!).
;;;
;;; The 46th layer is the MTP block (drafts a token ahead); not run here.
;;;
;;; The model is a DS4F (its HC and experts' code is DS4F's) with GLM's own.

(in-package #:illatio)

(defstruct (g53 (:include ds4f (glm t)) (:constructor %make-g53))
  (n-trunk 0 :type fixnum) (n-dense 0 :type fixnum)
  (kv-lora 512 :type fixnum) (v-dim 256 :type fixnum) (ix-pool 4 :type fixnum)
  (kda-heads 64 :type fixnum) (kda-dim 128 :type fixnum) (kda-lb -5f0 :type single-float)
  (clamp 10f0 :type single-float))

(defstruct (g53layer (:include d4layer) (:conc-name gl-))
  kind                                  ; :KDA or :DSA
  k-b v-b attn-out ix-k-bias ix-gate ix-ape
  kda-q kda-k kda-v q-conv k-conv v-conv conv3 f-a f-b dt-bias a-log beta g-a g-b o-norm kda-out
  ffn-gate ffn-up ffn-down)

(defun g53-dense-limit (m)
  "The longest prefix a DSA layer attends whole: top-k + pool - 1 tokens."
  (+ (ds4f-ix-top-k m) (g53-ix-pool m) -1))

(defvar *g53-trunk-limit* nil
  "When set, LOAD-G53 loads only this many trunk layers, and the embedding
   as the head (a test of a partial file's first layers against each
   other: CPU and GPU, prefill and decode).")

(defun load-g53 (path &key (threads (default-thread-count)))
  "GLM-5.3 Flash at PATH: as LOAD-DS4F, the routed experts left in the file."
  (let* ((g (read-gguf path)) (a "glm5-next"))
    (flet ((k (key &optional default) (or (arch-key g a key) default))
           (q (name) (load-qmat g name))
           (f32 (name) (load-f32 g name))
           (fm (name) (load-f32mat g name)))
      (let* ((n-layer (k "block_count"))
             (m (%make-g53
                 :gguf g :file (map-file-vector path) :pool (make-pool threads)
                 :n-layer (min (k "trunk_block_count") (or *g53-trunk-limit* 1000))
                 :n-trunk (min (k "trunk_block_count") (or *g53-trunk-limit* 1000)) :d (k "embedding_length")
                 :n-head (k "attention.head_count") :head-dim (k "attention.key_length")
                 :v-dim (k "attention.value_length") :n-rot 0
                 :lora-q (k "attention.q_lora_rank") :kv-lora (k "attention.kv_lora_rank")
                 :n-expert (k "expert_count") :n-used (k "expert_used_count")
                 :ff-exp (k "expert_feed_forward_length") :n-dense (k "leading_dense_block_count")
                 :n-hc (k "hyper_connection.count") :sinkhorn (k "hyper_connection.sinkhorn_iterations")
                 :hc-eps (coerce (k "hyper_connection.epsilon") 'single-float)
                 :ix-heads (k "attention.indexer.head_count") :ix-dim (k "attention.indexer.key_length")
                 :ix-top-k (k "attention.indexer.top_k") :ix-pool (k "attention.indexer.pool_size" 4)
                 :kda-heads (k "linear_attention.head_count") :kda-dim (k "linear_attention.head_dimension")
                 :kda-lb (coerce (k "linear_attention.gate_lower_bound" -5) 'single-float)
                 :eps (coerce (k "attention.layer_norm_rms_epsilon") 'single-float)
                 :expert-scale (coerce (k "expert_weights_scale") 'single-float)
                 :clamp (coerce (k "swiglu_limit" 10) 'single-float)
                 :n-vocab (k "vocab_size")
                 :embd-off (+ (gguf-data-offset g) (tensor-info-offset (gguf-tensor g "token_embd.weight")))
                 :output (load-q8mat g (if *g53-trunk-limit* "token_embd.weight" "output.weight"))
                 :output-norm (if *g53-trunk-limit* (make-array 4096 :element-type 'single-float :initial-element 1f0)
                                  (f32 "output_norm.weight")))))
        (assert (eq (tensor-info-type (gguf-tensor g "token_embd.weight")) :q8_0))
        (setf (ds4f-clamps m) (make-array n-layer :initial-element (g53-clamp m)))
        (setf (ds4f-layers m)
              (coerce
               (loop for i below (g53-n-trunk m)
                     collect (flet ((n (s) (format nil "blk.~D.~A" i s)))
                               (let* ((kda (/= (mod i 4) 3))
                                      (l (make-g53layer
                                          :kind (if kda :kda :dsa)
                                          :attn-norm (f32 (n "attn_norm.weight")) :ffn-norm (f32 (n "ffn_norm.weight"))
                                          :hc-attn-fn (fm (n "hc_attn_fn.weight")) :hc-attn-scale (f32 (n "hc_attn_scale.weight"))
                                          :hc-attn-base (f32 (n "hc_attn_base.weight"))
                                          :hc-ffn-fn (fm (n "hc_ffn_fn.weight")) :hc-ffn-scale (f32 (n "hc_ffn_scale.weight"))
                                          :hc-ffn-base (f32 (n "hc_ffn_base.weight")))))
                                 (if kda
                                     (setf (gl-kda-q l) (q (n "kda_q.weight")) (gl-kda-k l) (q (n "kda_k.weight"))
                                           (gl-kda-v l) (q (n "kda_v.weight"))
                                           (gl-q-conv l) (f32 (n "kda_q_conv.weight")) (gl-k-conv l) (f32 (n "kda_k_conv.weight"))
                                           (gl-v-conv l) (f32 (n "kda_v_conv.weight"))
                                           (gl-f-a l) (q (n "kda_f_a.weight")) (gl-f-b l) (q (n "kda_f_b.weight"))
                                           (gl-dt-bias l) (f32 (n "kda_dt_bias.weight")) (gl-a-log l) (f32 (n "kda_a_log.weight"))
                                           (gl-beta l) (q (n "kda_beta.weight"))
                                           (gl-g-a l) (q (n "kda_g_a.weight")) (gl-g-b l) (q (n "kda_g_b.weight"))
                                           (gl-o-norm l) (f32 (n "kda_o_norm.weight")) (gl-kda-out l) (q (n "kda_output.weight"))
                                           (gl-conv3 l) (concatenate 'f32vec (gl-q-conv l) (gl-k-conv l) (gl-v-conv l)))
                                     (setf (gl-q-a l) (q (n "attn_q_a.weight")) (gl-q-a-norm l) (f32 (n "attn_q_a_norm.weight"))
                                           (gl-q-b l) (q (n "attn_q_b.weight"))
                                           (gl-kv l) (q (n "attn_kv_a_mqa.weight")) (gl-kv-norm l) (f32 (n "attn_kv_a_norm.weight"))
                                           (gl-k-b l) (fm (n "attn_k_b.weight")) (gl-v-b l) (fm (n "attn_v_b.weight"))
                                           (gl-attn-out l) (q (n "attn_output.weight"))
                                           (gl-ix-q-b l) (fm (n "indexer.attn_q_b.weight")) (gl-ix-k l) (fm (n "indexer.attn_k.weight"))
                                           (gl-ix-k-norm l) (f32 (n "indexer.k_norm.weight")) (gl-ix-k-bias l) (f32 (n "indexer.k_norm.bias"))
                                           (gl-ix-proj l) (fm (n "indexer.proj.weight"))
                                           (gl-ix-ape l) (f32 (n "indexer.pool_ape.weight")) (gl-ix-gate l) (fm (n "indexer.pool_gate.weight"))))
                                 (if (< i (g53-n-dense m))
                                     (setf (gl-ffn-gate l) (q (n "ffn_gate.weight")) (gl-ffn-up l) (q (n "ffn_up.weight"))
                                           (gl-ffn-down l) (q (n "ffn_down.weight")))
                                     (setf (gl-gate-inp l) (fm (n "ffn_gate_inp.weight")) (gl-probs-b l) (f32 (n "exp_probs_b.bias"))
                                           (gl-gate-exps l) (expert-tensor g (n "ffn_gate_exps.weight"))
                                           (gl-up-exps l) (expert-tensor g (n "ffn_up_exps.weight"))
                                           (gl-down-exps l) (expert-tensor g (n "ffn_down_exps.weight"))
                                           (gl-sh-gate l) (q (n "ffn_gate_shexp.weight")) (gl-sh-up l) (q (n "ffn_up_shexp.weight"))
                                           (gl-sh-down l) (q (n "ffn_down_shexp.weight"))))
                                 l)))
               'simple-vector))
        m))))

(defun g53-mv! (pool w x out)
  "OUT = W X, whatever W's kind: X quantized as W's products want it."
  (etypecase w
    (q8mat (q8mv! pool w x out))
    (kmat (let ((xq (make-q8kvec (kmat-cols w)))) (quantize-q8k! x xq) (kmatvec! pool w xq out)))
    (f32mat (f32mv! pool w x out))))

;;; --- per-context state ---------------------------------------------------------------

(defstruct (g53ctx (:include d4ctx (caches #())) (:constructor %make-g53ctx))
  conv state                            ; per KDA layer: 3 x 3 x P raw history; heads x V x K
  kv ixk tail                           ; per DSA layer: positions x latent (F16's values); pools x key; 2 x 4 x key raw
  dev)                                  ; on CUDA, all of it on the device instead (G53-CTX-DEV)

(defun make-g53ctx (m &key (n-ctx 4096))
  (if (and (fboundp 'ds4f-cuda-p) (funcall 'ds4f-cuda-p))
      (%make-g53ctx :model m :n-ctx n-ctx :logits (f32vec (ds4f-n-vocab m)))
      (make-g53ctx-cpu m n-ctx)))

(defun make-g53ctx-cpu (m n-ctx)
  (let* ((nl (g53-n-trunk m)) (p (* (g53-kda-heads m) (g53-kda-dim m))) (ixd (ds4f-ix-dim m))
         (c (%make-g53ctx :model m :n-ctx n-ctx :logits (f32vec (ds4f-n-vocab m))
                          :conv (make-array nl :initial-element nil) :state (make-array nl :initial-element nil)
                          :kv (make-array nl :initial-element nil) :ixk (make-array nl :initial-element nil)
                          :tail (make-array nl :initial-element nil))))
    (dotimes (li nl c)
      (if (eq (gl-kind (svref (ds4f-layers m) li)) :kda)
          (setf (svref (g53ctx-conv c) li) (f32vec (* 9 p))
                (svref (g53ctx-state c) li) (f32vec (* p (g53-kda-dim m))))
          (setf (svref (g53ctx-kv c) li) (f32vec (* n-ctx (g53-kv-lora m)))
                (svref (g53ctx-ixk c) li) (f32vec (* (ceiling n-ctx (g53-ix-pool m)) ixd))
                (svref (g53ctx-tail c) li) (f32vec (* 2 (g53-ix-pool m) ixd)))))))

;;; --- KDA -------------------------------------------------------------------------------

(defun g53-kda (ctx l li x)
  "glm53_kda_decode: the delta rule's step for X (attn_norm'd), its output."
  (let* ((m (d4ctx-model ctx)) (pool (ds4f-pool m)) (nh (g53-kda-heads m)) (hd (g53-kda-dim m))
         (p (* nh hd)) (d (ds4f-d m)) (eps (ds4f-eps m)) (lb (g53-kda-lb m))
         (q (f32vec p)) (k (f32vec p)) (v (f32vec p)) (flow (f32vec (f32mat-or-rows (gl-f-a l))))
         (graw (f32vec p)) (braw (f32vec nh)) (glow (f32vec (f32mat-or-rows (gl-g-a l)))) (og (f32vec p))
         (o (f32vec p)) (out (f32vec d))
         (conv (svref (g53ctx-conv ctx) li)) (s (svref (g53ctx-state ctx) li))
         (dtb (gl-dt-bias l)) (alog (gl-a-log l)) (onorm (gl-o-norm l)))
    (declare (type f32vec q k v graw braw og o conv s dtb alog onorm) (type fixnum nh hd p))
    (g53-mv! pool (gl-kda-q l) x q)
    (g53-mv! pool (gl-kda-k l) x k)
    (g53-mv! pool (gl-kda-v l) x v)
    (g53-mv! pool (gl-f-a l) x flow)
    (g53-mv! pool (gl-f-b l) flow graw)
    (g53-mv! pool (gl-beta l) x braw)
    (g53-mv! pool (gl-g-a l) x glow)
    (g53-mv! pool (gl-g-b l) glow og)
    ;; the conv over the raw history (taps 0-2 the oldest first, 3 the token), then SiLU
    (loop for raw in (list q k v) for w in (list (gl-q-conv l) (gl-k-conv l) (gl-v-conv l)) for a from 0
          do (let ((raw raw) (w w) (base (* a 3 p)))
               (declare (type f32vec raw w) (type fixnum base))
               (dotimes (c p)
                 (let ((acc 0f0))
                   (declare (type single-float acc))
                   (dotimes (j 3) (setf acc (+ acc (* (aref conv (+ base (* j p) c)) (aref w (+ (* c 4) j))))))
                   (setf acc (+ acc (* (aref raw c) (aref w (+ (* c 4) 3)))))
                   (setf (aref conv (+ base c)) (aref conv (+ base p c))
                         (aref conv (+ base p c)) (aref conv (+ base (* 2 p) c))
                         (aref conv (+ base (* 2 p) c)) (aref raw c)
                         (aref raw c) (d4-silu acc))))))
    (parallel-for (h nh pool)
      (locally (declare (optimize speed (safety 0)) (type fixnum h))
        (let* ((o0 (* h hd)) (decay (make-array 128 :element-type 'single-float))
               (beta (d4-sigmoid (aref braw h))) (ea (exp (aref alog h)))
               (qs 0f0) (ks 0f0))
          (declare (type fixnum o0) (type single-float beta ea qs ks) (dynamic-extent decay))
          (dotimes (i hd)
            (let ((c (+ o0 i)))
              (setf (aref decay i) (exp (* lb (d4-sigmoid (* ea (+ (aref graw c) (aref dtb c)))))))
              (incf qs (* (aref q c) (aref q c)))
              (incf ks (* (aref k c) (aref k c)))))
          (let ((qn (* (/ 1f0 (sqrt (+ qs 1f-6))) 0.08838834764831845f0)) (kn (/ 1f0 (sqrt (+ ks 1f-6)))))
            (dotimes (i hd)
              (setf (aref q (+ o0 i)) (* (aref q (+ o0 i)) qn) (aref k (+ o0 i)) (* (aref k (+ o0 i)) kn))))
          (let ((os 0f0))
            (declare (type single-float os))
            (dotimes (vi hd)
              (let ((row (* (+ o0 vi) hd)) (hk 0f0) (acc 0f0))
                (declare (type fixnum row) (type single-float hk acc))
                (dotimes (j hd)
                  (let ((sv (* (aref s (+ row j)) (aref decay j))))
                    (setf (aref s (+ row j)) sv)
                    (incf hk (* sv (aref k (+ o0 j))))))
                (let ((delta (* (- (aref v (+ o0 vi)) hk) beta)))
                  (dotimes (j hd)
                    (let ((sv (+ (aref s (+ row j)) (* (aref k (+ o0 j)) delta))))
                      (setf (aref s (+ row j)) sv)
                      (incf acc (* sv (aref q (+ o0 j)))))))
                (setf (aref o (+ o0 vi)) acc)
                (incf os (* acc acc))))
            (let ((sc (/ 1f0 (sqrt (+ (/ os hd) eps)))))
              (dotimes (i hd)
                (setf (aref o (+ o0 i)) (* (aref o (+ o0 i)) sc (aref onorm i) (d4-sigmoid (aref og (+ o0 i)))))))))))
    (g53-mv! pool (gl-kda-out l) o out)
    out))

(defun f32mat-or-rows (w) (etypecase w (q8mat (q8mat-rows w)) (kmat (kmat-rows w)) (f32mat (f32mat-rows w))))

;;; --- DSA -------------------------------------------------------------------------------

(defun g53-pool-key! (m l tail ixk p)
  "Pool P's indexer key from TAIL's 4 raw keys and gates: each LayerNorm'd
   (eps 1e-6, with bias), weighted per channel by a softmax over the 4 of
   gate + ape; F16'd into IXK."
  (let* ((hd (ds4f-ix-dim m)) (np (g53-ix-pool m)) (gofs (* np hd))
         (w (gl-ix-k-norm l)) (b (gl-ix-k-bias l)) (ape (gl-ix-ape l))
         (mean (f32vec np)) (inv (f32vec np)))
    (declare (type f32vec tail ixk w b ape mean inv) (type fixnum hd np gofs))
    (dotimes (r np)
      (let ((sum 0f0) (var 0f0))
        (dotimes (i hd) (incf sum (aref tail (+ (* r hd) i))))
        (let ((mu (/ sum hd)))
          (dotimes (i hd) (let ((dv (- (aref tail (+ (* r hd) i)) mu))) (incf var (* dv dv))))
          (setf (aref mean r) mu (aref inv r) (/ 1f0 (sqrt (+ (/ var hd) 1f-6)))))))
    (dotimes (i hd)
      (let ((mx -1f30) (z 0f0) (acc 0f0))
        (dotimes (r np) (setf mx (max mx (+ (aref tail (+ gofs (* r hd) i)) (aref ape (+ (* r hd) i))))))
        (dotimes (r np) (incf z (exp (- (+ (aref tail (+ gofs (* r hd) i)) (aref ape (+ (* r hd) i))) mx))))
        (dotimes (r np)
          (let ((wr (/ (exp (- (+ (aref tail (+ gofs (* r hd) i)) (aref ape (+ (* r hd) i))) mx)) z))
                (kn (+ (* (* (- (aref tail (+ (* r hd) i)) (aref mean r)) (aref inv r)) (aref w i)) (aref b i))))
            (incf acc (* wr kn))))
        (setf (aref ixk (+ (* p hd) i)) (f16-round acc))))))

(defun g53-select (m l x qrn ixk visible)
  "The positions a DSA layer attends at VISIBLE tokens: all of them up to
   the dense limit, else the top-k complete pools' and the incomplete one's."
  (let ((np (g53-ix-pool m)))
    (if (<= visible (g53-dense-limit m))
        (loop for i below visible collect i)
        (let* ((pool (ds4f-pool m)) (nh (ds4f-ix-heads m)) (hd (ds4f-ix-dim m))
               (iq (f32vec (* nh hd))) (iw (f32vec nh)) (rows (floor visible np))
               (scores (f32vec rows)) (scale (/ 1f0 (sqrt (float (* nh hd) 1f0)))))
          (declare (type f32vec iq iw scores ixk))
          (f32mv! pool (gl-ix-q-b l) qrn iq)
          (f32mv! pool (gl-ix-proj l) x iw)
          (parallel-for (p rows pool)
            (let ((acc 0f0))
              (dotimes (h nh)
                (let ((dot 0f0))
                  (dotimes (i hd) (incf dot (* (aref iq (+ (* h hd) i)) (aref ixk (+ (* p hd) i)))))
                  (incf acc (* (max dot 0f0) (aref iw h)))))
              (setf (aref scores p) (* acc scale))))
          (let* ((k (floor (ds4f-ix-top-k m) np))
                 (best (subseq (sort (let ((v (make-array rows))) (dotimes (i rows v) (setf (aref v i) i)))
                                     (lambda (a b) (or (> (aref scores a) (aref scores b))
                                                       (and (= (aref scores a) (aref scores b)) (< a b)))))
                               0 (min k rows))))
            (append (loop for p across best append (loop for r below np collect (+ (* p np) r)))
                    (loop for i below (mod visible np) collect (+ (- visible (mod visible np)) i))))))))

(defun g53-dsa (ctx l li x pos)
  "A DSA layer's attention for X (attn_norm'd) at POS: its output."
  (let* ((m (d4ctx-model ctx)) (pool (ds4f-pool m)) (d (ds4f-d m)) (eps (ds4f-eps m))
         (nh (ds4f-n-head m)) (kd (ds4f-head-dim m)) (vd (g53-v-dim m)) (lat (g53-kv-lora m))
         (ixd (ds4f-ix-dim m)) (np (g53-ix-pool m))
         (qr (f32vec (ds4f-lora-q m))) (qrn (f32vec (ds4f-lora-q m))) (q (f32vec (* nh kd)))
         (kvr (f32vec lat)) (heads (f32vec (* nh vd))) (out (f32vec d))
         (kv (svref (g53ctx-kv ctx) li)) (ixk (svref (g53ctx-ixk ctx) li)) (tail (svref (g53ctx-tail ctx) li))
         (kb (f32mat-data (gl-k-b l))) (vb (f32mat-data (gl-v-b l))))
    (declare (type f32vec q kv kb vb heads tail ixk) (type fixnum nh kd vd lat))
    (g53-mv! pool (gl-q-a l) x qr)
    (d4-rms! qr qrn (length qr) eps (gl-q-a-norm l))
    (g53-mv! pool (gl-q-b l) qrn q)
    (g53-mv! pool (gl-kv l) x kvr)
    (d4-rms! kvr kvr lat eps (gl-kv-norm l))
    (dotimes (j lat) (setf (aref kv (+ (* pos lat) j)) (f16-round (aref kvr j))))
    ;; the indexer's pool: this token's raw key and gate, the pool's key once complete
    (let ((r (mod pos np)) (kr (f32vec ixd)) (gr (f32vec ixd)))
      (f32mv! pool (gl-ix-k l) x kr)
      (f32mv! pool (gl-ix-gate l) x gr)
      (replace tail kr :start1 (* r ixd))
      (replace tail gr :start1 (* (+ np r) ixd))
      (when (= r (1- np)) (g53-pool-key! m l tail ixk (floor pos np))))
    (let* ((sel (coerce (g53-select m l x qrn ixk (1+ pos)) '(simple-array fixnum (*))))
           (ns (length sel)))
      (declare (type (simple-array fixnum (*)) sel) (type fixnum ns))
      (parallel-for (h nh pool)
        (locally (declare (optimize speed (safety 0)) (type fixnum h))
          (let ((qk (make-array 512 :element-type 'single-float))
                (lv (make-array 512 :element-type 'single-float :initial-element 0f0))
                (sc (make-array ns :element-type 'single-float))
                (mx -1f30) (z 0f0))
            (declare (type single-float mx z))
            ;; the query in the latent: qk[j] = sum_i k_b[h][j][i] q[h][i]
            (dotimes (j lat)
              (let ((acc 0f0) (row (* (+ (* h lat) j) kd)))
                (declare (type single-float acc) (type fixnum row))
                (dotimes (i kd) (incf acc (* (aref kb (+ row i)) (aref q (+ (* h kd) i)))))
                (setf (aref qk j) acc)))
            (dotimes (ti ns)
              (let ((acc 0f0) (row (* (aref sel ti) lat)))
                (declare (type single-float acc) (type fixnum row))
                (dotimes (j lat) (incf acc (* (aref qk j) (aref kv (+ row j)))))
                (let ((s (* acc 0.0625f0))) (setf (aref sc ti) s mx (max mx s)))))
            (dotimes (ti ns) (let ((e (exp (- (aref sc ti) mx)))) (setf (aref sc ti) e) (incf z e)))
            (let ((inv (/ 1f0 (max z 1f-20))))
              (dotimes (ti ns)
                (let ((pw (* (aref sc ti) inv)) (row (* (aref sel ti) lat)))
                  (dotimes (j lat) (incf (aref lv j) (* pw (aref kv (+ row j))))))))
            ;; out[h][e] = sum_j v_b[h][e][j] lv[j]
            (dotimes (e vd)
              (let ((acc 0f0) (row (* (+ (* h vd) e) lat)))
                (declare (type single-float acc) (type fixnum row))
                (dotimes (j lat) (incf acc (* (aref vb (+ row j)) (aref lv j))))
                (setf (aref heads (+ (* h vd) e)) acc)))))))
    (g53-mv! pool (gl-attn-out l) heads out)
    out))

;;; --- FFN -------------------------------------------------------------------------------

(defun g53-route (m l x &optional logits-in)
  "(values experts weights): the top N-USED of sigmoid(router) + bias, ties
   to the lower index; weights the unbiased sigmoids, renormalized, scaled.
   LOGITS-IN: the router's logits, computed already."
  (let* ((ne (ds4f-n-expert m)) (nu (ds4f-n-used m)) (b (gl-probs-b l))
         (logits (f32vec ne)) (probs (f32vec ne)) (s (f32vec ne))
         (idx (make-array nu :initial-element -1)) (w (f32vec nu)))
    (if logits-in (replace logits logits-in) (f32mv! (ds4f-pool m) (gl-gate-inp l) x logits))
    (dotimes (i ne) (setf (aref probs i) (d4-sigmoid (aref logits i)) (aref s i) (+ (aref probs i) (aref b i))))
    (dotimes (i ne)
      (loop for j below nu
            when (or (minusp (aref idx j)) (> (aref s i) (aref s (aref idx j))))
              do (loop for k from (1- nu) above j do (setf (aref idx k) (aref idx (1- k))))
                 (setf (aref idx j) i)
                 (return)))
    (let ((sum 0f0))
      (dotimes (i nu) (setf (aref w i) (aref probs (aref idx i))) (incf sum (aref w i)))
      (setf sum (max sum 6.103515625f-5))
      (dotimes (i nu) (setf (aref w i) (* (/ (aref w i) sum) (ds4f-expert-scale m)))))
    (values idx w)))

(defun g53-ffn (m l li x)
  "The FFN block of X (ffn_norm'd): dense in the first layers, else the
   routed experts plus the shared one."
  (let* ((d (ds4f-d m)) (pool (ds4f-pool m)) (clamp (g53-clamp m)))
    (if (< li (g53-n-dense m))
        (let* ((ff (q8mat-rows (gl-ffn-gate l))) (g (f32vec ff)) (u (f32vec ff)) (out (f32vec d)))
          (g53-mv! pool (gl-ffn-gate l) x g)
          (g53-mv! pool (gl-ffn-up l) x u)
          (dotimes (i ff)
            (setf (aref g i) (* (d4-silu (min (aref g i) clamp)) (max (- clamp) (min (aref u i) clamp)))))
          (g53-mv! pool (gl-ffn-down l) g out))
        (multiple-value-bind (sel w) (g53-route m l x)
          (when *ds4f-route-trace* (format *ds4f-route-trace* "~D~{ ~D~}~%" li (coerce sel 'list)))
          (prefetch-experts m l sel)
          (let ((moe (d4-routed! m l x sel w clamp (f32vec d))) (shared (f32vec d)))
            (d4-shared! m l x clamp shared)
            (dotimes (i d moe) (setf (aref moe i) (+ (aref moe i) (aref shared i)))))))))

;;; --- a token -------------------------------------------------------------------------

(defun g53-embed! (m token streams)
  "TOKEN's embedding (a Q8_0 row in the file) in each of the streams."
  (let* ((d (ds4f-d m)) (wf (ds4f-file m)) (o (+ (ds4f-embd-off m) (* token (floor d 32) 34))))
    (declare (type u8vec wf) (type f32vec streams))
    (dotimes (b (floor d 32))
      (let ((sc (f16-at wf (+ o (* b 34)))))
        (dotimes (j 32)
          (let ((v (* sc (s8 (aref wf (+ o (* b 34) 2 j))))))
            (dotimes (h (ds4f-n-hc m)) (setf (aref streams (+ (* h d) (* b 32) j)) v))))))
    streams))

(defun g53-forward-1 (ctx token logits)
  #+(and sbcl linux)
  (when (ds4f-cuda-p) (return-from g53-forward-1 (g53-cuda-forward ctx token logits)))
  (let* ((m (d4ctx-model ctx)) (d (ds4f-d m)) (n (ds4f-n-hc m)) (eps (ds4f-eps m)) (pos (d4ctx-pos ctx))
         (streams (f32vec (* n d))) (next (f32vec (* n d))) (cur (f32vec d)) (norm (f32vec d))
         (flat (f32vec (* n d))))
    (assert (< pos (d4ctx-n-ctx ctx)) () "GLM context full (~D tokens)" (d4ctx-n-ctx ctx))
    (g53-embed! m token streams)
    (dotimes (li (g53-n-trunk m))
      (let ((l (svref (ds4f-layers m) li)))
        (multiple-value-bind (post comb)
            (hc-pre m (gl-hc-attn-fn l) (gl-hc-attn-scale l) (gl-hc-attn-base l) streams cur flat)
          (d4-rms! cur norm d eps (gl-attn-norm l))
          (hc-post! m next (if (eq (gl-kind l) :kda) (g53-kda ctx l li norm) (g53-dsa ctx l li norm pos))
                    streams post comb))
        (rotatef streams next)
        (multiple-value-bind (post comb)
            (hc-pre m (gl-hc-ffn-fn l) (gl-hc-ffn-scale l) (gl-hc-ffn-base l) streams cur flat)
          (d4-rms! cur norm d eps (gl-ffn-norm l))
          (hc-post! m next (g53-ffn m l li norm) streams post comb))
        (rotatef streams next)))
    (incf (d4ctx-pos ctx))
    (when logits (g53-head! m streams (d4ctx-logits ctx)))
    ctx))

(defun g53-head! (m streams logits)
  "The streams' mean, normed, to LOGITS."
  (let* ((n (ds4f-n-hc m)) (d (ds4f-d m)) (h (f32vec d)) (norm (f32vec d)) (w (/ 1f0 n)))
    (declare (type f32vec streams h))
    (dotimes (j d) (let ((acc 0f0)) (dotimes (s n) (incf acc (* w (aref streams (+ (* s d) j))))) (setf (aref h j) acc)))
    (d4-rms! h norm d (ds4f-eps m) (ds4f-output-norm m))
    (q8mv! (ds4f-pool m) (ds4f-output m) norm logits)))

(defun g53-forward (ctx token &key (logits t))
  #+sbcl (sb-int:with-float-traps-masked (:overflow :inexact :underflow :invalid) (g53-forward-1 ctx token logits))
  #-sbcl (g53-forward-1 ctx token logits))

(defun g53-eval-tokens (c tokens)
  "TOKENS into C from its position, the last one's logits: on CUDA in
   layer-major passes (cuda-glm53.lisp), else a token at a time."
  #+(and sbcl linux)
  (when (and (rest tokens) (ds4f-cuda-p))
    (return-from g53-eval-tokens
      (sb-int:with-float-traps-masked (:overflow :inexact :underflow :invalid) (g53-cuda-prefill c tokens t))))
  (loop for (tk . more) on tokens do (g53-forward c tk :logits (null more))))

(defun g53-hotlist (m)
  "The experts the device pools are seeded with, (layer expert) pairs flat:
   the profile's (glm53-hotlist.lisp) for this shape, else every MoE layer's
   in turn, expert by expert."
  (if (and (= (g53-n-trunk m) 45) (= (ds4f-n-expert m) 288) (boundp '+glm53-hotlist+))
      (symbol-value '+glm53-hotlist+)
      (coerce (loop for e below (ds4f-n-expert m)
                    append (loop for li from (g53-n-dense m) below (g53-n-trunk m) append (list li e)))
              'simple-vector)))

(defun load-g53-model (path threads)
  (let* ((d (load-g53 path :threads threads)) (g (ds4f-gguf d)) (v (make-vocab-from-gguf g)))
    (setf (ds4f-vocab d) v)
    (%make-model :gguf g :vocab v :arch "glm5-next" :name (meta g "general.name" "GLM-5.3 Flash")
                 :n-layer (g53-n-trunk d) :d 0 :n-head 0 :n-kv-head 1 :head-dim 1 :q-dim 0 :ffn 0
                 :n-vocab (ds4f-n-vocab d)
                 :context-length (or (arch-key g "glm5-next" "context_length") 4096)
                 :pool (ds4f-pool d) :impl d)))
