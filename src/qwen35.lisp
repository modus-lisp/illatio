;;;; qwen35.lisp — Qwen3.5's hybrid decoder: gated delta net and gated attention.
;;;
;;; Three layers in four are LINEAR attention (a gated delta net, as Qwen3-Next
;;; introduced); every fourth is FULL attention with an output gate.  Both
;;; follow llama.cpp's graph (src/models/qwen35.cpp, delta-net-base.cpp) op for
;;; op.  A linear layer, for one token, with h = rmsnorm(x)*attn_norm:
;;;
;;;   qkv = Wqkv h,  z = Wgate h,  beta = sigmoid(Wbeta h),
;;;   g = softplus(Walpha h + dt_bias) * ssm_a          (a log decay, per V head)
;;;   c = silu(causal conv of qkv over the last 4 tokens, per channel)
;;;   q, k = l2norm per K head of c's first two parts;  v = its third
;;;   per V head j (K head j mod n_k_heads -- ggml_repeat tiles them):
;;;     S = exp(g) S;  d = beta (v - S k);  S = S + d k^T;  o = S (q / sqrt(dk))
;;;   out = Wout (rmsnorm(o)*ssm_norm * silu(z))         (per V head)
;;;
;;; S is a dv x dk matrix per V head, the layer's whole memory of the past, so
;;; there is no KV cache for these layers -- and no rewinding one: a context
;;; can only go forward from where its state is, or start over.  A full layer
;;; projects Q with its gate interleaved per head ([q 256 | gate 256]), norms
;;; Q and K per head, rotates the first ROPE-DIMS of each (llama.cpp's IMROPE,
;;; which with one position for t/h/w is NEOX on those dims), attends, and
;;; multiplies the result by sigmoid(gate) before Wo.  The FFN is the usual
;;; SwiGLU, normed by post_attention_norm.

(in-package #:illatio)

(defun model-recurrent-p (m)
  "Whether a context's state can't go back to an earlier position: the
   delta net's, or DeepSeek V4 Flash's compressors and window."
  (or (model-impl m)
      (some (lambda (l) (eq (layer-kind l) :linear)) (model-layers m))))

;;; --- per-context state and scratch ----------------------------------------------

(defstruct (q35 (:constructor %make-q35))
  conv ssm                              ; per layer: conv history, delta-net state
  qkv z alpha beta c o qg gate)         ; scratch

(defun q35-channels (m)
  (+ (* 2 (model-ssm-k-heads m) (model-ssm-k-dim m)) (* (model-ssm-v-heads m) (model-ssm-v-dim m))))

(defun q35-state (ctx)
  (or (getf (context-plist ctx) :q35)
      (prog1
       (setf (getf (context-plist ctx) :q35)
            (let* ((m (context-model ctx)) (ch (q35-channels m))
                   (vh (model-ssm-v-heads m)) (vd (model-ssm-v-dim m)) (kd (model-ssm-k-dim m))
                   (hd (model-head-dim m)) (nh (model-n-head m)))
              (%make-q35
               :conv (map 'simple-vector
                          (lambda (l) (if (eq (layer-kind l) :linear) (f32vec (* ch (1- (model-ssm-conv m)))) (f32vec 0)))
                          (model-layers m))
               :ssm (map 'simple-vector
                         (lambda (l) (if (eq (layer-kind l) :linear) (f32vec (* vh vd kd) :static t) (f32vec 0)))
                         (model-layers m))
               :qkv (f32vec ch) :z (f32vec (* vh vd)) :alpha (f32vec vh) :beta (f32vec vh)
               :c (f32vec ch) :o (f32vec (* vh vd))
               :qg (f32vec (* 2 nh hd)) :gate (f32vec (* nh hd)))))
        (own-context-caches ctx))))

(defun reset-recurrent! (ctx)
  "Zero every linear layer's state: the context starts over."
  (when (model-impl (context-model ctx))
    (return-from reset-recurrent! (ds4f-context-reset ctx)))
  #+(and sbcl linux)
  (when (model-cuda (context-model ctx))
    (return-from reset-recurrent! (progn (reset-cuda-state ctx) ctx)))
  (let ((st (q35-state ctx)))
    (loop for v across (q35-conv st) do (fill v 0f0))
    (loop for v across (q35-ssm st) do (fill v 0f0)))
  #+(and sbcl darwin) (reset-gpu-state ctx)
  ctx)

;;; --- small ops, as ggml computes them ------------------------------------------

(declaim (inline sigmoid1 softplus1 silu1))
(defun sigmoid1 (x) (declare (type single-float x)) (/ 1f0 (+ 1f0 (exp (- x)))))
(defun softplus1 (x) (declare (type single-float x)) (if (> x 20f0) x (log (+ 1f0 (exp x)))))
(defun silu1 (x) (declare (type single-float x)) (/ x (+ 1f0 (exp (- x)))))

(defun l2norm! (x o n eps)
  "ggml_l2_norm on X[o..o+n): the sum of squares in double, scale
   1/max(sqrt(sum), eps) as a float, each element times it."
  (declare (type f32vec x) (type fixnum o n) (type single-float eps) (optimize speed (safety 0)))
  (let ((sum 0d0))
    (declare (type double-float sum))
    (loop for i of-type fixnum from o below (+ o n)
          do (setf sum (+ sum (coerce (* (aref x i) (aref x i)) 'double-float))))
    (let ((scale (/ 1f0 (max (sqrt (coerce sum 'single-float)) eps))))
      (declare (type single-float scale))
      (loop for i of-type fixnum from o below (+ o n)
            do (setf (aref x i) (* (aref x i) scale))))))

;;; --- a linear (gated delta net) layer -------------------------------------------

;;; Each piece below takes N tokens laid out one after another (token TK's
;;; QKV at TK*CH, its gates at TK*VH, ...), so the decode (N = 1) and a
;;; batched prefill run the same code.  The recurrent parts go token by
;;; token inside a channel's or a head's own unit of work: each channel's
;;; convolution and each head's state depend only on their own past, so the
;;; work splits across threads without changing a single operation.

(defun q35-conv! (pool l qkv c state ch &optional (n 1))
  "C = silu(the causal convolution of each channel's last 4 inputs, QKV
   last), as ggml_ssm_conv; STATE (3 a channel, oldest first) moves on by
   one per token."
  (declare (type f32vec qkv c state) (type fixnum ch n))
  (let ((kern (layer-conv1d l)))
    (declare (type f32vec kern))
    (pool-run pool
              (lambda (i nt)
                (multiple-value-bind (s e) (share ch i nt)
                  (locally (declare (optimize speed (safety 0)) (type fixnum s e))
                    (loop for ci of-type fixnum from s below e
                          do (let* ((b (* ci 3)) (w (* ci 4))
                                    (x0 (aref state b)) (x1 (aref state (+ b 1))) (x2 (aref state (+ b 2))))
                               (declare (type fixnum b w) (type single-float x0 x1 x2))
                               (dotimes (tk n)
                                 (let* ((at (+ (* tk ch) ci)) (x3 (aref qkv at)) (sum 0f0))
                                   (declare (type fixnum at) (type single-float x3 sum))
                                   (setf sum (+ sum (* x0 (aref kern w))))
                                   (setf sum (+ sum (* x1 (aref kern (+ w 1)))))
                                   (setf sum (+ sum (* x2 (aref kern (+ w 2)))))
                                   (setf sum (+ sum (* x3 (aref kern (+ w 3)))))
                                   (setf (aref c at) (silu1 sum)
                                         x0 x1 x1 x2 x2 x3)))
                               (setf (aref state b) x0 (aref state (+ b 1)) x1 (aref state (+ b 2)) x2)))))))))

(defun q35-gates! (l alpha beta vh n)
  "BETA = sigmoid(BETA); ALPHA = softplus(ALPHA + dt_bias) * ssm_a: the
   per-head write strength and log decay."
  (declare (type f32vec alpha beta) (type fixnum vh n))
  (let ((a (layer-ssm-a l)) (dt (layer-ssm-dt l)))
    (declare (type f32vec a dt))
    (dotimes (tk n)
      (dotimes (j vh)
        (let ((i (+ (* tk vh) j)))
          (setf (aref beta i) (sigmoid1 (aref beta i))
                (aref alpha i) (* (softplus1 (+ (aref alpha i) (aref dt j))) (aref a j))))))))

(defun q35-l2norms! (pool m c n)
  "L2-normalize every K head of Q and K (C's first two parts), each token."
  (let* ((kh (model-ssm-k-heads m)) (kd (model-ssm-k-dim m)) (ch (q35-channels m))
         (eps (model-eps m)))
    (parallel-for (u (* n 2 kh) pool)
      (multiple-value-bind (tk j) (floor u (* 2 kh))
        (l2norm! c (+ (* tk ch) (* j kd)) kd eps)))))

(defun q35-delta! (pool m c beta g S o &optional (n 1))
  "The delta rule for every V head (a head per unit of work, its tokens in
   order), as build_delta_net_autoregressive: decay, read, correct, write,
   query.  A head's S is stored transposed, k-major (S[k][v] at k*dv + v), so
   the values v of one k are adjacent: the loops run k outer, v inner, and
   every v still gets exactly the sequence of operations ggml's v-major loop
   gives it -- so a block of v's can go a NEON lane each (%DELTA-BLOCK)."
  (declare (type f32vec c beta g S o) (type fixnum n))
  (let* ((kh (model-ssm-k-heads m)) (kd (model-ssm-k-dim m))
         (vh (model-ssm-v-heads m)) (vd (model-ssm-v-dim m))
         (ch (q35-channels m)) (od (* vh vd))
         (q-off 0) (k-off (* kh kd)) (v-off (* 2 kh kd))
         (scale (/ 1f0 (sqrt (coerce kd 'single-float)))))
    (declare (type fixnum kh kd vh vd ch od q-off k-off v-off) (type single-float scale))
    (parallel-for (h vh pool)
      (let ((sk (make-array vd :element-type 'single-float))
            (dv (make-array vd :element-type 'single-float))
            (ov (make-array vd :element-type 'single-float)))
        (declare (dynamic-extent sk dv ov))
        (locally (declare (optimize speed (safety 0)) (type f32vec sk dv ov))
          (let ((base (* h vd kd)))
            (declare (type fixnum base))
            (dotimes (tk n)
              (let* ((cb (* tk ch))
                     (kq (+ cb q-off (* (mod h kh) kd))) ; ggml_repeat tiles K heads over V heads
                     (kk (+ cb k-off (* (mod h kh) kd)))
                     (vv (+ cb v-off (* h vd)))
                     (oo (+ (* tk od) (* h vd)))
                     (decay (exp (aref g (+ (* tk vh) h))))
                     (b (aref beta (+ (* tk vh) h))))
                (declare (type fixnum cb kq kk vv oo) (type single-float decay b))
                (if (delta-block-p vd)
                    (delta-blocks! S base c kq kk vv o oo vd kd decay b scale)
                    (progn
                      (fill sk 0f0)
                      ;; S *= exp(g), and each value's S k with the decayed S
                      (dotimes (k kd)
                        (let ((ck (aref c (+ kk k))) (row (+ base (* k vd))))
                          (declare (type single-float ck) (type fixnum row))
                          (dotimes (v vd)
                            (let ((sv (* (aref S (+ row v)) decay)))
                              (setf (aref S (+ row v)) sv
                                    (aref sk v) (+ (aref sk v) (* sv ck)))))))
                      (dotimes (v vd)
                        (setf (aref dv v) (* (- (aref c (+ vv v)) (aref sk v)) b)
                              (aref ov v) 0f0))
                      ;; S += d k^T, then o = S (q * scale)
                      (dotimes (k kd)
                        (let ((ck (aref c (+ kk k))) (qs (* (aref c (+ kq k)) scale))
                              (row (+ base (* k vd))))
                          (declare (type single-float ck qs) (type fixnum row))
                          (dotimes (v vd)
                            (let ((sv (+ (aref S (+ row v)) (* ck (aref dv v)))))
                              (setf (aref S (+ row v)) sv
                                    (aref ov v) (+ (aref ov v) (* sv qs)))))))
                      (replace o ov :start1 oo)))))))))))

#-(and sbcl arm64) (declaim (inline delta-block-p))
#-(and sbcl arm64) (defun delta-block-p (vd) (declare (ignore vd)) nil)
#-(and sbcl arm64) (defun delta-blocks! (&rest args) (declare (ignore args)) (error "no NEON delta kernel"))

(defun q35-out-norm! (pool m l o z n)
  "O = rmsnorm(O)*ssm_norm * silu(Z), per V head of each token."
  (let ((w (layer-ssm-norm l)) (vd (model-ssm-v-dim m)) (eps (model-eps m)))
    (parallel-for (u (* n (model-ssm-v-heads m)) pool)
      (rmsnorm! o w o eps (* u vd))
      (loop for i of-type fixnum from (* u vd) below (* (1+ u) vd)
            do (setf (aref o i) (* (aref o i) (silu1 (aref z i))))))))

(defun q35-recur! (ctx l li qkv z alpha beta c o n)
  "Everything a linear layer does between its projections, for N tokens."
  (let* ((m (context-model ctx)) (pool (model-pool m)) (st (q35-state ctx)))
    (q35-gates! l alpha beta (model-ssm-v-heads m) n)
    (q35-conv! pool l qkv c (svref (q35-conv st) li) (q35-channels m) n)
    (q35-l2norms! pool m c n)
    (q35-delta! pool m c beta alpha (svref (q35-ssm st) li) o n)
    (q35-out-norm! pool m l o z n)))

(defun q35-linear! (ctx l li h)
  "Layer LI (linear) of the token whose normed input is H: its output in H."
  (let* ((m (context-model ctx)) (pool (model-pool m)) (st (q35-state ctx))
         (xq (context-xq ctx))
         (qkv (q35-qkv st)) (z (q35-z st)) (alpha (q35-alpha st)) (beta (q35-beta st))
         (c (q35-c st)) (o (q35-o st)))
    (quantize-act-par! pool h xq)
    (qmatvec! pool (layer-wqkv l) xq qkv)
    (qmatvec! pool (layer-wz l) xq z)
    (qmatvec! pool (layer-walpha l) xq alpha)
    (qmatvec! pool (layer-wbeta l) xq beta)
    (q35-recur! ctx l li qkv z alpha beta c o 1)
    (quantize-act-par! pool o xq)
    (qmatvec! pool (layer-ssm-out l) xq h)))

;;; --- a full (gated attention) layer ---------------------------------------------

(defun q35-split-qg! (m qg q gate n)
  "QG holds [q | gate] per head; split them into Q and GATE, each token."
  (let* ((hd (model-head-dim m)) (nh (model-n-head m)))
    (dotimes (u (* n nh))
      (replace q qg :start1 (* u hd) :start2 (* u 2 hd) :end2 (+ (* u 2 hd) hd))
      (replace gate qg :start1 (* u hd) :start2 (+ (* u 2 hd) hd) :end2 (* (1+ u) 2 hd)))))

(defun q35-qk! (pool m l q k pos qo ko)
  "Norm then rotate (the first ROPE-DIMS of) each head of one token's Q at
   QO and K at KO, for position POS."
  (let* ((hd (model-head-dim m)) (nh (model-n-head m)) (nkv (model-n-kv-head m))
         (eps (model-eps m)))
    (multiple-value-bind (cos sin) (rope-table (model-rope-dims m) pos (model-rope-base m))
      (parallel-for (u (+ nh nkv) pool)
        (if (< u nh)
            (let ((o (+ qo (* u hd))))
              (rmsnorm! q (layer-q-norm l) q eps o)
              (rope-head! q o (model-rope-dims m) t cos sin))
            (let ((o (+ ko (* (- u nh) hd))))
              (rmsnorm! k (layer-k-norm l) k eps o)
              (rope-head! k o (model-rope-dims m) t cos sin)))))))

(defun q35-gate! (att gate end)
  (declare (type f32vec att gate) (type fixnum end))
  (dotimes (i end)
    (setf (aref att i) (* (aref att i) (sigmoid1 (aref gate i))))))

(defun q35-full! (ctx l li h)
  "Layer LI (full attention) of the token whose normed input is H: its
   output in H."
  (let* ((m (context-model ctx)) (pool (model-pool m)) (st (q35-state ctx))
         (xq (context-xq ctx)) (pos (context-pos ctx))
         (hd (model-head-dim m)) (nh (model-n-head m)) (nkv (model-n-kv-head m))
         (kvd (* nkv hd))
         (qg (q35-qg st)) (gate (q35-gate st))
         (q (context-q ctx)) (k (context-k ctx)) (v (context-v ctx)) (att (context-att ctx)))
    (quantize-act-par! pool h xq)
    (qmatvec! pool (layer-wq l) xq qg)
    (qmatvec! pool (layer-wk l) xq k)
    (qmatvec! pool (layer-wv l) xq v)
    (q35-split-qg! m qg q gate 1)
    (q35-qk! pool m l q k pos 0 0)
    (replace (svref (context-k-cache ctx) li) k :start1 (* pos kvd))
    (replace (svref (context-v-cache ctx) li) v :start1 (* pos kvd))
    (attention! ctx li)
    (q35-gate! att gate (* nh hd))
    (quantize-act-par! pool att xq)
    (qmatvec! pool (layer-wo l) xq h)))

;;; --- one token --------------------------------------------------------------------

(defun forward-qwen35 (ctx token &key (logits t))
  (let* ((m (context-model ctx)) (pool (model-pool m)) (pos (context-pos ctx))
         (x (context-x ctx)) (h (context-h ctx)) (xq (context-xq ctx)) (hq (context-hq ctx)))
    (when (>= pos (context-n-ctx ctx))
      (error "context is full (~D tokens)" (context-n-ctx ctx)))
    (embed-row! (model-token-embd m) token x)
    (loop for li below (model-n-layer m)
          for l = (svref (model-layers m) li)
          do (rmsnorm! x (layer-attn-norm l) h (model-eps m))
             (if (eq (layer-kind l) :linear) (q35-linear! ctx l li h) (q35-full! ctx l li h))
             (add! x h)
             (rmsnorm! x (layer-ffn-norm l) h (model-eps m))
             (quantize-act-par! pool h xq)
             (qmatvec! pool (layer-wgate l) xq (context-ffn-a ctx))
             (qmatvec! pool (layer-wup l) xq (context-ffn-b ctx))
             (swiglu-par! pool (context-ffn-a ctx) (context-ffn-b ctx))
             (quantize-act-par! pool (context-ffn-a ctx) hq)
             (qmatvec! pool (layer-wdown l) hq h)
             (add! x h))
    (incf (context-pos ctx))
    (when logits
      (rmsnorm! x (model-output-norm m) h (model-eps m))
      (quantize-act-par! pool h xq)
      (qmatvec! pool (model-output m) xq (context-logits ctx)))
    ctx))

;;; --- a batch of tokens ------------------------------------------------------------
;;;
;;; As FORWARD-BATCH: every projection and the FFN multiply a chunk of tokens
;;; at once, so the weights are read once a chunk, not once a token.  Only
;;; the convolution and the delta rule go token by token (see Q35-RECUR!).
;;; Bit-identical to FORWARD-QWEN35 on each token in turn.

(defstruct (q35-batch (:constructor %make-q35-batch))
  x h qkv z alpha beta c o qg q k v att gate fa fb
  xb ob ab fq)                          ; the quantized inputs: d, V heads, Q, FFN

(defun q35-batch (ctx)
  (or (getf (context-plist ctx) :q35-batch)
      (setf (getf (context-plist ctx) :q35-batch)
            (let* ((m (context-model ctx)) (n *prefill-chunk*) (kinds (model-act-kinds m))
                   (d (model-d m)) (f (model-ffn m)) (ch (q35-channels m))
                   (vh (model-ssm-v-heads m)) (od (* vh (model-ssm-v-dim m)))
                   (qd (* (model-n-head m) (model-head-dim m)))
                   (kvd (* (model-n-kv-head m) (model-head-dim m))))
              (%make-q35-batch
               :x (f32vec (* n d)) :h (f32vec (* n d))
               :qkv (f32vec (* n ch)) :z (f32vec (* n od)) :alpha (f32vec (* n vh)) :beta (f32vec (* n vh))
               :c (f32vec (* n ch)) :o (f32vec (* n od))
               :qg (f32vec (* n 2 qd)) :q (f32vec (* n qd)) :k (f32vec (* n kvd)) :v (f32vec (* n kvd))
               :att (f32vec (* n qd)) :gate (f32vec (* n qd))
               :fa (f32vec (* n f)) :fb (f32vec (* n f))
               :xb (make-q8batch n d kinds) :ob (make-q8batch n od kinds)
               :ab (make-q8batch n qd kinds) :fq (make-q8batch n f kinds))))))

(defun forward-batch-qwen35 (ctx tokens &key (logits t))
  "Run TOKENS (at most *PREFILL-CHUNK*) at the next positions; the logits,
   if LOGITS, are the last token's."
  #+sbcl (sb-int:with-float-traps-masked (:overflow :inexact) (forward-batch-qwen35-1 ctx tokens logits))
  #-sbcl (forward-batch-qwen35-1 ctx tokens logits))

(defun forward-batch-qwen35-1 (ctx tokens logits)
  (let* ((m (context-model ctx)) (pool (model-pool m))
         (n (length tokens)) (pos0 (context-pos ctx))
         (d (model-d m)) (f (model-ffn m)) (eps (model-eps m))
         (hd (model-head-dim m)) (nh (model-n-head m))
         (qd (* nh hd)) (kvd (* (model-n-kv-head m) hd))
         (b (q35-batch ctx))
         (x (q35-batch-x b)) (h (q35-batch-h b))
         (xb (q35-batch-xb b)) (fa (q35-batch-fa b)) (fb (q35-batch-fb b)))
    (assert (<= n *prefill-chunk*))
    (when (> (+ pos0 n) (context-n-ctx ctx))
      (error "context is full (~D tokens)" (context-n-ctx ctx)))
    (let ((row (f32vec d)))
      (loop for tok in tokens for tk from 0
            do (embed-row! (model-token-embd m) tok row)
               (replace x row :start1 (* tk d))))
    (loop for li below (model-n-layer m)
          for l = (svref (model-layers m) li)
          do (dotimes (tk n) (rmsnorm! x (layer-attn-norm l) h eps (* tk d)))
             (quantize-batch! h n xb pool)
             (if (eq (layer-kind l) :linear)
                 (let ((qkv (q35-batch-qkv b)) (z (q35-batch-z b))
                       (alpha (q35-batch-alpha b)) (beta (q35-batch-beta b))
                       (o (q35-batch-o b)) (ob (q35-batch-ob b)))
                   (matmul! pool (layer-wqkv l) xb qkv)
                   (matmul! pool (layer-wz l) xb z)
                   (matmul! pool (layer-walpha l) xb alpha)
                   (matmul! pool (layer-wbeta l) xb beta)
                   (q35-recur! ctx l li qkv z alpha beta (q35-batch-c b) o n)
                   (quantize-batch! o n ob pool)
                   (matmul! pool (layer-ssm-out l) ob h))
                 (let ((qg (q35-batch-qg b)) (q (q35-batch-q b)) (k (q35-batch-k b))
                       (v (q35-batch-v b)) (att (q35-batch-att b)) (gate (q35-batch-gate b))
                       (ab (q35-batch-ab b)))
                   (matmul! pool (layer-wq l) xb qg)
                   (matmul! pool (layer-wk l) xb k)
                   (matmul! pool (layer-wv l) xb v)
                   (q35-split-qg! m qg q gate n)
                   (dotimes (tk n)
                     (q35-qk! pool m l q k (+ pos0 tk) (* tk qd) (* tk kvd)))
                   (replace (svref (context-k-cache ctx) li) k :start1 (* pos0 kvd) :end2 (* n kvd))
                   (replace (svref (context-v-cache ctx) li) v :start1 (* pos0 kvd) :end2 (* n kvd))
                   ;; token tk sees positions 0..pos0+tk: causality is the length
                   (parallel-for (job (* n nh) pool)
                     (multiple-value-bind (tk hh) (floor job nh)
                       (attend-head! ctx li q (+ (* tk qd) (* hh hd)) att (+ pos0 tk 1) hh)))
                   (q35-gate! att gate (* n qd))
                   (quantize-batch! att n ab pool)
                   (matmul! pool (layer-wo l) ab h)))
             (add! x h (* n d))
             (dotimes (tk n) (rmsnorm! x (layer-ffn-norm l) h eps (* tk d)))
             (quantize-batch! h n xb pool)
             (matmul! pool (layer-wgate l) xb fa)
             (matmul! pool (layer-wup l) xb fb)
             (swiglu-par! pool fa fb (* n f))
             (quantize-batch! fa n (q35-batch-fq b) pool)
             (matmul! pool (layer-wdown l) (q35-batch-fq b) h)
             (add! x h (* n d)))
    (incf (context-pos ctx) n)
    (when logits
      (let ((last (f32vec d)) (hr (f32vec d)) (xq (context-xq ctx)))
        (replace last x :start2 (* (1- n) d))
        (rmsnorm! last (model-output-norm m) hr eps)
        (quantize-act-par! pool hr xq)
        (qmatvec! pool (model-output m) xq (context-logits ctx))))
    ctx))
