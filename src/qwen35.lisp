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
  (some (lambda (l) (eq (layer-kind l) :linear)) (model-layers m)))

;;; --- per-context state and scratch ----------------------------------------------

(defstruct (q35 (:constructor %make-q35))
  conv ssm                              ; per layer: conv history, delta-net state
  qkv z alpha beta c o qg gate)         ; scratch

(defun q35-channels (m)
  (+ (* 2 (model-ssm-k-heads m) (model-ssm-k-dim m)) (* (model-ssm-v-heads m) (model-ssm-v-dim m))))

(defun q35-state (ctx)
  (or (getf (context-plist ctx) :q35)
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
               :qg (f32vec (* 2 nh hd)) :gate (f32vec (* nh hd)))))))

(defun reset-recurrent! (ctx)
  "Zero every linear layer's state: the context starts over."
  (let ((st (q35-state ctx)))
    (loop for v across (q35-conv st) do (fill v 0f0))
    (loop for v across (q35-ssm st) do (fill v 0f0)))
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

(defun q35-conv! (pool l qkv c state ch)
  "C = silu(the causal convolution of each channel's last 4 inputs, QKV
   last), as ggml_ssm_conv; STATE (3 a channel, oldest first) moves on by one."
  (declare (type f32vec qkv c state) (type fixnum ch))
  (let ((kern (layer-conv1d l)))
    (declare (type f32vec kern))
    (pool-run pool
              (lambda (i n)
                (multiple-value-bind (s e) (share ch i n)
                  (locally (declare (optimize speed (safety 0)) (type fixnum s e))
                    (loop for ci of-type fixnum from s below e
                          do (let* ((b (* ci 3)) (w (* ci 4))
                                    (x0 (aref state b)) (x1 (aref state (+ b 1))) (x2 (aref state (+ b 2)))
                                    (x3 (aref qkv ci))
                                    (sum 0f0))
                               (declare (type fixnum b w) (type single-float x0 x1 x2 x3 sum))
                               (setf sum (+ sum (* x0 (aref kern w))))
                               (setf sum (+ sum (* x1 (aref kern (+ w 1)))))
                               (setf sum (+ sum (* x2 (aref kern (+ w 2)))))
                               (setf sum (+ sum (* x3 (aref kern (+ w 3)))))
                               (setf (aref c ci) (silu1 sum)
                                     (aref state b) x1 (aref state (+ b 1)) x2 (aref state (+ b 2)) x3)))))))))

(defun q35-delta! (pool m c beta g S o)
  "The delta rule for every V head (a head per unit of work), as
   build_delta_net_autoregressive: decay, read, correct, write, query."
  (declare (type f32vec c beta g S o))
  (let* ((kh (model-ssm-k-heads m)) (kd (model-ssm-k-dim m))
         (vh (model-ssm-v-heads m)) (vd (model-ssm-v-dim m))
         (q-off 0) (k-off (* kh kd)) (v-off (* 2 kh kd))
         (scale (/ 1f0 (sqrt (coerce kd 'single-float)))))
    (declare (type fixnum kh kd vh vd q-off k-off v-off) (type single-float scale))
    (parallel-for (h vh pool)
      (locally (declare (optimize speed (safety 0)))
        (let* ((kq (+ q-off (* (mod h kh) kd)))       ; ggml_repeat tiles K heads over V heads
               (kk (+ k-off (* (mod h kh) kd)))
               (vv (+ v-off (* h vd)))
               (base (* h vd kd))
               (decay (exp (aref g h)))
               (b (aref beta h)))
          (declare (type fixnum kq kk vv base) (type single-float decay b))
          (dotimes (v vd)
            (let ((row (+ base (* v kd))) (sk 0f0))
              (declare (type fixnum row) (type single-float sk))
              ;; S *= exp(g), and the row's S k with the decayed values
              (dotimes (k kd)
                (let ((sv (* (aref S (+ row k)) decay)))
                  (setf (aref S (+ row k)) sv
                        sk (+ sk (* sv (aref c (+ kk k)))))))
              (let ((d (* (- (aref c (+ vv v)) sk) b)) (ov 0f0))
                (declare (type single-float d ov))
                ;; S += d k^T, then o = S (q * scale)
                (dotimes (k kd)
                  (let ((sv (+ (aref S (+ row k)) (* (aref c (+ kk k)) d))))
                    (setf (aref S (+ row k)) sv
                          ov (+ ov (* sv (* (aref c (+ kq k)) scale))))))
                (setf (aref o (+ (* h vd) v)) ov)))))))))

(defun q35-linear! (ctx l li h)
  "Layer LI (linear) of the token whose normed input is H: its output in H."
  (let* ((m (context-model ctx)) (pool (model-pool m)) (st (q35-state ctx))
         (xq (context-xq ctx))
         (ch (q35-channels m)) (vh (model-ssm-v-heads m)) (vd (model-ssm-v-dim m))
         (kh (model-ssm-k-heads m)) (kd (model-ssm-k-dim m))
         (eps (model-eps m))
         (qkv (q35-qkv st)) (z (q35-z st)) (alpha (q35-alpha st)) (beta (q35-beta st))
         (c (q35-c st)) (o (q35-o st)))
    (quantize-act-par! pool h xq)
    (qmatvec! pool (layer-wqkv l) xq qkv)
    (qmatvec! pool (layer-wz l) xq z)
    (qmatvec! pool (layer-walpha l) xq alpha)
    (qmatvec! pool (layer-wbeta l) xq beta)
    (let ((a (layer-ssm-a l)) (dt (layer-ssm-dt l)))
      (declare (type f32vec a dt))
      (dotimes (j vh)
        (setf (aref beta j) (sigmoid1 (aref beta j))
              (aref alpha j) (* (softplus1 (+ (aref alpha j) (aref dt j))) (aref a j)))))
    (q35-conv! pool l qkv c (svref (q35-conv st) li) ch)
    (dotimes (j kh)
      (l2norm! c (* j kd) kd eps)
      (l2norm! c (+ (* kh kd) (* j kd)) kd eps))
    (q35-delta! pool m c beta alpha (svref (q35-ssm st) li) o)
    ;; rmsnorm(o)*ssm_norm * silu(z), per V head
    (let ((w (layer-ssm-norm l)))
      (parallel-for (j vh pool)
        (rmsnorm! o w o eps (* j vd))
        (loop for i of-type fixnum from (* j vd) below (* (1+ j) vd)
              do (setf (aref o i) (* (aref o i) (silu1 (aref z i)))))))
    (quantize-act-par! pool o xq)
    (qmatvec! pool (layer-ssm-out l) xq h)))

;;; --- a full (gated attention) layer ---------------------------------------------

(defun q35-full! (ctx l li h)
  "Layer LI (full attention) of the token whose normed input is H: its
   output in H."
  (let* ((m (context-model ctx)) (pool (model-pool m)) (st (q35-state ctx))
         (xq (context-xq ctx)) (pos (context-pos ctx))
         (hd (model-head-dim m)) (nh (model-n-head m)) (nkv (model-n-kv-head m))
         (kvd (* nkv hd)) (eps (model-eps m))
         (qg (q35-qg st)) (gate (q35-gate st))
         (q (context-q ctx)) (k (context-k ctx)) (v (context-v ctx)) (att (context-att ctx)))
    (quantize-act-par! pool h xq)
    (qmatvec! pool (layer-wq l) xq qg)
    (qmatvec! pool (layer-wk l) xq k)
    (qmatvec! pool (layer-wv l) xq v)
    ;; [q | gate] per head
    (dotimes (j nh)
      (replace q qg :start1 (* j hd) :start2 (* j 2 hd) :end2 (+ (* j 2 hd) hd))
      (replace gate qg :start1 (* j hd) :start2 (+ (* j 2 hd) hd) :end2 (* (1+ j) 2 hd)))
    (multiple-value-bind (cos sin) (rope-table (model-rope-dims m) pos (model-rope-base m))
      (parallel-for (u (+ nh nkv) pool)
        (if (< u nh)
            (let ((o (* u hd)))
              (rmsnorm! q (layer-q-norm l) q eps o)
              (rope-head! q o (model-rope-dims m) t cos sin))
            (let ((o (* (- u nh) hd)))
              (rmsnorm! k (layer-k-norm l) k eps o)
              (rope-head! k o (model-rope-dims m) t cos sin)))))
    (replace (svref (context-k-cache ctx) li) k :start1 (* pos kvd))
    (replace (svref (context-v-cache ctx) li) v :start1 (* pos kvd))
    (attention! ctx li)
    (dotimes (i (* nh hd))
      (setf (aref att i) (* (aref att i) (sigmoid1 (aref gate i)))))
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
