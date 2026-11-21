;;;; inkling2.lisp — Inkling 2 Mini (Thinking Machines), on the CPU: the
;;;; reference path.
;;;
;;; Written from the model's documentation, not from another implementation:
;;; the model card and blog, config.json, the checkpoint's tensor names and
;;; shapes, and Goodfire's appendix equations (docs/inkling2.md says which
;;; detail came from where, and which the weights decided by perplexity).
;;;
;;; The model reads the official checkpoint as it ships (one safetensors
;;; file, the experts in NVFP4 and MXFP8): no conversion step.  24 layers in
;;; four blocks of six; the middle twelve run three times with shared
;;; weights.  Each sublayer's input is a depth-wise mix of the earlier
;;; blocks' sums (attention residuals); passes 2 and 3 also mix in the
;;; previous pass's middle blocks through rank-64 gates.

(in-package #:illatio)

;;; --- safetensors ------------------------------------------------------------------
;;; An 8-byte little-endian header length, a JSON header (name -> dtype,
;;; shape, byte span), then the bytes.  The file is mapped, never read.

(defun %st-json (bytes start end)
  "The JSON value in BYTES[START, END): objects as hash tables, arrays as
   lists, numbers as integers (or floats), true T, false :FALSE, null NIL.
   Enough for a safetensors header and tokenizer.json."
  (let ((i start))
    (labels ((ws () (loop while (and (< i end) (member (aref bytes i) '(32 9 10 13))) do (incf i)))
             (val ()
               (ws)
               (let ((c (code-char (aref bytes i))))
                 (case c
                   (#\{ (incf i) (let ((h (make-hash-table :test 'equal)))
                                   (ws)
                                   (if (= (aref bytes i) 125) (incf i)
                                       (loop (ws) (let ((k (val))) (ws) (incf i) ; :
                                                    (setf (gethash k h) (val)))
                                             (ws)
                                             (if (= (aref bytes i) 44) (incf i) (progn (incf i) (return)))))
                                   h))
                   (#\[ (incf i) (ws)
                    (if (= (aref bytes i) 93) (progn (incf i) nil)
                        (loop with out = '()
                              do (push (val) out) (ws)
                              (if (= (aref bytes i) 44) (incf i) (progn (incf i) (return (nreverse out)))))))
                   (#\" (incf i)
                    (let ((s i))
                      (loop until (= (aref bytes i) 34) do (when (= (aref bytes i) 92) (incf i)) (incf i))
                      (let ((raw (sb-ext:octets-to-string bytes :start s :end i :external-format :utf-8)))
                        (incf i)
                        (if (find #\\ raw) (%json-unescape raw) raw))))
                   (#\t (incf i 4) t)
                   (#\f (incf i 5) :false)
                   (#\n (incf i 4) nil)
                   (t (let ((s i))
                        (loop while (and (< i end) (or (digit-char-p (code-char (aref bytes i)))
                                                       (member (aref bytes i) '(45 43 46 101 69))))
                              do (incf i))
                        (let ((tok (sb-ext:octets-to-string bytes :start s :end i)))
                          (or (ignore-errors (parse-integer tok)) (read-from-string tok)))))))))
      (val))))

(defun %json-unescape (raw)
  (with-output-to-string (o)
    (let ((i 0) (n (length raw)))
      (loop while (< i n)
            do (let ((c (char raw i)))
                 (if (char/= c #\\)
                     (progn (write-char c o) (incf i))
                     (let ((e (char raw (1+ i))))
                       (case e
                         (#\n (write-char #\Newline o) (incf i 2))
                         (#\t (write-char #\Tab o) (incf i 2))
                         (#\r (write-char #\Return o) (incf i 2))
                         (#\b (write-char (code-char 8) o) (incf i 2))
                         (#\f (write-char (code-char 12) o) (incf i 2))
                         (#\u (let ((cp (parse-integer raw :start (+ i 2) :end (+ i 6) :radix 16)))
                                (incf i 6)
                                ;; a surrogate pair
                                (when (and (<= #xD800 cp #xDBFF) (< (1+ i) n) (char= (char raw i) #\\)
                                           (char= (char raw (1+ i)) #\u))
                                  (let ((lo (parse-integer raw :start (+ i 2) :end (+ i 6) :radix 16)))
                                    (setf cp (+ #x10000 (ash (- cp #xD800) 10) (- lo #xDC00))) (incf i 6)))
                                (write-char (code-char cp) o)))
                         (t (write-char e o) (incf i 2))))))))))

(defstruct (stfile (:constructor %make-stfile))
  path bytes (data 0 :type fixnum)
  (tensors (make-hash-table :test 'equal)))   ; name -> (dtype shape start end)

(defun open-safetensors (path)
  (let* ((bytes (map-file-vector path))
         (n (loop for k below 8 sum (ash (aref bytes k) (* 8 k))))
         (h (%st-json bytes 8 (+ 8 n)))
         (st (%make-stfile :path path :bytes bytes :data (+ 8 n))))
    (maphash (lambda (k v)
               (unless (string= k "__metadata__")
                 (let ((span (gethash "data_offsets" v)))
                   (setf (gethash k (stfile-tensors st))
                         (list (gethash "dtype" v) (gethash "shape" v) (first span) (second span))))))
             h)
    st))

(defun st-tensor (st name)
  (or (gethash name (stfile-tensors st)) (error "~A: no tensor ~A" (stfile-path st) name)))

(defun st-offset (st name) (+ (stfile-data st) (third (st-tensor st name))))
(defun st-has (st name) (and (gethash name (stfile-tensors st)) t))

(defun st-floats (st name)
  "Tensor NAME as a static single-float vector, from BF16 or F32."
  (destructuring-bind (dtype shape start end) (st-tensor st name)
    (let* ((n (reduce #'* shape)) (v (f32vec n :static t)) (b (stfile-bytes st)) (o (+ (stfile-data st) start)))
      (declare (type (simple-array (unsigned-byte 8) (*)) b) (type fixnum o n) (type f32vec v))
      (cond ((string= dtype "BF16")
             (assert (= (- end start) (* 2 n)))
             (dotimes (i n) (setf (aref v i) (bits-single (ash (logior (aref b (+ o (* 2 i))) (ash (aref b (+ o 1 (* 2 i))) 8)) 16)))))
            ((string= dtype "F32")
             (dotimes (i n) (let ((p (+ o (* 4 i))))
                              (setf (aref v i) (bits-single (logior (aref b p) (ash (aref b (+ p 1)) 8)
                                                                    (ash (aref b (+ p 2)) 16) (ash (aref b (+ p 3)) 24)))))))
            (t (error "~A is ~A, not a float tensor" name dtype)))
      v)))

;;; --- the expert formats -----------------------------------------------------------
;;; NVFP4 (layers 0-19): two E2M1 codes a byte, the even element in the low
;;; nibble; an E4M3 scale per 16; an outer scale per expert, amax / 2688
;;; (6 x 448, the largest code times the largest scale).  MXFP8 (20-23):
;;; E4M3 values, an E8M0 scale (2^(s-127)) per 32.

(defparameter +fp4-e2m1+
  (make-array 16 :element-type 'single-float
                 :initial-contents '(0f0 .5f0 1f0 1.5f0 2f0 3f0 4f0 6f0 -0f0 -.5f0 -1f0 -1.5f0 -2f0 -3f0 -4f0 -6f0)))

(defparameter +fp8-e4m3+
  (let ((v (make-array 256 :element-type 'single-float)))
    (dotimes (b 256 v)
      (let* ((s (if (logbitp 7 b) -1f0 1f0)) (e (ldb (byte 4 3) b)) (m (ldb (byte 3 0) b)))
        (setf (aref v b)
              (cond ((and (= e 15) (= m 7)) 0f0)                   ; NaN: never in a checkpoint
                    ((zerop e) (* s (/ m 8f0) (expt 2f0 -6)))
                    (t (* s (+ 1f0 (/ m 8f0)) (expt 2f0 (- e 7))))))))))

(defstruct (ink2-experts (:constructor %make-ink2-experts))
  kind                                  ; :nvfp4 or :mxfp8
  rows cols                             ; one expert's matrix
  (w 0 :type fixnum) (s 0 :type fixnum) ; byte offsets of the codes and the scales
  (outer (f32vec 0) :type f32vec))      ; NVFP4: per expert

(defun load-ink2-experts (st prefix)
  (destructuring-bind (dtype shape start end) (st-tensor st prefix)
    (declare (ignore start end))
    (let ((nv (string= dtype "U8")))
      (%make-ink2-experts
       :kind (if nv :nvfp4 :mxfp8) :rows (second shape) :cols (if nv (* 2 (third shape)) (third shape))
       :w (st-offset st prefix) :s (st-offset st (concatenate 'string prefix "_scale"))
       :outer (if nv
                  (let ((amax (st-floats st (concatenate 'string prefix "_amax")))
                        (den (aref (st-floats st (concatenate 'string prefix "_outer_scale_denominator")) 0)))
                    (map-into (f32vec (length amax)) (lambda (a) (/ a den)) amax))
                  (f32vec 0))))))

(defun ink2-expert-row (ex bytes e r x)
  "Row R of expert E's matrix, dotted with X."
  (declare (type (simple-array (unsigned-byte 8) (*)) bytes) (type f32vec x) (type fixnum e r)
           (optimize speed (safety 0)))
  (let* ((rows (ink2-experts-rows ex)) (cols (ink2-experts-cols ex)) (row (+ (* e rows) r)))
    (declare (type fixnum rows cols row))
    (if (eq (ink2-experts-kind ex) :nvfp4)
        (let ((w (+ (ink2-experts-w ex) (* row (ash cols -1)))) (s (+ (ink2-experts-s ex) (* row (ash cols -4))))
              (fp4 +fp4-e2m1+) (fp8 +fp8-e4m3+) (acc 0f0))
          (declare (type fixnum w s) (type single-float acc)
                   (type (simple-array single-float (16)) fp4) (type (simple-array single-float (256)) fp8))
          (dotimes (b (ash cols -4))
            (let ((sum 0f0) (xo (* b 16)) (wo (+ w (* b 8))))
              (declare (type single-float sum) (type fixnum xo wo))
              (dotimes (j 8)
                (let ((c (aref bytes (+ wo j))))
                  (incf sum (+ (* (aref fp4 (logand c 15)) (aref x (+ xo (* 2 j))))
                               (* (aref fp4 (ash c -4)) (aref x (+ xo (* 2 j) 1)))))))
              (incf acc (* (aref fp8 (aref bytes (+ s b))) sum))))
          (* acc (aref (ink2-experts-outer ex) e)))
        (let ((w (+ (ink2-experts-w ex) (* row cols))) (s (+ (ink2-experts-s ex) (* row (ash cols -5))))
              (fp8 +fp8-e4m3+) (acc 0f0))
          (declare (type fixnum w s) (type single-float acc) (type (simple-array single-float (256)) fp8))
          (dotimes (b (ash cols -5) acc)
            (let ((sum 0f0) (xo (* b 32)))
              (declare (type single-float sum) (type fixnum xo))
              (dotimes (j 32) (incf sum (* (aref fp8 (aref bytes (+ w xo j))) (aref x (+ xo j)))))
              (incf acc (* (scale-float 1f0 (- (aref bytes (+ s b)) 127)) sum))))))))

;;; --- the model ------------------------------------------------------------------

(defconstant +ink-d+ 2304)
(defconstant +ink-heads+ 24)
(defconstant +ink-kv-heads+ 8)
(defconstant +ink-hd+ 128)
(defconstant +ink-rel+ 16)
(defconstant +ink-extent+ 1024)
(defconstant +ink-window+ 1024)
(defconstant +ink-ffn+ 1152)

(defstruct (ink2-layer (:constructor %make-ink2-layer))
  index local
  attn-norm q-norm k-norm v-norm        ; local: ATTN-NORM; global: one each for Q, K, V
  wq wk wv wo wr wgate wsink            ; row-major f32 matrices
  q-clip rel-clip rel-p                 ; 24x128, 24x16, 16x1024
  k-conv v-conv                         ; local: 1024x4 taps; global: (static 4x1024 . proj 32x2304)
  attn-sconv mlp-sconv                  ; 2304x4
  mlp-norm router router-bias
  w13 w2                                ; INK2-EXPERTS
  controllers)                          ; core layers: plist :attn :k :v :mlp -> (down forget gain)

(defstruct (ink2 (:constructor %make-ink2))
  st pool
  embed-off embed-norm unembed norm output-query
  attn-queries mlp-queries cross-q cross-kv
  layers
  (n-vocab 200058) (softcap 30f0) (eps 1f-6)
  (qk-clip 22.627417f0) (rel-clip 8f0) (route-scale 8f0)
  (passes 3)
  metal)                                ; an INK2-GPU when the GPU runs it (metal-inkling2.lisp)

(defun load-inkling2 (path &key (threads (default-thread-count)))
  "The official Inkling 2 Mini checkpoint at PATH (its safetensors file, or
   the directory holding it)."
  (let* ((file (if (uiop:directory-pathname-p path)
                   (merge-pathnames "model-00001-of-00001.safetensors" path)
                   path))
         (st (open-safetensors file)))
    (flet ((f (name) (st-floats st (concatenate 'string "model.llm." name)))
           (has (name) (st-has st (concatenate 'string "model.llm." name))))
      (%make-ink2
       :st st :pool (make-pool threads)
       :embed-off (st-offset st "model.llm.embed.weight")
       :embed-norm (f "embed_norm.weight")
       :unembed (let ((u (f "unembed.weight"))) u)
       :norm (f "norm.weight") :output-query (f "attn_res.output_query")
       :attn-queries (coerce (loop for i below 23 collect (f (format nil "attn_res.attn_queries.~D" i))) 'vector)
       :mlp-queries (coerce (loop for i below 24 collect (f (format nil "attn_res.mlp_queries.~D" i))) 'vector)
       :cross-q (f "middle_cycle_cross_pass_queries") :cross-kv (f "middle_cycle_cross_pass_kv_queries")
       :layers
       (coerce
        (loop for l below 24
              collect (let* ((p (format nil "layers.~D." l))
                             (local (has (concatenate 'string p "attn_norm.weight"))))
                        (flet ((g (n) (f (concatenate 'string p n))))
                          (%make-ink2-layer
                           :index l :local local
                           :attn-norm (and local (g "attn_norm.weight"))
                           :q-norm (and (not local) (g "q_input_norm.weight"))
                           :k-norm (and (not local) (g "k_input_norm.weight"))
                           :v-norm (and (not local) (g "v_input_norm.weight"))
                           :wq (g "attn.wq_du.weight") :wk (g "attn.wk_dv.weight") :wv (g "attn.wv_dv.weight")
                           :wo (g "attn.wo_ud.weight") :wr (g "attn.wr_du.weight")
                           :wgate (g "attn.gate_attn.weight") :wsink (g "attn.attn_sink_proj.weight")
                           :q-clip (g "attn.q_l2_clip.scale") :rel-clip (g "attn.rel_l2_clip.scale")
                           :rel-p (g "attn.rel_logits_proj.proj")
                           :k-conv (if local (g "attn.k_sconv.weight")
                                       (cons (g "attn.k_sconv.static_w") (g "attn.k_sconv.weight_proj")))
                           :v-conv (if local (g "attn.v_sconv.weight")
                                       (cons (g "attn.v_sconv.static_w") (g "attn.v_sconv.weight_proj")))
                           :attn-sconv (g "attn_sconv.weight") :mlp-sconv (g "mlp_sconv.weight")
                           :mlp-norm (g "mlp_norm.weight")
                           :router (g "mlp.gate.weight") :router-bias (g "mlp.gate.bias")
                           :w13 (load-ink2-experts st (format nil "model.llm.~Amlp.experts.w13_weight" p))
                           :w2 (load-ink2-experts st (format nil "model.llm.~Amlp.experts.w2_weight" p))
                           :controllers
                           (and (has (concatenate 'string p "controllers.attn.down_weight"))
                                (loop for s in '("attn" "k" "v" "mlp")
                                      for key in '(:attn :k :v :mlp)
                                      append (list key (list (g (format nil "controllers.~A.down_weight" s))
                                                             (g (format nil "controllers.~A.forget_weight" s))
                                                             (g (format nil "controllers.~A.gain_weight" s))))))))))
        'vector)))))

;;; --- small vector pieces -------------------------------------------------------

(defun ink-matvec! (pool w rows cols x out &optional (xo 0))
  "OUT = W X, W row-major ROWS x COLS (COLS a multiple of 16)."
  (declare (type f32vec w x out) (type fixnum rows cols xo))
  (parallel-for (r rows pool)
    (with-attention-kernels (dot axpy) (w w x out)
      (setf (aref out r) (dot (* r cols) xo cols))))
  out)

(defun ink-rms (x &optional w (eps 1f-6) (start 0) (n (- (length x) start)) (out (f32vec n)) (oo 0))
  "OUT[oo..] = X[start..start+n) / RMS (x W, when W)."
  (declare (type f32vec x out) (type fixnum start n oo))
  (let ((ss 0d0))
    (loop for i from start below (+ start n) do (incf ss (* (aref x i) (aref x i))))
    (let ((k (/ 1f0 (sqrt (+ (coerce (/ ss n) 'single-float) eps)))))
      (dotimes (i n out)
        (setf (aref out (+ oo i)) (* (aref x (+ start i)) k (if w (aref w i) 1f0)))))))

(defun ink-depth-read (q sources &optional (qo 0))
  "x = Σ_j softmax_j(q·v̂_j/√d) v̂_j, v̂ = v/RMS(v) (Goodfire's appendix)."
  (let* ((vh (mapcar (lambda (v) (ink-rms v)) sources))
         (a (mapcar (lambda (v) (/ (loop for i below +ink-d+ sum (* (aref q (+ qo i)) (aref v i))) (sqrt (float +ink-d+))))
                    vh))
         (m (reduce #'max a))
         (e (mapcar (lambda (s) (exp (- s m))) a))
         (z (reduce #'+ e))
         (x (f32vec +ink-d+)))
    (loop for v in vh for w in e
          do (let ((w (/ w z))) (dotimes (i +ink-d+) (incf (aref x i) (* w (aref v i))))))
    x))

(defun ink-sigmoid (x) (/ 1f0 (+ 1f0 (exp (- x)))))

(defun ink-l2clip! (v o n thr &optional scale (so 0) (log 1f0))
  "V[o..o+n) times SCALE[so..] (when SCALE), cut to L2 norm THR if longer,
   then times LOG (the global layers' log scale)."
  (when scale (dotimes (i n) (setf (aref v (+ o i)) (* (aref v (+ o i)) (aref scale (+ so i))))))
  (let* ((nn (sqrt (loop for i below n sum (* (aref v (+ o i)) (aref v (+ o i))))))
         (k (* log (min 1f0 (/ thr (+ nn 1f-6))))))
    (dotimes (i n v) (setf (aref v (+ o i)) (* (aref v (+ o i)) k)))))

(defun ink-log-scale (pos)
  "The global layers' query and relative-state scale at POS: 1 + 0.1
   log(max((pos+1)/8192, 1))."
  (+ 1f0 (* 0.1f0 (log (max (/ (1+ pos) 8192f0) 1f0)))))

;;; --- per-context state -------------------------------------------------------------
;;; One application per (layer, pass): 6 + 12x3 + 6 = 48.  Each keeps its K
;;; and V (after the short conv and the norm), and the last three inputs of
;;; each of its short convs.

(defstruct (ink2-app (:constructor %make-ink2-app))
  layer cap k v                         ; CAP positions (a ring of 1024 for a local layer)
  k-hist v-hist a-hist m-hist)          ; the last 3 rows, oldest first

(defstruct (ink2ctx (:constructor %make-ink2ctx))
  model (pos 0) n-ctx apps logits
  gpu)                                  ; the GPU's state for it (metal-inkling2.lisp)

(defun make-ink2ctx (m &key (n-ctx 8192))
  (%make-ink2ctx
   :model m :n-ctx n-ctx :logits (f32vec (ink2-n-vocab m))
   :apps (coerce
          (loop for (l . pass) in (ink2-app-order m)
                collect (let* ((layer (aref (ink2-layers m) l))
                               (cap (if (ink2-layer-local layer) (min n-ctx +ink-window+) n-ctx)))
                          (%make-ink2-app :layer layer :cap cap
                                          :k (f32vec (* cap 1024) :static t) :v (f32vec (* cap 1024) :static t)
                                          :k-hist (f32vec (* 3 1024)) :v-hist (f32vec (* 3 1024))
                                          :a-hist (f32vec (* 3 +ink-d+)) :m-hist (f32vec (* 3 +ink-d+)))))
          'vector)))

(defun ink2-app-order (m)
  "(layer . pass) for each application, in the order they run."
  (append (loop for l below 6 collect (cons l 1))
          (loop for p from 1 to (ink2-passes m) append (loop for l from 6 below 18 collect (cons l p)))
          (loop for l from 18 below 24 collect (cons l 1))))

(defun ink2-app-index (m l pass)
  (cond ((< l 6) l) ((< l 18) (+ 6 (* 12 (1- pass)) (- l 6))) (t (+ 6 (* 12 (ink2-passes m)) (- l 18)))))

(defun ink-conv-res! (hist w x n &key dyn)
  "The short conv of X over HIST (the last 3 inputs, oldest first) and X,
   residual and rescaled; HIST shifts.  Static (W N x 4, oldest first, as
   PyTorch's causal depthwise conv): (x + conv) / sqrt(5).  With DYN (global
   K/V: W is (static 4 x N . proj), DYN the token's per-head part 8 x 4): tap
   k weights the token k back, static[k] + dyn; (x + conv) / 3."
  (let ((y (f32vec n)))
    (dotimes (c n)
      (let ((xs (vector (aref hist c) (aref hist (+ n c)) (aref hist (+ (* 2 n) c)) (aref x c))))
        (setf (aref y c)
              (if dyn
                  (/ (+ (aref x c)
                        (loop for lag below 4
                              sum (* (+ (aref (car w) (+ (* lag n) c)) (aref dyn (+ (* 4 (floor c +ink-hd+)) lag)))
                                     (svref xs (- 3 lag)))))
                     3f0)
                  (/ (+ (aref x c) (loop for k below 4 sum (* (aref w (+ (* c 4) k)) (svref xs k))))
                     (sqrt 5f0))))))
    (replace hist hist :start1 0 :start2 n)
    (replace hist x :start1 (* 2 n))
    y))

(defun ink2-mix (m l s x mid)
  "Cross-pass: x' = (1 - tanh(F z)) x + tanh(G z) m_s, z = P[RMSNorm(x); RMSNorm(m_s)]."
  (let* ((i (- l 6))
         (u (ecase s (:attn (ink2-cross-q m)) (:mlp (ink2-cross-q m)) (:k (ink2-cross-kv m)) (:v (ink2-cross-kv m))))
         (row (* +ink-d+ (+ (* 2 i) (if (member s '(:mlp :v)) 1 0))))
         (msg (ink-depth-read u mid row))
         (ctl (getf (ink2-layer-controllers (aref (ink2-layers m) l)) s))
         (zin (f32vec (* 2 +ink-d+)))
         (z (f32vec 64)) (fz (f32vec +ink-d+)) (gz (f32vec +ink-d+)) (pool (ink2-pool m)))
    (ink-rms x nil 1f-6 0 +ink-d+ zin 0)
    (ink-rms msg nil 1f-6 0 +ink-d+ zin +ink-d+)
    (ink-matvec! pool (first ctl) 64 (* 2 +ink-d+) zin z)
    (ink-matvec! pool (second ctl) +ink-d+ 64 z fz)
    (ink-matvec! pool (third ctl) +ink-d+ 64 z gz)
    (let ((out (f32vec +ink-d+)))
      (dotimes (k +ink-d+ out)
        (setf (aref out k) (+ (* (- 1f0 (tanh (aref fz k))) (aref x k)) (* (tanh (aref gz k)) (aref msg k))))))))

;;; --- attention ---------------------------------------------------------------

(defun ink2-attention (m app pos xq xk xv)
  "The attention sublayer of APP for the token at POS, its Q, K and V inputs
   normed.  Returns its output (before the residual short conv)."
  (let* ((ly (ink2-app-layer app)) (pool (ink2-pool m)) (local (ink2-layer-local ly))
         (q (ink-matvec! pool (ink2-layer-wq ly) 3072 +ink-d+ xq (f32vec 3072)))
         (k0 (ink-matvec! pool (ink2-layer-wk ly) 1024 +ink-d+ xk (f32vec 1024)))
         (v0 (ink-matvec! pool (ink2-layer-wv ly) 1024 +ink-d+ xv (f32vec 1024)))
         (k (if local
                (ink-conv-res! (ink2-app-k-hist app) (ink2-layer-k-conv ly) k0 1024)
                (ink-conv-res! (ink2-app-k-hist app) (ink2-layer-k-conv ly) k0 1024
                               :dyn (ink-matvec! pool (cdr (ink2-layer-k-conv ly)) 32 +ink-d+ xk (f32vec 32)))))
         (v (if local
                (ink-conv-res! (ink2-app-v-hist app) (ink2-layer-v-conv ly) v0 1024)
                (ink-conv-res! (ink2-app-v-hist app) (ink2-layer-v-conv ly) v0 1024
                               :dyn (ink-matvec! pool (cdr (ink2-layer-v-conv ly)) 32 +ink-d+ xv (f32vec 32)))))
         (r (ink-matvec! pool (ink2-layer-wr ly) 384 +ink-d+ xq (f32vec 384)))
         (sink (ink-matvec! pool (ink2-layer-wsink ly) 24 +ink-d+ xq (f32vec 24)))
         (gate (ink-matvec! pool (ink2-layer-wgate ly) 3072 +ink-d+ xq (f32vec 3072)))
         (cap (ink2-app-cap app)) (slot (mod pos cap))
         (kc (ink2-app-k app)) (vc (ink2-app-v app))
         (o (f32vec 3072)))
    ;; K cut to the clip threshold per head, V RMS-normed per head (no gain)
    (dotimes (h +ink-kv-heads+)
      (ink-l2clip! k (* h 128) 128 (ink2-qk-clip m))
      (replace kc k :start1 (+ (* slot 1024) (* h 128)) :start2 (* h 128) :end2 (* (1+ h) 128))
      (ink-rms v nil 1f-6 (* h 128) 128 vc (+ (* slot 1024) (* h 128))))
    ;; Q and r scaled, then clipped (and in a global layer log-scaled)
    (let ((ls (if local 1f0 (ink-log-scale pos))))
      (dotimes (h +ink-heads+)
        (ink-l2clip! q (* h 128) 128 (ink2-qk-clip m) (ink2-layer-q-clip ly) (* h 128) ls)
        (ink-l2clip! r (* h 16) 16 (ink2-rel-clip m) (ink2-layer-rel-clip ly) (* h 16) ls)))
    (let* ((first (if local (max 0 (- pos (1- +ink-window+))) 0))
           (n (1+ (- pos first)))
           (p (ink2-layer-rel-p ly))
           (scale (/ 1f0 (sqrt 128f0))))
      (parallel-for (h +ink-heads+ pool)
        (let* ((g (floor h 3)) (lg (f32vec n)) (mx (aref sink h)))
          (dotimes (jj n)
            (let* ((j (+ first jj)) (s (mod j cap)) (dist (- pos j))
                   (dot (loop for d below 128 sum (* (aref q (+ (* h 128) d)) (aref kc (+ (* s 1024) (* g 128) d)))))
                   (b (if (< dist +ink-extent+)
                          (loop for rr below 16 sum (* (aref r (+ (* h 16) rr)) (aref p (+ (* rr +ink-extent+) dist))))
                          0f0)))
              (setf (aref lg jj) (+ (* dot scale) b))
              (setf mx (max mx (aref lg jj)))))
          (let ((z (exp (- (aref sink h) mx))))
            (dotimes (jj n) (setf (aref lg jj) (exp (- (aref lg jj) mx))) (incf z (aref lg jj)))
            (dotimes (jj n)
              (let ((w (/ (aref lg jj) z)) (s (mod (+ first jj) cap)))
                (dotimes (d 128)
                  (incf (aref o (+ (* h 128) d)) (* w (aref vc (+ (* s 1024) (* g 128) d)))))))))))
    (dotimes (i 3072) (setf (aref o i) (* (aref o i) (ink-sigmoid (aref gate i)))))
    (ink-matvec! pool (ink2-layer-wo ly) +ink-d+ 3072 o (f32vec +ink-d+))))

;;; --- MoE ---------------------------------------------------------------------

(defun ink2-moe (m ly x)
  (let* ((pool (ink2-pool m))
         (logits (ink-matvec! pool (ink2-layer-router ly) 128 +ink-d+ x (f32vec 128)))
         (sc (map 'list #'ink-sigmoid logits))
         (order (sort (loop for e below 128 collect e) #'>
                      :key (lambda (e) (+ (nth e sc) (aref (ink2-layer-router-bias ly) e)))))
         (sel (subseq order 0 8))
         (tot (+ 1f-20 (reduce #'+ sel :key (lambda (e) (nth e sc)))))
         (out (f32vec +ink-d+))
         (bytes (stfile-bytes (ink2-st m)))
         (h (f32vec (* 2 +ink-ffn+))) (a (f32vec +ink-ffn+)) (y (f32vec +ink-d+)))
    (dolist (e sel out)
      (let ((w (* (/ (nth e sc) tot) (ink2-route-scale m))))
        (parallel-for (rr (* 2 +ink-ffn+) pool)
          (setf (aref h rr) (ink2-expert-row (ink2-layer-w13 ly) bytes e rr x)))
        ;; gate on the even rows, up on the odd: 4 tanh(g/4) sigmoid(g) 25 tanh(u/25)
        ;; (the sigmoid sees the raw gate; the softcaps bound its linear factor and up)
        (dotimes (i +ink-ffn+)
          (let ((g (aref h (* 2 i))) (u (aref h (1+ (* 2 i)))))
            (setf (aref a i) (* 4f0 (tanh (/ g 4f0)) (/ 1f0 (+ 1f0 (exp (- g)))) 25f0 (tanh (/ u 25f0))))))
        (parallel-for (rr +ink-d+ pool)
          (setf (aref y rr) (ink2-expert-row (ink2-layer-w2 ly) bytes e rr a)))
        (dotimes (i +ink-d+) (incf (aref out i) (* w (aref y i))))))))

;;; --- one token ---------------------------------------------------------------

(defun ink2-token! (c token)
  "Run TOKEN at C's next position; the logits land in C."
  (let* ((m (ink2ctx-model c)) (pos (ink2ctx-pos c)) (eps (ink2-eps m))
         (emb (let ((v (f32vec +ink-d+)) (b (stfile-bytes (ink2-st m))) (o (+ (ink2-embed-off m) (* token +ink-d+ 2))))
                (dotimes (i +ink-d+ v)
                  (setf (aref v i) (bits-single (ash (logior (aref b (+ o (* 2 i))) (ash (aref b (+ o 1 (* 2 i))) 8)) 16))))))
         (v0 (ink-rms emb (ink2-embed-norm m) eps))
         (mid nil))
    (labels ((add (a b) (let ((o (f32vec +ink-d+))) (dotimes (i +ink-d+ o) (setf (aref o i) (+ (aref a i) (aref b i))))))
             (layer (l blocks part pass)
               (let* ((ly (aref (ink2-layers m) l))
                      (app (aref (ink2ctx-apps c) (ink2-app-index m l pass)))
                      (core (<= 6 l 17))
                      (xs (if (zerop l)
                              (list v0 v0 v0)
                              ;; a block's running sum is a source once it holds something:
                              ;; a block's first attention reads the completed blocks only
                              (let ((q (aref (ink2-attn-queries m) (1- l)))
                                    (srcs (if (zerop (mod l 6)) blocks (append blocks (list part)))))
                                (if (ink2-layer-local ly)
                                    (let ((x (ink-depth-read q srcs))) (list x x x))
                                    (loop for i below 3 collect (ink-depth-read q srcs (* i +ink-d+))))))))
                 (when (and core (>= pass 2))
                   (setf xs (loop for x in xs for s in '(:attn :k :v) collect (ink2-mix m l s x mid))))
                 (setf xs (if (ink2-layer-local ly)
                              (mapcar (lambda (x) (ink-rms x (ink2-layer-attn-norm ly) eps)) xs)
                              (list (ink-rms (first xs) (ink2-layer-q-norm ly) eps)
                                    (ink-rms (second xs) (ink2-layer-k-norm ly) eps)
                                    (ink-rms (third xs) (ink2-layer-v-norm ly) eps))))
                 (let* ((oa (ink2-attention m app pos (first xs) (second xs) (third xs)))
                        (oa (ink-conv-res! (ink2-app-a-hist app) (ink2-layer-attn-sconv ly) oa +ink-d+))
                        (part (add part oa))
                        (xm (ink-depth-read (aref (ink2-mlp-queries m) l) (append blocks (list part))))
                        (xm (if (and core (>= pass 2)) (ink2-mix m l :mlp xm mid) xm))
                        (xm (ink-rms xm (ink2-layer-mlp-norm ly) eps))
                        (om (ink2-moe m ly xm))
                        (om (ink-conv-res! (ink2-app-m-hist app) (ink2-layer-mlp-sconv ly) om +ink-d+)))
                   (add part om))))
             (block-sum (from to blocks pass)
               (let ((part (f32vec +ink-d+)))
                 (loop for l from from below to do (setf part (layer l blocks part pass)))
                 part)))
      (let* ((v1 (block-sum 0 6 (list v0) 1)) (v2 nil) (v3 nil))
        (loop for pass from 1 to (ink2-passes m)
              do (setf v2 (block-sum 6 12 (list v0 v1) pass)
                       v3 (block-sum 12 18 (list v0 v1 v2) pass)
                       mid (list v2 v3)))
        (let* ((v4 (block-sum 18 24 (list v0 v1 v2 v3) 1))
               (x (ink-rms (ink-depth-read (ink2-output-query m) (list v0 v1 v2 v3 v4)) (ink2-norm m) eps))
               (lg (ink2ctx-logits c)) (cap (ink2-softcap m)))
          (ink-matvec! (ink2-pool m) (ink2-unembed m) (ink2-n-vocab m) +ink-d+ x lg)
          (dotimes (i (length lg)) (setf (aref lg i) (* cap (tanh (/ (aref lg i) cap)))))
          (incf (ink2ctx-pos c))
          lg)))))

;;; --- the tokenizer: tokenizer.json ---------------------------------------------
;;; Byte-level BPE with o200k's split (tokenizer.lisp), merges by rank, and a
;;; word that is a token taken whole (ignore_merges).  The added tokens
;;; (<|message_user|>, <|content_text|>, ...) are specials.

(defun load-ink2-vocab (path)
  (let* ((bytes (with-open-file (s path :element-type '(unsigned-byte 8))
                  (let ((b (make-array (file-length s) :element-type '(unsigned-byte 8)))) (read-sequence b s) b)))
         (j (%st-json bytes 0 (length bytes)))
         (model (gethash "model" j))
         (vocab (gethash "vocab" model))
         (added (gethash "added_tokens" j))
         (n (1+ (max (loop for v being the hash-values of vocab maximize v)
                     (reduce #'max added :key (lambda (a) (gethash "id" a))))))
         (pieces (make-array n :initial-element ""))
         (types (make-array n :initial-element +tok-normal+))
         (v (%make-vocab :pieces pieces :types types :pre "o200k" :add-bos nil
                         :eos 200006 :eot 200006)))
    (maphash (lambda (piece id) (setf (svref pieces id) piece (gethash piece (vocab-ids v)) id)) vocab)
    (dolist (a added)
      (let ((id (gethash "id" a)) (piece (gethash "content" a)))
        (setf (svref pieces id) piece (svref types id) +tok-control+ (gethash piece (vocab-ids v)) id)))
    (loop for (a b) in (gethash "merges" model) for rank from 0
          do (setf (gethash (concatenate 'string a " " b) (vocab-ranks v)) rank))
    (setf (vocab-specials v)
          (sort (loop for a in added collect (cons (gethash "content" a) (gethash "id" a)))
                #'> :key (lambda (e) (length (car e)))))
    v))

;;; --- as an illatio model ---------------------------------------------------------

(defun inkling2-path-p (path)
  "PATH is Inkling 2 Mini's checkpoint: a safetensors file, or a directory
   holding one with a config.json that names it."
  (let ((p (namestring path)))
    (or (and (uiop:directory-pathname-p (pathname p))
             (probe-file (merge-pathnames "config.json" (pathname p)))
             (search "inkling2" (uiop:read-file-string (merge-pathnames "config.json" (pathname p)))))
        (and (>= (length p) 12) (string-equal ".safetensors" p :start2 (- (length p) 12))))))

(defun load-inkling2-model (path threads &key gpu)
  (let* ((dir (if (uiop:directory-pathname-p (pathname (namestring path)))
                  (pathname (namestring path))
                  (uiop:pathname-directory-pathname path)))
         (d (load-inkling2 dir :threads threads))
         (v (load-ink2-vocab (merge-pathnames "tokenizer.json" dir))))
    (when gpu (setf (ink2-metal d) (funcall 'ink2-gpu-open d)))
    (%make-model :gguf nil :vocab v :arch "inkling2" :name "Inkling 2 Mini"
                 :n-layer 24 :d 0 :n-head 0 :n-kv-head 1 :head-dim 1 :q-dim 0 :ffn 0
                 :n-vocab (ink2-n-vocab d) :context-length 1048576
                 :pool (ink2-pool d) :impl d)))

(defun ink2-context (ctx)
  (or (getf (context-plist ctx) :ink2)
      (setf (getf (context-plist ctx) :ink2)
            (let ((c (make-ink2ctx (model-impl (context-model ctx)) :n-ctx (context-n-ctx ctx))))
              (setf (ink2ctx-logits c) (context-logits ctx))
              c))))

(defun ink2-context-forward (ctx token logits)
  (declare (ignore logits))
  (let ((c (ink2-context ctx)))
    (assert (= (ink2ctx-pos c) (context-pos ctx)) () "Inkling 2's state does not rewind")
    (sb-int:with-float-traps-masked (:overflow :inexact :invalid :divide-by-zero)
      (if (ink2-metal (ink2ctx-model c))
          (funcall 'ink2-gpu-token! c token)
          (ink2-token! c token)))
    (setf (context-pos ctx) (ink2ctx-pos c))
    ctx))

(defmethod impl-rewinds-p ((impl ink2)) (and (ink2-metal impl) t))

(defun ink2-context-eval (ctx tokens)
  "TOKENS at CTX's position.  On the GPU CTX may have gone back
   (SYNC-PROMPT's common prefix): the state is restored from the last
   prompt end at or before it (or from the start), the tokens between fed
   again, and a snapshot kept at the new end."
  (if (and tokens (ink2-metal (model-impl (context-model ctx))))
      (let* ((c (ink2-context ctx)) (want (context-pos ctx)))
        (when (/= (ink2ctx-pos c) want)
          ;; a snapshot past WANT saw tokens this prompt no longer has
          (setf (getf (context-plist ctx) :ink2-snaps)
                (remove-if (lambda (e) (> (car e) want)) (getf (context-plist ctx) :ink2-snaps)))
          (let ((snap (find-if (lambda (e) (<= (car e) want)) (getf (context-plist ctx) :ink2-snaps))))
            (if (and snap (ink2ctx-gpu c))
                (funcall 'ink2-gpu-restore c (cdr snap) (car snap))
                (progn (ink2-context-reset ctx) (setf c (ink2-context ctx))))
            (setf tokens (append (subseq (context-tokens ctx) (ink2ctx-pos c) want) tokens))))
        (sb-int:with-float-traps-masked (:overflow :inexact :invalid :divide-by-zero)
          (funcall 'ink2-gpu-eval! c tokens))
        (setf (context-pos ctx) (ink2ctx-pos c))
        ;; this prompt's end, to come back to (the newest first; a few kept)
        (let ((snaps (remove (ink2ctx-pos c) (getf (context-plist ctx) :ink2-snaps) :key #'car :test #'<=)))
          (setf (getf (context-plist ctx) :ink2-snaps)
                (subseq (cons (cons (ink2ctx-pos c) (funcall 'ink2-gpu-snapshot c)) snaps) 0 (min 8 (1+ (length snaps)))))))
      (dolist (tk tokens) (ink2-context-forward ctx tk t)))
  (context-logits ctx))

(defun ink2-context-reset (ctx)
  "Start CTX over: fresh caches and conv histories."
  (let ((c (getf (context-plist ctx) :ink2)))
    (when c (loop for a across (ink2ctx-apps c)
                  do (free-static-vector (ink2-app-k a)) (free-static-vector (ink2-app-v a)))))
  (remf (context-plist ctx) :ink2)
  (remf (context-plist ctx) :ink2-snaps)
  (setf (context-pos ctx) 0)
  ctx)
