;;;; metal-dense.lisp — the dense models (qwen2, qwen3, qwen3vl's text model,
;;;; llama) on the GPU, their matrices Q8_0, Q4_K or Q6_K (src/metal/dense.metal).
;;;
;;; As metal-qwen35.lisp: the GGUF mapped and wrapped as one buffer, every
;;; matrix multiplied where it lies (a prefill's tokens by the tiled products,
;;; half products summed in floats: d4_mm_q8_0, d4_mm_q4_k, mm_q6k_64; one
;;; token by the matrix-vector ones: d4_mv_q8_0, dn_mv_q4_k, mv_q6k), a pass
;;; one command buffer.  The KV cache is halves, read by a flash attention on the half
;;; simdgroup matrices.  A caller may put rows into the stream after the
;;; embedding and after each layer (*GPU-AFTER-EMBED*, *GPU-AFTER-LAYER*, as
;;; cuda-dense.lisp's hooks: a vision model's picture), and may replace the
;;; RoPE table (GPU-SET-ROPE: Qwen3-VL's positions for a picture).  Steering
;;; (steer.lisp) applies to the attention's and FFN's outputs as on the CPU.

(in-package #:illatio)

(defvar *gpu-after-embed* nil
  "A function called with (ENCODER X POS0 N) once a dense model's pass has
   encoded its N tokens' embedding into buffer X (rows of the model's width,
   the first at position POS0), before the first layer; it may dispatch into
   ENCODER.  Or NIL.")

(defvar *gpu-after-layer* nil
  "A function called with (ENCODER LAYER X POS0 N) after each layer of a
   dense model's pass, X its rows as the next layer reads them; or NIL.")

(defun dense-gpu-check (m)
  "Signal an error unless dense model M's shapes and formats are what the kernels take."
  (let ((hd (model-head-dim m)))
    (assert (member hd '(64 128)) () "the GPU's attention takes heads of 64 or 128, not ~D" hd)
    (dolist (w (cons (model-token-embd m)
                     (loop for l across (model-layers m)
                           append (list (layer-wq l) (layer-wk l) (layer-wv l) (layer-wo l)
                                        (layer-wgate l) (layer-wup l) (layer-wdown l)))))
      (unless (member (gmat-type w) '(:q8_0 :q4_k :q6_k))
        (error "the GPU runs dense models' Q8_0, Q4_K and Q6_K matrices; ~A is ~A" (gmat-name w) (gmat-type w))))
    (let ((out (model-output m)))
      (unless (member (gmat-type out) '(:q8_0 :q4_k :q6_k))
        (error "the GPU runs dense models' Q8_0, Q4_K and Q6_K matrices; the output is ~A" (gmat-type out))))))

;;; --- per-context state ------------------------------------------------------------

(defstruct (dctx (:constructor %make-dctx))
  x h q k v qh att fa fb logits rope
  capture                               ; steering's capture: a row of D a layer and kind, or NIL
  kc vc)                                ; per layer: halves, n-ctx positions

(defun dense-state (ctx)
  (or (getf (context-plist ctx) :gpu-dense)
      (setf (getf (context-plist ctx) :gpu-dense)
            (let* ((m (context-model ctx)) (gpu (gmodel-gpu (model-gpu m)))
                   (n *gpu-chunk*) (n-ctx (context-n-ctx ctx))
                   (d (model-d m)) (f (model-ffn m)) (rd (model-rope-dims m))
                   (qd (* (model-n-head m) (model-head-dim m)))
                   (kvd (* (model-n-kv-head m) (model-head-dim m))))
              (flet ((fb (k) (gpu-zero (gpu-buffer gpu (* 4 k))))
                     (hb (k) (gpu-zero (gpu-buffer gpu (* 2 k)))))
                (let ((st (%make-dctx :x (fb (* n d)) :h (fb (* n d)) :q (fb (* n qd)) :k (fb (* n kvd))
                                      :v (fb (* n kvd)) :qh (hb (* n qd)) :att (fb (* n qd))
                                      :fa (fb (* n f)) :fb (fb (* n f)) :logits (fb (model-n-vocab m))
                                      :rope (fb (* n-ctx rd))
                                      :kc (map 'simple-vector (lambda (l) (declare (ignore l)) (hb (* n-ctx kvd)))
                                               (model-layers m))
                                      :vc (map 'simple-vector (lambda (l) (declare (ignore l)) (hb (* n-ctx kvd)))
                                               (model-layers m)))))
                  ;; every position's RoPE row, as the CPU computes it (ROPE-TABLE), once
                  (dotimes (p n-ctx)
                    (multiple-value-bind (cos sin) (rope-table rd p (model-rope-base m))
                      (gbuf-set-floats (dctx-rope st) cos (* p rd))
                      (gbuf-set-floats (dctx-rope st) sin (+ (* p rd) (floor rd 2)))))
                  st))))))

(defun gpu-set-rope (ctx table)
  "Replace CTX's RoPE table with TABLE: a row of the model's rope-dims floats
   a position from 0 (cosines, then sines), as many as it has."
  (let ((st (dense-state ctx)))
    (gbuf-set-floats (dctx-rope st) (coerce table 'f32vec))
    ctx))

(defun free-gpu-states (ctx)
  "Give back the GPU buffers CTX's passes made (a dense model's, a qwen35's)."
  (let ((st (getf (context-plist ctx) :gpu-dense)))
    (when st
      (dolist (b (list (dctx-x st) (dctx-h st) (dctx-q st) (dctx-k st) (dctx-v st) (dctx-qh st) (dctx-att st)
                       (dctx-fa st) (dctx-fb st) (dctx-logits st) (dctx-rope st) (dctx-capture st)))
        (when b (gpu-free b)))
      (loop for b across (dctx-kc st) do (gpu-free b))
      (loop for b across (dctx-vc st) do (gpu-free b))
      (remf (context-plist ctx) :gpu-dense)))
  (let ((st (getf (context-plist ctx) :gpu)))
    (when st
      (dolist (slot '(x h qkv z alpha beta c o qg q gate k v att fa fb logits rope part ml))
        (let ((b (slot-value st slot))) (when b (gpu-free b))))
      (dolist (slot '(kc vc conv ssm))
        (loop for b across (slot-value st slot) when b do (gpu-free b)))
      (remf (context-plist ctx) :gpu))))

;;; --- steering ---------------------------------------------------------------------------

(defun gpu-steer-dirs (gpu)
  "The steering's directions on GPU (made the first time)."
  (let ((s *steering*))
    (or (gethash gpu (steering-device s))
        (setf (gethash gpu (steering-device s))
              (let ((b (gpu-buffer gpu (* 4 (length (steering-dirs s))))))
                (gbuf-set-floats b (steering-dirs s)))))))

(defun gpu-steer (gpu e st y li kind n d)
  "STEER-ROWS! in the pass: Y's N rows of D steered for layer LI's KIND
   output, the last row first copied to the state's capture when capturing."
  (let ((scale (steer-scale li kind)) (slot (+ (* 2 li) (if (eq kind :attn) 0 1))))
    (dispatch gpu e "dn_steer" (list d scale (1- n) (if *steer-capture* 1 0))
              (list y (if (zerop scale) y (cons (gpu-steer-dirs gpu) (* 4 li d)))
                    (cons (dctx-capture st) (* 4 slot d)))
              n 256)))

;;; --- a pass ---------------------------------------------------------------------------

(defun mat-q8 (gm e w x y n)
  "Y = W X for N tokens (token t's X at t*cols, its Y at t*rows), W a GMAT
   in Q8_0, Q4_K or Q6_K."
  (let ((wb (cons (gmodel-file gm) (gmat-offset w))) (gpu (gmodel-gpu gm))
        (rows (gmat-rows w)) (cols (gmat-cols w)) (rbytes (gmat-row-bytes w)))
    (ecase (gmat-type w)
      (:q8_0
       (if (= n 1)
           (dispatch gpu e "d4_mv_q8_0" (list cols rows rbytes rows 0) (list wb x y) (ceiling rows 8) 64)
           (dispatch gpu e "d4_mm_q8_0" (list cols rows n rbytes cols rows) (list wb x y)
                     (list (ceiling n 64) (ceiling rows 64) 1) 128 :shared 16384)))
      (:q4_k
       (if (= n 1)
           (dispatch gpu e "dn_mv_q4_k" (list cols rows) (list wb x y) (ceiling rows 4) 64)
           (dispatch gpu e "d4_mm_q4_k" (list cols rows n rbytes cols rows) (list wb x y)
                     (list (ceiling n 64) (ceiling rows 64) 1) 128 :shared 16384)))
      (:q6_k
       (if (= n 1)
           (dispatch gpu e "mv_q6k" (list cols rows rbytes 0 0) (list wb x y) (ceiling rows 4) 64)
           (dispatch gpu e "mm_q6k_64" (list cols rows n rbytes) (list wb x y)
                     (list (ceiling n 64) (ceiling rows 64) 1) 128 :shared 16384))))))

(defun dense-forward-n (ctx tokens logits)
  "TOKENS (at most *GPU-CHUNK*) at the next positions through a dense model,
   as one command buffer; the logits, if LOGITS, are the last token's.  The
   state's X holds each token's row after the last layer (before the final
   norm) when it returns."
  (let* ((m (context-model ctx)) (gm (model-gpu m)) (gpu (gmodel-gpu gm))
         (st (dense-state ctx)) (pos0 (context-pos ctx)) (n (length tokens)) (eps (model-eps m))
         (d (model-d m)) (f (model-ffn m))
         (hd (model-head-dim m)) (nh (model-n-head m)) (nkv (model-n-kv-head m)) (qd (* nh hd)) (kvd (* nkv hd))
         (rd (model-rope-dims m))
         (x (dctx-x st)) (h (dctx-h st)))
    (assert (<= 1 n *gpu-chunk*))
    (when (> (+ pos0 n) (context-n-ctx ctx))
      (error "context is full (~D tokens)" (context-n-ctx ctx)))
    (unless (dctx-capture st)   ; (every layer's two outputs: the steering's, or its capture's)
      (setf (dctx-capture st) (gpu-buffer gpu (* 4 2 (model-n-layer m) d))))
    (with-gpu-work (e gpu)
      (let ((emb (model-token-embd m)))
        (loop for tok in tokens for tk from 0
              do (dispatch gpu e (ecase (gmat-type emb) (:q8_0 "dn_row_q8") (:q4_k "dn_row_q4_k") (:q6_k "row_q6k"))
                           (list (gmat-cols emb) (gmat-row-bytes emb) tok)
                           (list (cons (gmodel-file gm) (gmat-offset emb)) (cons x (* 4 tk d)))
                           (ceiling d 256) 256)))
      (when *gpu-after-embed* (funcall *gpu-after-embed* e x pos0 n))
      (loop for li below (model-n-layer m)
            for l = (svref (model-layers m) li)
            do (dispatch gpu e "rmsnorm" (list d eps 0) (list x (gm-layer gm li "attn_norm.weight") h) n 1024)
               (mat-q8 gm e (layer-wq l) h (dctx-q st) n)
               (mat-q8 gm e (layer-wk l) h (dctx-k st) n)
               (mat-q8 gm e (layer-wv l) h (dctx-v st) n)
               (loop for (b buf cols) in (list (list (layer-bq l) (dctx-q st) qd) (list (layer-bk l) (dctx-k st) kvd)
                                               (list (layer-bv l) (dctx-v st) kvd))
                     for name in '("attn_q.bias" "attn_k.bias" "attn_v.bias")
                     when b do (dispatch gpu e "dn_bias" (list (* n cols) cols) (list buf (gm-layer gm li name))
                                         (ceiling (* n cols) 256) 256))
               (dispatch gpu e "dn_qk_rope"
                         (list nh nkv hd rd pos0 eps (if (model-rope-neox m) 1 0)
                               (if (layer-q-norm l) 1 0) (if (layer-k-norm l) 1 0))
                         (list (dctx-q st) (dctx-k st) (dctx-v st)
                               (if (layer-q-norm l) (gm-layer gm li "attn_q_norm.weight") (dctx-q st))
                               (if (layer-k-norm l) (gm-layer gm li "attn_k_norm.weight") (dctx-k st))
                               (dctx-rope st) (dctx-qh st) (svref (dctx-kc st) li) (svref (dctx-vc st) li))
                         (list (+ nh nkv) n) (max 32 (floor hd 2)))
               (dispatch gpu e (if (= hd 64) "dn_attend_h64" "dn_attend_h128")
                         (list n pos0 nh nkv (/ 1f0 (sqrt (coerce hd 'single-float))))
                         (list (dctx-qh st) (svref (dctx-kc st) li) (svref (dctx-vc st) li) (dctx-att st))
                         (list nh (ceiling n 64)) 256 :shared (+ (* 2 32 hd 2) 16384))
               (mat-q8 gm e (layer-wo l) (dctx-att st) h n)
               (when (steer-wants-p li :attn) (gpu-steer gpu e st h li :attn n d))
               (dispatch gpu e "add" (list (* n d)) (list x h) (ceiling (* n d) 256) 256)
               (dispatch gpu e "rmsnorm" (list d eps 0) (list x (gm-layer gm li "ffn_norm.weight") h) n 1024)
               (mat-q8 gm e (layer-wgate l) h (dctx-fa st) n)
               (mat-q8 gm e (layer-wup l) h (dctx-fb st) n)
               (dispatch gpu e "swiglu" (list (* n f)) (list (dctx-fa st) (dctx-fb st)) (ceiling (* n f) 256) 256)
               (mat-q8 gm e (layer-wdown l) (dctx-fa st) h n)
               (when (steer-wants-p li :ffn) (gpu-steer gpu e st h li :ffn n d))
               (dispatch gpu e "add" (list (* n d)) (list x h) (ceiling (* n d) 256) 256)
               (when *gpu-after-layer* (funcall *gpu-after-layer* e li x pos0 n)))
      (when logits
        (dispatch gpu e "rmsnorm" (list d eps 0)
                  (list (cons x (* 4 (1- n) d)) (gw gm "output_norm.weight") h) 1 1024)
        (mat-q8 gm e (model-output m) h (dctx-logits st) 1)))
    (incf (context-pos ctx) n)
    (when *steer-capture*                 ; the last token's outputs, in the CPU's order
      (dotimes (li (model-n-layer m))
        (dolist (kind '(:attn :ffn))
          (funcall *steer-capture* li kind
                   (gbuf-floats (dctx-capture st) d (* d (+ (* 2 li) (if (eq kind :attn) 0 1))))))))
    (when logits
      (let ((out (context-logits ctx)) (sap (gbuf-sap (dctx-logits st))))
        (declare (type f32vec out))
        (dotimes (i (length out)) (setf (aref out i) (sb-sys:sap-ref-single sap (* 4 i))))))
    ctx))
