;;;; metal-qwen35.lisp — a qwen35 model on the GPU (src/metal/qwen35.metal).
;;;
;;; The GGUF is mapped once and handed to Metal as one buffer, so the GPU
;;; reads the weights where they lie in the file, in its own block format; no
;;; repacked copy is made, which on a 48 GB machine is what lets a 22 GB model
;;; fit.  A token is one command buffer: every layer's ops encoded in order
;;; (a serial encoder runs each dispatch after the last), then one wait.  The
;;; embedding row is fetched and every matrix multiplied on the GPU; the CPU
;;; only fills the RoPE table and reads the logits back.

(in-package #:illatio)

(defstruct (gmodel (:constructor %make-gmodel))
  gpu
  file                                  ; the GGUF, mapped: a GBUF over it
  (offsets (make-hash-table :test 'equal)))

(defun mmap-file (path)
  "PATH mapped read-only, and the mapping's length (whole 16 KB pages)."
  (let* ((size (with-open-file (s path :element-type '(unsigned-byte 8)) (file-length s)))
         (bytes (* 16384 (ceiling size 16384)))
         (fd (sb-unix:unix-open (namestring (truename path)) sb-unix:o_rdonly 0)))
    (unless fd (error "cannot open ~A" path))
    (unwind-protect
         (let ((addr (sb-alien:alien-funcall
                      (sb-alien:extern-alien "mmap" (function sb-alien:unsigned-long sb-alien:unsigned-long
                                                              sb-alien:unsigned-long sb-alien:int sb-alien:int
                                                              sb-alien:int sb-alien:long))
                      0 bytes 1 1 fd 0)))   ; PROT_READ, MAP_SHARED
           (when (= addr (ldb (byte 64 0) -1)) (error "mmap of ~A failed" path))
           (values (sb-sys:int-sap addr) bytes))
      (sb-unix:unix-close fd))))

(defun load-model-gpu (path)
  "The model at PATH run by the GPU: its metadata, tokenizer and small
   vectors read as usual, its matrices left in the file (GMATs)."
  (let ((m (let ((*matrices-in-file* t)) (load-model path :threads 1))))
    (unless (equal (model-arch m) "qwen35")
      (error "the GPU path runs qwen35 models; this is ~A" (model-arch m)))
    (let ((gpu (open-gpu)) (g (model-gguf m)))
      (multiple-value-bind (sap bytes) (mmap-file path)
        (let ((gm (%make-gmodel :gpu gpu :file (gpu-wrap gpu sap bytes))))
          (maphash (lambda (name ti)
                     (setf (gethash name (gmodel-offsets gm))
                           (+ (gguf-data-offset g) (tensor-info-offset ti))))
                   (gguf-tensors g))
          (setf (model-gpu m) gm)
          m)))))

(defun gw (gm name)
  "Tensor NAME, as (file buffer . offset), for DISPATCH."
  (cons (gmodel-file gm) (or (gethash name (gmodel-offsets gm)) (error "no tensor ~A" name))))

(defun gm-layer (gm i name) (gw gm (format nil "blk.~D.~A" i name)))

;;; --- per-context state on the GPU ------------------------------------------------

(defstruct (gctx (:constructor %make-gctx))
  x h qkv z alpha beta c o qg q gate k v att fa fb logits rope
  kc vc conv ssm)                       ; per layer (NIL where the kind has none)

(defun gpu-zero (b)
  (sb-alien:alien-funcall
   (sb-alien:extern-alien "memset" (function sb-sys:system-area-pointer sb-sys:system-area-pointer
                                             sb-alien:int sb-alien:unsigned-long))
   (gbuf-sap b) 0 (gbuf-bytes b))
  b)

(defun gpu-state (ctx)
  (or (getf (context-plist ctx) :gpu)
      (setf (getf (context-plist ctx) :gpu)
            (let* ((m (context-model ctx)) (gpu (gmodel-gpu (model-gpu m)))
                   (d (model-d m)) (f (model-ffn m)) (ch (q35-channels m))
                   (vh (model-ssm-v-heads m)) (od (* vh (model-ssm-v-dim m)))
                   (qd (* (model-n-head m) (model-head-dim m)))
                   (kvd (* (model-n-kv-head m) (model-head-dim m)))
                   (n-ctx (context-n-ctx ctx)))
              (flet ((fb (n) (gpu-zero (gpu-buffer gpu (* 4 n))))
                     (per-layer (kind n)
                       (map 'simple-vector
                            (lambda (l) (and (eq (layer-kind l) kind) (gpu-zero (gpu-buffer gpu (* 4 n)))))
                            (model-layers m))))
                (%make-gctx :x (fb d) :h (fb d) :qkv (fb ch) :z (fb od) :alpha (fb vh) :beta (fb vh)
                            :c (fb ch) :o (fb od) :qg (fb (* 2 qd)) :q (fb qd) :gate (fb qd)
                            :k (fb kvd) :v (fb kvd) :att (fb qd) :fa (fb f) :fb (fb f)
                            :logits (fb (model-n-vocab m)) :rope (fb (model-rope-dims m))
                            :kc (per-layer :full (* n-ctx kvd)) :vc (per-layer :full (* n-ctx kvd))
                            :conv (per-layer :linear (* ch (1- (model-ssm-conv m))))
                            :ssm (per-layer :linear (* vh (model-ssm-k-dim m) (model-ssm-v-dim m)))))))))

(defun reset-gpu-state (ctx)
  (let ((st (getf (context-plist ctx) :gpu)))
    (when st
      (loop for v across (gctx-conv st) when v do (gpu-zero v))
      (loop for v across (gctx-ssm st) when v do (gpu-zero v)))))

;;; --- one token -------------------------------------------------------------------

(defun mv (gm e w x y)
  "Y = W X, W a GMAT (Q6_K so far)."
  (unless (eq (gmat-type w) :q6_k)
    (error "the GPU multiplies Q6_K so far; ~A is ~A" (gmat-name w) (gmat-type w)))
  (dispatch (gmodel-gpu gm) e "mv_q6k"
            (list (gmat-cols w) (gmat-rows w) (gmat-row-bytes w) 0 0)
            (list (cons (gmodel-file gm) (gmat-offset w)) x y)
            (list (ceiling (gmat-rows w) 4) 1 1) 64))

(defun gpu-forward (ctx token logits)
  "FORWARD on the GPU: TOKEN at the next position."
  (let* ((m (context-model ctx)) (gm (model-gpu m)) (gpu (gmodel-gpu gm))
         (st (gpu-state ctx)) (pos (context-pos ctx)) (eps (model-eps m))
         (d (model-d m)) (f (model-ffn m)) (ch (q35-channels m))
         (hd (model-head-dim m)) (nh (model-n-head m)) (nkv (model-n-kv-head m))
         (kh (model-ssm-k-heads m)) (kd (model-ssm-k-dim m))
         (vh (model-ssm-v-heads m)) (vd (model-ssm-v-dim m))
         (x (gctx-x st)) (h (gctx-h st)))
    (when (>= pos (context-n-ctx ctx))
      (error "context is full (~D tokens)" (context-n-ctx ctx)))
    ;; the RoPE table for this position, as the CPU computes it
    (multiple-value-bind (cos sin) (rope-table (model-rope-dims m) pos (model-rope-base m))
      (gbuf-set-floats (gctx-rope st) cos)
      (gbuf-set-floats (gctx-rope st) sin (length cos)))
    (with-gpu-work (e gpu)
      (let ((emb (model-token-embd m)))
        (dispatch gpu e "row_q6k" (list (gmat-cols emb) (gmat-row-bytes emb) token)
                  (list (cons (gmodel-file gm) (gmat-offset emb)) x)
                  (ceiling d 256) 256))
      (loop for li below (model-n-layer m)
            for l = (svref (model-layers m) li)
            do (dispatch gpu e "rmsnorm" (list d eps 0) (list x (gm-layer gm li "attn_norm.weight") h) 1 1024)
               (if (eq (layer-kind l) :linear)
                   (let ((qkv (gctx-qkv st)) (z (gctx-z st)) (alpha (gctx-alpha st))
                         (beta (gctx-beta st)) (c (gctx-c st)) (o (gctx-o st)))
                     (mv gm e (layer-wqkv l) h qkv)
                     (mv gm e (layer-wz l) h z)
                     (mv gm e (layer-walpha l) h alpha)
                     (mv gm e (layer-wbeta l) h beta)
                     (dispatch gpu e "q35_gates" (list vh)
                               (list alpha beta (gm-layer gm li "ssm_a") (gm-layer gm li "ssm_dt.bias"))
                               1 (max 32 vh))
                     (dispatch gpu e "q35_conv" (list ch)
                               (list qkv (gm-layer gm li "ssm_conv1d.weight") (svref (gctx-conv st) li) c)
                               (ceiling ch 256) 256)
                     (dispatch gpu e "l2norm" (list kd eps) (list c) (* 2 kh) 128)
                     (dispatch gpu e "q35_delta" (list kh kd vh vd (/ 1f0 (sqrt (coerce kd 'single-float))))
                               (list c beta alpha (svref (gctx-ssm st) li) o)
                               vh vd)
                     (dispatch gpu e "q35_outnorm" (list vd eps)
                               (list o (gm-layer gm li "ssm_norm.weight") z) vh 128)
                     (mv gm e (layer-ssm-out l) o h))
                   (progn
                     (mv gm e (layer-wq l) h (gctx-qg st))
                     (mv gm e (layer-wk l) h (gctx-k st))
                     (mv gm e (layer-wv l) h (gctx-v st))
                     (dispatch gpu e "q35_qk" (list nh nkv hd (model-rope-dims m) eps pos)
                               (list (gctx-qg st) (gctx-k st) (gctx-v st)
                                     (gm-layer gm li "attn_q_norm.weight") (gm-layer gm li "attn_k_norm.weight")
                                     (gctx-rope st) (gctx-q st) (gctx-gate st)
                                     (svref (gctx-kc st) li) (svref (gctx-vc st) li))
                               (+ nh nkv) hd)
                     (dispatch gpu e "attend" (list nh nkv hd (1+ pos) (/ 1f0 (sqrt (coerce hd 'single-float))))
                               (list (gctx-q st) (svref (gctx-kc st) li) (svref (gctx-vc st) li) (gctx-att st))
                               nh 256 :shared (* 4 (max 4 (1+ pos))))
                     (dispatch gpu e "gate_mul" (list (* nh hd)) (list (gctx-att st) (gctx-gate st))
                               (ceiling (* nh hd) 256) 256)
                     (mv gm e (layer-wo l) (gctx-att st) h)))
               (dispatch gpu e "add" (list d) (list x h) (ceiling d 256) 256)
               (dispatch gpu e "rmsnorm" (list d eps 0)
                         (list x (gm-layer gm li "post_attention_norm.weight") h) 1 1024)
               (mv gm e (layer-wgate l) h (gctx-fa st))
               (mv gm e (layer-wup l) h (gctx-fb st))
               (dispatch gpu e "swiglu" (list f) (list (gctx-fa st) (gctx-fb st)) (ceiling f 256) 256)
               (mv gm e (layer-wdown l) (gctx-fa st) h)
               (dispatch gpu e "add" (list d) (list x h) (ceiling d 256) 256))
      (when logits
        (dispatch gpu e "rmsnorm" (list d eps 0) (list x (gw gm "output_norm.weight") h) 1 1024)
        (mv gm e (model-output m) h (gctx-logits st))))
    (incf (context-pos ctx))
    (when logits
      (let ((out (context-logits ctx)) (sap (gbuf-sap (gctx-logits st))))
        (declare (type f32vec out))
        (dotimes (i (length out)) (setf (aref out i) (sb-sys:sap-ref-single sap (* 4 i))))))
    ctx))
