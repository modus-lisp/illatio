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
    ;; the kernels' fixed shapes (src/metal/qwen35.metal)
    (assert (= (model-ssm-k-dim m) 128) () "q35_delta is written for 128-wide K heads")
    (assert (zerop (mod (model-ssm-v-dim m) 32)) () "q35_delta takes values 32 at a time")
    (assert (= (model-head-dim m) 256) () "attend_fa is written for 256-wide heads")
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

(defparameter *gpu-chunk* 512 "Tokens per GPU pass in a prefill.")

(defun gpu-state (ctx)
  (or (getf (context-plist ctx) :gpu)
      (setf (getf (context-plist ctx) :gpu)
            (let* ((m (context-model ctx)) (gpu (gmodel-gpu (model-gpu m)))
                   (n *gpu-chunk*)
                   (d (model-d m)) (f (model-ffn m)) (ch (q35-channels m))
                   (vh (model-ssm-v-heads m)) (od (* vh (model-ssm-v-dim m)))
                   (qd (* (model-n-head m) (model-head-dim m)))
                   (kvd (* (model-n-kv-head m) (model-head-dim m)))
                   (n-ctx (context-n-ctx ctx)))
              (flet ((fb (k) (gpu-zero (gpu-buffer gpu (* 4 k))))
                     (per-layer (kind k)
                       (map 'simple-vector
                            (lambda (l) (and (eq (layer-kind l) kind) (gpu-zero (gpu-buffer gpu (* 4 k)))))
                            (model-layers m))))
                (%make-gctx :x (fb (* n d)) :h (fb (* n d)) :qkv (fb (* n ch)) :z (fb (* n od))
                            :alpha (fb (* n vh)) :beta (fb (* n vh))
                            :c (fb (* n ch)) :o (fb (* n od)) :qg (fb (* n 2 qd)) :q (fb (* n qd))
                            :gate (fb (* n qd)) :k (fb (* n kvd)) :v (fb (* n kvd)) :att (fb (* (+ n 8) qd))
                            :fa (fb (* n f)) :fb (fb (* n f))
                            :logits (fb (model-n-vocab m)) :rope (fb (* n (model-rope-dims m)))
                            ;; 32 positions of zeros past the end: ATTEND_FA reads keys 32 at a time
                            :kc (per-layer :full (* (+ n-ctx 32) kvd)) :vc (per-layer :full (* (+ n-ctx 32) kvd))
                            :conv (per-layer :linear (* ch (1- (model-ssm-conv m))))
                            :ssm (per-layer :linear (* vh (model-ssm-k-dim m) (model-ssm-v-dim m)))))))))

(defun reset-gpu-state (ctx)
  (let ((st (getf (context-plist ctx) :gpu)))
    (when st
      (loop for v across (gctx-conv st) when v do (gpu-zero v))
      (loop for v across (gctx-ssm st) when v do (gpu-zero v)))))

;;; --- a pass: one token, or a chunk of a prefill --------------------------------------

(defun mat (gm e w x y n)
  "Y = W X for N tokens (token t's X at t*cols, its Y at t*rows): a matrix-
   vector product for one, the tiled matrix product for more.  W a GMAT,
   Q6_K so far."
  (unless (eq (gmat-type w) :q6_k)
    (error "the GPU multiplies Q6_K so far; ~A is ~A" (gmat-name w) (gmat-type w)))
  (let ((wb (cons (gmodel-file gm) (gmat-offset w))) (gpu (gmodel-gpu gm)))
    (if (= n 1)
        (dispatch gpu e "mv_q6k" (list (gmat-cols w) (gmat-rows w) (gmat-row-bytes w) 0 0)
                  (list wb x y) (list (ceiling (gmat-rows w) 4) 1 1) 64)
        (if *mm-64*
            (dispatch gpu e "mm_q6k_64" (list (gmat-cols w) (gmat-rows w) n (gmat-row-bytes w))
                      (list wb x y) (list (ceiling n 64) (ceiling (gmat-rows w) 64) 1) 128
                      :shared 16384)
            (dispatch gpu e "mm_q6k" (list (gmat-cols w) (gmat-rows w) n (gmat-row-bytes w))
                      (list wb x y) (list (ceiling n 32) (ceiling (gmat-rows w) 64) 1) 128
                      :shared 8192)))))

(defvar *flash* t "Whether a prefill's attention is ATTEND_FA (flash style).")

(defvar *mm-64* t "Whether a prefill's products take 64-token tiles (MM_Q6K_64).")

(defun gpu-forward (ctx token logits)
  "FORWARD on the GPU: TOKEN at the next position."
  (gpu-forward-n ctx (list token) logits))

(defun gpu-eval (ctx tokens)
  "TOKENS through the GPU, *GPU-CHUNK* at a time; the last one's logits."
  (loop for rest = tokens then (nthcdr *gpu-chunk* rest)
        while rest
        do (let ((chunk (subseq rest 0 (min *gpu-chunk* (length rest)))))
             (gpu-forward-n ctx chunk (<= (length rest) *gpu-chunk*))))
  (context-logits ctx))

(defun gpu-forward-n (ctx tokens logits)
  "TOKENS (at most *GPU-CHUNK*) at the next positions, as one command
   buffer; the logits, if LOGITS, are the last token's."
  (let* ((m (context-model ctx)) (gm (model-gpu m)) (gpu (gmodel-gpu gm))
         (st (gpu-state ctx)) (pos0 (context-pos ctx)) (n (length tokens)) (eps (model-eps m))
         (d (model-d m)) (f (model-ffn m)) (ch (q35-channels m))
         (hd (model-head-dim m)) (nh (model-n-head m)) (nkv (model-n-kv-head m)) (qd (* nh hd))
         (kh (model-ssm-k-heads m)) (kd (model-ssm-k-dim m))
         (vh (model-ssm-v-heads m)) (vd (model-ssm-v-dim m))
         (rd (model-rope-dims m))
         (x (gctx-x st)) (h (gctx-h st)))
    (assert (<= 1 n *gpu-chunk*))
    (when (> (+ pos0 n) (context-n-ctx ctx))
      (error "context is full (~D tokens)" (context-n-ctx ctx)))
    ;; each token's RoPE table, as the CPU computes it
    (dotimes (tk n)
      (multiple-value-bind (cos sin) (rope-table rd (+ pos0 tk) (model-rope-base m))
        (gbuf-set-floats (gctx-rope st) cos (* tk rd))
        (gbuf-set-floats (gctx-rope st) sin (+ (* tk rd) (length cos)))))
    (with-gpu-work (e gpu)
      (let ((emb (model-token-embd m)))
        (loop for tok in tokens for tk from 0
              do (dispatch gpu e "row_q6k" (list (gmat-cols emb) (gmat-row-bytes emb) tok)
                           (list (cons (gmodel-file gm) (gmat-offset emb)) (cons x (* 4 tk d)))
                           (ceiling d 256) 256)))
      (loop for li below (model-n-layer m)
            for l = (svref (model-layers m) li)
            do (dispatch gpu e "rmsnorm" (list d eps 0) (list x (gm-layer gm li "attn_norm.weight") h) n 1024)
               (if (eq (layer-kind l) :linear)
                   (let ((qkv (gctx-qkv st)) (z (gctx-z st)) (alpha (gctx-alpha st))
                         (beta (gctx-beta st)) (c (gctx-c st)) (o (gctx-o st)))
                     (mat gm e (layer-wqkv l) h qkv n)
                     (mat gm e (layer-wz l) h z n)
                     (mat gm e (layer-walpha l) h alpha n)
                     (mat gm e (layer-wbeta l) h beta n)
                     (dispatch gpu e "q35_gates" (list vh n)
                               (list alpha beta (gm-layer gm li "ssm_a") (gm-layer gm li "ssm_dt.bias"))
                               (ceiling (* n vh) 256) 256)
                     (dispatch gpu e "q35_conv" (list ch n)
                               (list qkv (gm-layer gm li "ssm_conv1d.weight") (svref (gctx-conv st) li) c)
                               (ceiling ch 256) 256)
                     (dispatch gpu e "l2norm" (list kd eps (* 2 kh) ch) (list c) (* n 2 kh) 128)
                     (dispatch gpu e "q35_delta" (list kh kd vh vd (/ 1f0 (sqrt (coerce kd 'single-float))) n ch)
                               (list c beta alpha (svref (gctx-ssm st) li) o)
                               (list vh (floor vd 32)) 32)
                     (dispatch gpu e "q35_outnorm" (list vd eps)
                               (list o (gm-layer gm li "ssm_norm.weight") z) (* n vh) 128)
                     (mat gm e (layer-ssm-out l) o h n))
                   (progn
                     (mat gm e (layer-wq l) h (gctx-qg st) n)
                     (mat gm e (layer-wk l) h (gctx-k st) n)
                     (mat gm e (layer-wv l) h (gctx-v st) n)
                     (dispatch gpu e "q35_qk" (list nh nkv hd rd eps pos0)
                               (list (gctx-qg st) (gctx-k st) (gctx-v st)
                                     (gm-layer gm li "attn_q_norm.weight") (gm-layer gm li "attn_k_norm.weight")
                                     (gctx-rope st) (gctx-q st) (gctx-gate st)
                                     (svref (gctx-kc st) li) (svref (gctx-vc st) li))
                               (list (+ nh nkv) n) hd)
                     (if (and *flash* (> n 1))
                         (dispatch gpu e "attend_fa" (list nh nkv n (/ 1f0 (sqrt (coerce hd 'single-float))) pos0)
                                   (list (gctx-q st) (svref (gctx-kc st) li) (svref (gctx-vc st) li) (gctx-att st))
                                   (list nh (ceiling n 8)) 32 :shared (* 4 (+ (* 8 256) (* 8 32) 64)))
                         (dispatch gpu e "attend" (list nh nkv hd (1+ pos0) (/ 1f0 (sqrt (coerce hd 'single-float))))
                                   (list (gctx-q st) (svref (gctx-kc st) li) (svref (gctx-vc st) li) (gctx-att st))
                                   (list nh n) 256 :shared (* 4 (max 4 (+ pos0 n)))))
                     (dispatch gpu e "gate_mul" (list (* n qd)) (list (gctx-att st) (gctx-gate st))
                               (ceiling (* n qd) 256) 256)
                     (mat gm e (layer-wo l) (gctx-att st) h n)))
               (dispatch gpu e "add" (list (* n d)) (list x h) (ceiling (* n d) 256) 256)
               (dispatch gpu e "rmsnorm" (list d eps 0)
                         (list x (gm-layer gm li "post_attention_norm.weight") h) n 1024)
               (mat gm e (layer-wgate l) h (gctx-fa st) n)
               (mat gm e (layer-wup l) h (gctx-fb st) n)
               (dispatch gpu e "swiglu" (list (* n f)) (list (gctx-fa st) (gctx-fb st)) (ceiling (* n f) 256) 256)
               (mat gm e (layer-wdown l) (gctx-fa st) h n)
               (dispatch gpu e "add" (list (* n d)) (list x h) (ceiling (* n d) 256) 256))
      (when logits
        (dispatch gpu e "rmsnorm" (list d eps 0)
                  (list (cons x (* 4 (1- n) d)) (gw gm "output_norm.weight") h) 1 1024)
        (mat gm e (model-output m) h (gctx-logits st) 1)))
    (incf (context-pos ctx) n)
    (when logits
      (let ((out (context-logits ctx)) (sap (gbuf-sap (gctx-logits st))))
        (declare (type f32vec out))
        (dotimes (i (length out)) (setf (aref out i) (sb-sys:sap-ref-single sap (* 4 i))))))
    ctx))
