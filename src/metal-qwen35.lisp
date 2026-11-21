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

(defparameter +dense-archs+ '("qwen2" "qwen3" "qwen3vl" "llama")
  "The dense architectures the GPU runs (metal-dense.lisp).")

(defun load-model-gpu (path &key gpu)
  "The model at PATH run by the GPU: its metadata, tokenizer and small
   vectors read as usual, its matrices left in the file (GMATs).  A qwen35
   model, or a dense one (metal-dense.lisp).  GPU: an opened GPU whose
   library holds these kernels (and a caller's), or NIL for a new one."
  (when (inkling2-path-p path)
    (return-from load-model-gpu (load-inkling2-model path (default-thread-count) :gpu t)))
  (let ((m (let ((*matrices-in-file* t)) (load-model path :threads 1))))
    (cond ((member (model-arch m) '("qwen35" "qwen35moe") :test #'equal)
           ;; the kernels' fixed shapes (src/metal/qwen35.metal)
           (assert (= (model-ssm-k-dim m) 128) () "q35_delta is written for 128-wide K heads")
           (assert (zerop (mod (model-ssm-v-dim m) 32)) () "q35_delta takes values 32 at a time")
           (assert (= (model-head-dim m) 256) () "attend_fa is written for 256-wide heads")
           (assert (<= (floor (model-n-head m) (model-n-kv-head m)) 8) () "attend_split takes at most 8 query heads a KV head")
           ;; checked here, not when a command buffer is half encoded: an
           ;; error there leaves an encoder open, and Metal aborts the process
           (unless (member (gmat-type (model-token-embd m)) '(:q6_k :q8_0))
             (error "the GPU takes a qwen35 model's embeddings in Q6_K or Q8_0; they are ~A" (gmat-type (model-token-embd m))))
           ;; qwen35moe: the routed experts' formats have expert kernels (qwen35moe.metal)
           (loop for l across (model-layers m)
                 when (layer-router l)
                   do (loop for (w ok) in (list (list (layer-gexps l) '(:q4_k :q5_k :q6_k)) (list (layer-uexps l) '(:q4_k :q5_k :q6_k))
                                                (list (layer-dexps l) '(:q4_k :q5_k :q6_k)) (list (layer-router l) '(:f32)))
                            unless (member (gmat-type w) ok)
                              do (error "the GPU runs qwen35moe's ~A in ~{~A~^, ~}; it is ~A" (gmat-name w) ok (gmat-type w))))
           (loop for l across (model-layers m)
                 do (dolist (w (list (layer-wqkv l) (layer-wz l) (layer-walpha l) (layer-wbeta l) (layer-ssm-out l)
                                     (layer-wq l) (layer-wk l) (layer-wv l) (layer-wo l)
                                     (layer-wgate l) (layer-wup l) (layer-wdown l)))
                      (when (and w (not (member (gmat-type w) '(:q8_0 :q4_0 :q4_1 :q4_k :q5_k :q6_k))))
                        ;; F32 only where the matrix is small: ALPHA and BETA
                        (unless (and (eq (gmat-type w) :f32) (<= (gmat-rows w) 256))
                          (error "the GPU runs a qwen35 model's Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K matrices; ~A is ~A"
                                 (gmat-name w) (gmat-type w)))))))
          ((member (model-arch m) +dense-archs+ :test #'equal) (dense-gpu-check m))
          (t (error "the GPU path runs qwen35 and dense models (~{~A~^, ~}); this is ~A"
                    +dense-archs+ (model-arch m))))
    (let ((gpu (or gpu (open-gpu))) (g (model-gguf m)))
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

(defvar *q35-debug-layers* nil "Debugging: run only this many layers.")

(defstruct (gctx (:constructor %make-gctx))
  x h qkv z alpha beta c o qg q gate k v att fa fb logits rope
  part ml                               ; decode's attention, split: chunks' outputs, their maxima and sums
  capture                               ; steering's capture: a row of D a layer and kind
  kc vc conv ssm                        ; per layer (NIL where the kind has none)
  rl ids wts eg eu ye sh shg shu sg counts bases pos tok xg)    ; qwen35moe's FFN: router logits, the chosen experts, their outputs

(defun gpu-zero (b)
  (sb-alien:alien-funcall
   (sb-alien:extern-alien "memset" (function sb-sys:system-area-pointer sb-sys:system-area-pointer
                                             sb-alien:int sb-alien:unsigned-long))
   (gbuf-sap b) 0 (gbuf-bytes b))
  b)

(defparameter *gpu-chunk* 512 "Tokens per GPU pass in a prefill.")

(defparameter *split-chunk* 64
  "Positions a threadgroup of decode's attention takes (ATTEND_SPLIT): the
   keys cut so the whole GPU reads them, not a threadgroup a head.")


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
                            :part (fb (* (model-n-head m) (ceiling n-ctx *split-chunk*) (model-head-dim m)))
                            :ml (fb (* (model-n-head m) (ceiling n-ctx *split-chunk*) 2))
                            :capture (fb (* 2 (model-n-layer m) d))
                            :rl (fb (* n (max 1 (model-n-expert m)))) :ids (fb (* n 8)) :wts (fb (* n 8))
                            :eg (fb (* n 8 f)) :eu (fb (* n 8 f)) :ye (fb (* n 8 d))
                            :sh (fb (* n d)) :shg (fb (* n f)) :shu (fb (* n f)) :sg (fb n)
                            :counts (fb (max 1 (model-n-expert m))) :bases (fb (max 1 (model-n-expert m)))
                            :pos (fb (* n 8)) :tok (fb (* n 8)) :xg (fb (* n 8 d))
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
   Q6_K by this file's kernels; Q8_0, Q4_K and Q5_K by the dense models'
   (MAT-Q8), as a 4-bit GGUF mixes them."
  (unless (eq (gmat-type w) :q6_k)
    (return-from mat (mat-q8 gm e w x y n)))
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

(defvar *split-decode* t
  "Whether one token's attention is split over the keys (ATTEND_SPLIT and
   ATTEND_MERGE) rather than a threadgroup a head (ATTEND).")

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
  (unless (model-recurrent-p (context-model ctx))
    (return-from gpu-forward-n (dense-forward-n ctx tokens logits)))
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
              do (dispatch gpu e (ecase (gmat-type emb) (:q6_k "row_q6k") (:q8_0 "dn_row_q8")) (list (gmat-cols emb) (gmat-row-bytes emb) tok)
                           (list (cons (gmodel-file gm) (gmat-offset emb)) (cons x (* 4 tk d)))
                           (ceiling d 256) 256)))
      (loop for li below (or *q35-debug-layers* (model-n-layer m))
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
                     ;; a prefill's attention flash style; one token's split over the
                     ;; keys (a threadgroup a KV head and chunk), then merged
                     (cond
                       ((> n 1)
                         (dispatch gpu e "attend_fa" (list nh nkv n (/ 1f0 (sqrt (coerce hd 'single-float))) pos0)
                                   (list (gctx-q st) (svref (gctx-kc st) li) (svref (gctx-vc st) li) (gctx-att st))
                                   (list nh (ceiling n 8)) 32 :shared (* 4 (+ (* 8 256) (* 8 32) 64))))
                       ((not *split-decode*)
                        (dispatch gpu e "attend" (list nh nkv hd (1+ pos0) (/ 1f0 (sqrt (coerce hd 'single-float))))
                                  (list (gctx-q st) (svref (gctx-kc st) li) (svref (gctx-vc st) li) (gctx-att st))
                                  (list nh n) 256 :shared (* 4 (max 4 (+ pos0 n)))))
                       (t
                         (let* ((len (1+ pos0)) (nsplit (ceiling len *split-chunk*)) (grp (floor nh nkv)))
                           (dispatch gpu e "attend_split"
                                     (list nkv grp len *split-chunk* nsplit (/ 1f0 (sqrt (coerce hd 'single-float))))
                                     (list (gctx-q st) (svref (gctx-kc st) li) (svref (gctx-vc st) li)
                                           (gctx-part st) (gctx-ml st))
                                     (list nkv nsplit) hd :shared (* 4 grp *split-chunk*))
                           (dispatch gpu e "attend_merge" (list nsplit) (list (gctx-part st) (gctx-ml st) (gctx-att st))
                                     nh hd))))
                     (dispatch gpu e "gate_mul" (list (* n qd)) (list (gctx-att st) (gctx-gate st))
                               (ceiling (* n qd) 256) 256)
                     (mat gm e (layer-wo l) (gctx-att st) h n)))
               ;; either kind's mixer output is the layer's attention output, to steering
               (when (steer-wants-p li :attn) (q35-steer gpu e st h li :attn n d))
               (dispatch gpu e "add" (list (* n d)) (list x h) (ceiling (* n d) 256) 256)
               (dispatch gpu e "rmsnorm" (list d eps 0)
                         (list x (gm-layer gm li "post_attention_norm.weight") h) n 1024)
               (if (layer-router l)
                   (q35-moe gm e st m l li h n)
                   (progn
                     (mat gm e (layer-wgate l) h (gctx-fa st) n)
                     (mat gm e (layer-wup l) h (gctx-fb st) n)
                     (dispatch gpu e "swiglu" (list (* n f)) (list (gctx-fa st) (gctx-fb st)) (ceiling (* n f) 256) 256)
                     (mat gm e (layer-wdown l) (gctx-fa st) h n)))
               (when (steer-wants-p li :ffn) (q35-steer gpu e st h li :ffn n d))
               (dispatch gpu e "add" (list (* n d)) (list x h) (ceiling (* n d) 256) 256))
      (when logits
        (dispatch gpu e "rmsnorm" (list d eps 0)
                  (list (cons x (* 4 (1- n) d)) (gw gm "output_norm.weight") h) 1 1024)
        (mat gm e (model-output m) h (gctx-logits st) 1)))
    (incf (context-pos ctx) n)
    (when *steer-capture*                 ; the last token's outputs, in the CPU's order
      (dotimes (li (model-n-layer m))
        (dolist (kind '(:attn :ffn))
          (funcall *steer-capture* li kind
                   (gbuf-floats (gctx-capture st) d (* d (+ (* 2 li) (if (eq kind :attn) 0 1))))))))
    (when logits
      (let ((out (context-logits ctx)) (sap (gbuf-sap (gctx-logits st))))
        (declare (type f32vec out))
        (dotimes (i (length out)) (setf (aref out i) (sb-sys:sap-ref-single sap (* 4 i))))))
    ctx))

(defun q35-steer (gpu e st y li kind n d)
  "Y's N rows of D steered for layer LI's KIND output (dense.metal's dn_steer),
   the last first copied to the state's capture when capturing."
  (let ((scale (steer-scale li kind)))
    (dispatch gpu e "dn_steer" (list d scale (1- n) (if *steer-capture* 1 0))
              (list y (if (zerop scale) y (cons (gpu-steer-dirs gpu) (* 4 li d)))
                    (cons (gctx-capture st) (* 4 d (+ (* 2 li) (if (eq kind :attn) 0 1)))))
              n 256)))

;;; --- qwen35moe's FFN (kernels: qwen35moe.metal) -------------------------------------
;;; llama.cpp's graph for it (src/models/qwen35moe.cpp): the router's softmax
;;; over 256 experts, the 8 largest renormalized; each a SwiGLU
;;; (silu(gate) up, then down); plus the shared expert's SwiGLU times
;;; sigmoid(its gate . x).  A token's 8 experts are 8 pairs t*8+k.

(defvar *q35-group-min* 16
  "Passes of at least this many tokens run the routed experts as matrix
   products grouped by expert (q35_group, q35_mm_id_*); shorter ones a
   pair at a time (q35_xmv_*).")

(defun q35-expert-offsets (gm w n-expert)
  "Expert z's offset in W's tensor, for each z: a GPU buffer of u64s (made once)."
  (let ((key (list :q35-offs (gmat-name w))))
    (or (gethash key (gmodel-offsets gm))
        (setf (gethash key (gmodel-offsets gm))
              (let* ((b (gpu-buffer (gmodel-gpu gm) (* 8 n-expert)))
                     (stride (* (floor (gmat-rows w) n-expert) (gmat-row-bytes w))))
                (dotimes (z n-expert b)
                  (setf (sb-sys:sap-ref-64 (gbuf-sap b) (* 8 z)) (* z stride))))))))

(defun q35-xmv (gm e st w x x-pair x-stride y npairs n-expert)
  "Y's pair rows = the chosen experts' W (one tensor, the experts' matrices
   in order) times X's row for each pair (its token's, or with X-PAIR its own)."
  (let* ((gpu (gmodel-gpu gm)) (wb (cons (gmodel-file gm) (gmat-offset w)))
         (rows (floor (gmat-rows w) n-expert)) (cols (gmat-cols w)) (xp (if x-pair 1 0))
         (rb (gmat-row-bytes w)))
    (multiple-value-bind (kernel args)
        (ecase (gmat-type w)
          (:q4_k (values "q35_xmv_q4_k" (list cols rows x-stride rows xp)))
          (:q5_k (values "q35_xmv_q5_k" (list cols rows x-stride rows xp)))
          (:q6_k (values "q35_xmv_q6k" (list cols rows rb x-stride rows xp))))
      (dispatch gpu e kernel args (list wb x y (gctx-ids st)) (list (ceiling rows 4) npairs) 64))))

(defun q35-mm-id (gm e st w x y n n-expert)
  "Y's grouped rows = each expert's W times its grouped rows of X (each
   expert at most N rows: a token picks an expert once)."
  (let* ((rows (floor (gmat-rows w) n-expert)) (cols (gmat-cols w)))
    (dispatch (gmodel-gpu gm) e
              (format nil (if *q35-tile-32* "q35_mm_id32_~A" "q35_mm_id_~A")
                      (ecase (gmat-type w) (:q4_k "q4_k") (:q5_k "q5_k") (:q6_k "q6_k")))
              (list cols rows (gmat-row-bytes w) cols rows 0)
              (list (cons (gmodel-file gm) (gmat-offset w)) (q35-expert-offsets gm w n-expert)
                    (gctx-counts st) (gctx-bases st) x y)
              (list (ceiling n (if *q35-tile-32* 32 64)) (ceiling rows 64) n-expert) 128
              :shared (if *q35-tile-32* 8192 16384))))

(defvar *q35-tile-32* t
  "Whether the grouped experts run in 32-pair tiles (else d4_mm_id's 64).")

(defun q35-moe (gm e st m l li h n)
  "H (N rows, normed) through layer L's routed and shared experts, into H."
  (let* ((gpu (gmodel-gpu gm)) (d (model-d m)) (f (model-ffn m)) (ne (model-n-expert m))
         (np (* 8 n)) (grouped (>= n *q35-group-min*)))
    ;; the router
    (dispatch gpu e "q35_mv_f32_n" (list d ne n) (list (cons (gmodel-file gm) (gmat-offset (layer-router l))) h (gctx-rl st))
              (list (ceiling ne 4) (ceiling n 8)) 128)
    (dispatch gpu e "q35_route" (list ne) (list (gctx-rl st) (gctx-ids st) (gctx-wts st)) n 32)
    (when grouped
      (dispatch gpu e "q35_group" (list np ne)
                (list (gctx-ids st) (gctx-counts st) (gctx-bases st) (gctx-pos st) (gctx-tok st)) 1 (* 32 (ceiling ne 32)))
      (dispatch gpu e "d4_gather_rows" (list d np) (list h (gctx-tok st) (gctx-xg st)) (ceiling (* np d) 256) 256))
    ;; the routed experts and the shared one, side by side
    (together (e)
      (cond (grouped
             (q35-mm-id gm e st (layer-gexps l) (gctx-xg st) (gctx-eg st) n ne)
             (q35-mm-id gm e st (layer-uexps l) (gctx-xg st) (gctx-eu st) n ne))
            (t
             (q35-xmv gm e st (layer-gexps l) h nil d (gctx-eg st) np ne)
             (q35-xmv gm e st (layer-uexps l) h nil d (gctx-eu st) np ne)))
      (mat gm e (layer-shgate l) h (gctx-shg st) n)
      (mat gm e (layer-shup l) h (gctx-shu st) n)
      (dispatch gpu e "q35_dot" (list d) (list (gm-layer gm li "ffn_gate_inp_shexp.weight") h (gctx-sg st)) n 256))
    (together (e)
      (dispatch gpu e "swiglu" (list (* np f)) (list (gctx-eg st) (gctx-eu st)) (ceiling (* np f) 256) 256)
      (dispatch gpu e "swiglu" (list (* n f)) (list (gctx-shg st) (gctx-shu st)) (ceiling (* n f) 256) 256))
    (together (e)
      (if grouped
          (q35-mm-id gm e st (layer-dexps l) (gctx-eg st) (gctx-ye st) n ne)
          (q35-xmv gm e st (layer-dexps l) (gctx-eg st) t f (gctx-ye st) np ne))
      (mat gm e (layer-shdown l) (gctx-shg st) (gctx-sh st) n))
    (if grouped
        (dispatch gpu e "q35_moe_combine_g" (list d n) (list (gctx-ye st) (gctx-wts st) (gctx-sh st) (gctx-sg st) h (gctx-pos st))
                  (list (ceiling d 256) n) 256)
        (dispatch gpu e "q35_moe_combine" (list d n) (list (gctx-ye st) (gctx-wts st) (gctx-sh st) (gctx-sg st) h)
                  (list (ceiling d 256) n) 256))))
