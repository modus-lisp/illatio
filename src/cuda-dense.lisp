;;;; cuda-dense.lisp — a qwen2 / qwen3 / llama model on an NVIDIA GPU
;;;; (src/cuda/dense.cu).
;;;
;;; The matrices are left in the file when the model loads (GMATs, as for
;;; Metal) and go to the device once: each tensor's raw blocks are copied up
;;; and split there into its int8s and its f16 scales, so a row is 16-byte
;;; aligned for the products.  The CPU never holds them.  A pass -- one token
;;; or a chunk of a prefill -- is a sequence of kernels on the device's
;;; stream; the CPU sends the token ids and reads back the logits.

(in-package #:illatio)

(defstruct (cmat (:constructor %make-cmat))
  name type (rows 0 :type fixnum) (cols 0 :type fixnum)
  qs d)                                 ; DBUFs: the int8s, the f16 scales

(defstruct (cmodel (:constructor %make-cmodel))
  dev
  (mats (make-hash-table :test 'eq))    ; GMAT -> CMAT
  (vecs (make-hash-table :test 'eq)))   ; f32 vector (norm, bias) -> DBUF

(defparameter *cuda-chunk* 512 "Tokens per GPU pass in a prefill.")

(defun load-model-cuda (path &key (device 0))
  "The model at PATH run by CUDA device DEVICE: its metadata, tokenizer and
   small vectors read as usual, its matrices copied to the device."
  (let ((m (let ((*matrices-in-file* t)) (load-model path :threads 1))))
    (unless (member (model-arch m) '("qwen2" "qwen3" "llama") :test #'equal)
      (error "the CUDA path runs qwen2, qwen3 and llama models so far; this is ~A" (model-arch m)))
    (let* ((dev (open-cuda device))
           (cm (%make-cmodel :dev dev))
           (file (map-file-vector path))
           (gmats (remove-duplicates
                   (list* (model-token-embd m) (model-output m)
                          (loop for l across (model-layers m)
                                append (list (layer-wq l) (layer-wk l) (layer-wv l) (layer-wo l)
                                             (layer-wgate l) (layer-wup l) (layer-wdown l))))))
           (raw (cuda-alloc dev (reduce #'max gmats :key (lambda (g) (* (gmat-rows g) (gmat-row-bytes g))))
                            :zero nil)))
      (unwind-protect
           (dolist (g gmats)
             (setf (gethash g (cmodel-mats cm)) (upload-gmat dev g file raw)))
        (cuda-free raw))
      (flet ((vec (v) (when v
                        (or (gethash v (cmodel-vecs cm))
                            (setf (gethash v (cmodel-vecs cm))
                                  (cuda-upload-vector (cuda-alloc dev (* 4 (length v)) :zero nil) v))))))
        (vec (model-output-norm m))
        (loop for l across (model-layers m)
              do (mapc #'vec (list (layer-attn-norm l) (layer-ffn-norm l) (layer-q-norm l) (layer-k-norm l)
                                   (layer-bq l) (layer-bk l) (layer-bv l)))))
      (cuda-sync dev)
      (setf (model-cuda m) cm)
      m)))

(defun upload-gmat (dev g file raw)
  "GMAT G's blocks from the mapped FILE to DEV, through the staging buffer RAW."
  (unless (eq (gmat-type g) :q8_0)
    (error "the CUDA path multiplies Q8_0 so far; ~A is ~A" (gmat-name g) (gmat-type g)))
  (let* ((rows (gmat-rows g)) (cols (gmat-cols g))
         (n (* rows cols)) (nb (floor n 32))
         (qs (cuda-alloc dev n :zero nil))
         (d (cuda-alloc dev (* 2 nb) :zero nil)))
    (cuda-upload raw (sb-sys:sap+ (sb-sys:vector-sap file) (gmat-offset g)) (* rows (gmat-row-bytes g)))
    (launch dev "repack_q8" (ceiling nb 8) '(32 8) (list raw qs d (list :i64 nb)))
    (%make-cmat :name (gmat-name g) :type :q8_0 :rows rows :cols cols :qs qs :d d)))

(defun cmat-of (cm g) (or (gethash g (cmodel-mats cm)) (error "~A is not on the device" (gmat-name g))))
(defun cvec-of (cm v) (and v (or (gethash v (cmodel-vecs cm)) (error "a vector is not on the device"))))

;;; --- per-context state ------------------------------------------------------------

(defstruct (cctx (:constructor %make-cctx))
  tokens x xq xs q k v att fa fb logits part rope kc vc
  (max-split 0 :type fixnum))

(defconstant +att-chunk+ 256 "Positions a block of ATTN_SPLIT takes, at least.")

(defun cuda-state (ctx)
  (or (getf (context-plist ctx) :cuda)
      (setf (getf (context-plist ctx) :cuda) (make-cuda-state ctx))))

(defun make-cuda-state (ctx)
  (let* ((m (context-model ctx)) (dev (cmodel-dev (model-cuda m)))
         (n *cuda-chunk*) (n-ctx (context-n-ctx ctx))
         (d (model-d m)) (f (model-ffn m)) (hd (model-head-dim m))
         (nh (model-n-head m)) (qd (model-q-dim m)) (kvd (* (model-n-kv-head m) hd))
         (wide (max d qd f))
         (max-split (ceiling n-ctx +att-chunk+)))
    (flet ((fb (k) (cuda-alloc dev (* 4 k))))
      (let ((st (%make-cctx
                 :tokens (fb n) :x (fb (* n d))
                 :xq (cuda-alloc dev (* n wide)) :xs (fb (ceiling (* n wide) 32))
                 :q (fb (* n qd)) :k (fb (* n kvd)) :v (fb (* n kvd)) :att (fb (* n qd))
                 :fa (fb (* n f)) :fb (fb (* n f)) :logits (fb (model-n-vocab m))
                 ;; a decode's splits reach the whole context; a prefill's at most 4
                 :part (fb (* (max max-split (* n 4)) nh (+ hd 2)))
                 :max-split max-split
                 :rope (fb (* n-ctx hd))
                 :kc (map 'simple-vector (lambda (l) (declare (ignore l)) (fb (* n-ctx kvd))) (model-layers m))
                 :vc (map 'simple-vector (lambda (l) (declare (ignore l)) (fb (* n-ctx kvd))) (model-layers m)))))
        ;; every position's RoPE table, as the CPU computes it (ROPE-TABLE), once
        (let ((tab (f32vec (* n-ctx hd))))
          (dotimes (p n-ctx)
            (multiple-value-bind (cos sin) (rope-table hd p (model-rope-base m))
              (replace tab cos :start1 (* p hd))
              (replace tab sin :start1 (+ (* p hd) (floor hd 2)))))
          (cuda-upload-vector (cctx-rope st) tab))
        st))))

(defun free-cuda-state (ctx)
  (let ((st (getf (context-plist ctx) :cuda)))
    (when st
      (dolist (s '(tokens x xq xs q k v att fa fb logits part rope))
        (cuda-free (slot-value st s)))
      (map nil #'cuda-free (cctx-kc st))
      (map nil #'cuda-free (cctx-vc st))
      (remf (context-plist ctx) :cuda))))

;;; --- a pass ---------------------------------------------------------------------------

(defun cmat-mul (dev w xq xs y n &key bias acc)
  "Y = W X for N tokens (token t's X at t*cols, its Y at t*rows), or Y += it
   when ACC: a matrix-vector product for one, the tiled product for more."
  (let ((rows (cmat-rows w)) (cols (cmat-cols w)))
    (if (= n 1)
        (launch dev "mv_q8" (ceiling rows 4) 128
                (list (cmat-qs w) (cmat-d w) xq xs y bias rows cols (if acc 1 0)))
        (launch dev "mm_q8" (list (ceiling n 64) (ceiling rows 64)) 256
                (list (cmat-qs w) (cmat-d w) xq xs y bias rows cols n (if acc 1 0))))))

(defun cuda-forward (ctx token logits)
  (cuda-forward-n ctx (list token) logits))

(defun cuda-eval (ctx tokens)
  "TOKENS through the GPU, *CUDA-CHUNK* at a time; the last one's logits."
  (loop for rest = tokens then (nthcdr *cuda-chunk* rest)
        while rest
        do (cuda-forward-n ctx (subseq rest 0 (min *cuda-chunk* (length rest)))
                           (<= (length rest) *cuda-chunk*)))
  (context-logits ctx))

(defun cuda-forward-n (ctx tokens logits)
  "TOKENS (at most *CUDA-CHUNK*) at the next positions; the logits, if
   LOGITS, are the last token's."
  (let* ((m (context-model ctx)) (cm (model-cuda m)) (dev (cmodel-dev cm))
         (st (cuda-state ctx)) (pos0 (context-pos ctx)) (n (length tokens))
         (eps (model-eps m)) (d (model-d m)) (f (model-ffn m))
         (hd (model-head-dim m)) (nh (model-n-head m)) (nkv (model-n-kv-head m))
         (qd (model-q-dim m))
         (x (cctx-x st)) (xq (cctx-xq st)) (xs (cctx-xs st))
         (visible (+ pos0 n))
         ;; a decode splits its keys finely; a prefill has blocks enough
         (chunk (if (= n 1)
                    +att-chunk+
                    (max +att-chunk+ (ceiling visible 4))))
         (nsplit (ceiling visible chunk))
         (scale (/ 1f0 (sqrt (coerce hd 'single-float)))))
    (assert (<= 1 n *cuda-chunk*))
    (when (> visible (context-n-ctx ctx))
      (error "context is full (~D tokens)" (context-n-ctx ctx)))
    (with-cuda
      (cuda-current dev)
      (cuda-upload-vector (cctx-tokens st) (coerce tokens '(simple-array (signed-byte 32) (*))))
      (let ((emb (cmat-of cm (model-token-embd m))))
        (launch dev "embed_q8" n 256 (list (cmat-qs emb) (cmat-d emb) (cctx-tokens st) x d)))
      (loop for li below (model-n-layer m)
            for l = (svref (model-layers m) li)
            do (launch dev "rmsnorm_q8" n 256 (list x (cvec-of cm (layer-attn-norm l)) nil xq xs d eps))
               (cmat-mul dev (cmat-of cm (layer-wq l)) xq xs (cctx-q st) n :bias (cvec-of cm (layer-bq l)))
               (cmat-mul dev (cmat-of cm (layer-wk l)) xq xs (cctx-k st) n :bias (cvec-of cm (layer-bk l)))
               (cmat-mul dev (cmat-of cm (layer-wv l)) xq xs (cctx-v st) n :bias (cvec-of cm (layer-bv l)))
               (launch dev "qk_rope" (list (+ nh nkv) n) (max 32 (floor hd 2))
                       (list (cctx-q st) (cctx-k st) (cctx-v st)
                             (cvec-of cm (layer-q-norm l)) (cvec-of cm (layer-k-norm l))
                             (cctx-rope st) (svref (cctx-kc st) li) (svref (cctx-vc st) li)
                             nh nkv hd pos0 eps (if (model-rope-neox m) 1 0)))
               (launch dev "attn_split" (list nh nsplit n) 128
                       (list (cctx-q st) (svref (cctx-kc st) li) (svref (cctx-vc st) li) (cctx-part st)
                             nh nkv hd pos0 chunk nsplit scale))
               (launch dev "attn_merge" (list nh n) (max 32 hd)
                       (list (cctx-part st) (cctx-att st) nh hd pos0 chunk nsplit))
               (launch dev "quant_q8" (ceiling (* n qd) 256) 256
                       (list (cctx-att st) xq xs (list :i64 (floor (* n qd) 32))))
               (cmat-mul dev (cmat-of cm (layer-wo l)) xq xs x n :acc t)
               (launch dev "rmsnorm_q8" n 256 (list x (cvec-of cm (layer-ffn-norm l)) nil xq xs d eps))
               (let ((g (cmat-of cm (layer-wgate l))) (u (cmat-of cm (layer-wup l))))
                 (if (= n 1)
                     (launch dev "mv_q8_glu" (ceiling f 4) 128
                             (list (cmat-qs g) (cmat-d g) (cmat-qs u) (cmat-d u) xq xs (cctx-fa st) f d))
                     (progn
                       (cmat-mul dev g xq xs (cctx-fa st) n)
                       (cmat-mul dev u xq xs (cctx-fb st) n)
                       (launch dev "swiglu" (ceiling (* n f) 256) 256
                               (list (cctx-fa st) (cctx-fb st) (list :i64 (* n f)))))))
               (launch dev "quant_q8" (ceiling (* n f) 256) 256
                       (list (cctx-fa st) xq xs (list :i64 (floor (* n f) 32))))
               (cmat-mul dev (cmat-of cm (layer-wdown l)) xq xs x n :acc t))
      (when logits
        (launch dev "rmsnorm_q8" 1 256
                (list (cons x (* 4 (1- n) d)) (cvec-of cm (model-output-norm m)) nil xq xs d eps))
        (cmat-mul dev (cmat-of cm (model-output m)) xq xs (cctx-logits st) 1)
        (let ((out (context-logits ctx)))
          (sb-sys:with-pinned-objects (out)
            (cuda-download (cctx-logits st) (sb-sys:vector-sap out) (* 4 (length out)))))))
    (incf (context-pos ctx) n)
    ctx))
