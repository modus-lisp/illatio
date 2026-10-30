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

(defconstant +att-chunk+ 64 "Positions a block of ATTN_DECODE takes at most (AD_CHUNK).")

(defun decode-chunk (visible)
  "Positions a block of ATTN_DECODE takes for a context of VISIBLE: fewer
   while the context is short, so the device still has blocks enough (about
   48 splits), between 16 and +ATT-CHUNK+."
  (max 16 (min +att-chunk+ (* 16 (ceiling visible (* 16 48))))))

(defun cuda-state (ctx)
  (or (getf (context-plist ctx) :cuda)
      (setf (getf (context-plist ctx) :cuda) (make-cuda-state ctx))))

(defun make-cuda-state (ctx)
  (let* ((m (context-model ctx)) (dev (cmodel-dev (model-cuda m)))
         (n *cuda-chunk*) (n-ctx (context-n-ctx ctx))
         (d (model-d m)) (f (model-ffn m)) (hd (model-head-dim m))
         (nh (model-n-head m)) (qd (model-q-dim m)) (kvd (* (model-n-kv-head m) hd))
         (wide (max d qd f))
         (max-split (ceiling n-ctx 16)))
    (flet ((fb (k) (cuda-alloc dev (* 4 k))))
      (let ((st (%make-cctx
                 :tokens (fb n) :x (fb (* n d))
                 :xq (cuda-alloc dev (* n wide)) :xs (fb (ceiling (* n wide) 32))
                 :q (fb (* n qd)) :k (fb (* n kvd)) :v (fb (* n kvd)) :att (fb (* n qd))
                 :fa (fb (* n f)) :fb (fb (* n f)) :logits (fb (model-n-vocab m))
                 ;; a decode's splits, for ATTN_MERGE
                 :part (fb (* max-split nh (+ hd 2)))
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
        (if (>= (cudev-arch dev) 80)
            (launch dev "mm_q8_mma" (list (ceiling n 64) (ceiling rows 128)) 256
                    (list (cmat-qs w) (cmat-d w) xq xs y bias rows cols n (if acc 1 0)))
            (launch dev "mm_q8" (list (ceiling n 64) (ceiling rows 64)) 256
                    (list (cmat-qs w) (cmat-d w) xq xs y bias rows cols n (if acc 1 0)))))))

(defun rmsnorm-q8 (dev x w xq xs d eps n)
  "RMSNORM! of N rows of X (D wide) by W, quantized into XQ / XS.  A single
   row is spread over a block per 256 of its elements."
  (launch dev "rmsnorm_q8" (list (if (= n 1) (ceiling d 256) 1) n) 256 (list x w nil xq xs d eps)))

(defun attn-prefill-shape (dev hd)
  "(VALUES queries-per-warp shared-bytes) for ATTN_PREFILL: as many queries
   a warp as the device's shared memory takes, up to 8."
  (loop for qpw from 8 downto 1
        for bytes = (* 4 (+ (* hd 33) (* 32 hd) (* 4 qpw hd)))
        when (<= bytes (cudev-max-shared dev)) return (values qpw bytes)
        finally (error "ATTN_PREFILL: heads of ~D do not fit in shared memory" hd)))

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
         (chunk (decode-chunk visible))
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
            do (rmsnorm-q8 dev x (cvec-of cm (layer-attn-norm l)) xq xs d eps n)
               (let ((wq (cmat-of cm (layer-wq l))) (wk (cmat-of cm (layer-wk l))) (wv (cmat-of cm (layer-wv l)))
                     (bq (cvec-of cm (layer-bq l))) (bk (cvec-of cm (layer-bk l))) (bv (cvec-of cm (layer-bv l))))
                 (if (= n 1)
                     (launch dev "mv_q8_3" (ceiling (+ (cmat-rows wq) (cmat-rows wk) (cmat-rows wv)) 4) 128
                             (list (cmat-qs wq) (cmat-d wq) (cmat-qs wk) (cmat-d wk) (cmat-qs wv) (cmat-d wv)
                                   xq xs (cctx-q st) (cctx-k st) (cctx-v st) bq bk bv
                                   (cmat-rows wq) (cmat-rows wk) (cmat-rows wv) d))
                     (progn
                       (cmat-mul dev wq xq xs (cctx-q st) n :bias bq)
                       (cmat-mul dev wk xq xs (cctx-k st) n :bias bk)
                       (cmat-mul dev wv xq xs (cctx-v st) n :bias bv))))
               (launch dev "qk_rope" (list (+ nh nkv) n) (max 32 (floor hd 2))
                       (list (cctx-q st) (cctx-k st) (cctx-v st)
                             (cvec-of cm (layer-q-norm l)) (cvec-of cm (layer-k-norm l))
                             (cctx-rope st) (svref (cctx-kc st) li) (svref (cctx-vc st) li)
                             nh nkv hd pos0 eps (if (model-rope-neox m) 1 0)))
               (if (= n 1)
                   (progn
                     (launch dev "attn_decode" (list nkv nsplit) 128
                             (list (cctx-q st) (svref (cctx-kc st) li) (svref (cctx-vc st) li) (cctx-part st)
                                   nh nkv hd pos0 chunk nsplit scale)
                             :shared (* 4 (max (* hd (1+ +att-chunk+)) (* 8 hd (floor nh nkv)))))
                     ;; merged and quantized for the output projection
                     (launch dev "attn_merge" (list nh n) (max 32 hd)
                             (list (cctx-part st) (cctx-att st) xq xs nh hd pos0 chunk nsplit)))
                   (progn
                     (let ((bytes2 (* 4 (+ (* 64 (+ hd 4)) (* 32 (+ hd 4)) (* 32 hd) (* 32 68)))))
                       (if (and (zerop (mod hd 64)) (<= bytes2 (cudev-max-shared dev)))
                           (launch dev "attn_prefill2" (list nh (ceiling n 64)) 256
                                   (list (cctx-q st) (svref (cctx-kc st) li) (svref (cctx-vc st) li) (cctx-att st)
                                         nh nkv hd pos0 n scale)
                                   :shared bytes2)
                           (multiple-value-bind (qpw bytes) (attn-prefill-shape dev hd)
                             (launch dev "attn_prefill" (list nh (ceiling n (* 4 qpw))) 128
                                     (list (cctx-q st) (svref (cctx-kc st) li) (svref (cctx-vc st) li) (cctx-att st)
                                           nh nkv hd pos0 n scale qpw)
                                     :shared bytes))))
                     (launch dev "quant_q8" (ceiling (* n qd) 256) 256
                             (list (cctx-att st) xq xs (list :i64 (floor (* n qd) 32))))))
               (cmat-mul dev (cmat-of cm (layer-wo l)) xq xs x n :acc t)
               (rmsnorm-q8 dev x (cvec-of cm (layer-ffn-norm l)) xq xs d eps n)
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
        (rmsnorm-q8 dev (cons x (* 4 (1- n) d)) (cvec-of cm (model-output-norm m)) xq xs d eps 1)
        (cmat-mul dev (cmat-of cm (model-output m)) xq xs (cctx-logits st) 1)
        (let ((out (context-logits ctx)))
          (sb-sys:with-pinned-objects (out)
            (cuda-download (cctx-logits st) (sb-sys:vector-sap out) (* 4 (length out)))))))
    (incf (context-pos ctx) n)
    ctx))
