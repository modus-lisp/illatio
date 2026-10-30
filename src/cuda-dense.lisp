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
  qs d)                                 ; DBUFs.  Q8_0: the int8s, the f16 scales;
                                        ; a K-quant: its blocks (kquant.cu), and NIL

(defun kquant-code (type)
  "A K-quant format as kquant.cu numbers it."
  (ecase type (:q4_k 0) (:q5_k 1) (:q6_k 2)))

(defun cmat-kquant-p (w) (member (cmat-type w) '(:q4_k :q5_k :q6_k)))

(defstruct (cmodel (:constructor %make-cmodel))
  dev
  (mats (make-hash-table :test 'eq))    ; GMAT -> CMAT
  (vecs (make-hash-table :test 'eq)))   ; f32 vector (norm, bias) -> DBUF

(defparameter *cuda-chunk* 512 "Tokens per GPU pass in a prefill.")

(defun load-model-cuda (path &key (device 0))
  "The model at PATH run by CUDA device DEVICE: its metadata, tokenizer and
   small vectors read as usual, its matrices copied to the device."
  (let ((m (let ((*matrices-in-file* t)) (load-model path :threads 1))))
    (unless (member (model-arch m) '("qwen2" "qwen3" "llama" "qwen35") :test #'equal)
      (error "the CUDA path runs qwen2, qwen3, llama and qwen35 models; this is ~A" (model-arch m)))
    (let* ((dev (open-cuda device))
           (cm (%make-cmodel :dev dev))
           (file (map-file-vector path))
           (gmats (remove-duplicates
                   (remove nil
                           (list* (model-token-embd m) (model-output m)
                                  (loop for l across (model-layers m)
                                        append (list (layer-wq l) (layer-wk l) (layer-wv l) (layer-wo l)
                                                     (layer-wqkv l) (layer-wz l) (layer-walpha l) (layer-wbeta l)
                                                     (layer-ssm-out l)
                                                     (layer-wgate l) (layer-wup l) (layer-wdown l)))))))
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
                                   (layer-bq l) (layer-bk l) (layer-bv l)
                                   (layer-ssm-a l) (layer-ssm-dt l) (layer-conv1d l) (layer-ssm-norm l)))))
      (cuda-sync dev)
      (setf (model-cuda m) cm)
      m)))

(defun upload-gmat (dev g file raw)
  "GMAT G's blocks from the mapped FILE to DEV, through the staging buffer
   RAW: Q8_0's split into int8s and scales, Q6_K's padded, Q4_K's and
   Q5_K's as they are."
  (let* ((type (gmat-type g)) (rows (gmat-rows g)) (cols (gmat-cols g))
         (n (* rows cols)) (bytes (* rows (gmat-row-bytes g)))
         (src (sb-sys:sap+ (sb-sys:vector-sap file) (gmat-offset g))))
    (unless (member type '(:q8_0 :q4_k :q5_k :q6_k :f32))
      (error "the CUDA path multiplies Q8_0, Q4_K, Q5_K, Q6_K and F32; ~A is ~A" (gmat-name g) type))
    (unless (zerop (mod cols (case type (:q8_0 32) (:f32 4) (t 256))))
      (error "~A: ~D columns is not a whole number of blocks" (gmat-name g) cols))
    (ecase type
      (:q8_0
       (let* ((nb (floor n 32)) (qs (cuda-alloc dev n :zero nil)) (d (cuda-alloc dev (* 2 nb) :zero nil)))
         (cuda-upload raw src bytes)
         (launch dev "repack_q8" (ceiling nb 8) '(32 8) (list raw qs d (list :i64 nb)))
         (%make-cmat :name (gmat-name g) :type type :rows rows :cols cols :qs qs :d d)))
      ((:q4_k :q5_k :f32)
       (let ((qs (cuda-alloc dev bytes :zero nil)))
         (cuda-upload qs src bytes)
         (%make-cmat :name (gmat-name g) :type type :rows rows :cols cols :qs qs)))
      (:q6_k
       (let* ((nb (floor n 256)) (qs (cuda-alloc dev (* 224 nb) :zero nil)))
         (cuda-upload raw src bytes)
         (launch dev "repack_q6k" (ceiling nb 8) '(32 8) (list raw qs (list :i64 nb)))
         (%make-cmat :name (gmat-name g) :type type :rows rows :cols cols :qs qs))))))

(defun model-kquant-p (m)
  "Whether any of M's matrices on the device is a K-quant."
  (loop for w being the hash-values of (cmodel-mats (model-cuda m)) thereis (cmat-kquant-p w)))

(defun cmat-of (cm g) (or (gethash g (cmodel-mats cm)) (error "~A is not on the device" (gmat-name g))))
(defun cvec-of (cm v) (and v (or (gethash v (cmodel-vecs cm)) (error "a vector is not on the device"))))

;;; --- per-context state ------------------------------------------------------------

(defstruct (cctx (:constructor %make-cctx))
  tokens x xq xs q k v att fa fb logits part rope kc vc
  xsum                                  ; the activation blocks' int8 sums, for K-quants (or NIL)
  h                                     ; the normed input as floats (for F32 matrices)
  ;; qwen35's linear layers: scratch, and per layer (NIL for a full one) the
  ;; convolution's history and the delta net's state
  qkv z alpha beta c o qg gate conv ssm
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
         (d (model-d m)) (f (model-ffn m)) (hd (model-head-dim m)) (rd (model-rope-dims m))
         (nh (model-n-head m)) (qd (model-q-dim m)) (kvd (* (model-n-kv-head m) hd))
         (q35 (model-recurrent-p m))
         (ch (if q35 (q35-channels m) 0)) (vh (model-ssm-v-heads m))
         (od (* vh (model-ssm-v-dim m)))
         (wide (max d qd f od))
         (max-split (ceiling n-ctx 16))
         (full (loop for l across (model-layers m) count (eq (layer-kind l) :full))))
    (flet ((fb (k) (cuda-alloc dev (* 4 k))))
      (let ((st (%make-cctx
                 :tokens (fb n) :x (fb (* n d))
                 :xq (cuda-alloc dev (* n wide)) :xs (fb (ceiling (* n wide) 32))
                 :xsum (and (model-kquant-p m) (fb (ceiling (* n wide) 32)))
                 :q (fb (* n qd)) :k (fb (* n kvd)) :v (fb (* n kvd)) :att (fb (* n qd))
                 :fa (fb (* n f)) :fb (fb (* n f)) :logits (fb (model-n-vocab m))
                 ;; a decode's splits, for ATTN_MERGE
                 :part (fb (* max-split nh (+ hd 2)))
                 :max-split max-split
                 :rope (fb (* n-ctx rd))
                 :h (fb (* n d))
                 ;; the full layers' caches in one allocation each for K and V:
                 ;; a layer's is (buffer . its offset), a linear layer's NIL
                 :kc (layer-slices m (fb (* full n-ctx kvd)) (* 4 n-ctx kvd))
                 :vc (layer-slices m (fb (* full n-ctx kvd)) (* 4 n-ctx kvd)))))
        (when q35
          (setf (cctx-qkv st) (fb (* n ch)) (cctx-z st) (fb (* n od))
                (cctx-alpha st) (fb (* n vh)) (cctx-beta st) (fb (* n vh))
                (cctx-c st) (fb (* n ch)) (cctx-o st) (fb (* n od))
                (cctx-qg st) (fb (* n 2 qd)) (cctx-gate st) (fb (* n qd))
                (cctx-conv st) (map 'simple-vector (lambda (l) (and (eq (layer-kind l) :linear)
                                                                    (fb (* ch (1- (model-ssm-conv m))))))
                                    (model-layers m))
                (cctx-ssm st) (map 'simple-vector (lambda (l) (and (eq (layer-kind l) :linear)
                                                                   (fb (* vh (model-ssm-k-dim m) (model-ssm-v-dim m)))))
                                   (model-layers m))))
        ;; every position's RoPE table, as the CPU computes it (ROPE-TABLE), once
        (let ((tab (f32vec (* n-ctx rd))))
          (dotimes (p n-ctx)
            (multiple-value-bind (cos sin) (rope-table rd p (model-rope-base m))
              (replace tab cos :start1 (* p rd))
              (replace tab sin :start1 (+ (* p rd) (floor rd 2)))))
          (cuda-upload-vector (cctx-rope st) tab))
        st))))

(defun reset-cuda-state (ctx)
  "Zero the linear layers' state: the context starts over."
  (let ((st (getf (context-plist ctx) :cuda)))
    (when (and st (cctx-conv st))
      (with-cuda
        (cuda-current (cmodel-dev (model-cuda (context-model ctx))))
        (loop for v across (cctx-conv st) when v do (cuda-zero v))
        (loop for v across (cctx-ssm st) when v do (cuda-zero v))))))

(defun layer-slices (m buf bytes)
  "BUF in slices of BYTES, one for each of M's full-attention layers in
   order (NIL for the others)."
  (let ((i -1))
    (map 'simple-vector (lambda (l) (and (eq (layer-kind l) :full) (cons buf (* (incf i) bytes))))
         (model-layers m))))

(defun free-cuda-state (ctx)
  (let ((st (getf (context-plist ctx) :cuda)))
    (when st
      (dolist (s '(tokens x xq xs xsum h q k v att fa fb logits part rope qkv z alpha beta c o qg gate))
        (cuda-free (slot-value st s)))
      (cuda-free (car (find-if-not #'null (cctx-kc st))))
      (cuda-free (car (find-if-not #'null (cctx-vc st))))
      (when (cctx-conv st)
        (map nil (lambda (v) (and v (cuda-free v))) (cctx-conv st))
        (map nil (lambda (v) (and v (cuda-free v))) (cctx-ssm st)))
      (remf (context-plist ctx) :cuda))))

;;; --- a pass ---------------------------------------------------------------------------

(defconstant +mm2-shared+ (* 2 (+ (* 4 192 36) (* 128 8) (* 64 16)))
  "MM_Q8_MMA2's shared memory: two stages of 128 rows and 64 tokens (M2_SHARED).")

(defun mm2-p (dev cols) (and (>= (cudev-arch dev) 80) (zerop (mod cols 128))))

(defun cmat-mul (dev w xq xs y n &key bias acc xsum h)
  "Y = W X for N tokens (token t's X at t*cols, its Y at t*rows), or Y += it
   when ACC: a matrix-vector product for one, the tiled product for more.
   XSUM, the activations' block sums, for a K-quant W; H, the activations
   unquantized, for an F32 one."
  (let ((rows (cmat-rows w)) (cols (cmat-cols w)))
    (cond ((eq (cmat-type w) :f32)
           (assert (and h (not acc) (not bias)))
           (launch dev "mm_f32" (ceiling (* rows n) 4) 128 (list (cmat-qs w) h y rows cols n)))
          ((cmat-kquant-p w)
           (if (= n 1)
               (launch dev "mv_k" (ceiling rows 4) 128
                       (list (kquant-code (cmat-type w)) (cmat-qs w) xq xs xsum y bias rows cols (if acc 1 0)))
               (if (>= (cudev-arch dev) 80)
                   (launch dev "mm_k_mma" (list (ceiling n 64) (ceiling rows 128)) 256
                           (list (kquant-code (cmat-type w)) (cmat-qs w) xq xs xsum y bias rows cols n (if acc 1 0)))
                   (launch dev "mm_k" (list (ceiling n 64) (ceiling rows 64)) 256
                           (list (kquant-code (cmat-type w)) (cmat-qs w) xq xs xsum y bias rows cols n (if acc 1 0))))))
          ((= n 1)
           (launch dev "mv_q8" (ceiling rows 4) 128
                   (list (cmat-qs w) (cmat-d w) xq xs y bias rows cols (if acc 1 0))))
          ((mm2-p dev cols) (cmat-mul-n dev (list (list w y bias)) xq xs n :acc acc))
          ((>= (cudev-arch dev) 80)
           (launch dev "mm_q8_mma" (list (ceiling n 64) (ceiling rows 128)) 256
                   (list (cmat-qs w) (cmat-d w) xq xs y bias rows cols n (if acc 1 0))))
          (t
           (launch dev "mm_q8" (list (ceiling n 64) (ceiling rows 64)) 256
                   (list (cmat-qs w) (cmat-d w) xq xs y bias rows cols n (if acc 1 0)))))))

(defun cmat-mul-n (dev parts xq xs n &key acc)
  "PARTS, up to three (W Y BIAS) sharing X, in one launch of MM_Q8_MMA2."
  (let ((args (loop for k below 3
                    for (w y bias) = (nth k parts)
                    append (if w (list (cmat-qs w) (cmat-d w) y bias (cmat-rows w)) (list nil nil nil nil 0)))))
    (launch dev "mm_q8_mma2"
            (list (ceiling n 64) (loop for (w) in parts sum (ceiling (cmat-rows w) 128))) 256
            (append args (list xq xs (cmat-cols (first (first parts))) n (if acc 1 0)))
            :shared +mm2-shared+)))

(defun rmsnorm-q8 (dev x w xq xs xsum d eps n)
  "RMSNORM! of N rows of X (D wide) by W, quantized into XQ / XS (and the
   block sums into XSUM, unless NIL).  A single row is spread over a block
   per 256 of its elements."
  (launch dev "rmsnorm_q8" (list (if (= n 1) (ceiling d 256) 1) n) 256 (list x w nil xq xs xsum d eps)))

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

(defun embed (dev emb tokens x n d)
  "Rows TOKENS (a device vector of N ids) of the embedding EMB as floats in X."
  (if (cmat-kquant-p emb)
      (launch dev "embed_k" n 128 (list (kquant-code (cmat-type emb)) (cmat-qs emb) tokens x d))
      (launch dev "embed_q8" n 256 (list (cmat-qs emb) (cmat-d emb) tokens x d))))

(defun qkv-mul (dev st wq wk wv bq bk bv n d)
  "Q, K and V's products into the state's Q, K, V: one launch when the
   formats allow it."
  (let ((xq (cctx-xq st)) (xs (cctx-xs st)) (xsum (cctx-xsum st))
        (q8 (notany #'cmat-kquant-p (list wq wk wv))))
    (cond ((and (= n 1) q8)
           (launch dev "mv_q8_3" (ceiling (+ (cmat-rows wq) (cmat-rows wk) (cmat-rows wv)) 4) 128
                   (list (cmat-qs wq) (cmat-d wq) (cmat-qs wk) (cmat-d wk) (cmat-qs wv) (cmat-d wv)
                         xq xs (cctx-q st) (cctx-k st) (cctx-v st) bq bk bv
                         (cmat-rows wq) (cmat-rows wk) (cmat-rows wv) d)))
          ((and (> n 1) q8 (mm2-p dev d) (every (lambda (w) (zerop (mod (cmat-rows w) 128))) (list wq wk wv)))
           (cmat-mul-n dev (list (list wq (cctx-q st) bq) (list wk (cctx-k st) bk) (list wv (cctx-v st) bv))
                       xq xs n))
          (t (cmat-mul dev wq xq xs (cctx-q st) n :bias bq :xsum xsum)
             (cmat-mul dev wk xq xs (cctx-k st) n :bias bk :xsum xsum)
             (cmat-mul dev wv xq xs (cctx-v st) n :bias bv :xsum xsum)))))

(defun glu-mul (dev st g u n f d)
  "silu(G x) * (U x) into the state's FA."
  (let ((xq (cctx-xq st)) (xs (cctx-xs st)) (xsum (cctx-xsum st)))
    (cond ((and (= n 1) (not (cmat-kquant-p g)) (not (cmat-kquant-p u)))
           (launch dev "mv_q8_glu" (ceiling f 4) 128
                   (list (cmat-qs g) (cmat-d g) (cmat-qs u) (cmat-d u) xq xs (cctx-fa st) f d)))
          ((and (= n 1) (cmat-kquant-p g) (cmat-kquant-p u))
           (launch dev "mv_k_glu" (ceiling f 4) 128
                   (list (kquant-code (cmat-type g)) (cmat-qs g) (kquant-code (cmat-type u)) (cmat-qs u)
                         xq xs xsum (cctx-fa st) f d)))
          (t (cmat-mul dev g xq xs (cctx-fa st) n :xsum xsum)
             (cmat-mul dev u xq xs (cctx-fb st) n :xsum xsum)
             (launch dev "swiglu" (ceiling (* n f) 256) 256
                     (list (cctx-fa st) (cctx-fb st) (list :i64 (* n f))))))))

(defun attention (dev m st li n pos0 visible &key (quantize t))
  "Layer LI's attention of the N tokens at POS0, into the state's ATT, and
   quantized for the output projection unless QUANTIZE is NIL."
  (let* ((hd (model-head-dim m)) (nh (model-n-head m)) (nkv (model-n-kv-head m)) (qd (model-q-dim m))
         (scale (/ 1f0 (sqrt (coerce hd 'single-float))))
         (q (cctx-q st)) (kc (svref (cctx-kc st) li)) (vc (svref (cctx-vc st) li))
         (xq (cctx-xq st)) (xs (cctx-xs st)) (xsum (cctx-xsum st)))
    (if (= n 1)
        (let* ((chunk (decode-chunk visible)) (nsplit (ceiling visible chunk)))
          (launch dev "attn_decode" (list nkv nsplit) 128
                  (list q kc vc (cctx-part st) nh nkv hd pos0 chunk nsplit scale)
                  :shared (* 4 (max (* hd (1+ +att-chunk+)) (* 8 hd (floor nh nkv)))))
          ;; merged and quantized at once
          (launch dev "attn_merge" (list nh n) (max 32 hd)
                  (list (cctx-part st) (cctx-att st) (and quantize xq) xs xsum nh hd pos0 chunk nsplit)))
        (let ((bytes2 (* 4 (+ (* 64 (+ hd 4)) (* 32 (+ hd 4)) (* 32 hd) (* 32 68)))))
          (if (and (zerop (mod hd 64)) (<= bytes2 (cudev-max-shared dev)))
              (launch dev "attn_prefill2" (list nh (ceiling n 64)) 256
                      (list q kc vc (cctx-att st) nh nkv hd pos0 n scale)
                      :shared bytes2)
              (multiple-value-bind (qpw bytes) (attn-prefill-shape dev hd)
                (launch dev "attn_prefill" (list nh (ceiling n (* 4 qpw))) 128
                        (list q kc vc (cctx-att st) nh nkv hd pos0 n scale qpw)
                        :shared bytes)))
          (when quantize
            (launch dev "quant_q8" (ceiling (* n qd) 256) 256
                    (list (cctx-att st) xq xs xsum (list :i64 (floor (* n qd) 32)))))))))

(defun cuda-forward-n (ctx tokens logits)
  "TOKENS (at most *CUDA-CHUNK*) at the next positions; the logits, if
   LOGITS, are the last token's."
  (when (model-recurrent-p (context-model ctx))
    (return-from cuda-forward-n (cuda-forward-q35 ctx tokens logits)))
  (let* ((m (context-model ctx)) (cm (model-cuda m)) (dev (cmodel-dev cm))
         (st (cuda-state ctx)) (pos0 (context-pos ctx)) (n (length tokens))
         (eps (model-eps m)) (d (model-d m)) (f (model-ffn m))
         (hd (model-head-dim m)) (nh (model-n-head m)) (nkv (model-n-kv-head m))
         (x (cctx-x st)) (xq (cctx-xq st)) (xs (cctx-xs st)) (xsum (cctx-xsum st))
         (visible (+ pos0 n)))
    (assert (<= 1 n *cuda-chunk*))
    (when (> visible (context-n-ctx ctx))
      (error "context is full (~D tokens)" (context-n-ctx ctx)))
    (with-cuda
      (cuda-current dev)
      (cuda-upload-vector (cctx-tokens st) (coerce tokens '(simple-array (signed-byte 32) (*))))
      (embed dev (cmat-of cm (model-token-embd m)) (cctx-tokens st) x n d)
      (loop for li below (model-n-layer m)
            for l = (svref (model-layers m) li)
            do (rmsnorm-q8 dev x (cvec-of cm (layer-attn-norm l)) xq xs xsum d eps n)
               (qkv-mul dev st (cmat-of cm (layer-wq l)) (cmat-of cm (layer-wk l)) (cmat-of cm (layer-wv l))
                        (cvec-of cm (layer-bq l)) (cvec-of cm (layer-bk l)) (cvec-of cm (layer-bv l)) n d)
               (launch dev "qk_rope" (list (+ nh nkv) n) (max 32 (floor hd 2))
                       (list (cctx-q st) (cctx-k st) (cctx-v st)
                             (cvec-of cm (layer-q-norm l)) (cvec-of cm (layer-k-norm l))
                             (cctx-rope st) (svref (cctx-kc st) li) (svref (cctx-vc st) li)
                             nh nkv hd (model-rope-dims m) pos0 eps (if (model-rope-neox m) 1 0)))
               (attention dev m st li n pos0 visible)
               (cmat-mul dev (cmat-of cm (layer-wo l)) xq xs x n :acc t :xsum xsum)
               (rmsnorm-q8 dev x (cvec-of cm (layer-ffn-norm l)) xq xs xsum d eps n)
               (glu-mul dev st (cmat-of cm (layer-wgate l)) (cmat-of cm (layer-wup l)) n f d)
               (launch dev "quant_q8" (ceiling (* n f) 256) 256
                       (list (cctx-fa st) xq xs xsum (list :i64 (floor (* n f) 32))))
               (cmat-mul dev (cmat-of cm (layer-wdown l)) xq xs x n :acc t :xsum xsum))
      (when logits
        (rmsnorm-q8 dev (cons x (* 4 (1- n) d)) (cvec-of cm (model-output-norm m)) xq xs xsum d eps 1)
        (cmat-mul dev (cmat-of cm (model-output m)) xq xs (cctx-logits st) 1 :xsum xsum)
        (let ((out (context-logits ctx)))
          (sb-sys:with-pinned-objects (out)
            (cuda-download (cctx-logits st) (sb-sys:vector-sap out) (* 4 (length out)))))))
    (incf (context-pos ctx) n)
    ctx))
