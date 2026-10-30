;;;; cuda-ds4f-prefill.lisp — a DS4F prefill chunk on the GPU, a layer at a time.
;;;
;;; METAL-DS4F-PREFILL's long chunk on the first device: every op over the
;;; chunk's tokens at once, its activations kept on the device between them
;;; (the matrix products on the tensor cores, ds4f_mm.cu; the rest,
;;; ds4f_chunk.cu and ds4f.cu), the routing on the GPU too.  A layer's
;;; experts (nearly all of them are chosen by some token of a long chunk) are
;;; all taken into slots of the expert pool before it runs: those missing
;;; copied over the bus while the layer before computes, the pool's LRU
;;; deciding what they displace.  The device syncs once a layer, which costs
;;; nothing at this size and lets the arena (the small per-layer inputs) be
;;; reused.  A short chunk is still decode a token at a time.

(in-package #:illatio)

(defparameter *ds4f-cuda-chunk-from* 96
  "From this many tokens a prefill chunk runs layer-major on the device: a
   layer's experts all read (about 1.4 GB over the bus, 60 ms), against
   about 35 ms a token decoding.")

(defparameter *ds4f-cuda-chunk* 2048
  "Tokens a layer-major pass takes at most.  Its layers' experts are read
   whatever its length (about 60 GB a pass: 3 s), so long passes pay.")

(defparameter *ds4f-cuda-chunk-bytes* (* 1100 1024)
  "Device memory a token of a pass takes, at most (its activations).")

;;; A pass's activations live in slots of the expert pool, borrowed for it:
;;; a run of them at the pool's end, their experts dropped and the slots
;;; held, handed back (empty, first to be taken) when it is done.  So the
;;; pool can have nearly all of the device's memory for decode.

(defvar *dk-scratch* nil "During a pass: (next end table), its buffers' allocator.")

(defun pb (st key floats)
  "Scratch buffer KEY of at least FLOATS floats: the pass's, else ST's."
  (let ((bytes (* 4 (max 1 floats))))
    (if (null *dk-scratch*)
        (dbuf-of st key bytes)
        (destructuring-bind (next end table) *dk-scratch*
          (let ((b (gethash key table)))
            (if (and b (>= (dbuf-bytes b) bytes))
                b
                (let ((at (* 256 (ceiling next 256))))
                  (assert (<= (+ at bytes) end) () "a pass's scratch is out of room (~A)" key)
                  (setf (first *dk-scratch*) (+ at bytes)
                        (gethash key table) (%make-dbuf :ptr at :bytes bytes :dev (dk-dev st))))))))))

(defun d4-borrow-slots (st bytes)
  "The last slots of the pool, enough for BYTES, emptied and held: (first . count)."
  (let* ((n (dk-n-slots st)) (k (ceiling bytes (dk-slot-bytes st))) (s0 (- n k))
         (owner (dk-owner st)) (slot-of (dk-slot-of st)) (pin (dk-pin st)))
    (declare (type (simple-array fixnum (*)) owner slot-of pin))
    (assert (= (dk-free st) n) () "the pool is not yet full")
    (loop for s from s0 below n
          do (assert (zerop (aref pin s)))
             (when (>= (aref owner s) 0) (setf (aref slot-of (aref owner s)) -1 (aref owner s) -1))
             (incf (aref pin s)))
    (cons s0 k)))

(defun d4-return-slots (st run)
  (loop for s from (car run) below (+ (car run) (cdr run))
        do (decf (aref (dk-pin st) s)) (setf (aref (dk-used st) s) 0)))

(defun dk-mm (st kernel w wd rbytes rows cols x y n &key (x-stride cols) (y-stride rows))
  "Y (N rows, Y-STRIDE apart) = W X (N rows, X-STRIDE apart): D4_MM_<format>."
  (launch (dk-dev st) kernel (list (ceiling n 64) (ceiling rows 64)) 128
          (list cols rows n rbytes x-stride y-stride w wd x y)))

(defvar *dk-small* nil
  "During a short pass (a speculative step's few tokens): the products as
   decode's matvecs, each weight read once for all the tokens.")

(defvar *dk-keep-proj* nil
  "During a verifying pass: each layer's compressor projections kept in
   buffers of its own, so its carried windows can be redone for fewer
   tokens (DS4F-CUDA-UNDO).")

(defun dk-q8mm (st name x y n &rest keys &key group-rows &allow-other-keys)
  "Y = W X for Q8_0 tensor NAME over N tokens: the tensor cores' product,
   or in a short pass decode's (X quantized as decode quantizes it; row r
   against its token's group r/GROUP-ROWS when grouped)."
  (let ((w (or (gethash name (dk-q8 st)) (error "no Q8_0 tensor ~A" name))))
    (if *dk-small*
        (let* ((dev (dk-dev st)) (rows (cmat-rows w)) (cols (cmat-cols w))
               (xlen (if group-rows (* cols (floor rows group-rows)) cols))
               (q (dk-qbuf st :v-x (* n xlen))))
          (launch dev "d4_quant" (ceiling (* n xlen) 256) 256 (list x (car q) (cdr q) (floor (* n xlen) 32)))
          (launch dev "d4_mvq8n" (ceiling rows 4) 128
                  (list rows cols (or group-rows rows) n xlen rows (cmat-qs w) (cmat-d w) (car q) (cdr q) y)))
        (progn
          (assert (null group-rows))
          (apply #'dk-mm st "d4_mm_q8_0" (cmat-qs w) (cmat-d w) 0 (cmat-rows w) (cmat-cols w) x y n keys)))))

(defun dk-f16mm (st name x y n &rest keys &key y-stride &allow-other-keys)
  (destructuring-bind (w rows cols &optional type) (or (gethash name (dk-f16 st)) (error "no F16 tensor ~A" name))
    (declare (ignore type))
    (if *dk-small*
        (launch (dk-dev st) "d4_mv_f16n" (ceiling rows 4) 256 (list cols rows n (or y-stride rows) w x y))
        (apply #'dk-mm st "d4_mm_f16" w nil (* 2 cols) rows cols x y n keys))))

(defun dk-cs-rows (m st key pos-list il &optional inverse)
  "D4-ROPE-CS!'s table for each position of POS-LIST, a row each: in scratch
   KEY, uploaded now (before the pass's copies are queued) when KEY, else
   in the pass's mapped arena."
  (let* ((nrot (ds4f-n-rot m)) (cs (f32vec nrot)) (n (length pos-list)) (v (f32vec (* n nrot))) (k 0))
    (dolist (p pos-list)
      (d4-rope-cs! m p il inverse cs)
      (replace v cs :start1 (* k nrot))
      (incf k))
    (if key
        (let ((b (pb st key (* n nrot)))) (cuda-upload-vector b v) b)
        (dk-mput st v :f32))))

;;; A pass's small inputs for each layer (the KV rows' counts, the experts'
;;; slots, a compressor's RoPE rows) are not copied in: on the compute
;;; stream, a copy waits for the one copy engine behind the next layer's
;;; experts, and the layer's kernels behind it.  They are written to pinned
;;; host memory the kernels read in place, reused once the layer is done.

(defvar *dk-mapped* nil "During a pass: (sap . at), the mapped arena's host memory and its fill.")
(defconstant +dk-mapped-bytes+ (* 4 1024 1024))

(defun dk-mput (st words type)
  "WORDS (a sequence: :F32, :U32 or :U64) into the mapped arena: a DBUF over them."
  (let* ((size (if (eq type :u64) 8 4)) (n (length words)) (sap (car *dk-mapped*)) (at (cdr *dk-mapped*)) (i 0))
    (assert (<= (+ at (* size n)) +dk-mapped-bytes+) () "the pass's mapped arena is out of room")
    (map nil (lambda (x)
               (ecase type
                 (:f32 (setf (sb-sys:sap-ref-single sap (+ at (* 4 i))) x))
                 (:u32 (setf (sb-sys:sap-ref-32 sap (+ at (* 4 i))) x))
                 (:u64 (setf (sb-sys:sap-ref-64 sap (+ at (* 8 i))) x)))
               (incf i))
         words)
    (setf (cdr *dk-mapped*) (* 256 (ceiling (+ at (* size n)) 256)))
    (%make-dbuf :ptr (+ (sb-sys:sap-int sap) at) :bytes (* size n) :dev (dk-dev st))))

(defun dk-u32s (st v) (dk-mput st v :u32))

(defun dk-hc-pre-chunk (m st li which streams n out pc)
  "HC pre of N tokens' STREAMS (layer LI's WHICH: :attn or :ffn): OUT, PC (32 a token)."
  (let* ((l (svref (ds4f-layers m) li)) (dev (dk-dev st)) (d (ds4f-d m)) (hcd (* (ds4f-n-hc m) d))
         (flat (pb st :p-flat (* n hcd))) (mix (pb st :p-mix (* n 32))))
    (dk-rms dev streams nil flat hcd n (ds4f-eps m))
    (dk-f16mm st (lname li (if (eq which :attn) "hc_attn_fn.weight" "hc_ffn_fn.weight")) flat mix n :y-stride 32)
    (launch dev "d4_hc_pre_fin" (list (ceiling d 256) n) 256
            (list d (ds4f-n-hc m) (ds4f-sinkhorn m) 1f-6 mix
                  (dvec st (if (eq which :attn) (dl-hc-attn-scale l) (dl-hc-ffn-scale l)))
                  (dvec st (if (eq which :attn) (dl-hc-attn-base l) (dl-hc-ffn-base l))) streams out pc))))

(defun dk-hc-post-chunk (m st block streams pc out n)
  (let ((d (ds4f-d m)) (nh (ds4f-n-hc m)))
    (launch (dk-dev st) "d4_hc_post" (ceiling (* n nh d) 256) 256 (list d nh n 1f-6 block streams pc out))))

(defun dk-rope-chunk (st x cs nh hd nrot n)
  (launch (dk-dev st) "d4_rope" (ceiling (* n nh (floor nrot 2)) 256) 256 (list nh hd nrot n x cs)))

(defun dk-comp-carry (m st l c pos0 n kind pkv psc)
  "The window compressor KIND of layer L carries after N tokens from POS0
   (their projections PKV, PSC), brought up to date in C."
  (let* ((ratio (dl-ratio l)) (coff (if (= ratio 4) 2 1))
         (hd (if (= kind 0) (ds4f-head-dim m) (ds4f-ix-dim m))) (width (* coff hd))
         (ape (if (= kind 0) (dl-comp-ape l) (dl-ixc-ape l)))
         (st-kv (if (= kind 0) (dc-st-kv c) (dc-ix-kv c))) (st-sc (if (= kind 0) (dc-st-score c) (dc-ix-score c)))
         (wstart (* ratio (floor pos0 ratio))) (e-pos (+ pos0 n -1))
         (cw (* ratio (floor e-pos ratio))) (ends (zerop (mod (1+ e-pos) ratio)))
         (set-prev (and (= ratio 4) (or ends (> cw wstart))))
         (prev-from (cond ((not set-prev) 0) (ends cw) (t (- cw ratio)))))
    (launch (dk-dev st) "d4_comp_carry" 1 1024
            (list ratio width pos0 n wstart (f32mat-cols ape) prev-from (if set-prev 1 0)
                  pkv psc (dvec st (f32mat-data ape)) (dvec st st-kv) (dvec st st-sc)))))

(defun dk-proj-key (kind which li)
  "The scratch key of compressor KIND's projection WHICH (0 kv, 1 score) for layer LI."
  (let ((base (svref #("P-CKV" "P-CSC" "P-IKV" "P-ISC") (+ (* 2 kind) which))))
    (intern (if *dk-keep-proj* (format nil "~A-~D" base li) base) :keyword)))

(defun dk-compressor-chunk (m st l li c norm pos0 n kind)
  "A compressor (KIND 0 the attention's, 1 the indexer's) over the chunk:
   its projections, every window ending in it pooled, the carried window
   brought up to date; how many rows it pushed."
  (let* ((ratio (dl-ratio l)) (coff (if (= ratio 4) 2 1))
         (hd (if (= kind 0) (ds4f-head-dim m) (ds4f-ix-dim m))) (width (* coff hd))
         (ape (if (= kind 0) (dl-comp-ape l) (dl-ixc-ape l)))
         (st-kv (if (= kind 0) (dc-st-kv c) (dc-ix-kv c))) (st-sc (if (= kind 0) (dc-st-score c) (dc-ix-score c)))
         (rows (if (= kind 0) (dc-comp c) (dc-ix-comp c))) (dst0 (if (= kind 0) (dc-n-comp c) (dc-n-ix c)))
         (wstart (* ratio (floor pos0 ratio))) (e-pos (+ pos0 n -1))
         (p0 (+ wstart ratio -1)) (nb (if (> p0 e-pos) 0 (1+ (floor (- e-pos p0) ratio))))
         (pkv (pb st (dk-proj-key kind 0 li) (* n width)))
         (psc (pb st (dk-proj-key kind 1 li) (* n width))))
    (dk-f16mm st (lname li (if (= kind 0) "attn_compressor_kv.weight" "indexer_compressor_kv.weight")) norm pkv n)
    (dk-f16mm st (lname li (if (= kind 0) "attn_compressor_gate.weight" "indexer_compressor_gate.weight")) norm psc n)
    (when (plusp nb)
      (let ((cs (dk-cs-rows m st nil (loop for b below nb collect (- (+ p0 (* b ratio) 1) ratio)) li)))
        (launch (dk-dev st) "d4_comp_pool_chunk" nb hd
                (list hd ratio width (ds4f-eps m) kind (ds4f-n-rot m) dst0 pos0 p0 wstart (f32mat-cols ape)
                      pkv psc (dvec st (f32mat-data ape)) (dvec st st-kv) (dvec st st-sc)
                      (dvec st (if (= kind 0) (dl-comp-norm l) (dl-ixc-norm l))) cs (dvec st rows))
                :shared (* 4 (+ hd 32)))))
    (dk-comp-carry m st l c pos0 n kind pkv psc)
    nb))

(defun dk-attention-chunk (m st l li c pos0 n sa sb cs ics)
  "Layer LI's attention block for the chunk's N tokens from POS0 (their
   RoPE tables CS and ICS): streams SA into SB."
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (hd (ds4f-head-dim m)) (nh (ds4f-n-head m)) (lq (ds4f-lora-q m))
         (nrot (ds4f-n-rot m)) (eps (ds4f-eps m)) (ratio (dl-ratio l)) (swa (ds4f-swa m))
         (groups (ds4f-out-groups m)) (rank (ds4f-lora-o m)) (gdim (* hd (floor nh groups)))
         (cur (pb st :p-cur (* n d))) (norm (pb st :p-norm (* n d))) (pc (pb st :p-pca (* n 32)))
         (qr (pb st :p-qr (* n lq))) (qrn (pb st :p-qrn (* n lq))) (q (pb st :p-q (* n nh hd)))
         (kv0 (pb st :p-kv0 (* n hd))) (heads (pb st :p-heads (* n nh hd)))
         (low (pb st :p-low (* n groups rank))) (aout (pb st :p-aout (* n d)))
         (prev (dc-n-raw c)) (kr (pb st :p-kr (* (+ prev n) hd))) (raw (dvec st (dc-raw c))))
    (dk-hc-pre-chunk m st li :attn sa n cur pc)
    (dk-rms dev cur (dvec st (dl-attn-norm l)) norm d n eps)
    (dk-q8mm st (lname li "attn_q_a.weight") norm qr n)
    (dk-q8mm st (lname li "attn_kv.weight") norm kv0 n)
    (dk-rms dev qr (dvec st (dl-q-a-norm l)) qrn lq n eps)
    (dk-q8mm st (lname li "attn_q_b.weight") qrn q n)
    (dk-rms dev q nil q hd (* n nh) eps)
    (dk-rope-chunk st q cs nh hd nrot n)
    ;; the KV rows: the window's, oldest first, then the chunk's own; the
    ;; chunk's last ones into the window
    (when (plusp prev)
      (launch dev "d4_ring_gather" (ceiling (* prev hd) 256) 256
              (list hd swa (mod (- (dc-raw-total c) prev) swa) 0 prev raw kr)))
    (launch dev "d4_kv_fin" n hd (list hd nrot eps prev kv0 (dvec st (dl-kv-norm l)) cs kr) :shared (* 4 (+ hd 32)))
    (let ((keep (min swa (+ prev n))))
      (launch dev "d4_ring_store" (ceiling (* keep hd) 256) 256
              (list hd swa (mod (- (+ (dc-raw-total c) n) keep) swa) (- (+ prev n) keep) keep raw kr)))
    (let ((raws (make-array n)) (nraw (make-array n)) (ncomp (make-array n)) (vis (make-array n))
          (nc0 (if (dc-comp c) (dc-n-comp c) 0)) (top-k (ds4f-ix-top-k m)))
      (dotimes (tk n)
        (setf (aref raws tk) (max 0 (- (+ prev tk 1) swa)) (aref nraw tk) (min swa (+ prev tk 1))))
      (incf (dc-raw-total c) n)
      (setf (dc-n-raw c) (min swa (+ prev n)))
      ;; the compressors; a token sees the rows of the windows ending at or before it
      (when (plusp ratio)
        (incf (dc-n-comp c) (dk-compressor-chunk m st l li c norm pos0 n 0))
        (when (= ratio 4) (incf (dc-n-ix c) (dk-compressor-chunk m st l li c norm pos0 n 1))))
      (dotimes (tk n)
        (setf (aref ncomp tk) (+ nc0 (floor (+ pos0 tk 1) (max ratio 1)) (- (floor pos0 (max ratio 1))))
              (aref vis tk) (aref ncomp tk)))
      (when (zerop ratio) (fill ncomp 0))
      (let* ((indexed (and (= ratio 4) (> (reduce #'max vis) top-k)))
             (idx (pb st :p-idx (* n top-k))))
        (when indexed
          (let* ((ixh (ds4f-ix-heads m)) (ixd (ds4f-ix-dim m)) (nix (dc-n-ix c))
                 (ixq (pb st :p-ixq (* n ixh ixd))) (ixw (pb st :p-ixw (* n ixh)))
                 (scores (pb st :p-ixs (* n nix))) (bvis (dk-u32s st vis)))
            (dk-f16mm st (lname li "indexer.attn_q_b.weight") qrn ixq n)
            (dk-f16mm st (lname li "indexer.proj.weight") norm ixw n)
            (launch dev "d4_ixq_fin" (list ixh n) ixd (list ixh nrot (/ 1f0 (sqrt (float (* ixh ixd) 1f0))) ixq cs ixw))
            (launch dev "d4_ix_scores" (list (ceiling nix 32) (ceiling n 8)) 128
                    (list n nix ixh ixq ixw (dvec st (dc-ix-comp c)) bvis scores))
            (launch dev "d4_topk_chunk" n 1024 (list nix top-k nix scores idx bvis))
            (dotimes (tk n) (setf (aref ncomp tk) (min top-k (aref vis tk))))))
        (let ((kc (if (dc-comp c) (dvec st (dc-comp c)) kr)))
          (launch dev "d4_attend" n 512
                  (list nh hd (/ 1f0 (sqrt (float hd 1f0))) top-k 0 (if indexed 0 1)
                        q kr kc (dk-u32s st raws) (dk-u32s st nraw) (dk-u32s st ncomp) idx (dvec st (dl-sinks l)) heads)))))
    (dk-rope-chunk st heads ics nh hd nrot n)
    (if *dk-small*
        (dk-q8mm st (lname li "attn_output_a.weight") heads low n :group-rows rank)
     (let ((w (gethash (lname li "attn_output_a.weight") (dk-q8 st))))
      (dotimes (g groups)
        (dk-mm st "d4_mm_q8_0" (cons (cmat-qs w) (* g rank gdim)) (cons (cmat-d w) (* 2 g rank (floor gdim 32)))
               0 rank gdim (cons heads (* 4 g gdim)) (cons low (* 4 g rank)) n
               :x-stride (* nh hd) :y-stride (* groups rank)))))
    (dk-q8mm st (lname li "attn_output_b.weight") low aout n)
    (dk-hc-post-chunk m st aout sa pc sb n)))

(defun dk-ffn-head-chunk (m st l li n sb)
  "The chunk's FFN up to its routing: HC pre, norm, router logits, shared expert."
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (ff (ds4f-ff-exp m)) (ne (ds4f-n-expert m))
         (cur (pb st :p-cur (* n d))) (norm2 (pb st :p-norm2 (* n d))) (pcf (pb st :p-pcf (* n 32)))
         (logits (pb st :p-logits (* n ne))) (sg (pb st :p-sg (* n ff))) (su (pb st :p-su (* n ff)))
         (smid (pb st :p-smid (* n ff))) (sout (pb st :p-sout (* n d))))
    (dk-hc-pre-chunk m st li :ffn sb n cur pcf)
    (dk-rms dev cur (dvec st (dl-ffn-norm l)) norm2 d n (ds4f-eps m))
    (dk-f16mm st (lname li "ffn_gate_inp.weight") norm2 logits n)
    (dk-q8mm st (lname li "ffn_gate_shexp.weight") norm2 sg n)
    (dk-q8mm st (lname li "ffn_up_shexp.weight") norm2 su n)
    (launch dev "d4_swiglu_w" (ceiling (* n ff) 256) 256
            (list n ff (svref (ds4f-clamps m) li) sg su (pb st :p-ones n) smid nil nil))
    (dk-q8mm st (lname li "ffn_down_shexp.weight") smid sout n)))

(defun dk-tid2eid (st li l)
  "Layer LI's hash table (token -> its experts) on the device."
  (let ((key (intern (format nil "P-TID2EID-~D" li) :keyword)))
    (or (getf (dk-bufs st) key)
        (let* ((v (dl-tid2eid l)) (u (make-array (length v) :element-type '(unsigned-byte 32))))
          (dotimes (i (length v)) (setf (aref u i) (ldb (byte 32 0) (aref v i))))
          (let ((b (cuda-alloc (dk-dev st) (* 4 (length u)))))
            (sb-sys:with-pinned-objects (u) (cuda-upload b (sb-sys:vector-sap u) (* 4 (length u))))
            (setf (getf (dk-bufs st) key) b))))))

(defun dk-ffn-routed-chunk (m st l li n toks slots sb sa)
  "The chunk's FFN past its router, routing included, on the device: each
   token's experts and weights (D4_ROUTE), the rows grouped by expert, the
   experts as one indexed product each for gate, up and down (D4_MM_ID,
   from their pool SLOTS), the sums in slot order with the shared expert,
   HC post: SB into SA."
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (ff (ds4f-ff-exp m)) (nu (ds4f-n-used m)) (ne (ds4f-n-expert m))
         (rows (* n nu)) (clamp (svref (ds4f-clamps m) li))
         (gb (dk-gate-bytes st)) (sbytes (dk-slot-bytes st)) (base (dbuf-ptr (dk-pool st)))
         (grb (second (dl-gate-exps l))) (grows (third (dl-gate-exps l)))
         (drb (second (dl-down-exps l))) (drows (third (dl-down-exps l)))
         (sel (pb st :p-sel rows)) (w (pb st :p-w rows)) (counts (pb st :p-counts ne))
         (rank (pb st :p-rank rows)) (bases (pb st :p-bases ne)) (rmap (pb st :p-rmap rows))
         (smap (pb st :p-smap rows)) (wrow (pb st :p-wrow rows)) (x (pb st :p-x (* rows d)))
         (g (pb st :p-g (* rows ff))) (u (pb st :p-u (* rows ff))) (mid (pb st :p-mid (* rows ff)))
         (down (pb st :p-down (* rows d))) (moe (pb st :p-moe (* n d)))
         (ptrs (dk-mput st (map 'list (lambda (s) (+ base (* s sbytes))) slots) :u64)))
    (launch dev "d4_route" n ne
            (list ne nu (if (dl-tid2eid l) 1 0) (ds4f-expert-scale m) (if (dl-probs-b l) 1 0)
                  (pb st :p-logits (* n ne)) (and (dl-probs-b l) (dvec st (dl-probs-b l)))
                  (and (dl-tid2eid l) (dk-tid2eid st li l)) toks sel w)
            :shared (* 4 (+ (* 2 ne) 64 nu)))
    (launch dev "d4_group_zero" (ceiling ne 256) (min ne 256) (list ne counts))
    (launch dev "d4_group_count" (ceiling rows 256) 256 (list rows ne nu sel counts rank))
    (launch dev "d4_group_scan" 1 ne (list rows ne nu counts bases))
    (launch dev "d4_group_scatter" (ceiling rows 256) 256 (list rows ne nu sel rank bases w rmap smap wrow))
    (launch dev "d4_gather_rows" (ceiling (* rows d) 256) 256 (list d rows (pb st :p-norm2 (* n d)) rmap x))
    (flet ((mm-id (kernel ncols nrows rb add xin yout)
             (launch dev kernel (list (ceiling n 64) (ceiling nrows 64) ne) 128
                     (list ncols nrows rb ncols nrows (list :i64 add) ptrs counts bases xin yout))))
      (mm-id "d4_mm_id_iq2_xxs" d grows grb 0 x g)
      (mm-id "d4_mm_id_iq2_xxs" d grows grb gb x u)
      (launch dev "d4_swiglu_w" (ceiling (* rows ff) 256) 256 (list rows ff clamp g u wrow mid nil nil))
      (mm-id "d4_mm_id_q2_k" ff drows drb (* 2 gb) mid down))
    (launch dev "d4_moe_sum_rows" (ceiling (* n d) 256) 256 (list d nu n down smap (pb st :p-sout (* n d)) moe))
    (dk-hc-post-chunk m st moe sb (pb st :p-pcf (* n 32)) sa n)))

(defvar *dk-stage* nil
  "During a pass: the staging slots free to take ((slot ...)), the only slots
   its copies go into -- the pool's least recently used, so the experts decode
   keeps there are left where they are.")

(defun d4-stage-slots (st k exclude)
  "The K least recently used slots of the pool not in run EXCLUDE (first . count), emptied for staging."
  (let* ((n (dk-n-slots st)) (used (dk-used st)) (pin (dk-pin st))
         (cands (loop for s below n
                      unless (or (and (>= s (car exclude)) (< s (+ (car exclude) (cdr exclude)))) (plusp (aref pin s)))
                        collect s)))
    ;; emptied: what they held would otherwise be taken as held by a layer
    (let ((stage (subseq (sort cands #'< :key (lambda (s) (aref used s))) 0 k)) (owner (dk-owner st)))
      (dolist (s stage stage)
        (when (>= (aref owner s) 0)
          (setf (aref (dk-slot-of st) (aref owner s)) -1 (aref owner s) -1))))))

(defun d4-take-layer (m st li)
  "Slots for all of layer LI's experts, held (pinned) until let go: those
   the pool holds where they are, the others copied into staging slots, the
   copies queued on the copy stream.  (values slots event staged)."
  (let* ((ne (ds4f-n-expert m)) (slot-of (dk-slot-of st)) (pin (dk-pin st)) (used (dk-used st))
         (owner (dk-owner st)) (tick (incf (dk-tick st))) (slots (make-array ne :initial-element -1)) (staged '()))
    (declare (type (simple-array fixnum (*)) slot-of pin used owner))
    (dotimes (e ne)
      (let ((key (+ (* li ne) e)))
        (let ((s (aref slot-of key)))
          (if (>= s 0)
              (progn (incf (aref pin s)) (setf (aref used s) tick) (incf (dk-hits st)))
              (progn
                (setf s (or (pop (car *dk-stage*)) (error "a pass is out of staging slots")))
                (when (>= (aref owner s) 0) (setf (aref slot-of (aref owner s)) -1))
                ;; low priority once let go: decode's misses take these first
                (setf (aref owner s) key (aref slot-of key) s (aref used s) 0)
                (incf (aref pin s)) (incf (dk-misses st)) (push s staged)
                (d4-copy-expert m st li e s)))
          (setf (aref slots e) s))))
    (let ((ev (cuda-event-create (dk-dev st))))
      (cuda-event-record ev (dk-copy-stream st))
      (values slots ev staged))))

(defun d4-let-go (st slots staged)
  (loop for s across slots do (decf (aref (dk-pin st) s)))
  (setf (car *dk-stage*) (append staged (car *dk-stage*))))

(defun dk-ready (m st ctx)
  "What is copied in once (the hash tables, the layers' vectors, a fresh
   context's caches), before a pass queues anything: behind its copies a
   copy on the compute stream holds up its layer."
  (loop for l across (ds4f-layers m) for li from 0 when (dl-tid2eid l) do (dk-tid2eid st li l))
  (loop for l across (ds4f-layers m)
        do (dolist (v (list (dl-attn-norm l) (dl-q-a-norm l) (dl-kv-norm l) (dl-ffn-norm l) (dl-sinks l)
                            (dl-hc-attn-scale l) (dl-hc-attn-base l) (dl-hc-ffn-scale l) (dl-hc-ffn-base l)
                            (dl-comp-norm l) (dl-ixc-norm l) (dl-probs-b l)
                            (and (dl-comp-ape l) (f32mat-data (dl-comp-ape l)))
                            (and (dl-ixc-ape l) (f32mat-data (dl-ixc-ape l)))))
             (when v (dvec st v))))
  (loop for c across (d4ctx-caches ctx)
        do (dolist (v (list (dc-raw c) (dc-comp c) (dc-st-kv c) (dc-st-score c)
                            (dc-ix-comp c) (dc-ix-kv c) (dc-ix-score c)))
             (when v (dvec st v)))))

(defun ds4f-cuda-prefill-1 (ctx tokens logits)
  "DS4F-PREFILL-1 on the device: the chunk TOKENS at the next positions,
   layer-major (at most *DS4F-CUDA-CHUNK* of them)."
  (let* ((m (d4ctx-model ctx)) (st (ds4f-cuda-state m)) (dev (dk-dev st))
         (d (ds4f-d m)) (n-hc (ds4f-n-hc m)) (hcd (* n-hc d)) (pos0 (d4ctx-pos ctx))
         (n (length tokens)) (tokv (coerce tokens 'simple-vector)) (wf (ds4f-file m))
         (nl (ds4f-n-layer m)) (positions (loop for i below n collect (+ pos0 i))))
    (declare (type u8vec wf))
    (with-cuda
      (cuda-current dev)
      (let* ((run (d4-borrow-slots st (+ (* n *ds4f-cuda-chunk-bytes*) (* 64 1024 1024))))
             (base (+ (dbuf-ptr (dk-pool st)) (* (car run) (dk-slot-bytes st))))
             (*dk-scratch* (list base (+ base (* (cdr run) (dk-slot-bytes st))) (make-hash-table)))
             (*dk-stage* (list (d4-stage-slots st (* 2 (ds4f-n-expert m)) run)))
             (*dk-mapped* (cons (or (getf (dk-bufs st) :p-mapped-sap)
                                    (setf (getf (dk-bufs st) :p-mapped-sap) (cuda-host-alloc +dk-mapped-bytes+ :mapped t)))
                                0))
             (sa (pb st :p-sa (* n hcd))) (sb (pb st :p-sb (* n hcd))))
        ;; the tokens' embeddings into each stream
        (let ((e (f32vec (* n hcd))))
          (dotimes (tk n)
            (let ((o (+ (ds4f-embd-off m) (* (svref tokv tk) d 2))))
              (dotimes (j d)
                (let ((v (f16-at wf (+ o (* 2 j)))))
                  (dotimes (h n-hc) (setf (aref e (+ (* tk hcd) (* h d) j)) v))))))
          (cuda-upload-vector sa e))
        (let ((ones (f32vec n))) (fill ones 1f0) (cuda-upload-vector (pb st :p-ones n) ones))
        (let* ((toks (let ((b (pb st :p-tok n)) (u (make-array n :element-type '(unsigned-byte 32))))
                       (dotimes (i n) (setf (aref u i) (svref tokv i)))
                       (sb-sys:with-pinned-objects (u) (cuda-upload b (sb-sys:vector-sap u) (* 4 n)))
                       b))
               ;; the RoPE tables: a layer's depend only on whether it is compressed
               (li-plain (position 0 (ds4f-ratios m))) (li-comp (position-if #'plusp (ds4f-ratios m)))
               (tables (loop for li in (list li-plain li-comp) for k from 0
                             collect (and li (cons (dk-cs-rows m st (if (zerop k) :p-cs0 :p-cs1) positions li)
                                                   (dk-cs-rows m st (if (zerop k) :p-ics0 :p-ics1) positions li t))))))
          (dk-ready m st ctx)
          (multiple-value-bind (slots ev staged) (d4-take-layer m st 0)
            (dotimes (li nl)
              (let* ((l (svref (ds4f-layers m) li)) (c (svref (d4ctx-caches ctx) li))
                     (tab (if (plusp (dl-ratio l)) (second tables) (first tables))))
                ;; the next layer's experts come in while this one computes
                (multiple-value-bind (nslots nev nstaged)
                    (if (< (1+ li) nl) (d4-take-layer m st (1+ li)) (values nil nil nil))
                  (dk-attention-chunk m st l li c pos0 n sa sb (car tab) (cdr tab))
                  (dk-ffn-head-chunk m st l li n sb)
                  (cuda-stream-wait (cudev-stream dev) ev)
                  (dk-ffn-routed-chunk m st l li n toks slots sb sa)
                  (cuda-sync dev)
                  (dk-arena-reset st)
                  (setf (cdr *dk-mapped*) 0)
                  (cuda-event-destroy ev)
                  (d4-let-go st slots staged)
                  (setf slots nslots ev nev staged nstaged)))))
          (incf (d4ctx-pos ctx) n)
          (when logits
            (let ((out (dk-head m st (cons sa (* 4 (1- n) hcd)))) (v (d4ctx-logits ctx)))
              (sb-sys:with-pinned-objects (v)
                (cuda-download out (sb-sys:vector-sap v) (* 4 (ds4f-n-vocab m)))))
            (cuda-sync dev)))
        (d4-return-slots st run)))
    ctx))
