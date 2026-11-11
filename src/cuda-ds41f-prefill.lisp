;;;; cuda-ds41f-prefill.lisp — a V4.1 prefill chunk on the GPU, a layer at a
;;;; time; and the verifying pass of its speculative steps.
;;;
;;; cuda-ds4f-prefill.lisp's pass (its scratch in borrowed pool slots, the
;;; routing on the device, the experts grouped by expert into one indexed
;;; product each; a long pass takes all of a layer's experts ahead into
;;; staging slots while the layer before computes, a short one only those
;;; its tokens route to, after its routing) with V4.1's layers: the rounding,
;;; HC's pre a sublayer late, Engram's rows hashed token by token on the CPU
;;; and gated in, the shared compressed caches written by their source
;;; layers (ratio-2 pairs that may start before the chunk, ratio-1 rows), the
;;; indexer's pick for each token made on the index-source layers and kept
;;; for the layers after.  The CPU sees a token's state only at the chunk's
;;; end: the context's position, Engram's tail, the pair carried over.

(in-package #:illatio)

(defparameter *ds41f-cuda-chunk-from* 64
  "From this many tokens a V4.1 prompt goes through layer-major passes: a
   pass copies in the experts its tokens use that the pool lacks (64
   tokens about 2.9 s, a long pass every one, about 6 s), against about
   50-90 ms a token decoding (with the misses).")

(defparameter *ds41f-cuda-chunk* nil
  "Tokens a V4.1 pass takes at most; NIL, as many as the pool leaves room
   for -- the experts' copies cost a pass the same whatever its length, so
   the fewer passes the better.")

(defparameter *ds41f-cuda-routed-below* 1024
  "A pass of fewer tokens takes only the experts its tokens route to, each
   layer's after its routing (no copies under the compute, but a short
   pass's tokens pick half of them or fewer); a longer one takes all of the
   next layer's while one computes.")

(defparameter *ds41f-cuda-chunk-bytes* (* 1400 1024)
  "Device memory a token of a V4.1 pass takes, at most (its activations, the
   indexer's scores apart).")

(defun dk41-cs-rows (m st key positions li &optional inverse)
  "D41-ROPE-CS's pairs for each of POSITIONS, a row each, uploaded into
   scratch KEY (before the pass queues anything)."
  (let* ((nrot (ds4f-n-rot m)) (n (length positions)) (v (f32vec (* (max n 1) nrot))))
    (loop for p in positions for k from 0
          do (replace v (d41-rope-cs m p li inverse) :start1 (* k nrot)))
    (let ((b (pb st key (* (max n 1) nrot)))) (cuda-upload-vector b v) b)))

(defun dk41-hc-chunk (m st li which streams n pre pstride out pc)
  "HC's mix of N tokens' STREAMS (layer LI's WHICH, :attn or :ffn) into PC
   (32 a token, its own pre at 24), OUT the streams summed by PRE (PSTRIDE
   a token)."
  (let* ((l (svref (ds4f-layers m) li)) (dev (dk-dev st)) (d (ds4f-d m)) (nh (ds4f-n-hc m)) (hcd (* nh d))
         (flat (pb st :p-flat (* n hcd))) (mix (pb st :p-mix (* n 32))))
    (dk-rms dev streams nil flat hcd n (ds4f-eps m))
    (dk-f16mm st (lname li (if (eq which :attn) "hc_attn_fn.weight" "hc_ffn_fn.weight")) flat mix n :y-stride 32)
    (launch dev "d41_hc_split" n 32
            (list (ds4f-sinkhorn m) 1f-6 mix
                  (dvec st (if (eq which :attn) (dl-hc-attn-scale l) (dl-hc-ffn-scale l)))
                  (dvec st (if (eq which :attn) (dl-hc-attn-base l) (dl-hc-ffn-base l))) pc))
    (launch dev "d41_hc_sum" (list (ceiling d 256) n) 256 (list d nh pstride streams pre out))))

(defun dk41-rms-rows (dev x w out width n eps)
  (launch dev "d41_rms" n (min 1024 width) (list width eps x w out nil nil)))

(defvar *dk41-carry-rows* nil
  "During a verifying pass: a table, each ratio-2 kv-source layer to a
   buffer of its tokens' compressor projections (kv rows, then score rows).")

(defun dk41-publish-chunk (m st l li ctx norm pos0 n k)
  "A kv-source layer's compressor over the chunk: the rows that finish in
   it (ratio 1: every token's; ratio 2: each pair ending at an odd
   position) normed, keyed, turned and rounded into owner K's caches; the
   pair's first half carried when the chunk ends on an even position."
  (let* ((dev (dk-dev st)) (hd (ds4f-head-dim m)) (ixd (ds4f-ix-dim m)) (ratio (dl-ratio l)) (eps (ds4f-eps m))
         (nrot (ds4f-n-rot m)) (e (+ pos0 n -1))
         (pkv (pb st :p-ckv (* n hd)))
         (comp (dvec st (svref (d41ctx-comp ctx) k))) (keys (dvec st (svref (d41ctx-ix-comp ctx) k)))
         (latent nil) (row0 0) (nb 0) (cs nil))
    (dk-f16mm st (lname li "attn_compressor_kv.weight") norm pkv n)
    (if (= ratio 1)
        (progn (dk41-round dev pkv (* n hd))
               (setf latent pkv row0 pos0 nb n cs (pb st :p-cs1 0)))
        (let* ((psc (pb st :p-csc (* n hd))) (p1 (if (oddp pos0) pos0 (1+ pos0)))
               (ckv (dvec st (svref (d41ctx-prev-kv ctx) k))) (csc (dvec st (svref (d41ctx-prev-score ctx) k))))
          (dk-f16mm st (lname li "attn_compressor_gate.weight") norm psc n)
          ;; pair B ends at P1 + 2B: compressed row (P1 + 1)/2 - 1 + B
          (setf nb (if (> p1 e) 0 (1+ (floor (- e p1) 2))) row0 (1- (floor (1+ p1) 2))
                latent (pb st :p-clat (* (max nb 1) hd)) cs (pb st :p-cs2 0))
          (when (plusp nb)
            (launch dev "d41_pool2_chunk" nb hd (list hd pos0 p1 pkv psc ckv csc latent)))
          ;; a verifying pass's, for its undo to carry a pair's first token again
          (when *dk41-carry-rows*
            (let ((b (pb st (intern (format nil "V-CARRY-~D" li) :keyword) (* 2 n hd))))
              (cuda-copy-dd b 0 pkv 0 (* 4 n hd))
              (cuda-copy-dd b (* 4 n hd) psc 0 (* 4 n hd))
              (setf (gethash li *dk41-carry-rows*) b)))
          ;; the chunk ends on a pair's first token: it is carried
          (when (evenp e)
            (cuda-copy-dd ckv 0 pkv (* 4 (- e pos0) hd) (* 4 hd))
            (cuda-copy-dd csc 0 psc (* 4 (- e pos0) hd) (* 4 hd)))))
    (when (plusp nb)
      (let ((rows (cons comp (* 4 row0 hd))) (krows (cons keys (* 4 row0 ixd))) (ik (pb st :p-ik (* nb ixd))))
        (dk41-rms-rows dev latent (dvec st (dl-comp-norm l)) rows hd nb eps)
        (dk-f16mm st (lname li "indexer.attn_k.weight") rows ik nb)
        (dk41-round dev ik (* nb ixd))
        (dk41-rms-rows dev ik (dvec st (dl-ix-k-norm l)) krows ixd nb eps)
        (dk41-rope dev krows cs 1 ixd nrot nb)
        (dk41-quant dev krows (* nb ixd) :fp4)
        (dk41-rope dev rows cs 1 hd nrot nb)
        (dk41-quant dev rows (* nb hd) :fp4-e4m3)))))

(defvar *dk41-image-rows* nil
  "In a pass with a picture's positions: a vector, true at each of them.")

(defvar *dk41-pick* nil
  "During a pass: whether the last index-source layer made a pick (its rows
   in :P-IDX), for the layers after it.")

(defun dk41-attention-chunk (m st l li ctx pos0 n sa sb pre pstride cs ics)
  "Layer LI's attention for the chunk's N tokens from POS0: streams SA into SB."
  (let* ((v (ds4f-v41 m)) (dev (dk-dev st)) (d (ds4f-d m)) (hd (ds4f-head-dim m)) (nh (ds4f-n-head m))
         (lq (ds4f-lora-q m)) (nrot (ds4f-n-rot m)) (eps (ds4f-eps m)) (ratio (dl-ratio l)) (swa (ds4f-swa m))
         (top-k (ds4f-ix-top-k m)) (groups (ds4f-out-groups m)) (rank (ds4f-lora-o m)) (gdim (* hd (floor nh groups)))
         (cur (pb st :p-cur (* n d))) (norm (pb st :p-norm (* n d))) (pca (pb st :p-pca (* n 32)))
         (qr (pb st :p-qr (* n lq))) (qrn (pb st :p-qrn (* n lq))) (q (pb st :p-q (* n nh hd)))
         (kv0 (pb st :p-kv0 (* n hd))) (heads (pb st :p-heads (* n nh hd)))
         (low (pb st :p-low (* n groups rank))) (aout (pb st :p-aout (* n d)))
         (prev (min pos0 swa)) (kr (pb st :p-kr (* (+ prev n) hd)))
         (raw (dvec st (svref (d41ctx-raw ctx) li))) (ring (d41ctx-ring ctx)) (idx (pb st :p-idx (* n top-k)))
         (k nil) (ident 1) (ncomp (make-array n :initial-element 0)))
    (dk41-hc-chunk m st li :attn sa n pre pstride cur pca)
    (dk41-rms-rows dev cur (dvec st (dl-attn-norm l)) norm d n eps)
    (dk-q8mm st (lname li "attn_q_a.weight") norm qr n)
    (dk-q8mm st (lname li "attn_kv.weight") norm kv0 n)
    (dk41-round dev qr (* n lq))
    (dk41-round dev kv0 (* n hd))
    (dk41-rms-rows dev qr (dvec st (dl-q-a-norm l)) qrn lq n eps)
    (dk-q8mm st (lname li "attn_q_b.weight") qrn q n)
    (dk41-round dev q (* n nh hd))
    (dk41-rope dev q cs nh hd nrot n)
    ;; the KV rows: the window's, oldest first, then the chunk's own (normed,
    ;; turned, through FP8); the last ones into the window
    (when (plusp prev)
      (launch dev "d4_ring_gather" (ceiling (* prev hd) 256) 256 (list hd ring (mod (- pos0 prev) ring) 0 prev raw kr)))
    (let ((rows (cons kr (* 4 prev hd))))
      (dk41-rms-rows dev kv0 (dvec st (dl-kv-norm l)) rows hd n eps)
      (dk41-rope dev rows cs 1 hd nrot n)
      (dk41-quant dev rows (* n hd) :fp8))
    (let ((keep (min ring (+ prev n))))
      (launch dev "d4_ring_store" (ceiling (* keep hd) 256) 256
              (list hd ring (mod (- (+ pos0 n) keep) ring) (- (+ prev n) keep) keep raw kr)))
    (when (plusp ratio)
      (setf k (d41-owner m li))
      (when (member li (d41-kv-src v)) (dk41-publish-chunk m st l li ctx norm pos0 n k))
      (let ((vis (map 'vector (lambda (tk) (floor (+ pos0 tk 1) ratio)) (loop for tk below n collect tk))))
        (when (member li (d41-ix-src v))
          (setf *dk41-pick* (> (aref vis (1- n)) top-k))
          (when *dk41-pick*
            (let* ((ixh (ds4f-ix-heads m)) (ixd (ds4f-ix-dim m)) (nc (aref vis (1- n)))
                   (ixq (pb st :p-ixq (* n ixh ixd))) (ixw (pb st :p-ixw (* n ixh)))
                   (scores (pb st :p-ixs (* n nc))) (bvis (dk-u32s st vis)))
              (dk-f16mm st (lname li "indexer.attn_q_b.weight") qrn ixq n)
              (dk41-round dev ixq (* n ixh ixd))
              (dk41-rope dev ixq cs ixh ixd nrot n)
              (dk41-quant dev ixq (* n ixh ixd) :fp4)
              (dk-f16mm st (lname li "indexer.proj.weight") norm ixw n)
              (dk41-round dev ixw (* n ixh))
              (if (and (>= (cudev-arch dev) 80) (= ixh 32))
                  ;; on the tensor cores: Q and the keys are BF16 values
                  (launch dev "d41_ix_scores_mma" (list (ceiling nc 128) (ceiling n 32)) 256
                          (list nc n ixq ixw (dvec st (svref (d41ctx-ix-comp ctx) k)) bvis scores)
                          :shared 60416)
                  (launch dev "d41_ix_scores_chunk" (list (ceiling nc 8) n) 256
                          (list ixh nc ixq ixw (dvec st (svref (d41ctx-ix-comp ctx) k)) bvis scores)
                          :shared (* 4 ixh ixd)))
              (dk41-candidates m st li n nc bvis 0 scores)
              (launch dev "d4_topk_chunk" n 1024 (list nc top-k nc scores idx bvis)))))
        (when *dk41-pick* (setf ident 0))
        (dotimes (tk n) (setf (aref ncomp tk) (min top-k (aref vis tk))))))
    (let ((raws (make-array n)) (nraw (make-array n)))
      (dotimes (tk n) (setf (aref raws tk) (max 0 (- (+ prev tk 1) swa)) (aref nraw tk) (min swa (+ prev tk 1))))
      ;; on the tensor cores where there are some: Q and the keys are BF16 values
      (multiple-value-call #'launch dev
        (if (>= (cudev-arch dev) 80) (values "d41_attend_mma" n 256) (values "d4_attend" n 512))
        (list nh hd (/ 1f0 (sqrt (float hd 1f0))) top-k 0 ident
              q kr (if k (dvec st (svref (d41ctx-comp ctx) k)) kr)
              (dk-u32s st raws) (dk-u32s st nraw) (dk-u32s st ncomp) idx (dvec st (dl-sinks l)) heads)
        :shared (if (>= (cudev-arch dev) 80) 94464 0)))
    (dk41-round dev heads (* n nh hd))
    (dk41-rope dev heads ics nh hd nrot n)
    (dk41-round dev heads (* n nh hd))
    (if *dk-small*
        ;; decode's grouped product: row r against its token's group r/RANK
        (dk-q8mm st (lname li "attn_output_a.weight") heads low n :group-rows rank)
        (let ((w (gethash (lname li "attn_output_a.weight") (dk-q8 st))))
          (dotimes (g groups)
            (dk-mm st "d4_mm_q8_0" (cons (cmat-qs w) (* g rank gdim)) (cons (cmat-d w) (* 2 g rank (floor gdim 32)))
                   0 rank gdim (cons heads (* 4 g gdim)) (cons low (* 4 g rank)) n
                   :x-stride (* nh hd) :y-stride (* groups rank)))))
    (dk41-round dev low (* n groups rank))
    (dk-q8mm st (lname li "attn_output_b.weight") low aout n)
    (dk41-round dev aout (* n d))
    (dk-hc-post-chunk m st aout sa pca sb n (cons li :attn))
    (dk41-round dev sb (* n (ds4f-n-hc m) d))))

(defun dk41-ffn-chunk (m st l li n toks slots sb sa &optional small)
  "Layer LI's FFN for the chunk: HC (summed by the attention's pre), norm,
   router, shared expert, then V4's routing and grouped experts on the
   device: SB into SA."
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (ff (ds4f-ff-exp m)) (ne (ds4f-n-expert m))
         (cur (pb st :p-cur (* n d))) (norm2 (pb st :p-norm2 (* n d))) (pcf (pb st :p-pcf (* n 32)))
         (logits (pb st :p-logits (* n ne))) (sg (pb st :p-sg (* n ff))) (su (pb st :p-su (* n ff)))
         (smid (pb st :p-smid (* n ff))) (sout (pb st :p-sout (* n d))))
    (dk41-hc-chunk m st li :ffn sb n (cons (pb st :p-pca (* n 32)) 96) 32 cur pcf)
    (dk41-rms-rows dev cur (dvec st (dl-ffn-norm l)) norm2 d n (ds4f-eps m))
    (destructuring-bind (w rows cols &optional type) (gethash (lname li "ffn_gate_inp.weight") (dk-f16 st))
      (declare (ignore type))
      (if *dk-small*
          (launch dev "d41_mv_f32n" (list (ceiling rows 4) n) 256 (list cols rows cols rows w norm2 logits))
          (launch dev "d41_mm_f32" (list (ceiling n 64) (ceiling rows 64)) 256 (list cols rows n cols rows w norm2 logits))))
    (dk-q8mm st (lname li "ffn_gate_shexp.weight") norm2 sg n)
    (dk-q8mm st (lname li "ffn_up_shexp.weight") norm2 su n)
    (launch dev "d41_swiglu" (ceiling (* n ff) 256) 256 (list (* n ff) (svref (ds4f-clamps m) li) sg su smid nil nil))
    (dk-q8mm st (lname li "ffn_down_shexp.weight") smid sout n)
    (dk41-round dev sout (* n d))
    ;; V4's routed half (no hash layers here); then the sum and the streams rounded.
    ;; A picture's positions route with the layer's vision bias: a row a token then.
    (let ((*dk41-round-moe* t)
          (*dk-route-bias*
            (and *dk41-image-rows* (dl-probs-b-vl l)
                 (let ((rows (f32vec (* n ne))) (b (dl-probs-b l)) (bv (dl-probs-b-vl l)))
                   (dotimes (i n) (replace rows (if (svref *dk41-image-rows* i) bv b) :start1 (* i ne)))
                   (let ((buf (pb st :p-route-bias (* n ne)))) (cuda-upload-vector buf rows) buf)))))
      (if small
          (dk-ffn-small m st l li n small sb sa)   ; SMALL: the tokens; the slots it holds
          (dk-ffn-routed-chunk m st l li n toks slots sb sa)))))

(defun dk41-engram-inputs (m st ctx tokens)
  "Engram's table rows for each of TOKENS (hashed in turn from CTX's tail),
   their bytes uploaded and decoded on the device (D41_EG_DECODE): a buffer
   a table, 24 x 256 a token; and the tail after them."
  (let* ((v (ds4f-v41 m)) (n (length tokens)) (ntab (length (d41-eg-layers v))) (rb 264) (tw (* 24 rb))
         (wf (d41-eg-file v)) (raws (loop repeat ntab collect (make-array (* n tw) :element-type '(unsigned-byte 8))))
         (tail (d41ctx-tail ctx)) (dev (dk-dev st)))
    (declare (type u8vec wf))
    (loop for tk in tokens for i from 0
          do (multiple-value-bind (rows nt) (d41-engram-hash m tail tk)
               (loop for tbl below ntab for raw in raws for base in (d41-eg-offs v)
                     do (loop for r in (nth tbl rows) for col from 0
                              do (let ((o (+ base (* r rb))))
                                   (replace raw wf :start1 (+ (* i tw) (* col rb)) :start2 o :end2 (+ o rb)))))
               (setf tail nt)))
    (values (loop for raw in raws for tbl from 0
                  collect (let ((b (pb st (intern (format nil "P-EGB~D" tbl) :keyword) (ceiling (* n tw) 4)))
                                (d (pb st (intern (format nil "P-EG~D" tbl) :keyword) (* n 24 256))))
                            (sb-sys:with-pinned-objects (raw) (cuda-upload b (sb-sys:vector-sap raw) (* n tw)))
                            (launch dev "d41_eg_decode" (ceiling (* n 24 256) 256) 256
                                    (list (* n 24) b (dvec st +d41-e4m3-codes+) d))
                            d))
            tail)))

(defun dk41-ready (m st ctx)
  "The layers' vectors and the context's caches on the device before a pass queues anything."
  (loop for l across (ds4f-layers m)
        do (dolist (v (list (dl-attn-norm l) (dl-q-a-norm l) (dl-kv-norm l) (dl-ffn-norm l) (dl-sinks l)
                            (dl-hc-attn-scale l) (dl-hc-attn-base l) (dl-hc-ffn-scale l) (dl-hc-ffn-base l)
                            (dl-comp-norm l) (dl-ix-k-norm l) (dl-probs-b l) (dl-probs-b-vl l) (dl-eg-q-norm l) (dl-eg-k-norm l)))
             (when v (dvec st v))))
  (dolist (vs (list (d41ctx-raw ctx) (d41ctx-comp ctx) (d41ctx-ix-comp ctx) (d41ctx-prev-kv ctx) (d41ctx-prev-score ctx)))
    (loop for v across vs do (dvec st v)))
  (dvec st (ds4f-output-norm m)))

(defun ds41f-cuda-prefill-1 (ctx tokens logits &key small)
  "TOKENS at CTX's next positions, layer-major on the device (at most
   DS41F-CUDA-CHUNK-MAX of them); the last one's logits when LOGITS.  SMALL: a
   verifying pass (at most +DV-NT+ tokens), decode's products and experts
   (DK-FFN-SMALL), its scratch the device's own, its compressor projections
   kept (*DK41-CARRY-ROWS*); every token's logits, a vector."
  (let* ((m (d41ctx-model ctx)) (v (ds4f-v41 m)) (st (ds4f-cuda-state m)) (dev (dk-dev st))
         (d (ds4f-d m)) (n-hc (ds4f-n-hc m)) (hcd (* n-hc d)) (pos0 (d41ctx-pos ctx))
         (n (length tokens)) (tokv (coerce tokens 'simple-vector))
         (nl (ds4f-n-layer m)) (positions (loop for i below n collect (+ pos0 i)))
         (routed (< n *ds41f-cuda-routed-below*)))
    (with-cuda
      (cuda-current dev)
      (with-dk-pass (st (and (not small) (ds41f-pass-bytes n pos0)) :stage (* 2 (ds4f-n-expert m)))
       (let* ((*dk-small* small) (*dk41-carry-rows* (and small (or *dk41-carry-rows* (make-hash-table))))
              (held '()) (rows nil) (*dk41-pick* nil)
              (*dk41-image-rows* (and (some #'image-token-p tokv) (map 'simple-vector #'image-token-p tokv)))
              ;; a byte a token, 0 at a picture's: Engram passes those by
              (egmask (and *dk41-image-rows*
                           (let ((b (pb st :p-egmask (ceiling n 4)))
                                 (v (make-array (* 4 (ceiling n 4)) :element-type '(unsigned-byte 8) :initial-element 0)))
                             (dotimes (i n) (setf (aref v i) (if (svref *dk41-image-rows* i) 0 1)))
                             (sb-sys:with-pinned-objects (v) (cuda-upload b (sb-sys:vector-sap v) (length v)))
                             b)))
              (sa (pb st :p-sa (* n hcd))) (sb (pb st :p-sb (* n hcd))))
        (dk41-ready m st ctx)
        ;; the tokens' embeddings into each stream
        (cuda-upload-vector sa (ds4f-embed! m tokv (f32vec (* n hcd))))
        (let* ((toks (dk-upload-tokens st tokv))
               (li-plain (position 0 (ds4f-ratios m))) (li-comp (position-if #'plusp (ds4f-ratios m)))
               (cs0 (dk41-cs-rows m st :p-cs0 positions li-plain)) (ics0 (dk41-cs-rows m st :p-ics0 positions li-plain t))
               (cs1 (dk41-cs-rows m st :p-cs1 positions li-comp)) (ics1 (dk41-cs-rows m st :p-ics1 positions li-comp t))
               (pre0 (let ((b (pb st :p-pre0 4)) (p (f32vec 4))) (setf (aref p 0) 1f0) (cuda-upload-vector b p) b))
               (pre pre0) (pstride 0))
          ;; the ratio-2 pairs' first positions, for their rows' RoPE
          (let ((p1 (if (oddp pos0) pos0 (1+ pos0))))
            (dk41-cs-rows m st :p-cs2 (loop for p from p1 below (+ pos0 n) by 2 collect (1- p)) li-comp))
          (multiple-value-bind (eg tail) (dk41-engram-inputs m st ctx tokens)
            (multiple-value-bind (slots ev staged) (if (or routed small) (values nil nil nil) (d4-take-layer m st 0))
              (dotimes (li nl)
                (let* ((l (svref (ds4f-layers m) li)) (tbl (position li (d41-eg-layers v)))
                       (comp (plusp (dl-ratio l))))
                  (multiple-value-bind (nslots nev nstaged)
                      (if (and (< (1+ li) nl) (not routed) (not small)) (d4-take-layer m st (1+ li)) (values nil nil nil))
                    (when tbl
                      (let ((kv (pb st :p-egkv (* n (1+ n-hc) d))))
                        (dk-f16mm st (lname li "engram_kv.weight") (nth tbl eg) kv n)
                        (dk41-round dev kv (* n (1+ n-hc) d))
                        (launch dev "d41_engram" (list n-hc n) 1024
                                (list d n-hc (ds4f-eps m) sa kv (dvec st (dl-eg-q-norm l)) (dvec st (dl-eg-k-norm l))
                                      egmask))))
                    (dk41-attention-chunk m st l li ctx pos0 n sa sb pre pstride
                                          (if comp cs1 cs0) (if comp ics1 ics0))
                    (cond
                      (small
                       ;; (it syncs: the layer before is done with its slots, and the mapped arena)
                       (let ((now (dk41-ffn-chunk m st l li n toks nil sb sa tokv)))
                         (dolist (s held) (decf (aref (dk-pin st) s)))
                         (setf held now)))
                      (routed
                        (dk41-ffn-chunk m st l li n toks
                                        (lambda (sel)
                                          (multiple-value-setq (slots ev staged) (d4-take-routed m st li sel n))
                                          slots)
                                        sb sa))
                      (t (cuda-stream-wait (cudev-stream dev) ev)
                         (dk41-ffn-chunk m st l li n toks slots sb sa)))
                    (unless small
                      (cuda-sync dev)
                      (dk-arena-reset st))
                    (setf (cdr *dk-mapped*) 0)
                    (when ev (cuda-event-destroy ev))
                    (when slots (d4-let-go st slots staged))
                    (setf slots nslots ev nev staged nstaged
                          pre (cons (pb st :p-pcf (* n 32)) 96) pstride 32)))))
            (setf (d41ctx-tail ctx) tail))
          (incf (d41ctx-pos ctx) n)
          (when small
            ;; every token's logits: the output weights read once (DK-Q8MM as decode's products)
            (let* ((nv (ds4f-n-vocab m)) (embd (pb st :v-hembd (* n d))) (norm (pb st :v-hnorm (* n d)))
                   (out (pb st :v-out (* n nv))))
              (launch dev "d41_hc_sum" (list (ceiling d 256) n) 256
                      (list d n-hc 32 sa (cons (pb st :p-pcf (* n 32)) (* 4 24)) embd))
              (dk41-rms-rows dev embd (dvec st (ds4f-output-norm m)) norm d n (ds4f-eps m))
              (dk-q8mm st "output.weight" norm out n)
              (setf rows (f32vec (* n nv)))
              (sb-sys:with-pinned-objects (rows)
                (cuda-download out (sb-sys:vector-sap rows) (* 4 n nv)))
              (dolist (s held) (decf (aref (dk-pin st) s)))))
          (when (and logits (not small))
            (let* ((q (dk-qbuf st :hnorm d)) (embd (dbuf-of st :hembd (* 4 d))) (norm (dbuf-of st :hnorm (* 4 d)))
                   (out (dbuf-of st :out (* 4 (ds4f-n-vocab m)))) (lv (d41ctx-logits ctx)))
              (launch dev "d41_hc_sum" (ceiling d 256) 256
                      (list d n-hc 0 (cons sa (* 4 (1- n) hcd)) (cons (pb st :p-pcf (* n 32)) (* 4 (+ (* 32 (1- n)) 24))) embd))
              (dk41-rms dev embd (dvec st (ds4f-output-norm m)) norm d (ds4f-eps m) q)
              (dk-q8mv st "output.weight" norm out :q q)
              (sb-sys:with-pinned-objects (lv)
                (cuda-download out (sb-sys:vector-sap lv) (* 4 (ds4f-n-vocab m))))))
          (cuda-sync dev)
          (dk-arena-reset st))
        (or rows ctx))))))

;;; --- speculative steps ---------------------------------------------------------------
;;; DS4F-SPECULATE's verifying pass for V4.1: the token and its guesses as a
;;; small pass, then what the guesses past the kept ones wrote undone.  The
;;; ring (*D41-RING* rows, the window SWA) keeps the window's rows; compressed rows
;;; past the kept tokens are never read before they are written again; so it
;;; is the Engram tail and, when the next position is a pair's second, the
;;; pair's first token carried again (its projections, kept by the pass).

(defun ds41f-cuda-undo (ctx pos0 keep n tail0 kept)
  "CTX after a verifying pass of N tokens from POS0, as if only its first
   KEEP (the tokens KEPT) had gone through; TAIL0 the Engram tail before."
  (let* ((m (d41ctx-model ctx)) (st (ds4f-cuda-state m)) (hd (ds4f-head-dim m))
         (map (d41-token-map (ds4f-v41 m))) (tail tail0))
    (dolist (tk kept) (setf tail (list (if (image-token-p tk) -1 (aref map tk)) (first tail) (second tail))))
    (when (and (< keep n) (oddp (+ pos0 keep)))
      (maphash (lambda (li b)
                 (let ((k (d41-owner m li)))
                   (cuda-copy-dd (dvec st (svref (d41ctx-prev-kv ctx) k)) 0 b (* 4 (1- keep) hd) (* 4 hd))
                   (cuda-copy-dd (dvec st (svref (d41ctx-prev-score ctx) k)) 0 b (* 4 (+ (* n hd) (* (1- keep) hd))) (* 4 hd))))
               *dk41-carry-rows*))
    (setf (d41ctx-tail ctx) tail
          (d41ctx-written ctx) (max (d41ctx-written ctx) (+ pos0 n))
          (d41ctx-pos ctx) (+ pos0 keep))
    ctx))

(defun ds41f-cuda-speculate (ctx tok draft)
  "DS4F-CUDA-SPECULATE for V4.1: TOK and its guesses DRAFT through one pass
   at CTX's next position; how many its greedy choices confirm, CTX then
   past them (and the token after them its logits)."
  (when (plusp *dv-cool*)
    (decf *dv-cool*)
    (ds4f-forward ctx tok)
    (return-from ds41f-cuda-speculate 0))
  (let* ((m (d41ctx-model ctx)) (st (ds4f-cuda-state m)) (nv (ds4f-n-vocab m))
         (draft (subseq draft 0 (min (length draft) *dv-k* (1- +dv-nt+))))
         (pos0 (d41ctx-pos ctx)) (batch (cons tok draft)) (n (length batch)) (tail0 (d41ctx-tail ctx)))
    (sb-int:with-float-traps-masked (:overflow :inexact :underflow :invalid)
      (let ((*dk41-carry-rows* (make-hash-table)))
        (let* ((rows (ds41f-cuda-prefill-1 ctx batch t :small t))
               (a (loop for g in draft for i from 0 while (= (row-argmax rows i nv) g) count t))
               (keep (1+ a)))
          (with-cuda
            (cuda-current (dk-dev st))
            (when (< keep n) (ds41f-cuda-undo ctx pos0 keep n tail0 (subseq batch 0 keep)) (cuda-sync (dk-dev st))))
          (replace (d41ctx-logits ctx) rows :start2 (* (1- keep) nv) :end2 (* keep nv))
          (dv-adapt a (length draft))
          a)))))

(defun ds41f-pass-bytes (n pos0)
  "The scratch a pass of N tokens from POS0 takes: the tokens' activations;
   per (token, position) the indexer's scores -- a ratio-2 layer's (half the
   positions) and then, the arena only growing, a ratio-1 layer's: 6 bytes
   -- and past 2,048 candidate blocks a block maximum and a mask byte per
   block (7 bytes in all, a little spare); and per token those blocks' 2,048
   picks (8 KB)."
  (+ (* n *ds41f-cuda-chunk-bytes*) (* 7 n (+ pos0 n)) (* n 8448) (* 64 1024 1024)))

(defun ds41f-cuda-chunk-max (m pos0)
  "The most tokens a pass from POS0 has room for: the pool's slots past
   the staging ones (two layers' experts, and a few spare) hold its scratch."
  (let* ((st (ds4f-cuda-state m))
         (room (* (- (dk-n-slots st) (* 2 (ds4f-n-expert m)) 8) (dk-slot-bytes st))))
    (flet ((bytes (n) (ds41f-pass-bytes n pos0)))
      (let ((n (floor room *ds41f-cuda-chunk-bytes*)))
        (loop while (and (> n 1) (> (bytes n) room)) do (setf n (floor (* n 15) 16)))
        (max n 1)))))

(defun ds41f-cuda-prefill (ctx tokens &key (logits t))
  "TOKENS into V4.1 context CTX on the device, a pass's most at a time (a
   short last piece, or all of them on a device without the passes'
   products, decoded a token at a time); decode's pool as it was after."
  (let ((m (d41ctx-model ctx)))
    (sb-int:with-float-traps-masked (:overflow :inexact :underflow :invalid)
      (cond ((not (dk-passes-p (ds4f-cuda-state m)))
             (loop for (tk . more) on tokens do (ds41f-forward-1 ctx tk (and logits (null more)))))
            ((>= (length tokens) *ds41f-cuda-chunk-from*)
             (d4-call-keeping-pool m (ds4f-cuda-state m) (lambda () (ds41f-cuda-prefill-chunks ctx tokens logits))))
            (t (ds41f-cuda-prefill-chunks ctx tokens logits)))))
  ctx)

(defun ds41f-cuda-prefill-chunks (ctx tokens logits)
  (loop while tokens
        do (let* ((k (min (length tokens)
                          (or *ds41f-cuda-chunk* (ds41f-cuda-chunk-max (d41ctx-model ctx) (d41ctx-pos ctx)))))
                  (chunk (subseq tokens 0 k)))
             (setf tokens (nthcdr k tokens))
             (if (>= k *ds41f-cuda-chunk-from*)
                 (ds41f-cuda-prefill-1 ctx chunk (and logits (null tokens)))
                 (loop for (tk . more) on chunk
                       do (ds41f-forward-1 ctx tk (and logits (null more) (null tokens))))))))
