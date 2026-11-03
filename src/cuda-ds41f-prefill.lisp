;;;; cuda-ds41f-prefill.lisp — a V4.1 prefill chunk on the GPU, a layer at a time.
;;;
;;; cuda-ds4f-prefill.lisp's pass (its scratch in borrowed pool slots, a
;;; layer's experts all taken ahead into staging slots while the layer
;;; before computes, the routing on the device, the experts grouped by
;;; expert into one indexed product each) with V4.1's layers: the rounding,
;;; HC's pre a sublayer late, Engram's rows hashed token by token on the CPU
;;; and gated in, the shared compressed caches written by their source
;;; layers (ratio-2 pairs that may start before the chunk, ratio-1 rows), the
;;; indexer's pick for each token made on the index-source layers and kept
;;; for the layers after.  The CPU sees a token's state only at the chunk's
;;; end: the context's position, Engram's tail, the pair carried over.

(in-package #:illatio)

(defparameter *ds41f-cuda-chunk-from* 64
  "From this many tokens a V4.1 prompt goes through layer-major passes: a
   pass reads nearly all of every layer's experts (about 4 s over the bus
   when the pool holds few), against about 50 ms a token decoding.")

(defparameter *ds41f-cuda-chunk* 1024 "Tokens a V4.1 pass takes at most.")

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
         (raw (dvec st (svref (d41ctx-raw ctx) li))) (idx (pb st :p-idx (* n top-k)))
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
      (launch dev "d4_ring_gather" (ceiling (* prev hd) 256) 256 (list hd swa (mod (- pos0 prev) swa) 0 prev raw kr)))
    (let ((rows (cons kr (* 4 prev hd))))
      (dk41-rms-rows dev kv0 (dvec st (dl-kv-norm l)) rows hd n eps)
      (dk41-rope dev rows cs 1 hd nrot n)
      (dk41-quant dev rows (* n hd) :fp8))
    (let ((keep (min swa (+ prev n))))
      (launch dev "d4_ring_store" (ceiling (* keep hd) 256) 256
              (list hd swa (mod (- (+ pos0 n) keep) swa) (- (+ prev n) keep) keep raw kr)))
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
              (assert (<= (ceiling nc (d41-cand-block v)) (d41-cand-top v)) ()
                      "V4.1's candidate blocks (past ~D compressed rows) are not on the GPU yet"
                      (* (d41-cand-block v) (d41-cand-top v)))
              (dk-f16mm st (lname li "indexer.attn_q_b.weight") qrn ixq n)
              (dk41-round dev ixq (* n ixh ixd))
              (dk41-rope dev ixq cs ixh ixd nrot n)
              (dk41-quant dev ixq (* n ixh ixd) :fp4)
              (dk-f16mm st (lname li "indexer.proj.weight") norm ixw n)
              (dk41-round dev ixw (* n ixh))
              (launch dev "d41_ix_scores_chunk" (list (ceiling nc 8) n) 256
                      (list ixh nc ixq ixw (dvec st (svref (d41ctx-ix-comp ctx) k)) bvis scores)
                      :shared (* 4 ixh ixd))
              (launch dev "d4_topk_chunk" n 1024 (list nc top-k nc scores idx bvis)))))
        (when *dk41-pick* (setf ident 0))
        (dotimes (tk n) (setf (aref ncomp tk) (min top-k (aref vis tk))))))
    (let ((raws (make-array n)) (nraw (make-array n)))
      (dotimes (tk n) (setf (aref raws tk) (max 0 (- (+ prev tk 1) swa)) (aref nraw tk) (min swa (+ prev tk 1))))
      (launch dev "d4_attend" n 512
              (list nh hd (/ 1f0 (sqrt (float hd 1f0))) top-k 0 ident
                    q kr (if k (dvec st (svref (d41ctx-comp ctx) k)) kr)
                    (dk-u32s st raws) (dk-u32s st nraw) (dk-u32s st ncomp) idx (dvec st (dl-sinks l)) heads)))
    (dk41-round dev heads (* n nh hd))
    (dk41-rope dev heads ics nh hd nrot n)
    (dk41-round dev heads (* n nh hd))
    (let ((w (gethash (lname li "attn_output_a.weight") (dk-q8 st))))
      (dotimes (g groups)
        (dk-mm st "d4_mm_q8_0" (cons (cmat-qs w) (* g rank gdim)) (cons (cmat-d w) (* 2 g rank (floor gdim 32)))
               0 rank gdim (cons heads (* 4 g gdim)) (cons low (* 4 g rank)) n
               :x-stride (* nh hd) :y-stride (* groups rank))))
    (dk41-round dev low (* n groups rank))
    (dk-q8mm st (lname li "attn_output_b.weight") low aout n)
    (dk41-round dev aout (* n d))
    (dk-hc-post-chunk m st aout sa pca sb n)
    (dk41-round dev sb (* n (ds4f-n-hc m) d))))

(defun dk41-ffn-chunk (m st l li n toks slots sb sa)
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
      (launch dev "d41_mv_f32n" (list (ceiling rows 4) n) 256 (list cols rows cols rows w norm2 logits)))
    (dk-q8mm st (lname li "ffn_gate_shexp.weight") norm2 sg n)
    (dk-q8mm st (lname li "ffn_up_shexp.weight") norm2 su n)
    (launch dev "d41_swiglu" (ceiling (* n ff) 256) 256 (list (* n ff) (svref (ds4f-clamps m) li) sg su smid nil nil))
    (dk-q8mm st (lname li "ffn_down_shexp.weight") smid sout n)
    (dk41-round dev sout (* n d))
    ;; V4's routed half (no hash layers here); then the sum and the streams rounded
    (let ((*dk41-round-moe* t))
      (dk-ffn-routed-chunk m st l li n toks slots sb sa))))

(defun dk41-engram-inputs (m st ctx tokens)
  "Engram's table rows for each of TOKENS (hashed in turn from CTX's tail),
   decoded and uploaded: a buffer a table, 24 x 256 a token; and the tail
   after them."
  (let* ((v (ds4f-v41 m)) (n (length tokens)) (ntab (length (d41-eg-layers v))) (w (* 24 256))
         (bufs (loop for tbl below ntab collect (f32vec (* n w)))) (tail (d41ctx-tail ctx)))
    (loop for tk in tokens for i from 0
          do (multiple-value-bind (rows nt) (d41-engram-hash m tail tk)
               (loop for tbl below ntab for b in bufs
                     do (replace b (d41-engram-input m tbl (nth tbl rows)) :start1 (* i w)))
               (setf tail nt)))
    (values (loop for b in bufs for tbl from 0
                  collect (let ((d (pb st (intern (format nil "P-EG~D" tbl) :keyword) (* n w))))
                            (cuda-upload-vector d b) d))
            tail)))

(defun dk41-ready (m st ctx)
  "The layers' vectors and the context's caches on the device before a pass queues anything."
  (loop for l across (ds4f-layers m)
        do (dolist (v (list (dl-attn-norm l) (dl-q-a-norm l) (dl-kv-norm l) (dl-ffn-norm l) (dl-sinks l)
                            (dl-hc-attn-scale l) (dl-hc-attn-base l) (dl-hc-ffn-scale l) (dl-hc-ffn-base l)
                            (dl-comp-norm l) (dl-ix-k-norm l) (dl-probs-b l) (dl-eg-q-norm l) (dl-eg-k-norm l)))
             (when v (dvec st v))))
  (dolist (vs (list (d41ctx-raw ctx) (d41ctx-comp ctx) (d41ctx-ix-comp ctx) (d41ctx-prev-kv ctx) (d41ctx-prev-score ctx)))
    (loop for v across vs do (dvec st v)))
  (dvec st (ds4f-output-norm m)))

(defun ds41f-cuda-prefill-1 (ctx tokens logits)
  "TOKENS at CTX's next positions, layer-major on the device (at most
   *DS41F-CUDA-CHUNK*); the last one's logits when LOGITS."
  (let* ((m (d41ctx-model ctx)) (v (ds4f-v41 m)) (st (ds4f-cuda-state m)) (dev (dk-dev st))
         (d (ds4f-d m)) (n-hc (ds4f-n-hc m)) (hcd (* n-hc d)) (pos0 (d41ctx-pos ctx))
         (n (length tokens)) (tokv (coerce tokens 'simple-vector)) (wf (ds4f-file m))
         (nl (ds4f-n-layer m)) (positions (loop for i below n collect (+ pos0 i)))
         (nc-max (floor (+ pos0 n) 1)))
    (declare (type u8vec wf))
    (with-cuda
      (cuda-current dev)
      (let* ((run (d4-borrow-slots st (+ (* n *ds41f-cuda-chunk-bytes*) (* 4 n nc-max) (* 64 1024 1024))))
             (base (+ (dbuf-ptr (dk-pool st)) (* (car run) (dk-slot-bytes st))))
             (*dk-scratch* (list base (+ base (* (cdr run) (dk-slot-bytes st))) (make-hash-table)))
             (*dk-stage* (list (d4-stage-slots st (* 2 (ds4f-n-expert m)) run)))
             (*dk-mapped* (cons (or (getf (dk-bufs st) :p-mapped-sap)
                                    (setf (getf (dk-bufs st) :p-mapped-sap) (cuda-host-alloc +dk-mapped-bytes+ :mapped t)))
                                0))
             (*dk41-pick* nil)
             (sa (pb st :p-sa (* n hcd))) (sb (pb st :p-sb (* n hcd))))
        (dk41-ready m st ctx)
        ;; the tokens' embeddings into each stream
        (let ((e (f32vec (* n hcd))))
          (dotimes (tk n)
            (let ((o (+ (ds4f-embd-off m) (* (svref tokv tk) d 2))))
              (dotimes (j d)
                (let ((x (f16-at wf (+ o (* 2 j)))))
                  (dotimes (h n-hc) (setf (aref e (+ (* tk hcd) (* h d) j)) x))))))
          (cuda-upload-vector sa e))
        (let ((ones (f32vec n))) (fill ones 1f0) (cuda-upload-vector (pb st :p-ones n) ones))
        (let* ((toks (let ((b (pb st :p-tok n)) (u (make-array n :element-type '(unsigned-byte 32))))
                       (dotimes (i n) (setf (aref u i) (svref tokv i)))
                       (sb-sys:with-pinned-objects (u) (cuda-upload b (sb-sys:vector-sap u) (* 4 n)))
                       b))
               (li-plain (position 0 (ds4f-ratios m))) (li-comp (position-if #'plusp (ds4f-ratios m)))
               (cs0 (dk41-cs-rows m st :p-cs0 positions li-plain)) (ics0 (dk41-cs-rows m st :p-ics0 positions li-plain t))
               (cs1 (dk41-cs-rows m st :p-cs1 positions li-comp)) (ics1 (dk41-cs-rows m st :p-ics1 positions li-comp t))
               (pre0 (let ((b (pb st :p-pre0 4)) (p (f32vec 4))) (setf (aref p 0) 1f0) (cuda-upload-vector b p) b))
               (pre pre0) (pstride 0))
          ;; the ratio-2 pairs' first positions, for their rows' RoPE
          (let ((p1 (if (oddp pos0) pos0 (1+ pos0))))
            (dk41-cs-rows m st :p-cs2 (loop for p from p1 below (+ pos0 n) by 2 collect (1- p)) li-comp))
          (multiple-value-bind (eg tail) (dk41-engram-inputs m st ctx tokens)
            (multiple-value-bind (slots ev staged) (d4-take-layer m st 0)
              (dotimes (li nl)
                (let* ((l (svref (ds4f-layers m) li)) (tbl (position li (d41-eg-layers v)))
                       (comp (plusp (dl-ratio l))))
                  (multiple-value-bind (nslots nev nstaged)
                      (if (< (1+ li) nl) (d4-take-layer m st (1+ li)) (values nil nil nil))
                    (when tbl
                      (let ((kv (pb st :p-egkv (* n (1+ n-hc) d))))
                        (dk-f16mm st (lname li "engram_kv.weight") (nth tbl eg) kv n)
                        (dk41-round dev kv (* n (1+ n-hc) d))
                        (launch dev "d41_engram" (list n-hc n) 1024
                                (list d n-hc (ds4f-eps m) sa kv (dvec st (dl-eg-q-norm l)) (dvec st (dl-eg-k-norm l))))))
                    (dk41-attention-chunk m st l li ctx pos0 n sa sb pre pstride
                                          (if comp cs1 cs0) (if comp ics1 ics0))
                    (cuda-stream-wait (cudev-stream dev) ev)
                    (dk41-ffn-chunk m st l li n toks slots sb sa)
                    (cuda-sync dev)
                    (dk-arena-reset st)
                    (setf (cdr *dk-mapped*) 0)
                    (cuda-event-destroy ev)
                    (d4-let-go st slots staged)
                    (setf slots nslots ev nev staged nstaged
                          pre (cons (pb st :p-pcf (* n 32)) 96) pstride 32)))))
            (setf (d41ctx-tail ctx) tail))
          (incf (d41ctx-pos ctx) n)
          (when logits
            (let* ((q (dk-qbuf st :hnorm d)) (embd (dbuf-of st :hembd (* 4 d))) (norm (dbuf-of st :hnorm (* 4 d)))
                   (out (dbuf-of st :out (* 4 (ds4f-n-vocab m)))) (lv (d41ctx-logits ctx)))
              (launch dev "d41_hc_sum" (ceiling d 256) 256
                      (list d n-hc 0 (cons sa (* 4 (1- n) hcd)) (cons (pb st :p-pcf (* n 32)) (* 4 (+ (* 32 (1- n)) 24))) embd))
              (dk41-rms dev embd (dvec st (ds4f-output-norm m)) norm d (ds4f-eps m) q)
              (dk-q8mv st "output.weight" norm out :q q)
              (sb-sys:with-pinned-objects (lv)
                (cuda-download out (sb-sys:vector-sap lv) (* 4 (ds4f-n-vocab m))))))
          (cuda-sync dev))
        (d4-return-slots st run)))
    ctx))

(defun ds41f-cuda-prefill (ctx tokens &key (logits t))
  "TOKENS into V4.1 context CTX on the device, *DS41F-CUDA-CHUNK* at a time
   (a short last piece decoded a token at a time)."
  (sb-int:with-float-traps-masked (:overflow :inexact :underflow :invalid)
    (loop while tokens
          do (let* ((k (min (length tokens) *ds41f-cuda-chunk*)) (chunk (subseq tokens 0 k)))
               (setf tokens (nthcdr k tokens))
               (if (>= k *ds41f-cuda-chunk-from*)
                   (ds41f-cuda-prefill-1 ctx chunk (and logits (null tokens)))
                   (loop for (tk . more) on chunk
                         do (ds41f-forward-1 ctx tk (and logits (null more) (null tokens))))))))
  ctx)
