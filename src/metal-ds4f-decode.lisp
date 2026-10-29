;;;; metal-ds4f-decode.lisp — a DS4F token on the GPU, a command buffer a layer.
;;;
;;; DS4F-FORWARD's ops, as ds4 runs its decode: everything a layer does but
;;; the routing on the GPU, so the GPU is never left idle long enough to
;;; slow down (a lone command buffer runs at the clock an idle GPU drops to).
;;; A layer's command buffer runs the previous layer's routed experts (their
;;; sum, its FFN's HC post), then this layer's attention block -- HC pre, the
;;; norms, the Q8_0 projections, RoPE, the KV row into the raw window, the
;;; compressors, the indexer's top-K, D4_ATTEND, the output projections, HC
;;; post -- then its FFN's HC pre, norm, router logits and shared expert.
;;; Back on the CPU: the routing (D4-ROUTE, from the GPU's logits), the
;;; expert cache, the next layer's guess.  The streams stay on the GPU; the
;;; caches are the context's own vectors (static), read and written in
;;; place, so prefill on the CPU and decode on the GPU share them.

(in-package #:illatio)

;;; --- the GPU's views of Lisp vectors ------------------------------------------------

(defun gpu-vector (m v)
  "(buffer . offset) of f32 vector V: a static vector wrapped where it lies
   (its data 16 bytes into its pages), any other copied once."
  (let* ((st (ds4f-gpu-state m))
         (table (or (getf (d4gpu-plist* st) :gvec)
                    (setf (getf (d4gpu-plist* st) :gvec) (make-hash-table :test 'eq)))))
    (declare (type f32vec v))
    (or (gethash v table)
        (setf (gethash v table)
              (if (static-vector-p v)
                  (let* ((base (logandc2 (sb-kernel:get-lisp-obj-address v) sb-vm:lowtag-mask))
                         (data (sb-sys:sap-int (sb-sys:vector-sap v)))
                         (bytes (* 16384 (ceiling (+ (- data base) (* 4 (length v))) 16384))))
                    (cons (gpu-wrap (d4gpu-gpu st) (sb-sys:int-sap base) bytes) (- data base)))
                  (let ((b (gpu-buffer (d4gpu-gpu st) (* 4 (max 1 (length v))))))
                    (copy-floats-in b v (length v))
                    (cons b 0)))))))

(defun forget-gpu-vector (m v)
  "Drop the GPU's view of V (before V is freed)."
  (let ((table (getf (d4gpu-plist* (ds4f-gpu-state m)) :gvec)))
    (when table (remhash v table))))

(defun gmat (m f) (gpu-vector m (f32mat-data f)))

;;; --- a per-command-buffer arena for small inputs (RoPE tables, counts) ----------------

(defun arena-reset (st) (setf (getf (d4gpu-plist* st) :arena-at) 0))

(defun arena-put (st words &key float)
  "WORDS (a sequence of floats, or of u32s) into ST's arena: (buffer . offset)."
  (let* ((buf (d4gpu-buf st :arena (* 1024 1024)))
         (at (getf (d4gpu-plist* st) :arena-at 0)) (sap (gbuf-sap buf)))
    (assert (< (+ at (* 4 (length words))) (gbuf-bytes buf)))
    (let ((i 0))
      (map nil (lambda (x)
                 (if float
                     (setf (sb-sys:sap-ref-single sap (+ at (* 4 i))) x)
                     (setf (sb-sys:sap-ref-32 sap (+ at (* 4 i))) x))
                 (incf i))
           words))
    (setf (getf (d4gpu-plist* st) :arena-at) (* 16 (ceiling (+ at (* 4 (length words))) 16)))
    (cons buf at)))

(defun arena-cs (m st pos il &optional inverse)
  (arena-put st (d4-rope-cs! m pos il inverse (f32vec (ds4f-n-rot m))) :float t))

;;; --- encoding the ops ----------------------------------------------------------------

(defun enc-rms (gpu e x w out n groups eps)
  (dispatch gpu e "d4_rms" (list n eps (if w 1 0)) (list x (or w x) out) groups (min 1024 n) :shared 128))

(defun enc-mvf (gpu e w x y rows cols)
  (dispatch gpu e "d4_mv_f32" (list cols rows) (list w x y) (ceiling rows 4) 256 :shared 128))

(defun enc-rope (gpu e x cs nh hd nrot &optional (ntok 1))
  "RoPE on NTOK tokens' NH heads at X, CS a table a token."
  (dispatch gpu e "d4_rope" (list nh hd nrot ntok) (list x cs) (ceiling (* ntok nh (floor nrot 2)) 256) 256))

(defun lname (li short) (format nil "blk.~D.~A" li short))

(defun enc-mv16 (m gpu e name x y)
  "Y = W X, W the F16 tensor NAME (the file's, in DS4F-GPU-F16's buffer)."
  (destructuring-bind (buf . table) (ds4f-gpu-f16 m)
    (destructuring-bind (woff rbytes rows cols) (gethash name table)
      (declare (ignore rbytes))
      (dispatch gpu e "d4_mv_f16" (list cols rows) (list (cons buf woff) x y) (ceiling rows 4) 256 :shared 128))))

(defun enc-head (m gpu e st streams)
  "output_logits_one on the GPU, from STREAMS (a buffer, or (buffer . offset)):
   the logits into buffer :G-OUT."
  (let* ((d (ds4f-d m)) (n (ds4f-n-hc m)) (hcd (* n d)) (eps (ds4f-eps m))
         (flat (d4gpu-buf st :g-hflat (* 4 hcd))) (pre (d4gpu-buf st :g-hpre (* 4 16)))
         (embd (d4gpu-buf st :g-hembd (* 4 d))) (norm (d4gpu-buf st :g-hnorm (* 4 d)))
         (out (d4gpu-buf st :g-out (* 4 (ds4f-n-vocab m)))))
    (enc-rms gpu e streams nil flat hcd 1 eps)
    (enc-mv16 m gpu e "output_hc_fn.weight" flat pre)
    (dispatch gpu e "d4_head_mix" (list d n (ds4f-hc-eps m))
              (list pre (gpu-vector m (ds4f-out-hc-scale m)) (gpu-vector m (ds4f-out-hc-base m)) streams embd)
              (ceiling d 256) 256)
    (enc-rms gpu e embd (gpu-vector m (ds4f-output-norm m)) norm d 1 eps)
    (destructuring-bind (buf . table) (ds4f-gpu-weights m)
      (destructuring-bind (woff rbytes rows cols) (gethash "output.weight" table)
        (dispatch gpu e "d4_mv_q8_0" (list cols rows rbytes rows 0) (list (cons buf woff) norm out) (ceiling rows 8) 64)))
    out))

(defun enc-hc-pre (m gpu e st fn scale base streams out pc)
  "HC-PRE of STREAMS (4 x D): OUT, its POST and COMB into PC."
  (assert (= (ds4f-n-hc m) 4) () "d4_hc_pre_fin is written for 4 streams")
  (let* ((d (ds4f-d m)) (n (ds4f-n-hc m)) (hcd (* n d))
         (flat (d4gpu-buf st :g-flat (* 4 hcd))) (mix (d4gpu-buf st :g-mix (* 4 64))))
    (enc-rms gpu e streams nil flat hcd 1 (ds4f-eps m))
    (enc-mv16 m gpu e fn flat mix)
    (dispatch gpu e "d4_hc_pre_fin" (list d n (ds4f-sinkhorn m) 1f-6)
              (list mix (gpu-vector m scale) (gpu-vector m base) streams out pc)
              (ceiling d 256) 256 :shared (* 4 32))))

(defun enc-hc-post (m gpu e block streams pc out &optional (ntok 1))
  "HC post for NTOK tokens (each's BLOCK, STREAMS, PC and OUT a row)."
  (let ((d (ds4f-d m)) (n (ds4f-n-hc m)))
    (dispatch gpu e "d4_hc_post" (list d n ntok 1f-6) (list block streams pc out)
              (ceiling (* ntok n d) 256) 256)))

(defun enc-compressor (m gpu e st l li c norm pos ratio wkv wgate ape normw rows-of n-of hd kind)
  "One compressor's step for the token at POS (its window in C), and on a
   window's last token its row pushed: whether one was."
  (let* ((coff (if (= ratio 4) 2 1)) (width (* coff hd)) (pos-mod (mod pos ratio))
         (row (if (= ratio 4) (+ ratio pos-mod) pos-mod))
         (kv (d4gpu-buf st (if (= kind 0) :g-ckv :g-ikv) (* 4 width)))
         (sc (d4gpu-buf st (if (= kind 0) :g-csc :g-isc) (* 4 width)))
         (st-kv (if (= kind 0) (dc-st-kv c) (dc-ix-kv c))) (st-sc (if (= kind 0) (dc-st-score c) (dc-ix-score c))))
    (declare (ignore l))
    (enc-mv16 m gpu e (lname li wkv) norm kv)
    (enc-mv16 m gpu e (lname li wgate) norm sc)
    (dispatch gpu e "d4_comp_store" (list width row pos-mod (f32mat-cols ape))
              (list kv sc (gmat m ape) (gpu-vector m st-kv) (gpu-vector m st-sc))
              (ceiling width 256) 256)
    (when (zerop (mod (1+ pos) ratio))
      (dispatch gpu e "d4_comp_pool" (list hd ratio width (ds4f-eps m) kind (ds4f-n-rot m) (funcall n-of c))
                (list (gpu-vector m st-kv) (gpu-vector m st-sc) (gpu-vector m normw)
                      (arena-cs m st (- (1+ pos) ratio) li) (gpu-vector m (funcall rows-of c)))
                1 hd :shared (* 4 (+ hd 32)))
      t)))

(defun enc-attention (m gpu e st l li c pos sa sb)
  "Layer LI's attention block for the token at POS: streams SA into SB."
  (let* ((d (ds4f-d m)) (hd (ds4f-head-dim m)) (nh (ds4f-n-head m)) (lq (ds4f-lora-q m))
         (nrot (ds4f-n-rot m)) (eps (ds4f-eps m)) (ratio (dl-ratio l)) (swa (ds4f-swa m))
         (cur (d4gpu-buf st :g-cur (* 4 d))) (norm (d4gpu-buf st :g-norm (* 4 d)))
         (pca (d4gpu-buf st :g-pca (* 4 32))) (qr (d4gpu-buf st :g-qr (* 4 lq)))
         (qrn (d4gpu-buf st :g-qrn (* 4 lq))) (q (d4gpu-buf st :g-q (* 4 nh hd)))
         (kv0 (d4gpu-buf st :g-kv0 (* 4 hd))) (heads (d4gpu-buf st :g-heads (* 4 nh hd)))
         (low (d4gpu-buf st :g-low (* 4 (ds4f-out-groups m) (ds4f-lora-o m))))
         (aout (d4gpu-buf st :g-aout (* 4 d)))
         (cs (arena-cs m st pos li)) (ics (arena-cs m st pos li t)))
    (enc-hc-pre m gpu e st (lname li "hc_attn_fn.weight") (dl-hc-attn-scale l) (dl-hc-attn-base l) sa cur pca)
    (enc-rms gpu e cur (gpu-vector m (dl-attn-norm l)) norm d 1 eps)
    (q8mv-enc m gpu e li "attn_q_a.weight" norm qr)
    (q8mv-enc m gpu e li "attn_kv.weight" norm kv0)
    (enc-rms gpu e qr (gpu-vector m (dl-q-a-norm l)) qrn lq 1 eps)
    (q8mv-enc m gpu e li "attn_q_b.weight" qrn q)
    (enc-rms gpu e q nil q hd nh eps)
    (enc-rope gpu e q cs nh hd nrot)
    ;; the KV row into the ring (its counts kept on the CPU, as PUSH-RAW! keeps them)
    (dispatch gpu e "d4_kv_fin" (list hd nrot eps (mod (dc-raw-total c) swa))
              (list kv0 (gpu-vector m (dl-kv-norm l)) cs (gpu-vector m (dc-raw c)))
              1 hd :shared (* 4 (+ hd 32)))
    (incf (dc-raw-total c))
    (setf (dc-n-raw c) (min swa (1+ (dc-n-raw c))))
    ;; the compressors
    (when (plusp ratio)
      (when (enc-compressor m gpu e st l li c norm pos ratio "attn_compressor_kv.weight" "attn_compressor_gate.weight" (dl-comp-ape l)
                            (dl-comp-norm l) #'dc-comp #'dc-n-comp hd 0)
        (incf (dc-n-comp c)))
      (when (= ratio 4)
        (when (enc-compressor m gpu e st l li c norm pos ratio "indexer_compressor_kv.weight" "indexer_compressor_gate.weight" (dl-ixc-ape l)
                              (dl-ixc-norm l) #'dc-ix-comp #'dc-n-ix (ds4f-ix-dim m) 1)
          (incf (dc-n-ix c)))))
    ;; the indexer, then attention
    (let* ((nc (if (dc-comp c) (dc-n-comp c) 0)) (top-k (ds4f-ix-top-k m))
           (indexed (and (= ratio 4) (> (dc-n-ix c) top-k)))
           (idx (d4gpu-buf st :g-ixidx (* 4 top-k))))
      (when indexed
        (let* ((ixh (ds4f-ix-heads m)) (ixd (ds4f-ix-dim m)) (nix (dc-n-ix c))
               (ixq (d4gpu-buf st :g-ixq (* 4 ixh ixd))) (ixw (d4gpu-buf st :g-ixw (* 4 ixh)))
               (scores (d4gpu-buf st :g-ixs (* 4 nix))))
          (enc-mv16 m gpu e (lname li "indexer.attn_q_b.weight") qrn ixq)
          (enc-mv16 m gpu e (lname li "indexer.proj.weight") norm ixw)
          (dispatch gpu e "d4_ixq_fin" (list ixh nrot (/ 1f0 (sqrt (float (* ixh ixd) 1f0))))
                    (list ixq cs ixw) ixh ixd :shared (* 4 (+ ixd 32)))
          (dispatch gpu e "d4_ix_scores" (list 1 nix ixh)
                    (list ixq ixw (gpu-vector m (dc-ix-comp c)) (arena-put st (list nix)) scores)
                    (list (ceiling nix 32) 1) 128 :shared (* 4 (+ (* 8 128) (* 32 128) (* 8 32))))
          (dispatch gpu e "d4_topk" (list nix top-k nix) (list scores idx (arena-put st (list nix))) 1 1024
                    :shared (* 4 96))))
      (let ((kc (if (dc-comp c) (gpu-vector m (dc-comp c)) (gpu-vector m (dc-raw c))))
            (ncomp (if indexed top-k nc)))
        (assert (= hd 512) () "d4_attend1 is written for 512-wide heads (128 threads x 4)")
        (dispatch gpu e "d4_attend1"
                  (list hd (/ 1f0 (sqrt (float hd 1f0))) swa (mod (- (dc-raw-total c) (dc-n-raw c)) swa)
                        (dc-n-raw c) ncomp (if indexed 0 1))
                  (list q (gpu-vector m (dc-raw c)) kc idx (gpu-vector m (dl-sinks l)) heads)
                  (floor nh 4) 128 :shared (* 4 (+ (* 4 hd) (* 4 (+ (dc-n-raw c) ncomp)) 8)))))
    (enc-rope gpu e heads ics nh hd nrot)
    (q8mv-enc m gpu e li "attn_output_a.weight" heads low :group-rows (ds4f-lora-o m)
                                                         :group-stride (* hd (floor nh (ds4f-out-groups m))))
    (q8mv-enc m gpu e li "attn_output_b.weight" low aout)
    (enc-hc-post m gpu e aout sa pca sb)))

(defun enc-ffn-head (m gpu e st l li sb)
  "Layer LI's FFN up to the routing: HC pre of SB, the norm, the router's
   logits, the shared expert."
  (let* ((d (ds4f-d m)) (ff (ds4f-ff-exp m)) (ne (ds4f-n-expert m))
         (cur (d4gpu-buf st :g-cur (* 4 d))) (norm2 (d4gpu-buf st :g-norm2 (* 4 d)))
         (pcf (d4gpu-buf st :g-pcf (* 4 32))) (logits (d4gpu-buf st :g-logits (* 4 ne)))
         (sg (d4gpu-buf st :g-sg (* 4 ff))) (su (d4gpu-buf st :g-su (* 4 ff)))
         (smid (d4gpu-buf st :g-smid (* 4 ff))) (sout (d4gpu-buf st :g-sout (* 4 d)))
         (one (arena-put st (list 1f0) :float t)))
    (enc-hc-pre m gpu e st (lname li "hc_ffn_fn.weight") (dl-hc-ffn-scale l) (dl-hc-ffn-base l) sb cur pcf)
    (enc-rms gpu e cur (gpu-vector m (dl-ffn-norm l)) norm2 d 1 (ds4f-eps m))
    (enc-mv16 m gpu e (lname li "ffn_gate_inp.weight") norm2 logits)
    ;; the next layer's router on this input: the guess at its experts (read ahead)
    (when (and (< (1+ li) (ds4f-n-layer m)) (null (dl-tid2eid (svref (ds4f-layers m) (1+ li)))))
      (enc-mv16 m gpu e (lname (1+ li) "ffn_gate_inp.weight") norm2 (d4gpu-buf st :g-plogits (* 4 ne))))
    (q8mv-enc m gpu e li "ffn_gate_shexp.weight" norm2 sg)
    (q8mv-enc m gpu e li "ffn_up_shexp.weight" norm2 su)
    (dispatch gpu e "d4_swiglu_w" (list 1 ff (svref (ds4f-clamps m) li)) (list sg su one smid) (ceiling ff 256) 256)
    (q8mv-enc m gpu e li "ffn_down_shexp.weight" smid sout)))

(defun enc-ffn-tail (m gpu e st l li slots w sb sa)
  "Layer LI's routed experts (cache SLOTS, weights W) on its FFN input, summed
   with the shared expert, and its FFN's HC post: streams SB into SA."
  (let* ((d (ds4f-d m)) (ff (ds4f-ff-exp m)) (k (length slots)) (clamp (svref (ds4f-clamps m) li))
         (xc (ds4f-expert-cache m)) (sbytes (d4gpu-slot-bytes st)) (gb (d4gpu-gate-bytes st))
         (norm2 (d4gpu-buf st :g-norm2 (* 4 d))) (bo (arena-put st (make-list (* 4 k) :initial-element 0)))
         (bw (arena-put st (coerce w 'list) :float t))
         (bgu (d4gpu-buf st :dgu (* 4 2 k ff))) (bm (d4gpu-buf st :dmid (* 4 k ff)))
         (bd (d4gpu-buf st :ddown (* 4 k d))) (moe (d4gpu-buf st :g-moe (* 4 d)))
         (grb (second (dl-gate-exps l))) (drb (second (dl-down-exps l))) (buf (dx-buf xc)))
    ;; the experts' gate slots then their up slots (64-bit offsets)
    (dotimes (i k)
      (setf (sb-sys:sap-ref-64 (gbuf-sap (car bo)) (+ (cdr bo) (* 8 i))) (* (aref slots i) sbytes)
            (sb-sys:sap-ref-64 (gbuf-sap (car bo)) (+ (cdr bo) (* 8 (+ k i)))) (+ (* (aref slots i) sbytes) gb)))
    (flet ((mv (kernel ncols nrows rb add x xs y z)
             (dispatch gpu e kernel (list ncols nrows rb add xs nrows) (list buf bo x y)
                       (list (ceiling nrows 8) 1 z) 64
                       :shared (if (string= kernel "d4_mv_iq2_xxs") (+ (* 8 256) 128) 0))))
      (mv "d4_mv_iq2_xxs" d ff grb 0 norm2 0 bgu (* 2 k))
      (dispatch gpu e "d4_swiglu_w" (list k ff clamp) (list bgu (cons bgu (* 4 k ff)) bw bm)
                (ceiling (* k ff) 256) 256)
      (mv "d4_mv_q2_k" ff d drb (* 2 gb) bm ff bd k))
    (dispatch gpu e "d4_moe_sum" (list d k) (list bd (d4gpu-buf st :g-sout (* 4 d)) moe) (ceiling d 256) 256)
    (enc-hc-post m gpu e moe sb (d4gpu-buf st :g-pcf (* 4 32)) sa)))

;;; --- a token ---------------------------------------------------------------------------

(defun ds4f-gpu-forward (ctx token logits)
  "DS4F-FORWARD-1 on the GPU: TOKEN at the next position."
  (let* ((m (d4ctx-model ctx)) (st (ds4f-gpu-state m)) (gpu (d4gpu-gpu st))
         (d (ds4f-d m)) (n (ds4f-n-hc m)) (hcd (* n d)) (pos (d4ctx-pos ctx)) (wf (ds4f-file m))
         (ne (ds4f-n-expert m)) (nl (ds4f-n-layer m))
         (sa (d4gpu-buf st :g-sa (* 4 hcd))) (sb (d4gpu-buf st :g-sb (* 4 hcd)))
         (lg (f32vec ne)) (norm2 (f32vec d)) (pending nil))
    (declare (type u8vec wf))
    (ds4f-gpu-weights m)
    ;; the token's embedding into each stream
    (let ((o (+ (ds4f-embd-off m) (* token d 2))) (sap (gbuf-sap sa)))
      (dotimes (j d)
        (let ((v (f16-at wf (+ o (* 2 j)))))
          (dotimes (h n) (setf (sb-sys:sap-ref-single sap (* 4 (+ (* h d) j))) v)))))
    (dotimes (li nl)
      (let ((l (svref (ds4f-layers m) li)) (c (svref (d4ctx-caches ctx) li)))
        (arena-reset st)
        (with-gpu-work (e gpu)
          (when pending (apply #'enc-ffn-tail m gpu e st (append pending (list sb sa))))
          (enc-attention m gpu e st l li c pos sa sb)
          (enc-ffn-head m gpu e st l li sb))
        ;; the routing, on the CPU
        (copy-floats-out lg (d4gpu-buf st :g-logits (* 4 ne)) ne)
        (copy-floats-out norm2 (d4gpu-buf st :g-norm2 (* 4 d)) d)
        (multiple-value-bind (sel w) (d4-route m l li norm2 token lg)
          (let ((slots (cache-experts m l li sel)))
            (when (and (plusp *ds4f-prefetch-k*) (< (1+ li) nl))
              (let ((pl (f32vec ne)))
                (copy-floats-out pl (d4gpu-buf st :g-plogits (* 4 ne)) ne)
                (prefetch-cache-experts m (1+ li) (predict-experts m (1+ li) norm2 token *ds4f-prefetch-k* pl))))
            (setf pending (list l li slots w))))))
    (arena-reset st)
    (with-gpu-work (e gpu)
      (apply #'enc-ffn-tail m gpu e st (append pending (list sb sa)))
      (when logits (enc-head m gpu e st sa)))
    (incf (d4ctx-pos ctx))
    (when logits
      (copy-floats-out (d4ctx-logits ctx) (d4gpu-buf st :g-out 0) (ds4f-n-vocab m)))
    ctx))
