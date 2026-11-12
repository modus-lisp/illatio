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

(defun arena-reset (st &optional (which 0))
  "Start arena WHICH (0 or 1: a command buffer may be encoded while the one
   before it runs) over."
  (setf (getf (d4gpu-plist* st) :arena-key) (if (zerop which) :arena :arena1)
        (getf (d4gpu-plist* st) :arena-at) 0))

(defun arena-put (st words &key float)
  "WORDS (a sequence of floats, or of u32s) into ST's arena: (buffer . offset)."
  (let* ((buf (d4gpu-buf st (getf (d4gpu-plist* st) :arena-key :arena) (* 1024 1024)))
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

(defun enc-head (m gpu e st streams &optional out)
  "output_logits_one on the GPU, from STREAMS (a buffer, or (buffer . offset)):
   the logits into OUT ((buffer . offset); buffer :G-OUT by default)."
  (let* ((d (ds4f-d m)) (n (ds4f-n-hc m)) (hcd (* n d)) (eps (ds4f-eps m))
         (flat (d4gpu-buf st :g-hflat (* 4 hcd))) (pre (d4gpu-buf st :g-hpre (* 4 16)))
         (embd (d4gpu-buf st :g-hembd (* 4 d))) (norm (d4gpu-buf st :g-hnorm (* 4 d)))
         (out (or out (d4gpu-buf st :g-out (* 4 (ds4f-n-vocab m))))))
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

(defun enc-hc-post (m gpu e block streams pc out &optional (ntok 1) steer)
  "HC post for NTOK tokens (each's BLOCK, STREAMS, PC and OUT a row); BLOCK
   first steered when STEER, (layer . kind), says which it is (steer.lisp):
   its rows steered, the last captured (read after the pass, DS4F-STEER-CAPTURED)."
  (let ((d (ds4f-d m)) (n (ds4f-n-hc m)))
    (when (and steer (steer-wants-p (car steer) (cdr steer)))
      (let* ((li (car steer)) (kind (cdr steer)) (scale (steer-scale li kind))
             (cap (d4gpu-buf (ds4f-gpu-state m) :steer-capture (* 4 2 (ds4f-n-layer m) d))))
        (dispatch gpu e "dn_steer" (list d scale (1- ntok) (if *steer-capture* 1 0))
                  (list block (if (zerop scale) block (cons (gpu-steer-dirs gpu) (* 4 li d)))
                        (cons cap (* 4 d (+ (* 2 li) (if (eq kind :attn) 0 1)))))
                  ntok 256)))
    (dispatch gpu e "d4_hc_post" (list d n ntok 1f-6) (list block streams pc out)
              (ceiling (* ntok n d) 256) 256)))

(defun ds4f-steer-captured (m)
  "After a pass with *STEER-CAPTURE*: each layer's captured rows given to it, in the CPU's order."
  (when *steer-capture*
    (let ((d (ds4f-d m)) (cap (d4gpu-buf (ds4f-gpu-state m) :steer-capture (* 4 2 (ds4f-n-layer m) (ds4f-d m)))))
      (dotimes (li (ds4f-n-layer m))
        (dolist (kind '(:attn :ffn))
          (funcall *steer-capture* li kind (gbuf-floats cap d (* d (+ (* 2 li) (if (eq kind :attn) 0 1))))))))))

(defun comp-bufs (st kind width)
  (values (d4gpu-buf st (if (= kind 0) :g-ckv :g-ikv) (* 4 width))
          (d4gpu-buf st (if (= kind 0) :g-csc :g-isc) (* 4 width))))

(defun enc-compressor-proj (m gpu e st li norm ratio wkv wgate hd kind)
  "One compressor's two projections of NORM (in the caller's TOGETHER)."
  (multiple-value-bind (kv sc) (comp-bufs st kind (* (if (= ratio 4) 2 1) hd))
    (enc-mv16 m gpu e (lname li wkv) norm kv)
    (enc-mv16 m gpu e (lname li wgate) norm sc)))

(defun enc-compressor-store (m gpu e st c pos ratio ape hd kind)
  "The projections into the compressor's window (in the caller's TOGETHER)."
  (let* ((width (* (if (= ratio 4) 2 1) hd)) (pos-mod (mod pos ratio))
         (row (if (= ratio 4) (+ ratio pos-mod) pos-mod)))
    (multiple-value-bind (kv sc) (comp-bufs st kind width)
      (dispatch gpu e "d4_comp_store" (list width row pos-mod (f32mat-cols ape))
                (list kv sc (gmat m ape)
                      (gpu-vector m (if (= kind 0) (dc-st-kv c) (dc-ix-kv c)))
                      (gpu-vector m (if (= kind 0) (dc-st-score c) (dc-ix-score c))))
                (ceiling width 256) 256))))

(defun enc-compressor-pool (m gpu e st li c pos ratio normw rows-of n-of hd kind)
  "On a window's last token, its row pooled and pushed (in the caller's TOGETHER)."
  (let ((width (* (if (= ratio 4) 2 1) hd)))
    (dispatch gpu e "d4_comp_pool" (list hd ratio width (ds4f-eps m) kind (ds4f-n-rot m) (funcall n-of c))
              (list (gpu-vector m (if (= kind 0) (dc-st-kv c) (dc-ix-kv c)))
                    (gpu-vector m (if (= kind 0) (dc-st-score c) (dc-ix-score c)))
                    (gpu-vector m normw) (arena-cs m st (- (1+ pos) ratio) li) (gpu-vector m (funcall rows-of c)))
              1 hd :shared (* 4 (+ hd 32)))))

(defun enc-attention (m gpu e st l li c pos sa sb)
  "Layer LI's attention block for the token at POS: streams SA into SB.
   What reads the same input goes TOGETHER: the projections of the norm,
   then what each makes of its own, and the indexer beside the queries."
  (let* ((d (ds4f-d m)) (hd (ds4f-head-dim m)) (nh (ds4f-n-head m)) (lq (ds4f-lora-q m))
         (nrot (ds4f-n-rot m)) (eps (ds4f-eps m)) (ratio (dl-ratio l)) (swa (ds4f-swa m))
         (cur (d4gpu-buf st :g-cur (* 4 d))) (norm (d4gpu-buf st :g-norm (* 4 d)))
         (pca (d4gpu-buf st :g-pca (* 4 32))) (qr (d4gpu-buf st :g-qr (* 4 lq)))
         (qrn (d4gpu-buf st :g-qrn (* 4 lq))) (q (d4gpu-buf st :g-q (* 4 nh hd)))
         (kv0 (d4gpu-buf st :g-kv0 (* 4 hd))) (heads (d4gpu-buf st :g-heads (* 4 nh hd)))
         (low (d4gpu-buf st :g-low (* 4 (ds4f-out-groups m) (ds4f-lora-o m))))
         (aout (d4gpu-buf st :g-aout (* 4 d)))
         (cs (arena-cs m st pos li)) (ics (arena-cs m st pos li t))
         (comp (plusp ratio)) (ixc (= ratio 4))
         (push (and comp (zerop (mod (1+ pos) ratio))))   ; a window's last token: a row pushed
         (top-k (ds4f-ix-top-k m))
         (indexed (and ixc (> (+ (dc-n-ix c) (if push 1 0)) top-k)))
         (ixh (ds4f-ix-heads m)) (ixd (ds4f-ix-dim m))
         (ixq (d4gpu-buf st :g-ixq (* 4 ixh ixd))) (ixw (d4gpu-buf st :g-ixw (* 4 ixh)))
         (idx (d4gpu-buf st :g-ixidx (* 4 top-k))))
    (enc-hc-pre m gpu e st (lname li "hc_attn_fn.weight") (dl-hc-attn-scale l) (dl-hc-attn-base l) sa cur pca)
    (enc-rms gpu e cur (gpu-vector m (dl-attn-norm l)) norm d 1 eps)
    (together (e)
      (q8mv-enc m gpu e li "attn_q_a.weight" norm qr)
      (q8mv-enc m gpu e li "attn_kv.weight" norm kv0)
      (when comp (enc-compressor-proj m gpu e st li norm ratio "attn_compressor_kv.weight" "attn_compressor_gate.weight" hd 0))
      (when ixc (enc-compressor-proj m gpu e st li norm ratio "indexer_compressor_kv.weight" "indexer_compressor_gate.weight" ixd 1))
      (when indexed (enc-mv16 m gpu e (lname li "indexer.proj.weight") norm ixw)))
    (together (e)
      (enc-rms gpu e qr (gpu-vector m (dl-q-a-norm l)) qrn lq 1 eps)
      ;; the KV row into the ring (its counts kept on the CPU, as PUSH-RAW! keeps them)
      (dispatch gpu e "d4_kv_fin" (list hd nrot eps (mod (dc-raw-total c) swa))
                (list kv0 (gpu-vector m (dl-kv-norm l)) cs (gpu-vector m (dc-raw c)))
                1 hd :shared (* 4 (+ hd 32)))
      (when comp (enc-compressor-store m gpu e st c pos ratio (dl-comp-ape l) hd 0))
      (when ixc (enc-compressor-store m gpu e st c pos ratio (dl-ixc-ape l) ixd 1)))
    (incf (dc-raw-total c))
    (setf (dc-n-raw c) (min swa (1+ (dc-n-raw c))))
    (together (e)
      (q8mv-enc m gpu e li "attn_q_b.weight" qrn q)
      (when indexed (enc-mv16 m gpu e (lname li "indexer.attn_q_b.weight") qrn ixq))
      (when push
        (enc-compressor-pool m gpu e st li c pos ratio (dl-comp-norm l) #'dc-comp #'dc-n-comp hd 0)
        (when ixc (enc-compressor-pool m gpu e st li c pos ratio (dl-ixc-norm l) #'dc-ix-comp #'dc-n-ix ixd 1))))
    (when push
      (incf (dc-n-comp c))
      (when ixc (incf (dc-n-ix c))))
    (let ((nix (dc-n-ix c)) (nc (if (dc-comp c) (dc-n-comp c) 0))
          (scores (d4gpu-buf st :g-ixs (* 4 (max 1 (dc-n-ix c))))))
      (together (e)
        (enc-rms gpu e q nil q hd nh eps)
        (when indexed
          (dispatch gpu e "d4_ixq_fin" (list ixh nrot (/ 1f0 (sqrt (float (* ixh ixd) 1f0))))
                    (list ixq cs ixw) ixh ixd :shared (* 4 (+ ixd 32)))))
      (together (e)
        (enc-rope gpu e q cs nh hd nrot)
        (when indexed
          (dispatch gpu e "d4_ix_scores" (list 1 nix ixh)
                    (list ixq ixw (gpu-vector m (dc-ix-comp c)) (arena-put st (list nix)) scores)
                    (list (ceiling nix 32) 1) 128 :shared (* 4 (+ (* 8 128) (* 32 128) (* 8 32))))))
      (when indexed
        (dispatch gpu e "d4_topk" (list nix top-k nix) (list scores idx (arena-put st (list nix))) 1 1024
                  :shared (* 4 96)))
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
    (enc-hc-post m gpu e aout sa pca sb 1 (cons li :attn))))

(defun route-bufs (m st)
  "Decode's routing results, a row a layer: (values sel w bo ok plogits)."
  (let ((nl (ds4f-n-layer m)) (nu (ds4f-n-used m)) (ne (ds4f-n-expert m)))
    (values (d4gpu-buf st :g-sel (* 4 nl nu)) (d4gpu-buf st :g-w (* 4 nl nu))
            (d4gpu-buf st :g-bo (* 8 2 nl nu)) (d4gpu-buf st :g-ok (* 4 nl))
            (d4gpu-buf st :g-plogits (* 4 nl ne)))))

(defun enc-ffn-head (m gpu e st l li sb token)
  "Layer LI's FFN up to the routing, and the routing: HC pre of SB, the
   norm, the router's logits (and the next layer's, its guess), the experts
   chosen (D4_ROUTE) and looked up in the expert pool (D4_SLOTS), into the
   layer's rows of ROUTE-BUFS.  The shared expert waits for the routed ones
   (ENC-FFN-TAIL), to run beside them."
  (let* ((d (ds4f-d m)) (ne (ds4f-n-expert m)) (nu (ds4f-n-used m)) (xc (ds4f-expert-cache m))
         (cur (d4gpu-buf st :g-cur (* 4 d))) (norm2 (d4gpu-buf st :g-norm2 (* 4 d)))
         (pcf (d4gpu-buf st :g-pcf (* 4 32))) (logits (d4gpu-buf st :g-logits (* 4 ne)))
         (bias (if (and (dl-probs-b-vl l) (or *d4-vision-position* (image-token-p token)))
                   (dl-probs-b-vl l)
                   (dl-probs-b l))))
    (multiple-value-bind (gsel gw gbo gok gpl) (route-bufs m st)
      (enc-hc-pre m gpu e st (lname li "hc_ffn_fn.weight") (dl-hc-ffn-scale l) (dl-hc-ffn-base l) sb cur pcf)
      (enc-rms gpu e cur (gpu-vector m (dl-ffn-norm l)) norm2 d 1 (ds4f-eps m))
      (together (e)
        (enc-mv16 m gpu e (lname li "ffn_gate_inp.weight") norm2 logits)
        ;; the next layer's router on this input: the guess at its experts (read ahead)
        (when (and (< (1+ li) (ds4f-n-layer m)) (null (dl-tid2eid (svref (ds4f-layers m) (1+ li)))))
          (enc-mv16 m gpu e (lname (1+ li) "ffn_gate_inp.weight") norm2 (cons gpl (* 4 ne (1+ li))))))
      (dispatch gpu e "d4_route" (list ne nu (if (dl-tid2eid l) 1 0) (ds4f-expert-scale m) (if bias 1 0))
                (list logits (if bias (gpu-vector m bias) logits)
                      (if (dl-tid2eid l) (tid2eid-buffer m st li l) logits) (arena-put st (list token))
                      (cons gsel (* 4 nu li)) (cons gw (* 4 nu li)))
                1 ne :shared (* 4 (+ (* 2 ne) 64 nu)))
      (dispatch gpu e "d4_slots" (list nu (* li ne) (d4gpu-slot-bytes st) (d4gpu-gate-bytes st))
                (list (cons gsel (* 4 nu li)) (dx-ready xc) (cons gbo (* 16 nu li)) (cons gok (* 4 li)))
                1 32))))

(defun enc-ffn-tail (m gpu e st l li sb sa)
  "Layer LI's routed experts (its rows of ROUTE-BUFS: their offsets in the
   pool, their weights) on its FFN input and the shared expert beside them,
   summed, and its FFN's HC post: streams SB into SA."
  (let* ((d (ds4f-d m)) (ff (ds4f-ff-exp m)) (k (ds4f-n-used m)) (clamp (svref (ds4f-clamps m) li))
         (xc (ds4f-expert-cache m)) (gb (d4gpu-gate-bytes st))
         (norm2 (d4gpu-buf st :g-norm2 (* 4 d)))
         (bo (multiple-value-bind (gsel gw gbo) (route-bufs m st) (declare (ignore gsel gw)) (cons gbo (* 16 k li))))
         (bw (multiple-value-bind (gsel gw) (route-bufs m st) (declare (ignore gsel)) (cons gw (* 4 k li))))
         (bgu (d4gpu-buf st :dgu (* 4 2 k ff))) (bm (d4gpu-buf st :dmid (* 4 k ff)))
         (bd (d4gpu-buf st :ddown (* 4 k d))) (moe (d4gpu-buf st :g-moe (* 4 d)))
         (sg (d4gpu-buf st :g-sg (* 4 ff))) (su (d4gpu-buf st :g-su (* 4 ff)))
         (smid (d4gpu-buf st :g-smid (* 4 ff))) (sout (d4gpu-buf st :g-sout (* 4 d)))
         (one (arena-put st (list 1f0) :float t))
         (grb (second (dl-gate-exps l))) (drb (second (dl-down-exps l))) (buf (dx-buf xc)))
    (flet ((mv (kernel ncols nrows rb add x xs y z)
             (dispatch gpu e kernel (list ncols nrows rb add xs nrows) (list buf bo x y)
                       (list (ceiling nrows 8) 1 z) 64
                       :shared (if (string= kernel "d4_mv_iq2_xxs") (+ (* 8 256) 128) 0))))
      (together (e)
        (mv "d4_mv_iq2_xxs" d ff grb 0 norm2 0 bgu (* 2 k))
        (q8mv-enc m gpu e li "ffn_gate_shexp.weight" norm2 sg)
        (q8mv-enc m gpu e li "ffn_up_shexp.weight" norm2 su))
      (together (e)
        (dispatch gpu e "d4_swiglu_w" (list k ff clamp) (list bgu (cons bgu (* 4 k ff)) bw bm)
                  (ceiling (* k ff) 256) 256)
        (dispatch gpu e "d4_swiglu_w" (list 1 ff clamp) (list sg su one smid) (ceiling ff 256) 256))
      (together (e)
        (mv "d4_mv_q2_k" ff d drb (* 2 gb) bm ff bd k)
        (q8mv-enc m gpu e li "ffn_down_shexp.weight" smid sout)))
    (dispatch gpu e "d4_moe_sum" (list d k) (list bd sout moe) (ceiling d 256) 256)
    (enc-hc-post m gpu e moe sb (d4gpu-buf st :g-pcf (* 4 32)) sa 1 (cons li :ffn))))

;;; --- a token ---------------------------------------------------------------------------

;;; A token's command buffers.  Buffer J runs layer J-1's routed experts,
;;; then layer J's attention and its FFN up to the routing, which the GPU
;;; does too, looking the experts up in the pool (ENC-FFN-HEAD); buffer N
;;; (the layer count) runs the last layer's experts and the head.  Buffer
;;; J+1 is queued before J is done, gated on J's lookup having found every
;;; expert ready (GATE-WORK): the GPU goes from layer to layer without
;;; waiting for the CPU.  When an expert was not there, J+1 does nothing;
;;; the CPU reads the expert (CACHE-EXPERTS, as before) and queues J+1
;;; again, its CPU-side counts set back first.  Meanwhile the CPU reads
;;; ahead the next layer's guessed experts.  While it does, the experts of
;;; the layers the GPU may be looking up stay where they are (the guard).

(defvar *ds4f-gpu-ahead* t
  "Whether decode queues a layer's command buffer before the one before it
   is done (gated on its experts being there); NIL: each after the CPU has
   looked at the one before, as it was (the GPU's routing all the same).")

(defun dc-counts (c) (list (dc-raw-total c) (dc-n-raw c) (dc-n-comp c) (dc-n-ix c)))
(defun (setf dc-counts) (v c)
  (destructuring-bind (rt nr nc ni) v
    (setf (dc-raw-total c) rt (dc-n-raw c) nr (dc-n-comp c) nc (dc-n-ix c) ni)))

(defun note-gpu-hits (m li sel)
  "Layer LI's experts SEL, all found in the pool by the GPU: noted as
   CACHE-EXPERTS notes hits (used now, pinned until the next layer's)."
  (let* ((xc (ds4f-expert-cache m)) (ne (ds4f-n-expert m)) (slot-of (dx-slot-of xc)) (used (dx-used xc))
         (pin (dx-pin xc)) (tick (incf (dx-tick xc))))
    (declare (type (simple-array fixnum (*)) slot-of used pin))
    (dolist (s (dx-decode-pins xc)) (decf (aref pin s)))
    (setf (dx-decode-pins xc) '())
    (loop for e across sel
          do (let ((s (aref slot-of (+ (* li ne) e))))
               (assert (>= s 0) () "the GPU used layer ~D's expert ~D, which the pool does not hold" li e)
               (setf (aref used s) tick) (incf (aref pin s)) (push s (dx-decode-pins xc)) (incf (dx-hits xc))))))

(defun ds4f-gpu-forward (ctx token logits)
  "DS4F-FORWARD-1 on the GPU: TOKEN at the next position."
  (let* ((m (d4ctx-model ctx)) (st (ds4f-gpu-state m)) (gpu (d4gpu-gpu st))
         (d (ds4f-d m)) (n (ds4f-n-hc m)) (hcd (* n d)) (pos (d4ctx-pos ctx)) (wf (ds4f-file m))
         (ne (ds4f-n-expert m)) (nu (ds4f-n-used m)) (nl (ds4f-n-layer m)) (xc (ds4f-expert-cache m))
         (sa (d4gpu-buf st :g-sa (* 4 hcd))) (sb (d4gpu-buf st :g-sb (* 4 hcd)))
         (counts (make-array (1+ nl) :initial-element nil)))
    (declare (type u8vec wf))
    (ds4f-gpu-weights m)
    (sync-ready xc)
    ;; the token's embedding into each stream
    (let ((o (+ (ds4f-embd-off m) (* token d 2))) (sap (gbuf-sap sa)))
      (dotimes (j d)
        (let ((v (f16-at wf (+ o (* 2 j)))))
          (dotimes (h n) (setf (sb-sys:sap-ref-single sap (* 4 (+ (* h d) j))) v)))))
    (multiple-value-bind (gsel gw gbo gok gpl) (route-bufs m st)
      (declare (ignore gw))
      (labels ((queue (j gated)
                 ;; command buffer J, committed (retained: it outlives the autorelease pool)
                 (let ((p (mod j 2)))
                   (when (< j nl) (setf (svref counts j) (dc-counts (svref (d4ctx-caches ctx) j))))
                   (arena-reset st p)
                   (with-metal
                     (with-autorelease
                       (let ((e (begin-work gpu)))
                         (msg ptr (enc-cb e) "retain")
                         (when gated
                           (gate-work gpu e (cons gok (* 4 (1- j)))
                                      (d4gpu-buf st (if (zerop p) :gate-src0 :gate-src1) (* 12 +gate-max+))
                                      (d4gpu-buf st (if (zerop p) :gate-dst0 :gate-dst1) (* 12 +gate-max+))))
                         (when (plusp j)
                           (enc-ffn-tail m gpu e st (svref (ds4f-layers m) (1- j)) (1- j) sb sa))
                         (if (< j nl)
                             (let ((l (svref (ds4f-layers m) j)))
                               (enc-attention m gpu e st l j (svref (d4ctx-caches ctx) j) pos sa sb)
                               (enc-ffn-head m gpu e st l j sb token))
                             (when logits (enc-head m gpu e st sa)))
                         (commit-work e)
                         e)))))
               (finish (e)
                 (with-metal (wait-work e))
                 (msg sb-alien:void (enc-cb e) "release")))
        (let* ((ahead *ds4f-gpu-ahead*) (cur (queue 0 nil)) (next (and ahead (queue 1 t))))
          (unwind-protect
               (dotimes (j nl)
                 (finish cur)
                 (setf (dx-guard xc) (cons j (1+ j)))
                 (let ((sel (make-array nu)))
                   (dotimes (i nu) (setf (aref sel i) (sb-sys:sap-ref-32 (gbuf-sap gsel) (* 4 (+ (* nu j) i)))))
                   (if (and next (/= 0 (sb-sys:sap-ref-32 (gbuf-sap gok) (* 4 j))))
                       (note-gpu-hits m j sel)
                       ;; an expert was missing: J+1 did nothing; read it, and queue J+1 again
                       (let ((slots (progn (when next (finish next))
                                           (when (and next (< (1+ j) nl))
                                             (setf (dc-counts (svref (d4ctx-caches ctx) (1+ j))) (svref counts (1+ j))))
                                           (cache-experts m (svref (ds4f-layers m) j) j sel)))
                             (sbytes (d4gpu-slot-bytes st)) (gb (d4gpu-gate-bytes st)))
                         (dotimes (i nu)
                           (setf (sb-sys:sap-ref-64 (gbuf-sap gbo) (* 8 (+ (* 2 nu j) i))) (* (aref slots i) sbytes)
                                 (sb-sys:sap-ref-64 (gbuf-sap gbo) (* 8 (+ (* 2 nu j) nu i))) (+ (* (aref slots i) sbytes) gb)))
                         (setf next (queue (1+ j) nil)))))
                 ;; the next layer's guessed experts read ahead
                 (when (and (plusp *ds4f-prefetch-k*) (< (1+ j) nl))
                   (let ((pl (gbuf-floats gpl ne (* ne (1+ j)))))
                     (prefetch-cache-experts m (1+ j) (predict-experts m (1+ j) token *ds4f-prefetch-k* pl))))
                 (setf cur next next (when (and ahead (<= (+ j 2) nl)) (queue (+ j 2) t))))
            (setf (dx-guard xc) nil))
          (finish cur))))
    (incf (d4ctx-pos ctx))
    (ds4f-steer-captured m)
    (when logits
      (copy-floats-out (d4ctx-logits ctx) (d4gpu-buf st :g-out 0) (ds4f-n-vocab m)))
    ctx))
