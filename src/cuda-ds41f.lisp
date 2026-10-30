;;;; cuda-ds41f.lisp — DeepSeek V4.1 Flash's decode on an NVIDIA GPU.
;;;
;;; ds41f.lisp's token on cuda-ds4f.lisp's machinery: its device state (the
;;; weights on the device, the routed experts pinned in host memory, the
;;; pool of slots, read-through), its routing on the CPU between a layer's
;;; head and its experts, its expert kernels (D4-PLAN, DK-FFN-TAIL).  What
;;; is V4.1's: the rounding (src/cuda/ds4f_v41.cu), HC's split apart from
;;; its sum, the shared caches and the indexer's pick reused down a group,
;;; Engram's rows read on the CPU and gated in on the device.  The caches
;;; are the context's vectors' device copies (DVEC), the context's vectors
;;; themselves left as they were.

(in-package #:illatio)

(defun dk41-cs (m st pos li &optional inverse)
  (dk-put st (d41-rope-cs m pos li inverse) :type :f32))

(defun dk41-round (dev x n &optional q)
  (launch dev "d41_round" (ceiling n 256) 256 (list n x (car q) (cdr q))))

(defun dk41-rms (dev x w out n eps &optional q)
  (launch dev "d41_rms" 1 (min 1024 n) (list n eps x w out (car q) (cdr q))))

(defun dk41-rope (dev x cs nh width nrot)
  (launch dev "d41_rope" (ceiling (* nh (floor nrot 2)) 256) 256 (list nh width nrot x cs)))

(defun dk41-quant (dev x n kind)
  (launch dev "d41_quant" (ceiling n 256) 256 (list n (ecase kind (:fp8 0) (:fp4 1) (:fp4-e4m3 2)) x)))

(defun dk41-hc (m st fn scale base streams pre out pc)
  "HC's mix of STREAMS by FN into PC (post, comb, and its own pre at 24),
   and OUT = the streams summed by PRE (the previous sublayer's), BF16."
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (n (ds4f-n-hc m)) (hcd (* n d))
         (flat (dbuf-of st :flat (* 4 hcd))) (mix (dbuf-of st :mix (* 4 64))))
    (dk-rms dev streams nil flat hcd 1 (ds4f-eps m))
    (dk-mv16 st fn flat mix)
    (launch dev "d41_hc_split" 1 32 (list (ds4f-sinkhorn m) 1f-6 mix (dvec st scale) (dvec st base) pc))
    (launch dev "d41_hc_sum" (ceiling d 256) 256 (list d n streams pre out))))

(defun dk41-engram (m st l li streams x)
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (n (ds4f-n-hc m))
         (kv (dbuf-of st :egkv (* 4 (1+ n) d))))
    (dk-mv16 st (lname li "engram_kv.weight") (dk-put st x :type :f32) kv)
    (dk41-round dev kv (* (1+ n) d))
    (launch dev "d41_engram" n 1024 (list d n (ds4f-eps m) streams kv (dvec st (dl-eg-q-norm l)) (dvec st (dl-eg-k-norm l))))))

(defun dk41-publish (m st l li ctx norm pos k)
  "D41-PUBLISH! on the device: the finished row and its key written in
   place in owner K's caches."
  (let* ((dev (dk-dev st)) (hd (ds4f-head-dim m)) (ixd (ds4f-ix-dim m)) (ratio (dl-ratio l)) (eps (ds4f-eps m))
         (nrot (ds4f-n-rot m)) (kv (dbuf-of st :ckv (* 4 hd))) (latent (dbuf-of st :clat (* 4 hd))))
    (dk-mv16 st (lname li "attn_compressor_kv.weight") norm kv)
    (if (= ratio 1)
        (progn (cuda-copy-dd latent 0 kv 0 (* 4 hd)) (dk41-round dev latent hd))
        (let ((sc (dbuf-of st :csc (* 4 hd))))
          (dk-mv16 st (lname li "attn_compressor_gate.weight") norm sc)
          (launch dev "d41_pool2" (ceiling hd 256) 256
                  (list hd (if (oddp pos) 1 0) kv sc (dvec st (svref (d41ctx-prev-kv ctx) k))
                        (dvec st (svref (d41ctx-prev-score ctx) k)) latent))))
    (when (zerop (mod (1+ pos) ratio))
      (let* ((n (floor (1+ pos) ratio)) (rpos (- (1+ pos) ratio)) (cs (dk41-cs m st rpos li))
             (row (cons (dvec st (svref (d41ctx-comp ctx) k)) (* 4 (1- n) hd)))
             (key (cons (dvec st (svref (d41ctx-ix-comp ctx) k)) (* 4 (1- n) ixd)))
             (ik (dbuf-of st :ik (* 4 ixd))))
        (dk41-rms dev latent (dvec st (dl-comp-norm l)) row hd eps)
        (dk-mv16 st (lname li "indexer.attn_k.weight") row ik)
        (dk41-round dev ik ixd)
        (dk41-rms dev ik (dvec st (dl-ix-k-norm l)) key ixd eps)
        (dk41-rope dev key cs 1 ixd nrot)
        (dk41-quant dev key ixd :fp4)
        (dk41-rope dev row cs 1 hd nrot)
        (dk41-quant dev row hd :fp4-e4m3)))))

(defun dk41-attention (m st l li ctx pos sa sb pre)
  "Layer LI's attention for the token at POS: streams SA into SB, summed by PRE."
  (let* ((v (ds4f-v41 m)) (dev (dk-dev st)) (d (ds4f-d m)) (hd (ds4f-head-dim m)) (nh (ds4f-n-head m))
         (lq (ds4f-lora-q m)) (nrot (ds4f-n-rot m)) (eps (ds4f-eps m)) (ratio (dl-ratio l)) (swa (ds4f-swa m))
         (top-k (ds4f-ix-top-k m))
         (cur (dbuf-of st :cur (* 4 d))) (norm (dbuf-of st :norm (* 4 d)))
         (pca (dbuf-of st :pca (* 4 32))) (qr (dbuf-of st :qr (* 4 lq)))
         (qrn (dbuf-of st :qrn (* 4 lq))) (q (dbuf-of st :q (* 4 nh hd)))
         (kv0 (dbuf-of st :kv0 (* 4 hd))) (heads (dbuf-of st :heads (* 4 nh hd)))
         (nlow (* (ds4f-out-groups m) (ds4f-lora-o m)))
         (low (dbuf-of st :low (* 4 nlow))) (aout (dbuf-of st :aout (* 4 d)))
         (raw (dvec st (svref (d41ctx-raw ctx) li))) (kvrow (cons raw (* 4 (mod pos swa) hd)))
         (cs (dk41-cs m st pos li)) (ics (dk41-cs m st pos li t))
         (qnorm (dk-qbuf st :norm d)) (qqrn (dk-qbuf st :qrn lq)) (qheads (dk-qbuf st :heads (* nh hd)))
         (qlow (dk-qbuf st :low nlow))
         (k nil) (attended 0) (ident 1) (idx (dbuf-of st :ixidx (* 4 top-k))))
    (dk41-hc m st (lname li "hc_attn_fn.weight") (dl-hc-attn-scale l) (dl-hc-attn-base l) sa pre cur pca)
    (dk41-rms dev cur (dvec st (dl-attn-norm l)) norm d eps qnorm)
    (dk-q8mv st (lname li "attn_q_a.weight") norm qr :q qnorm)
    (dk-q8mv st (lname li "attn_kv.weight") norm kv0 :q qnorm)
    (dk41-round dev qr lq)
    (dk41-round dev kv0 hd)
    (dk41-rms dev qr (dvec st (dl-q-a-norm l)) qrn lq eps qqrn)
    (dk-q8mv st (lname li "attn_q_b.weight") qrn q :q qqrn)
    (dk41-round dev q (* nh hd))
    (dk41-rope dev q cs nh hd nrot)
    ;; the KV row, normed, turned and through FP8, straight into its ring row
    (dk41-rms dev kv0 (dvec st (dl-kv-norm l)) kvrow hd eps)
    (dk41-rope dev kvrow cs 1 hd nrot)
    (dk41-quant dev kvrow hd :fp8)
    (when (plusp ratio)
      (setf k (d41-owner m li))
      (when (member li (d41-kv-src v)) (dk41-publish m st l li ctx norm pos k))
      (let ((nc (floor (1+ pos) ratio)))
        (when (> nc top-k)
          ;; more rows than are attended: the group's pick (made here on an index-source layer)
          (setf ident 0)
          (when (member li (d41-ix-src v))
            (let* ((ixh (ds4f-ix-heads m)) (ixd (ds4f-ix-dim m))
                   (ixq (dbuf-of st :ixq (* 4 ixh ixd))) (ixw (dbuf-of st :ixw (* 4 ixh)))
                   (scores (dbuf-of st :ixs (* 4 nc))))
              (assert (= ixd 128))
              (assert (<= (ceiling nc (d41-cand-block v)) (d41-cand-top v)) ()
                      "V4.1's candidate blocks (past ~D compressed rows) are not on the GPU yet"
                      (* (d41-cand-block v) (d41-cand-top v)))
              (dk-mv16 st (lname li "indexer.attn_q_b.weight") qrn ixq)
              (dk41-round dev ixq (* ixh ixd))
              (dk41-rope dev ixq cs ixh ixd nrot)
              (dk41-quant dev ixq (* ixh ixd) :fp4)
              (dk-mv16 st (lname li "indexer.proj.weight") norm ixw)
              (dk41-round dev ixw ixh)
              (launch dev "d41_ix_scores" (ceiling nc 8) 256
                      (list ixh nc ixq ixw (dvec st (svref (d41ctx-ix-comp ctx) k)) scores)
                      :shared (* 4 ixh ixd))
              (launch dev "d4_topk" 1 1024 (list nc top-k scores idx)))))
        (setf attended (min nc top-k))))
    (launch dev "d4_attend1" (floor nh 4) 128
            (list hd (/ 1f0 (sqrt (float hd 1f0))) swa (mod (- (1+ pos) (min (1+ pos) swa)) swa)
                  (min (1+ pos) swa) attended ident
                  q raw (if k (dvec st (svref (d41ctx-comp ctx) k)) raw) idx (dvec st (dl-sinks l)) heads)
            :shared (* 4 (+ (* 4 hd) (* 4 (+ (min (1+ pos) swa) attended)) 8)))
    (dk41-round dev heads (* nh hd))
    (dk41-rope dev heads ics nh hd nrot)
    (dk41-round dev heads (* nh hd) qheads)
    (dk-q8mv st (lname li "attn_output_a.weight") heads low :group-rows (ds4f-lora-o m) :q qheads)
    (dk41-round dev low nlow qlow)
    (dk-q8mv st (lname li "attn_output_b.weight") low aout :q qlow)
    (dk41-round dev aout d)
    (dk-hc-post m st aout sa pca sb)
    (dk41-round dev sb (* (ds4f-n-hc m) d))))

(defun dk41-ffn-head (m st l li sb)
  "Layer LI's FFN up to the routing: HC of SB (summed by the attention's
   pre), the norm, the router's logits, the shared expert into :SOUT."
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (ff (ds4f-ff-exp m)) (ne (ds4f-n-expert m))
         (cur (dbuf-of st :cur (* 4 d))) (norm2 (dbuf-of st :norm2 (* 4 d)))
         (pca (dbuf-of st :pca (* 4 32))) (pcf (dbuf-of st :pcf (* 4 32))) (logits (dbuf-of st :logits (* 4 2 ne)))
         (sg (dbuf-of st :sg (* 4 ff))) (su (dbuf-of st :su (* 4 ff)))
         (smid (dbuf-of st :smid (* 4 ff))) (sout (dbuf-of st :sout (* 4 d)))
         (qn2 (dk-qbuf st :norm2 d)) (qmid (dk-qbuf st :smid ff)))
    (dk41-hc m st (lname li "hc_ffn_fn.weight") (dl-hc-ffn-scale l) (dl-hc-ffn-base l) sb (cons pca 96) cur pcf)
    (dk41-rms dev cur (dvec st (dl-ffn-norm l)) norm2 d (ds4f-eps m) qn2)
    (dk-mv16 st (lname li "ffn_gate_inp.weight") norm2 logits)
    (dk-q8mv st (lname li "ffn_gate_shexp.weight") norm2 sg :q qn2)
    (dk-q8mv st (lname li "ffn_up_shexp.weight") norm2 su :q qn2)
    (launch dev "d41_swiglu" (ceiling ff 256) 256
            (list ff (svref (ds4f-clamps m) li) sg su smid (car qmid) (cdr qmid)))
    (dk-q8mv st (lname li "ffn_down_shexp.weight") smid sout :q qmid)
    (dk41-round dev sout d)))

(defun ds41f-cuda-forward (ctx token logits)
  "DS41F-FORWARD-1 on the GPU: TOKEN at the next position."
  (let* ((m (d41ctx-model ctx)) (v (ds4f-v41 m)) (st (ds4f-cuda-state m)) (dev (dk-dev st))
         (d (ds4f-d m)) (n (ds4f-n-hc m)) (hcd (* n d)) (pos (d41ctx-pos ctx)) (wf (ds4f-file m))
         (ne (ds4f-n-expert m)) (nl (ds4f-n-layer m)) (rb (dk-rb st))
         (sa (dbuf-of st :sa (* 4 hcd))) (sb (dbuf-of st :sb (* 4 hcd)))
         (lg (f32vec ne)) (prev nil))
    (declare (type u8vec wf))
    (with-cuda
      (cuda-current dev)
      (let ((e (f32vec hcd)) (o (+ (ds4f-embd-off m) (* token d 2))))
        (dotimes (j d)
          (let ((x (f16-at wf (+ o (* 2 j))))) (dotimes (h n) (setf (aref e (+ (* h d) j)) x))))
        (cuda-upload-vector sa e))
      (multiple-value-bind (rows tail) (d41-engram-hash m (d41ctx-tail ctx) token)
        (let ((pre (dk-put st '(1f0 0f0 0f0 0f0) :type :f32)))
          (dotimes (li nl)
            (let ((l (svref (ds4f-layers m) li)) (tbl (position li (d41-eg-layers v))))
              (when tbl (dk41-engram m st l li sa (d41-engram-input m tbl (nth tbl rows))))
              (dk41-attention m st l li ctx pos sa sb pre)
              (dk41-ffn-head m st l li sb)
              ;; the routing, on the CPU
              (cuda-copy-out rb (dbuf-of st :logits 0) 0 (* 4 ne))
              (cuda-sync dev)
              (dk-arena-reset st)
              (dotimes (i ne) (setf (aref lg i) (sb-sys:sap-ref-single rb (* 4 i))))
              (multiple-value-bind (sel w) (d4-route m l li nil token lg)
                (when *ds4f-route-trace* (format *ds4f-route-trace* "~D~{ ~D~}~%" li (coerce sel 'list)))
                (let ((plan (d4-plan m st li sel prev)))
                  (dk-ffn-tail m st l li plan w sb sa :round t)
                  (setf prev plan)))
              (setf pre (cons (dbuf-of st :pcf 0) 96)))))
        (setf (d41ctx-tail ctx) tail))
      (when logits
        (let* ((q (dk-qbuf st :hnorm d)) (embd (dbuf-of st :hembd (* 4 d))) (norm (dbuf-of st :hnorm (* 4 d)))
               (out (dbuf-of st :out (* 4 (ds4f-n-vocab m)))) (lv (d41ctx-logits ctx)))
          (launch dev "d41_hc_sum" (ceiling d 256) 256 (list d n sa (cons (dbuf-of st :pcf 0) 96) embd))
          (dk41-rms dev embd (dvec st (ds4f-output-norm m)) norm d (ds4f-eps m) q)
          (dk-q8mv st "output.weight" norm out :q q)
          (sb-sys:with-pinned-objects (lv)
            (cuda-download out (sb-sys:vector-sap lv) (* 4 (ds4f-n-vocab m))))))
      (loop for s across (dp-a-slots prev) when (>= s 0) do (decf (aref (dk-pin st) s)))
      (cuda-sync dev))
    (incf (d41ctx-pos ctx))
    ctx))
