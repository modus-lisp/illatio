;;;; cuda-glm53.lisp — GLM-5.3 Flash's decode on an NVIDIA GPU, its routed
;;;; experts in host memory.
;;;
;;; As cuda-ds4f.lisp's DeepSeek decode, whose machinery it is: the experts
;;; pinned in host memory, a pool of them on the device (and a fixed set on
;;; the second), the routing on the CPU between a layer's FFN head and its
;;; experts; HC, the norms, the shared and routed experts are its kernels.
;;; GLM's own (src/cuda/glm53.cu): the BF16 products, KDA's conv and delta
;;; rule, DSA's F16 latent cache, pooled indexer and absorbed attention.
;;; A context's state lives on the device alone (G53CTX-DEV).

(in-package #:illatio)

(defun g53-layer-tensors (m li)
  "The device's tensors of trunk layer LI (short names)."
  (let ((l (svref (ds4f-layers m) li)))
    (append '("hc_attn_fn.weight" "hc_ffn_fn.weight")
            (if (eq (gl-kind l) :kda)
                '("kda_q.weight" "kda_k.weight" "kda_v.weight" "kda_f_a.weight" "kda_f_b.weight"
                  "kda_beta.weight" "kda_g_a.weight" "kda_g_b.weight" "kda_output.weight")
                '("attn_q_a.weight" "attn_q_b.weight" "attn_kv_a_mqa.weight" "attn_k_b.weight" "attn_v_b.weight"
                  "attn_output.weight" "indexer.attn_q_b.weight" "indexer.attn_k.weight" "indexer.proj.weight"
                  "indexer.pool_gate.weight"))
            (if (< li (g53-n-dense m))
                '("ffn_gate.weight" "ffn_up.weight" "ffn_down.weight")
                '("ffn_gate_inp.weight" "ffn_gate_shexp.weight" "ffn_up_shexp.weight" "ffn_down_shexp.weight")))))

(defun g53-upload-weights (m st)
  "The trunk's non-routed tensors to the device: Q8_0 repacked (DK-Q8), a
   K-quant's blocks as they are (DK-Q8 too, its CMAT's type says), BF16,
   F16 and F32 as they are (DK-F16)."
  (let* ((g (ds4f-gguf m)) (dev (dk-dev st)) (fd (d4-open m)) (raw nil)
         (names (cons (if *g53-trunk-limit* "token_embd.weight" "output.weight")
                      (loop for li below (g53-n-trunk m)
                            append (mapcar (lambda (s) (lname li s)) (g53-layer-tensors m li)))))
         (stage (cuda-host-alloc (loop for name in names maximize (tensor-bytes (gguf-tensor g name))))))
    (unwind-protect
         (dolist (name names)
           (let* ((ti (gguf-tensor g name)) (dims (tensor-info-dims ti)) (type (tensor-info-type ti))
                  (cols (first dims)) (rows (reduce #'* (rest dims))) (bytes (tensor-bytes ti))
                  (off (+ (gguf-data-offset g) (tensor-info-offset ti))) (piece (* 16 1024 1024)))
             (parallel-for (k (ceiling bytes piece) (ds4f-pool m))
               (let ((o (* k piece))) (d4-pread fd (sb-sys:sap+ stage o) (+ off o) (min piece (- bytes o)))))
             (ecase type
               (:q8_0
                (let* ((nb (floor (* rows cols) 32))
                       (qs (cuda-alloc dev (* rows cols) :zero nil)) (d (cuda-alloc dev (* 2 nb) :zero nil)))
                  (unless (and raw (>= (dbuf-bytes raw) bytes))
                    (cuda-free raw)
                    (setf raw (cuda-alloc dev bytes :zero nil)))
                  (cuda-upload raw stage bytes)
                  (launch dev "repack_q8" (ceiling nb 8) '(32 8) (list raw qs d (list :i64 nb)))
                  (setf (gethash name (dk-q8 st)) (%make-cmat :name name :type :q8_0 :rows rows :cols cols :qs qs :d d))))
               ((:q4_k :q5_k)
                (let ((qs (cuda-alloc dev bytes :zero nil)))
                  (cuda-upload qs stage bytes)
                  (setf (gethash name (dk-q8 st)) (%make-cmat :name name :type type :rows rows :cols cols :qs qs))))
               ((:bf16 :f16 :f32)
                (let ((b (cuda-alloc dev bytes :zero nil)))
                  (cuda-upload b stage bytes)
                  (setf (gethash name (dk-f16 st)) (list b rows cols type)))))))
      (cuda-free raw)
      (cuda-host-free stage)
      (sb-unix:unix-close fd))))

(defun g53-cuda-state (m)
  "G53 M's device state, made the first time (DS4F-CUDA-STATE's, GLM's tensors)."
  (or (getf (ds4f-gpu-plist m) :cuda)
      (let* ((dev (open-cuda *ds4f-cuda-device*))
             (l0 (find-if #'gl-gate-exps (ds4f-layers m)))
             (gb (* (third (dl-gate-exps l0)) (second (dl-gate-exps l0))))
             (db (* (third (dl-down-exps l0)) (second (dl-down-exps l0))))
             (st (make-d4cuda :dev dev :slot-bytes (* 256 (ceiling (+ (* 2 gb) db) 256)) :gate-bytes gb)))
        (with-cuda
          (cuda-current dev)
          (setf (dk-copy-stream st) (cuda-stream-create dev) (dk-fill-stream st) (cuda-stream-create dev)
                (dk-event st) (cuda-event-create dev)
                (dk-arena-host st) (cuda-host-alloc (* 1024 1024)) (dk-arena-dev st) (cuda-alloc dev (* 1024 1024))
                (dk-rb st) (cuda-host-alloc (* 64 1024)))
          (multiple-value-bind (sap bytes) (d4-host-experts m (dk-slot-bytes st) gb :node (cudev-numa-node dev))
            (setf (dk-host st) sap (dk-host-bytes st) bytes))
          (g53-upload-weights m st)
          (d4-make-pool m st)
          (when (and *ds4f-cuda-tier-device* (> (cuda-device-count) *ds4f-cuda-tier-device*)
                     (/= *ds4f-cuda-tier-device* *ds4f-cuda-device*))
            (setf (dk-tier st) (d4-make-tier m st (open-cuda *ds4f-cuda-tier-device*)))
            (cuda-current dev))
          #+x86-64
          (when *ds4f-cpu-experts*
            (setf (dk-cpu st) (d4-make-cpu m))))
        (setf (getf (ds4f-gpu-plist m) :cuda) st))))

;;; --- a context's state on the device ------------------------------------------------

(defun g53-ctx-dev (ctx st)
  "CTX's per-layer device state, made (zeroed) the first time: a KDA
   layer's (hist . S), a DSA layer's (latents ixkeys tail)."
  (or (g53ctx-dev ctx)
      (let* ((m (d4ctx-model ctx)) (dev (dk-dev st)) (p (* (g53-kda-heads m) (g53-kda-dim m)))
             (n (d4ctx-n-ctx ctx)) (ixd (ds4f-ix-dim m)))
        (setf (g53ctx-dev ctx)
              (map 'simple-vector
                   (lambda (l)
                     (if (eq (gl-kind l) :kda)
                         (list (cuda-alloc dev (* 4 9 p)) (cuda-alloc dev (* 4 p (g53-kda-dim m))))
                         (list (cuda-alloc dev (* 2 n (g53-kv-lora m))) (cuda-alloc dev (* 2 (ceiling n 4) ixd))
                               (cuda-alloc dev (* 4 8 ixd)))))
                   (ds4f-layers m))))))

(defun free-g53ctx (ctx)
  (when (g53ctx-dev ctx)
    (loop for bufs across (g53ctx-dev ctx) do (mapc #'cuda-free bufs))
    (setf (g53ctx-dev ctx) nil)))

;;; --- the products ------------------------------------------------------------------

(defun dk-mv (st name x y &key q)
  "Y = W X for tensor NAME, whatever its kind: Q8_0 (X quantized, or Q's),
   a K-quant (X quantized as kquant.cu wants), BF16 / F16 / F32."
  (let ((w (gethash name (dk-q8 st))))
    (cond ((null w) (dk-mv16 st name x y))
          ((eq (cmat-type w) :q8_0) (dk-q8mv st name x y :q q))
          (t (let* ((dev (dk-dev st)) (cols (cmat-cols w)) (rows (cmat-rows w)) (nb (floor cols 32))
                    (xq (dbuf-of st :kx-q cols)) (xs (dbuf-of st :kx-s (* 4 nb))) (xsum (dbuf-of st :kx-sum (* 4 nb))))
               (launch dev "quant_q8" (ceiling cols 256) 256 (list x xq xs xsum (list :i64 nb)))
               (launch dev "mv_k" (ceiling rows 4) 128
                       (list (kquant-code (cmat-type w)) (cmat-qs w) xq xs xsum y nil rows cols 0)))))))

;;; --- the blocks --------------------------------------------------------------------

(defun g53k-attn-in (m st l li sa)
  "HC pre of SA for the attention, normed: (values norm qnorm pca)."
  (let* ((dev (dk-dev st)) (d (ds4f-d m))
         (cur (dbuf-of st :cur (* 4 d))) (norm (dbuf-of st :norm (* 4 d))) (pca (dbuf-of st :pca (* 4 32)))
         (qnorm (dk-qbuf st :norm d)))
    (dk-hc-pre m st (lname li "hc_attn_fn.weight") (gl-hc-attn-scale l) (gl-hc-attn-base l) sa cur pca)
    (dk-rms dev cur (dvec st (gl-attn-norm l)) norm d 1 (ds4f-eps m) qnorm)
    (values norm qnorm pca)))

(defun g53k-kda (m st l li bufs sa sb)
  "KDA layer LI for the next token: streams SA into SB."
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (nh (g53-kda-heads m)) (hd (g53-kda-dim m)) (p (* nh hd))
         (qkv (dbuf-of st :qkv (* 4 3 p))) (flow (dbuf-of st :flow (* 4 128))) (graw (dbuf-of st :graw (* 4 p)))
         (braw (dbuf-of st :braw (* 4 nh))) (glow (dbuf-of st :glow (* 4 128))) (og (dbuf-of st :og (* 4 p)))
         (o (dbuf-of st :ko (* 4 p))) (aout (dbuf-of st :aout (* 4 d))))
    (assert (= hd 128))
    (multiple-value-bind (norm qnorm pca) (g53k-attn-in m st l li sa)
      (dk-mv st (lname li "kda_q.weight") norm qkv :q qnorm)
      (dk-mv st (lname li "kda_k.weight") norm (cons qkv (* 4 p)) :q qnorm)
      (dk-mv st (lname li "kda_v.weight") norm (cons qkv (* 4 2 p)) :q qnorm)
      (dk-mv st (lname li "kda_f_a.weight") norm flow :q qnorm)
      (dk-mv st (lname li "kda_f_b.weight") flow graw)
      (dk-mv st (lname li "kda_beta.weight") norm braw :q qnorm)
      (dk-mv st (lname li "kda_g_a.weight") norm glow :q qnorm)
      (dk-mv st (lname li "kda_g_b.weight") glow og)
      (launch dev "g53_kda_conv" (ceiling (* 3 p) 256) 256 (list p qkv (dvec st (gl-conv3 l)) (first bufs)))
      (launch dev "g53_kda_step" nh 128
              (list p (g53-kda-lb m) (ds4f-eps m) qkv graw braw og (dvec st (gl-dt-bias l)) (dvec st (gl-a-log l))
                    (dvec st (gl-o-norm l)) (second bufs) o))
      (dk-mv st (lname li "kda_output.weight") o aout)
      (dk-hc-post m st aout sa pca sb))))

(defun g53k-dsa (m st l li bufs pos sa sb)
  "DSA layer LI for the token at POS: streams SA into SB."
  (destructuring-bind (cache ixk tail) bufs
    (let* ((dev (dk-dev st)) (d (ds4f-d m)) (eps (ds4f-eps m)) (nh (ds4f-n-head m)) (kd (ds4f-head-dim m))
           (vd (g53-v-dim m)) (lat (g53-kv-lora m)) (lq (ds4f-lora-q m)) (ixh (ds4f-ix-heads m)) (ixd (ds4f-ix-dim m))
           (np (g53-ix-pool m)) (visible (1+ pos))
           (qr (dbuf-of st :qr (* 4 lq))) (qrn (dbuf-of st :qrn (* 4 lq))) (q (dbuf-of st :q (* 4 nh kd)))
           (kvr (dbuf-of st :kvr (* 4 lat))) (kr (dbuf-of st :ixkr (* 4 ixd))) (gr (dbuf-of st :ixgr (* 4 ixd)))
           (qk (dbuf-of st :qk (* 4 nh lat))) (lv (dbuf-of st :lv (* 4 nh lat)))
           (heads (dbuf-of st :heads (* 4 nh vd))) (aout (dbuf-of st :aout (* 4 d)))
           (qqrn (dk-qbuf st :qrn lq)) (sel nil) (ns visible))
      (assert (and (= lat 512) (= ixd 128) (= np 4)))
      (multiple-value-bind (norm qnorm pca) (g53k-attn-in m st l li sa)
        (dk-mv st (lname li "attn_q_a.weight") norm qr :q qnorm)
        (dk-mv st (lname li "attn_kv_a_mqa.weight") norm kvr :q qnorm)
        (dk-rms dev qr (dvec st (gl-q-a-norm l)) qrn lq 1 eps qqrn)
        (dk-mv st (lname li "attn_q_b.weight") qrn q :q qqrn)
        (launch dev "g53_kv_store" 1 lat (list lat eps pos kvr (dvec st (gl-kv-norm l)) cache))
        ;; the indexer's pool
        (dk-mv st (lname li "indexer.attn_k.weight") norm kr)
        (dk-mv st (lname li "indexer.pool_gate.weight") norm gr)
        (launch dev "g53_ix_pool" 1 ixd
                (list ixd (mod pos np) (floor pos np) kr gr tail (dvec st (gl-ix-k-norm l)) (dvec st (gl-ix-k-bias l))
                      (dvec st (gl-ix-ape l)) ixk))
        (when (> visible (g53-dense-limit m))
          (let* ((rows (floor visible np)) (k (floor (ds4f-ix-top-k m) np)) (ntail (mod visible np))
                 (iq (dbuf-of st :ixq (* 4 ixh ixd))) (iw (dbuf-of st :ixw (* 4 ixh)))
                 (scores (dbuf-of st :ixs (* 4 rows))) (idx (dbuf-of st :ixidx (* 4 k))))
            (setf sel (dbuf-of st :sel (* 4 (+ (* np k) np))) ns (+ (* np k) ntail))
            (dk-mv st (lname li "indexer.attn_q_b.weight") qrn iq)
            (dk-mv st (lname li "indexer.proj.weight") norm iw)
            (launch dev "g53_ix_scores" (ceiling rows 8) 256
                    (list ixh rows (/ 1f0 (sqrt (float (* ixh ixd) 1f0))) iq iw ixk scores))
            (launch dev "d4_topk" 1 1024 (list rows k scores idx))
            (launch dev "g53_expand" (ceiling ns 256) 256 (list k ntail (- visible ntail) idx sel))))
        ;; the query into the latent, attention there, the heads out of it
        (dk-q8mv st (lname li "attn_k_b.weight") q qk :group-rows lat)
        (launch dev "g53_attend" nh 256 (list lat ns (/ 1f0 (sqrt (float kd 1f0))) qk cache sel lv)
                :shared (* 4 ns))
        (dk-q8mv st (lname li "attn_v_b.weight") lv heads :group-rows vd)
        (dk-mv st (lname li "attn_output.weight") heads aout)
        (dk-hc-post m st aout sa pca sb)))))

(defun g53k-dense-ffn (m st l li sb sa)
  "Dense FFN layer LI: streams SB into SA."
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (ff (cmat-rows (gethash (lname li "ffn_gate.weight") (dk-q8 st))))
         (cur (dbuf-of st :cur (* 4 d))) (norm2 (dbuf-of st :norm2 (* 4 d))) (pcf (dbuf-of st :pcf (* 4 32)))
         (g (dbuf-of st :fg (* 4 ff))) (u (dbuf-of st :fu (* 4 ff))) (mid (dbuf-of st :fmid (* 4 ff)))
         (out (dbuf-of st :sout (* 4 d))) (qn2 (dk-qbuf st :norm2 d)) (qmid (dk-qbuf st :fmid ff))
         (one (dk-put st (list 1f0) :type :f32)))
    (dk-hc-pre m st (lname li "hc_ffn_fn.weight") (gl-hc-ffn-scale l) (gl-hc-ffn-base l) sb cur pcf)
    (dk-rms dev cur (dvec st (gl-ffn-norm l)) norm2 d 1 (ds4f-eps m) qn2)
    (dk-mv st (lname li "ffn_gate.weight") norm2 g :q qn2)
    (dk-mv st (lname li "ffn_up.weight") norm2 u :q qn2)
    (launch dev "d4_swiglu_w" (ceiling ff 256) 256 (list 1 ff (g53-clamp m) g u one mid (car qmid) (cdr qmid)))
    (dk-mv st (lname li "ffn_down.weight") mid out :q qmid)
    (dk-hc-post m st out sb pcf sa)))

(defun g53k-head (m st streams)
  "The streams' mean, normed, to the logits (:OUT)."
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (h (dbuf-of st :hembd (* 4 d))) (norm (dbuf-of st :hnorm (* 4 d)))
         (out (dbuf-of st :out (* 4 (ds4f-n-vocab m)))) (q (dk-qbuf st :hnorm d)))
    (launch dev "g53_mean" (ceiling d 256) 256 (list d (ds4f-n-hc m) streams h))
    (dk-rms dev h (dvec st (ds4f-output-norm m)) norm d 1 (ds4f-eps m) q)
    (dk-q8mv st (if *g53-trunk-limit* "token_embd.weight" "output.weight") norm out :q q)
    out))

;;; --- a token -------------------------------------------------------------------------

(defun g53-cuda-forward (ctx token logits)
  "G53-FORWARD-1 on the GPU: TOKEN at the next position."
  (let* ((m (d4ctx-model ctx)) (st (g53-cuda-state m)) (dev (dk-dev st))
         (d (ds4f-d m)) (n (ds4f-n-hc m)) (hcd (* n d)) (pos (d4ctx-pos ctx))
         (ne (ds4f-n-expert m)) (rb (dk-rb st))
         (sa (dbuf-of st :sa (* 4 hcd))) (sb (dbuf-of st :sb (* 4 hcd)))
         (lg (f32vec ne)) (prev nil))
    (assert (< pos (d4ctx-n-ctx ctx)) () "GLM context full (~D tokens)" (d4ctx-n-ctx ctx))
    (with-cuda
      (cuda-current dev)
      (let ((bufs (g53-ctx-dev ctx st)))
        (cuda-upload-vector sa (g53-embed! m token (f32vec hcd)))
        (dotimes (li (g53-n-trunk m))
          (let ((l (svref (ds4f-layers m) li)) (b (svref bufs li)))
            (if (eq (gl-kind l) :kda)
                (g53k-kda m st l li b sa sb)
                (g53k-dsa m st l li b pos sa sb))
            (if (< li (g53-n-dense m))
                (g53k-dense-ffn m st l li sb sa)
                (progn
                  (dk-ffn-head m st l li sb)
                  (cuda-copy-out rb (dbuf-of st :logits 0) 0 (* 4 ne))
                  (when (or (dk-tier st) (dk-cpu st)) (cuda-copy-out (sb-sys:sap+ rb 4096) (dbuf-of st :norm2 0) 0 (* 4 d)))
                  (cuda-sync dev)
                  (dk-arena-reset st)
                  (dotimes (i ne) (setf (aref lg i) (sb-sys:sap-ref-single rb (* 4 i))))
                  (multiple-value-bind (sel w) (g53-route m l nil lg)
                    (when *ds4f-route-trace* (format *ds4f-route-trace* "~D~{ ~D~}~%" li (coerce sel 'list)))
                    (let ((plan (d4-plan m st li sel prev)))
                      (dk-ffn-tail m st l li plan w sb sa)
                      (setf prev plan))))))))
      (when logits
        (let ((out (g53k-head m st sa)) (v (d4ctx-logits ctx)))
          (sb-sys:with-pinned-objects (v)
            (cuda-download out (sb-sys:vector-sap v) (* 4 (ds4f-n-vocab m))))))
      (when prev (loop for s across (dp-a-slots prev) when (>= s 0) do (decf (aref (dk-pin st) s))))
      (cuda-sync dev))
    (incf (d4ctx-pos ctx))
    ctx))

;;; --- a prompt, layer-major --------------------------------------------------------------
;;;
;;; As DS4F-CUDA-PREFILL-1: a pass's tokens through a layer before the next,
;;; its scratch borrowed from the pool, the next layer's experts copied in
;;; while a layer computes.  KDA's recurrence goes a token at a time (a warp
;;; a value row, its state in registers); DSA's attention is D4_ATTEND's
;;; (64 heads of 512, the key the value: GLM's latent, the queries absorbed),
;;; over a float copy of the latents, each token its prefix or its selection.

(defparameter *g53-cuda-chunk-from* 32
  "From this many tokens a prompt runs layer-major on the device.")

(defparameter *g53-cuda-chunk* 2048 "Tokens a pass takes at most.")

(defparameter *g53-cuda-routed-below* 1024
  "A pass shorter than this takes only the experts its tokens route to (read
   once routed), not all of each layer's ahead.")

(defparameter *g53-cuda-chunk-bytes* (* 1700 1024) "Device memory a token of a pass takes, at most.")

(defun dk-kmm (st name x y n)
  "Y = W X over N tokens for K-quant tensor NAME: kquant.cu's tiled product."
  (let* ((w (gethash name (dk-q8 st))) (dev (dk-dev st)) (cols (cmat-cols w)) (rows (cmat-rows w))
         (nb (floor (* n cols) 32))
         (xq (pb st :p-kxq (ceiling (* n cols) 4))) (xs (pb st :p-kxs nb)) (xsum (pb st :p-kxsum nb)))
    (launch dev "quant_q8" (ceiling (* n cols) 256) 256 (list x xq xs xsum (list :i64 nb)))
    (if (>= (cudev-arch dev) 80)
        (launch dev (ecase (cmat-type w) (:q4_k "mm_q4k_mma") (:q5_k "mm_q5k_mma"))
                (list (ceiling n 64) (ceiling rows 128)) 256 (list (cmat-qs w) xq xs xsum y nil rows cols n 0))
        (launch dev "mm_k" (list (ceiling n 64) (ceiling rows 64)) 256
                (list (kquant-code (cmat-type w)) (cmat-qs w) xq xs xsum y nil rows cols n 0)))))

(defun dk-mmx (st name x y n &rest keys)
  "Y = W X over N tokens, whatever W's kind."
  (let ((w (gethash name (dk-q8 st))))
    (cond ((null w) (apply #'dk-f16mm st name x y n keys))
          ((eq (cmat-type w) :q8_0) (apply #'dk-q8mm st name x y n keys))
          (t (assert (null keys)) (dk-kmm st name x y n)))))

(defun g53p-attn-in (m st l li sa n)
  "HC pre of the chunk's SA for the attention, normed: (values norm pca)."
  (let* ((dev (dk-dev st)) (d (ds4f-d m))
         (cur (pb st :p-cur (* n d))) (norm (pb st :p-norm (* n d))) (pca (pb st :p-pca (* n 32))))
    (dk-hc-pre-chunk m st li :attn sa n cur pca)
    (dk-rms dev cur (dvec st (gl-attn-norm l)) norm d n (ds4f-eps m))
    (values norm pca)))

(defun g53p-kda (m st l li bufs n sa sb)
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (nh (g53-kda-heads m)) (p (* nh (g53-kda-dim m)))
         (q (pb st :p-kq (* n p))) (k (pb st :p-kk (* n p))) (v (pb st :p-kv (* n p)))
         (flow (pb st :p-flow (* n 128))) (graw (pb st :p-graw (* n p))) (braw (pb st :p-braw (* n nh)))
         (glow (pb st :p-glow (* n 128))) (og (pb st :p-og (* n p))) (o (pb st :p-ko (* n p)))
         (aout (pb st :p-aout (* n d))))
    (multiple-value-bind (norm pca) (g53p-attn-in m st l li sa n)
      (dk-mmx st (lname li "kda_q.weight") norm q n)
      (dk-mmx st (lname li "kda_k.weight") norm k n)
      (dk-mmx st (lname li "kda_v.weight") norm v n)
      (dk-mmx st (lname li "kda_f_a.weight") norm flow n)
      (dk-mmx st (lname li "kda_f_b.weight") flow graw n)
      (dk-mmx st (lname li "kda_beta.weight") norm braw n)
      (dk-mmx st (lname li "kda_g_a.weight") norm glow n)
      (dk-mmx st (lname li "kda_g_b.weight") glow og n)
      (launch dev "g53_kda_conv_chunk" (ceiling (* 3 p) 256) 256 (list p n q k v (dvec st (gl-conv3 l)) (first bufs)))
      (launch dev "g53_kda_prep" (list nh n) 128
              (list p (g53-kda-lb m) q k graw braw (dvec st (gl-dt-bias l)) (dvec st (gl-a-log l))))
      (launch dev "g53_kda_rec" (list nh 32) 128 (list p n q k v graw braw (second bufs) o))
      (launch dev "g53_kda_out" (list nh n) 128 (list p (ds4f-eps m) o og (dvec st (gl-o-norm l))))
      (dk-mmx st (lname li "kda_output.weight") o aout n)
      (dk-hc-post-chunk m st aout sa pca sb n))))

(defun g53p-dsa (m st l li bufs pos0 n sa sb)
  (destructuring-bind (cache ixk tail) bufs
    (let* ((dev (dk-dev st)) (d (ds4f-d m)) (eps (ds4f-eps m)) (nh (ds4f-n-head m)) (kd (ds4f-head-dim m))
           (vd (g53-v-dim m)) (lat (g53-kv-lora m)) (lq (ds4f-lora-q m)) (ixh (ds4f-ix-heads m)) (ixd (ds4f-ix-dim m))
           (end (+ pos0 n)) (dense (g53-dense-limit m)) (sparse (> end dense))
           (top (floor (ds4f-ix-top-k m) (g53-ix-pool m))) (maxc (+ (* 4 top) 3)) (rows (floor end 4))
           (qr (pb st :p-qr (* n lq))) (qrn (pb st :p-qrn (* n lq))) (q (pb st :p-q (* n nh kd)))
           (kvr (pb st :p-kvr (* n lat))) (kr (pb st :p-ixkr (* n ixd))) (gr (pb st :p-ixgr (* n ixd)))
           (qk (pb st :p-qk (* n nh lat))) (lv (pb st :p-lv (* n nh lat))) (heads (pb st :p-heads (* n nh vd)))
           (aout (pb st :p-aout (* n d))) (kf (pb st :p-kf (* end lat)))
           (sel (pb st :p-sel2 (if sparse (* n maxc) 1))) (nsel (pb st :p-nsel n))
           (raws (pb st :p-raws n)) (nraw (pb st :p-nraw n)) (ncomp (pb st :p-ncomp n))
           (sinks (pb st :p-sinks nh)))
      (multiple-value-bind (norm pca) (g53p-attn-in m st l li sa n)
        (dk-mmx st (lname li "attn_q_a.weight") norm qr n)
        (dk-mmx st (lname li "attn_kv_a_mqa.weight") norm kvr n)
        (dk-rms dev qr (dvec st (gl-q-a-norm l)) qrn lq n eps)
        (dk-mmx st (lname li "attn_q_b.weight") qrn q n)
        (launch dev "g53_kv_store" n lat (list lat eps pos0 kvr (dvec st (gl-kv-norm l)) cache))
        ;; the indexer's pools
        (dk-mmx st (lname li "indexer.attn_k.weight") norm kr n)
        (dk-mmx st (lname li "indexer.pool_gate.weight") norm gr n)
        (let ((p0 (floor pos0 4)) (p1 (floor end 4)))
          (when (> p1 p0)
            (launch dev "g53_ix_pool_chunk" (- p1 p0) ixd
                    (list ixd pos0 p0 kr gr tail (dvec st (gl-ix-k-norm l)) (dvec st (gl-ix-k-bias l))
                          (dvec st (gl-ix-ape l)) ixk)))
          (launch dev "g53_ix_tail" 1 ixd (list ixd pos0 end kr gr tail)))
        ;; the sparse tokens' selections
        (when sparse
          (let ((iq (pb st :p-ixq (* n ixh ixd))) (iw (pb st :p-ixw (* n ixh)))
                (scores (pb st :p-ixs (* n rows))) (idx (pb st :p-ixidx (* n top))))
            (dk-mmx st (lname li "indexer.attn_q_b.weight") qrn iq n)
            (dk-mmx st (lname li "indexer.proj.weight") norm iw n)
            (launch dev "g53_ix_scores_chunk" (list (ceiling rows 8) n) 256
                    (list ixh rows pos0 (/ 1f0 (sqrt (float (* ixh ixd) 1f0))) iq iw ixk scores))
            (launch dev "g53_select_chunk" n 1024 (list rows top pos0 dense maxc scores idx sel nsel))))
        (launch dev "g53_attend_counts" (ceiling n 256) 256 (list n pos0 dense nsel raws nraw ncomp))
        ;; the queries into the latent, attention there, the heads out of it
        (let ((wk (gethash (lname li "attn_k_b.weight") (dk-q8 st))) (wv (gethash (lname li "attn_v_b.weight") (dk-q8 st))))
          (dotimes (h nh)
            (dk-mm st "d4_mm_q8_0" (cons (cmat-qs wk) (* h lat kd)) (cons (cmat-d wk) (* 2 h lat (floor kd 32)))
                   0 lat kd (cons q (* 4 h kd)) (cons qk (* 4 h lat)) n :x-stride (* nh kd) :y-stride (* nh lat)))
          (launch dev "g53_cache_f32" (ceiling (* end lat) 256) 256 (list (list :i64 (* end lat)) cache kf))
          (launch dev "d4_attend" n 512
                  (list nh lat (/ 1f0 (sqrt (float kd 1f0))) maxc 0 0 qk kf kf raws nraw ncomp sel sinks lv))
          (dotimes (h nh)
            (dk-mm st "d4_mm_q8_0" (cons (cmat-qs wv) (* h vd lat)) (cons (cmat-d wv) (* 2 h vd (floor lat 32)))
                   0 vd lat (cons lv (* 4 h lat)) (cons heads (* 4 h vd)) n :x-stride (* nh lat) :y-stride (* nh vd))))
        (dk-mmx st (lname li "attn_output.weight") heads aout n)
        (dk-hc-post-chunk m st aout sa pca sb n)))))

(defun g53p-dense-ffn (m st l li n sb sa)
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (ff (cmat-rows (gethash (lname li "ffn_gate.weight") (dk-q8 st))))
         (cur (pb st :p-cur (* n d))) (norm2 (pb st :p-norm2 (* n d))) (pcf (pb st :p-pcf (* n 32)))
         (g (pb st :p-fg (* n ff))) (u (pb st :p-fu (* n ff))) (mid (pb st :p-fmid (* n ff))) (out (pb st :p-sout (* n d))))
    (dk-hc-pre-chunk m st li :ffn sb n cur pcf)
    (dk-rms dev cur (dvec st (gl-ffn-norm l)) norm2 d n (ds4f-eps m))
    (dk-mmx st (lname li "ffn_gate.weight") norm2 g n)
    (dk-mmx st (lname li "ffn_up.weight") norm2 u n)
    (launch dev "d4_swiglu_w" (ceiling (* n ff) 256) 256 (list n ff (g53-clamp m) g u (pb st :p-ones n) mid nil nil))
    (dk-mmx st (lname li "ffn_down.weight") mid out n)
    (dk-hc-post-chunk m st out sb pcf sa n)))

(defun g53-cuda-prefill-1 (ctx tokens logits)
  "TOKENS at CTX's next positions, layer-major on the device."
  (let* ((m (d4ctx-model ctx)) (st (g53-cuda-state m)) (dev (dk-dev st))
         (d (ds4f-d m)) (hcd (* (ds4f-n-hc m) d)) (pos0 (d4ctx-pos ctx)) (n (length tokens))
         (tokv (coerce tokens 'simple-vector)) (nl (g53-n-trunk m)) (ne (ds4f-n-expert m))
         (routed (< n *g53-cuda-routed-below*)) (end (+ pos0 n))
         (bytes (+ (* n *g53-cuda-chunk-bytes*) (* end 2048) (* n 4 (floor end 4)) (* 64 1024 1024))))
    (assert (<= end (d4ctx-n-ctx ctx)) () "GLM context full (~D tokens)" (d4ctx-n-ctx ctx))
    (with-cuda
      (cuda-current dev)
      (let ((bufs (g53-ctx-dev ctx st)))
        (with-dk-pass (st bytes :stage (* 2 ne))
          (let ((sa (pb st :p-sa (* n hcd))) (sb (pb st :p-sb (* n hcd))) (emb (f32vec (* n hcd))) (row (f32vec hcd)))
            (dotimes (i n) (replace emb (g53-embed! m (svref tokv i) row) :start1 (* i hcd)))
            (cuda-upload-vector sa emb)
            (let ((ones (f32vec n))) (fill ones 1f0) (cuda-upload-vector (pb st :p-ones n) ones))
            (let ((sk (f32vec (ds4f-n-head m)))) (fill sk -1f30) (cuda-upload-vector (pb st :p-sinks (length sk)) sk))
            (let ((toks (dk-upload-tokens st tokv)))
              (flet ((take (li) (if (or routed (< li (g53-n-dense m))) (values nil nil nil) (d4-take-layer m st li))))
                (multiple-value-bind (slots ev staged) (take 0)
                  (dotimes (li nl)
                    (let ((l (svref (ds4f-layers m) li)) (b (svref bufs li)))
                      (multiple-value-bind (nslots nev nstaged) (if (< (1+ li) nl) (take (1+ li)) (values nil nil nil))
                        (if (eq (gl-kind l) :kda) (g53p-kda m st l li b n sa sb) (g53p-dsa m st l li b pos0 n sa sb))
                        (if (< li (g53-n-dense m))
                            (g53p-dense-ffn m st l li n sb sa)
                            (progn
                              (dk-ffn-head-chunk m st l li n sb)
                              (if routed
                                  (dk-ffn-routed-chunk m st l li n toks
                                                       (lambda (sel)
                                                         (multiple-value-bind (s e g) (d4-take-routed m st li sel n)
                                                           (setf slots s ev e staged g)
                                                           s))
                                                       sb sa)
                                  (progn (cuda-stream-wait (cudev-stream dev) ev)
                                         (dk-ffn-routed-chunk m st l li n toks slots sb sa)))))
                        (cuda-sync dev)
                        (dk-arena-reset st)
                        (setf (cdr *dk-mapped*) 0)
                        (when ev (cuda-event-destroy ev))
                        (when slots (d4-let-go st slots staged))
                        (setf slots nslots ev nev staged nstaged)))))))
            (incf (d4ctx-pos ctx) n)
            (when logits
              (let ((out (g53k-head m st (cons sa (* 4 (1- n) hcd)))) (v (d4ctx-logits ctx)))
                (sb-sys:with-pinned-objects (v)
                  (cuda-download out (sb-sys:vector-sap v) (* 4 (ds4f-n-vocab m))))))
            (cuda-sync dev)))))
    ctx))

(defun g53-cuda-prefill (ctx tokens logits)
  "TOKENS into CTX on the device: layer-major passes of *G53-CUDA-CHUNK* at
   most, a short piece a token at a time; decode's pool as it was after."
  (let* ((m (d4ctx-model ctx)) (st (g53-cuda-state m)))
    (flet ((decode (toks last) (loop for (tk . more) on toks do (g53-cuda-forward ctx tk (and last (null more))))))
      (if (or (< (length tokens) *g53-cuda-chunk-from*) (not (dk-passes-p st)))
          (decode tokens logits)
          (d4-call-keeping-pool
           m st (lambda ()
                  (loop for rest = tokens then (nthcdr k rest)
                        for k = (min *g53-cuda-chunk* (length rest))
                        while rest
                        do (let ((piece (subseq rest 0 k)) (last (and logits (= k (length rest)))))
                             (if (>= k *g53-cuda-chunk-from*)
                                 (g53-cuda-prefill-1 ctx piece last)
                                 (decode piece last))))))))
    ctx))
