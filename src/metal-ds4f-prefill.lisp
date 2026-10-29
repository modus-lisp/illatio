;;;; metal-ds4f-prefill.lisp — a DS4F prefill chunk on the GPU, a layer at a time.
;;;
;;; DS4F-PREFILL-1's ops, with the chunk's activations kept on the GPU
;;; between them: two command buffers a layer, the routing between them on
;;; the CPU.  The first: the attention block for every token (HC pre, the
;;; norms, the Q8_0 products, RoPE, the KV rows through FP8 into a line
;;; behind the window's, the compressors -- every window that ends in the
;;; chunk pooled at once from the chunk's projections and the window the
;;; context carries, which is then brought up to date -- the indexer's top
;;; 512 per token, D4_ATTEND, the output products, HC post), then the FFN's
;;; HC pre, norm, router logits and shared expert.  The second: the routed
;;; experts (each expert once, on the tokens that chose it, from its slot in
;;; the expert pool), each token's sum in slot order, HC post.  A long chunk
;;; takes slots for all of a layer's experts as the layer starts, reading the
;;; missing ones while its attention runs; the layer done, the slots go back
;;; but for the experts the prompt's last tokens chose, kept for decode.
;;; The F16 weights (the compressors', the indexer's, the router's, HC's) are
;;; the file's, in a GPU buffer of their own: D4_MM_F16 takes them as they are.

(in-package #:illatio)

(defparameter +ds4f-gpu-f16-tensors+
  '("hc_attn_fn.weight" "hc_ffn_fn.weight" "attn_compressor_kv.weight" "attn_compressor_gate.weight"
    "indexer.attn_q_b.weight" "indexer.proj.weight" "indexer_compressor_kv.weight"
    "indexer_compressor_gate.weight" "ffn_gate_inp.weight"))

(defun ds4f-gpu-f16 (m)
  "(buffer . table): the F16 tensors of +DS4F-GPU-F16-TENSORS+ every layer has,
   raw, in one GPU buffer; TABLE maps a name to (offset row-bytes rows cols)."
  (let ((st (ds4f-gpu-state m)))
    (or (getf (ds4f-gpu-plist m) :f16)
        (let* ((g (ds4f-gguf m)) (table (make-hash-table :test 'equal)) (total 0) (jobs '()))
          (dotimes (li (ds4f-n-layer m))
            (dolist (short +ds4f-gpu-f16-tensors+)
              (let* ((name (format nil "blk.~D.~A" li short)) (ti (gguf-tensor g name nil)))
                (when ti
                  (assert (eq (tensor-info-type ti) :f16))
                  (let* ((dims (tensor-info-dims ti)) (bytes (tensor-bytes ti)) (rows (reduce #'* (rest dims))))
                    (setf (gethash name table) (list total (floor bytes rows) rows (first dims)))
                    (push (list total (+ (gguf-data-offset g) (tensor-info-offset ti)) bytes) jobs)
                    (incf total (* 64 (ceiling bytes 64))))))))
          (let ((buf (lock-buffer (gpu-buffer (d4gpu-gpu st) total) "F16 weights")) (fd (ds4f-fd m))
                (jobs (coerce jobs 'simple-vector)))
            (parallel-for (k (length jobs) (ds4f-pool m))
              (destructuring-bind (at off bytes) (svref jobs k)
                (pread-into fd (sb-sys:sap+ (gbuf-sap buf) at) off bytes)))
            (setf (getf (ds4f-gpu-plist m) :f16) (cons buf table)))))))

(defun enc-mm (gpu e kernel w woff rbytes rows cols x y ntok &key (x-stride cols) (y-stride rows) (xoff 0) (yoff 0))
  "Y (NTOK rows, Y-STRIDE apart) = W X (NTOK rows, X-STRIDE apart): D4_MM."
  (dispatch gpu e kernel (list cols rows ntok rbytes x-stride y-stride)
            (list (cons w woff) (if (consp x) (cons (car x) (+ (cdr x) xoff)) (cons x xoff))
                  (if (consp y) (cons (car y) (+ (cdr y) yoff)) (cons y yoff)))
            (list (ceiling ntok 64) (ceiling rows 64) 1) 128 :shared 16384))

(defun enc-q8mm (m gpu e li short x y ntok &rest keys)
  (destructuring-bind (buf . table) (ds4f-gpu-weights m)
    (destructuring-bind (woff rbytes rows cols) (gethash (format nil "blk.~D.~A" li short) table)
      (apply #'enc-mm gpu e "d4_mm_q8_0" buf woff rbytes rows cols x y ntok keys))))

(defun enc-f16mm (m gpu e li short x y ntok &rest keys)
  (destructuring-bind (buf . table) (ds4f-gpu-f16 m)
    (destructuring-bind (woff rbytes rows cols) (gethash (format nil "blk.~D.~A" li short) table)
      (apply #'enc-mm gpu e "d4_mm_f16" buf woff rbytes rows cols x y ntok keys))))

(defun pbuf (st key floats) (d4gpu-buf st key (* 4 (max 1 floats))))

(defun put-u32s (buf v)
  (dotimes (i (length v) buf) (setf (sb-sys:sap-ref-32 (gbuf-sap buf) (* 4 i)) (aref v i))))

(defun put-cs (m buf pos-list il &optional inverse)
  "BUF: D4-ROPE-CS!'s table for each position of POS-LIST, a row each."
  (let ((nrot (ds4f-n-rot m)) (cs (f32vec (ds4f-n-rot m))) (sap (gbuf-sap buf)) (k 0))
    (dolist (p pos-list buf)
      (d4-rope-cs! m p il inverse cs)
      (dotimes (i nrot) (setf (sb-sys:sap-ref-single sap (* 4 (+ (* k nrot) i))) (aref cs i)))
      (incf k))))

(defun enc-hc-pre-chunk (m gpu e st li which streams n out pc)
  "HC pre of N tokens' STREAMS (layer LI's WHICH: :attn or :ffn): OUT, PC (32 a token)."
  (let* ((l (svref (ds4f-layers m) li)) (d (ds4f-d m)) (hcd (* (ds4f-n-hc m) d))
         (flat (pbuf st :p-flat (* n hcd))) (mix (pbuf st :p-mix (* n 32))))
    (enc-rms gpu e streams nil flat hcd n (ds4f-eps m))
    (enc-f16mm m gpu e li (if (eq which :attn) "hc_attn_fn.weight" "hc_ffn_fn.weight") flat mix n :y-stride 32)
    (dispatch gpu e "d4_hc_pre_fin" (list d (ds4f-n-hc m) (ds4f-sinkhorn m) 1f-6)
              (list mix (gpu-vector m (if (eq which :attn) (dl-hc-attn-scale l) (dl-hc-ffn-scale l)))
                    (gpu-vector m (if (eq which :attn) (dl-hc-attn-base l) (dl-hc-ffn-base l))) streams out pc)
              (list (ceiling d 256) n) 256 :shared (* 4 32))))

(defun enc-compressor-chunk (m gpu e st l li c norm pos0 n kind)
  "A compressor (KIND 0 the attention's, 1 the indexer's) over the chunk:
   its projections, every window ending in it pooled, the carried window
   brought up to date; how many rows it pushed."
  (let* ((ratio (dl-ratio l)) (coff (if (= ratio 4) 2 1))
         (hd (if (= kind 0) (ds4f-head-dim m) (ds4f-ix-dim m))) (width (* coff hd))
         (pkv (pbuf st (if (= kind 0) :p-ckv :p-ikv) (* n width)))
         (psc (pbuf st (if (= kind 0) :p-csc :p-isc) (* n width)))
         (ape (if (= kind 0) (dl-comp-ape l) (dl-ixc-ape l)))
         (st-kv (if (= kind 0) (dc-st-kv c) (dc-ix-kv c))) (st-sc (if (= kind 0) (dc-st-score c) (dc-ix-score c)))
         (rows (if (= kind 0) (dc-comp c) (dc-ix-comp c))) (dst0 (if (= kind 0) (dc-n-comp c) (dc-n-ix c)))
         (wstart (* ratio (floor pos0 ratio))) (e-pos (+ pos0 n -1))
         (p0 (+ wstart ratio -1)) (nb (if (> p0 e-pos) 0 (1+ (floor (- e-pos p0) ratio)))))
    (enc-f16mm m gpu e li (if (= kind 0) "attn_compressor_kv.weight" "indexer_compressor_kv.weight") norm pkv n)
    (enc-f16mm m gpu e li (if (= kind 0) "attn_compressor_gate.weight" "indexer_compressor_gate.weight") norm psc n)
    (when (plusp nb)
      (let ((cs (put-cs m (pbuf st (if (= kind 0) :p-ccs :p-ics2) (* nb (ds4f-n-rot m)))
                        (loop for b below nb collect (- (+ p0 (* b ratio) 1) ratio)) li)))
        (dispatch gpu e "d4_comp_pool_chunk"
                  (list hd ratio width (ds4f-eps m) kind (ds4f-n-rot m) dst0 pos0 p0 wstart (f32mat-cols ape))
                  (list pkv psc (gmat m ape) (gpu-vector m st-kv) (gpu-vector m st-sc)
                        (gpu-vector m (if (= kind 0) (dl-comp-norm l) (dl-ixc-norm l))) cs (gpu-vector m rows))
                  nb hd :shared (* 4 (+ hd 32)))))
    ;; the window the context carries on
    (let* ((cw (* ratio (floor e-pos ratio))) (ends (zerop (mod (1+ e-pos) ratio)))
           (set-prev (and (= ratio 4) (or ends (> cw wstart))))
           (prev-from (cond ((not set-prev) 0) (ends cw) (t (- cw ratio)))))
      (dispatch gpu e "d4_comp_carry" (list ratio width pos0 n wstart (f32mat-cols ape) prev-from (if set-prev 1 0))
                (list pkv psc (gmat m ape) (gpu-vector m st-kv) (gpu-vector m st-sc)) 1 1024))
    nb))

(defun enc-attention-chunk (m gpu e st l li c pos0 n sa sb)
  "Layer LI's attention block for the chunk's N tokens from POS0: streams SA into SB."
  (let* ((d (ds4f-d m)) (hd (ds4f-head-dim m)) (nh (ds4f-n-head m)) (lq (ds4f-lora-q m))
         (nrot (ds4f-n-rot m)) (eps (ds4f-eps m)) (ratio (dl-ratio l)) (swa (ds4f-swa m))
         (groups (ds4f-out-groups m)) (rank (ds4f-lora-o m)) (gdim (* hd (floor nh groups)))
         (cur (pbuf st :p-cur (* n d))) (norm (pbuf st :p-norm (* n d))) (pc (pbuf st :p-pca (* n 32)))
         (qr (pbuf st :p-qr (* n lq))) (qrn (pbuf st :p-qrn (* n lq))) (q (pbuf st :p-q (* n nh hd)))
         (kv0 (pbuf st :p-kv0 (* n hd))) (heads (pbuf st :p-heads (* n nh hd)))
         (low (pbuf st :p-low (* n groups rank))) (aout (pbuf st :p-aout (* n d)))
         (positions (loop for i below n collect (+ pos0 i)))
         (cs (put-cs m (pbuf st :p-cs (* n nrot)) positions li))
         (ics (put-cs m (pbuf st :p-icsx (* n nrot)) positions li t))
         (prev (dc-n-raw c)) (kr (pbuf st :p-kr (* (+ prev n) hd))))
    (enc-hc-pre-chunk m gpu e st li :attn sa n cur pc)
    (enc-rms gpu e cur (gpu-vector m (dl-attn-norm l)) norm d n eps)
    (enc-q8mm m gpu e li "attn_q_a.weight" norm qr n)
    (enc-q8mm m gpu e li "attn_kv.weight" norm kv0 n)
    (enc-rms gpu e qr (gpu-vector m (dl-q-a-norm l)) qrn lq n eps)
    (enc-q8mm m gpu e li "attn_q_b.weight" qrn q n)
    (enc-rms gpu e q nil q hd (* n nh) eps)
    (enc-rope gpu e q cs nh hd nrot n)
    ;; the KV rows: the window's, oldest first, then the chunk's own; the
    ;; chunk's last ones into the window
    (when (plusp prev)
      (dispatch gpu e "d4_ring_gather" (list hd swa (mod (- (dc-raw-total c) prev) swa) 0 prev)
                (list (gpu-vector m (dc-raw c)) kr) (ceiling (* prev hd) 256) 256))
    (dispatch gpu e "d4_kv_fin" (list hd nrot eps prev) (list kv0 (gpu-vector m (dl-kv-norm l)) cs kr)
              n hd :shared (* 4 (+ hd 32)))
    (let ((keep (min swa (+ prev n))))
      (dispatch gpu e "d4_ring_store" (list hd swa (mod (- (+ (dc-raw-total c) n) keep) swa) (- (+ prev n) keep) keep)
                (list (gpu-vector m (dc-raw c)) kr) (ceiling (* keep hd) 256) 256))
    ;; each token's raw rows in KR
    (let ((raws (make-array n :element-type '(unsigned-byte 32))) (nraw (make-array n :element-type '(unsigned-byte 32)))
          (ncomp (make-array n :element-type '(unsigned-byte 32))) (vis (make-array n :element-type '(unsigned-byte 32)))
          (nc0 (if (dc-comp c) (dc-n-comp c) 0)) (top-k (ds4f-ix-top-k m)))
      (dotimes (tk n)
        (setf (aref raws tk) (max 0 (- (+ prev tk 1) swa)) (aref nraw tk) (min swa (+ prev tk 1))))
      (incf (dc-raw-total c) n)
      (setf (dc-n-raw c) (min swa (+ prev n)))
      ;; the compressors; a token sees the rows of the windows ending at or before it
      (when (plusp ratio)
        (incf (dc-n-comp c) (enc-compressor-chunk m gpu e st l li c norm pos0 n 0))
        (when (= ratio 4) (incf (dc-n-ix c) (enc-compressor-chunk m gpu e st l li c norm pos0 n 1))))
      (dotimes (tk n)
        (setf (aref ncomp tk) (+ nc0 (floor (+ pos0 tk 1) (max ratio 1)) (- (floor pos0 (max ratio 1))))
              (aref vis tk) (aref ncomp tk)))
      (when (zerop ratio) (fill ncomp 0))
      (let* ((indexed (and (= ratio 4) (> (reduce #'max vis) top-k)))
             (idx (pbuf st :p-idx (* n top-k))))
        (when indexed
          (let* ((ixh (ds4f-ix-heads m)) (ixd (ds4f-ix-dim m)) (nix (dc-n-ix c))
                 (ixq (pbuf st :p-ixq (* n ixh ixd))) (ixw (pbuf st :p-ixw (* n ixh)))
                 (scores (pbuf st :p-ixs (* n nix))) (bvis (put-u32s (pbuf st :p-vis n) vis)))
            (enc-f16mm m gpu e li "indexer.attn_q_b.weight" qrn ixq n)
            (enc-f16mm m gpu e li "indexer.proj.weight" norm ixw n)
            (dispatch gpu e "d4_ixq_fin" (list ixh nrot (/ 1f0 (sqrt (float (* ixh ixd) 1f0))))
                      (list ixq cs ixw) (list ixh n) ixd :shared (* 4 (+ ixd 32)))
            (dispatch gpu e "d4_ix_scores" (list n nix ixh)
                      (list ixq ixw (gpu-vector m (dc-ix-comp c)) bvis scores)
                      (list (ceiling nix 32) (ceiling n 8)) 128 :shared (* 4 (+ (* 8 128) (* 32 128) (* 8 32))))
            (dispatch gpu e "d4_topk" (list nix top-k nix) (list scores idx bvis) n 1024 :shared (* 4 96))
            (dotimes (tk n) (setf (aref ncomp tk) (min top-k (aref vis tk))))))
        (let ((kc (if (dc-comp c) (gpu-vector m (dc-comp c)) kr)))
          (dispatch gpu e "d4_attend" (list nh hd (/ 1f0 (sqrt (float hd 1f0))) top-k 0 (if indexed 0 1))
                    (list q kr kc (put-u32s (pbuf st :p-raws n) raws) (put-u32s (pbuf st :p-nraw n) nraw)
                          (put-u32s (pbuf st :p-ncomp n) ncomp) idx (gpu-vector m (dl-sinks l)) heads)
                    n 512 :shared (* 4 (+ (* 8 512) (* 64 8) (* 8 64) 128))))))
    (enc-rope gpu e heads ics nh hd nrot n)
    (dotimes (g groups)
      (destructuring-bind (buf . table) (ds4f-gpu-weights m)
        (destructuring-bind (woff rbytes rows cols) (gethash (format nil "blk.~D.attn_output_a.weight" li) table)
          (declare (ignore rows cols))
          (enc-mm gpu e "d4_mm_q8_0" buf (+ woff (* g rank rbytes)) rbytes rank gdim heads low n
                  :x-stride (* nh hd) :y-stride (* groups rank) :xoff (* 4 g gdim) :yoff (* 4 g rank)))))
    (enc-q8mm m gpu e li "attn_output_b.weight" low aout n)
    (enc-hc-post m gpu e aout sa pc sb n)))

(defun enc-ffn-head-chunk (m gpu e st l li n sb)
  "The chunk's FFN up to its routing: HC pre, norm, router logits, shared expert."
  (let* ((d (ds4f-d m)) (ff (ds4f-ff-exp m)) (ne (ds4f-n-expert m))
         (cur (pbuf st :p-cur (* n d))) (norm2 (pbuf st :p-norm2 (* n d))) (pcf (pbuf st :p-pcf (* n 32)))
         (logits (pbuf st :p-logits (* n ne))) (sg (pbuf st :p-sg (* n ff))) (su (pbuf st :p-su (* n ff)))
         (smid (pbuf st :p-smid (* n ff))) (sout (pbuf st :p-sout (* n d)))
         (ones (pbuf st :p-ones n)))
    (dotimes (i n) (setf (sb-sys:sap-ref-single (gbuf-sap ones) (* 4 i)) 1f0))
    (enc-hc-pre-chunk m gpu e st li :ffn sb n cur pcf)
    (enc-rms gpu e cur (gpu-vector m (dl-ffn-norm l)) norm2 d n (ds4f-eps m))
    (enc-f16mm m gpu e li "ffn_gate_inp.weight" norm2 logits n)
    (enc-q8mm m gpu e li "ffn_gate_shexp.weight" norm2 sg n)
    (enc-q8mm m gpu e li "ffn_up_shexp.weight" norm2 su n)
    (dispatch gpu e "d4_swiglu_w" (list n ff (svref (ds4f-clamps m) li)) (list sg su ones smid) (ceiling (* n ff) 256) 256)
    (enc-q8mm m gpu e li "ffn_down_shexp.weight" smid sout n)))

(defparameter *ds4f-keep-tail* 16
  "A prompt's last this many tokens' experts stay in the pool after its prefill.")

(defun enc-ffn-tail-chunk (m gpu e st l li n ev users ws slots sb sa)
  "The chunk's routed experts (pool SLOTS, by position in EV), each once on
   the tokens that chose it, their sums in slot order with the shared
   expert, and HC post: SB into SA."
  (let* ((d (ds4f-d m)) (ff (ds4f-ff-exp m)) (nu (ds4f-n-used m)) (ne (length ev))
         (clamp (svref (ds4f-clamps m) li))
         (counts (map 'vector #'length users))
         (bases (let ((b 0)) (map 'vector (lambda (k) (prog1 b (incf b k))) counts)))
         (rows (reduce #'+ counts))
         (rmap (pbuf st :p-rmap rows)) (smap (pbuf st :p-smap (* n nu))) (wv (pbuf st :p-wrow rows))
         (x (pbuf st :p-x (* rows d))) (g (pbuf st :p-g (* rows ff))) (u (pbuf st :p-u (* rows ff)))
         (mid (pbuf st :p-mid (* rows ff))) (down (pbuf st :p-down (* rows d))) (moe (pbuf st :p-moe (* n d)))
         (stage (dx-buf (ds4f-expert-cache m))) (sbytes (d4gpu-slot-bytes st)) (gb (d4gpu-gate-bytes st))
         (grb (second (dl-gate-exps l))) (grows (third (dl-gate-exps l)))
         (drb (second (dl-down-exps l))) (drows (third (dl-down-exps l))))
    ;; the row each (expert, user) takes, and back from each (token, slot)
    (dotimes (k ne)
      (loop for (tk . slot) across (svref users k) for i from 0
            do (let ((r (+ (aref bases k) i)))
                 (setf (sb-sys:sap-ref-32 (gbuf-sap rmap) (* 4 r)) tk
                       (sb-sys:sap-ref-32 (gbuf-sap smap) (* 4 (+ (* tk nu) slot))) r
                       (sb-sys:sap-ref-single (gbuf-sap wv) (* 4 r)) (aref (svref ws tk) slot)))))
    (dispatch gpu e "d4_gather_rows" (list d rows) (list (pbuf st :p-norm2 (* n d)) rmap x) (ceiling (* rows d) 256) 256)
    (dotimes (k ne)
      (let ((nt (aref counts k)) (b (aref bases k)) (s0 (* (aref slots k) sbytes)))
        (enc-mm gpu e "d4_mm_iq2_xxs" stage s0 grb grows d x g nt :xoff (* 4 b d) :yoff (* 4 b ff))
        (enc-mm gpu e "d4_mm_iq2_xxs" stage (+ s0 gb) grb grows d x u nt :xoff (* 4 b d) :yoff (* 4 b ff))))
    (dispatch gpu e "d4_swiglu_w" (list rows ff clamp) (list g u wv mid) (ceiling (* rows ff) 256) 256)
    (dotimes (k ne)
      (let ((nt (aref counts k)) (b (aref bases k)) (s0 (* (aref slots k) sbytes)))
        (enc-mm gpu e "d4_mm_q2_k" stage (+ s0 (* 2 gb)) drb drows ff mid down nt :xoff (* 4 b ff) :yoff (* 4 b d))))
    (dispatch gpu e "d4_moe_sum_rows" (list d nu n) (list down smap (pbuf st :p-sout (* n d)) moe)
              (ceiling (* n d) 256) 256)
    (enc-hc-post m gpu e moe sb (pbuf st :p-pcf (* n 32)) sa n)))

(defun ds4f-gpu-prefill-1 (ctx tokens logits)
  "DS4F-PREFILL-1 on the GPU: the chunk TOKENS at the next positions."
  (let* ((m (d4ctx-model ctx)) (st (ds4f-gpu-state m)) (gpu (d4gpu-gpu st))
         (d (ds4f-d m)) (n-hc (ds4f-n-hc m)) (hcd (* n-hc d)) (pos0 (d4ctx-pos ctx))
         (n (length tokens)) (tokv (coerce tokens 'simple-vector)) (wf (ds4f-file m))
         (ne (ds4f-n-expert m)) (nu (ds4f-n-used m))
         (sa (pbuf st :p-sa (* n hcd))) (sb (pbuf st :p-sb (* n hcd))))
    (declare (type u8vec wf))
    (ds4f-gpu-weights m) (ds4f-gpu-f16 m)
    (let ((sap (gbuf-sap sa)))
      (dotimes (tk n)
        (let ((o (+ (ds4f-embd-off m) (* (svref tokv tk) d 2))))
          (dotimes (j d)
            (let ((v (f16-at wf (+ o (* 2 j)))))
              (dotimes (h n-hc) (setf (sb-sys:sap-ref-single sap (* 4 (+ (* tk hcd) (* h d) j))) v)))))))
    (dotimes (li (ds4f-n-layer m))
      (let* ((l (svref (ds4f-layers m) li)) (c (svref (d4ctx-caches ctx) li))
             ;; a long chunk wants nearly all of a layer's experts: slots for them all, read under the attention
             (all (and (>= n *ds4f-read-all-from*) (pool-take-layer m li (loop for k below ne collect k) nil))))
        (with-gpu-work (e gpu)
          (enc-attention-chunk m gpu e st l li c pos0 n sa sb)
          (enc-ffn-head-chunk m gpu e st l li n sb))
        ;; the routing, on the CPU
        (let* ((lg (f32vec (* n ne))) (row (f32vec ne)) (norm (f32vec d))
               (sels (make-array n)) (ws (make-array n)) (groups (make-hash-table)))
          (copy-floats-out lg (pbuf st :p-logits (* n ne)) (* n ne))
          (dotimes (tk n)
            (replace row lg :start2 (* tk ne) :end2 (* (1+ tk) ne))
            (multiple-value-bind (sel w) (d4-route m l li norm (svref tokv tk) row)
              (setf (svref sels tk) sel (svref ws tk) w)
              (dotimes (slot nu) (push (cons tk slot) (gethash (aref sel slot) groups)))))
          (let* ((experts (sort (loop for k being the hash-keys of groups collect k) #'<))
                 (users (map 'simple-vector (lambda (k) (coerce (reverse (gethash k groups)) 'simple-vector)) experts))
                 (ev (coerce experts 'simple-vector)) (xc (ds4f-expert-cache m))
                 (taken (or all (pool-take-layer m li experts t)))
                 (slots (if all (map '(vector fixnum) (lambda (k) (aref all k)) ev) taken))
                 (tail (and logits (let ((h (make-hash-table)))
                                     (loop for tk from (max 0 (- n *ds4f-keep-tail*)) below n
                                           do (loop for x across (svref sels tk) do (setf (gethash x h) t)))
                                     h))))
            (cache-wait m xc (coerce slots 'list))
            (with-gpu-work (e gpu)
              (enc-ffn-tail-chunk m gpu e st l li n ev users ws slots sb sa))
            (pool-done-layer m taken
                             (lambda (s) (and tail (gethash (mod (aref (dx-owner xc) s) ne) tail))))))))
    (incf (d4ctx-pos ctx) n)
    (when logits
      (let ((last (f32vec hcd)))
        (sb-sys:with-pinned-objects (last)
          (sap-copy (sb-sys:vector-sap last) (sb-sys:sap+ (gbuf-sap sa) (* 4 (1- n) hcd)) (* 4 hcd)))
        (ds4f-head! m last (d4ctx-logits ctx))))
    ctx))
