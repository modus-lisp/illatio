;;;; metal-ds4f.lisp — DeepSeek V4 Flash's prefill experts on the GPU, streamed
;;;; from the SSD.
;;;
;;; The GPU can't read the 86.7 GB mapped file in place -- Metal's buffers stop
;;; at 30 GB, and a buffer over a file mapping would have its pages made
;;; resident while a command runs -- so a layer's routed experts are copied,
;;; from the mapping (the SSD, or the page cache when they are there), into a
;;; GPU-visible staging buffer, one slot an expert, as ds4 does with its expert
;;; cache.  The copies run on the pool's threads, an expert each.  Then one
;;; command buffer: for each expert, its gate and up products against the
;;; tokens that chose it (gathered into a block of rows), the clamped SwiGLU
;;; times each row's routing weight, and its down product.  The CPU sums each
;;; token's experts in slot order.

(in-package #:illatio)

(defstruct (d4gpu (:constructor %make-d4gpu))
  gpu stage (slot-bytes 0) (gate-bytes 0) (down-bytes 0) (plist* '())
  x g u w mid down (rows 0))            ; scratch, sized for ROWS routed rows

(defun ds4f-gpu-state (m)
  "The GPU state of DS4F M: a staging slot for each of a layer's experts."
  (or (getf (ds4f-gpu-plist m) :gpu)
      (setf (getf (ds4f-gpu-plist m) :gpu)
            (let* ((gpu (open-gpu)) (l (svref (ds4f-layers m) 0))
                   (gate-bytes (* (third (dl-gate-exps l)) (second (dl-gate-exps l))))
                   (down-bytes (* (third (dl-down-exps l)) (second (dl-down-exps l))))
                   (slot (+ (* 2 gate-bytes) down-bytes)))
              (%make-d4gpu :gpu gpu :stage (gpu-buffer gpu (* (ds4f-n-expert m) slot))
                           :slot-bytes slot :gate-bytes gate-bytes :down-bytes down-bytes)))))

(defun d4gpu-scratch (st m rows)
  "ST's row buffers, grown to hold ROWS routed rows."
  (when (< (d4gpu-rows st) rows)
    (let ((gpu (d4gpu-gpu st)) (d (ds4f-d m)) (ff (ds4f-ff-exp m)))
      (setf (d4gpu-x st) (gpu-buffer gpu (* 4 rows d)) (d4gpu-g st) (gpu-buffer gpu (* 4 rows ff))
            (d4gpu-u st) (gpu-buffer gpu (* 4 rows ff)) (d4gpu-w st) (gpu-buffer gpu (* 4 rows))
            (d4gpu-mid st) (gpu-buffer gpu (* 4 rows ff)) (d4gpu-down st) (gpu-buffer gpu (* 4 rows d))
            (d4gpu-rows st) rows)))
  st)

(defun ds4f-fd (m)
  "A descriptor open on DS4F M's file, for PREAD-INTO."
  (or (getf (ds4f-gpu-plist m) :fd)
      (setf (getf (ds4f-gpu-plist m) :fd)
            (or (sb-unix:unix-open (namestring (gguf-path (ds4f-gguf m))) sb-unix:o_rdonly 0)
                (error "cannot open ~A" (gguf-path (ds4f-gguf m)))))))

(defun pread-into (fd dst offset bytes)
  "BYTES of FD at OFFSET into DST: straight from the disk (or the page
   cache), large reads, no faults on a mapping."
  (loop with done = 0
        while (< done bytes)
        do (let ((got (sb-alien:alien-funcall
                       (sb-alien:extern-alien "pread" (function sb-alien:long sb-alien:int sb-sys:system-area-pointer
                                                                sb-alien:unsigned-long sb-alien:long))
                       fd (sb-sys:sap+ dst done) (- bytes done) (+ offset done))))
             (when (<= got 0) (error "pread of ~:D bytes at ~:D failed" bytes offset))
             (incf done got))))

(defun sap-copy (dst src bytes)
  (sb-alien:alien-funcall
   (sb-alien:extern-alien "memcpy" (function sb-sys:system-area-pointer sb-sys:system-area-pointer
                                             sb-sys:system-area-pointer sb-alien:unsigned-long))
   dst src bytes))

(defun ds4f-gpu-experts! (m l li ev users norms ws n down-out)
  "The routed FFN of a chunk's layer LI on the GPU.  EV: the distinct experts;
   USERS: for each, its (token . slot)s; NORMS: the tokens' normed inputs;
   WS: their routing weights.  DOWN-OUT gets each (token, slot)'s down
   product at (token*N-USED + slot)*D."
  (let* ((st (ds4f-gpu-state m)) (gpu (d4gpu-gpu st)) (pool (ds4f-pool m))
         (d (ds4f-d m)) (ff (ds4f-ff-exp m)) (nu (ds4f-n-used m)) (ne (length ev))
         (clamp (svref (ds4f-clamps m) li))
         (counts (map 'vector #'length users))
         (bases (let ((b 0)) (map 'vector (lambda (k) (prog1 b (incf b k))) counts)))
         (rows (reduce #'+ counts)))
    (declare (type f32vec norms down-out) (ignore n))
    (d4gpu-scratch st m rows)
    ;; the experts into their slots (the reads that stream from the SSD)
    (destructuring-bind (goff grb grows gcols) (dl-gate-exps l)
      (declare (ignore gcols))
      (destructuring-bind (uoff urb urows ucols) (dl-up-exps l)
        (declare (ignore ucols urows))
        (destructuring-bind (doff drb drows dcols) (dl-down-exps l)
          (declare (ignore dcols))
          (let ((fd (ds4f-fd m)) (sbase (gbuf-sap (d4gpu-stage st)))
                (sb (d4gpu-slot-bytes st)) (gb (d4gpu-gate-bytes st)))
            (parallel-for (k ne pool)
              (let ((e (svref ev k)) (slot (sb-sys:sap+ sbase (* k sb))))
                (pread-into fd slot (+ goff (* e grows grb)) gb)
                (pread-into fd (sb-sys:sap+ slot gb) (+ uoff (* e grows urb)) gb)
                (pread-into fd (sb-sys:sap+ slot (* 2 gb)) (+ doff (* e drows drb)) (* drows drb)))))
          ;; each expert's tokens gathered, with their routing weights
          (let ((xs (gbuf-sap (d4gpu-x st))) (wsap (gbuf-sap (d4gpu-w st))))
            (sb-sys:with-pinned-objects (norms)
              (let ((nsap (sb-sys:vector-sap norms)))
                (dotimes (k ne)
                  (loop for (tk . slot) across (svref users k) for i from 0
                        do (let ((row (+ (aref bases k) i)))
                             (sap-copy (sb-sys:sap+ xs (* 4 row d)) (sb-sys:sap+ nsap (* 4 tk d)) (* 4 d))
                             (setf (sb-sys:sap-ref-single wsap (* 4 row)) (aref (svref ws tk) slot))))))))
          (let ((stage (d4gpu-stage st)) (sb (d4gpu-slot-bytes st)) (gb (d4gpu-gate-bytes st)))
            (with-gpu-work (e gpu)
              (flet ((mm (kernel woff cols rbytes nr x xoff y yoff ntok)
                       (dispatch gpu e kernel (list cols nr ntok rbytes cols nr)
                                 (list (cons stage woff) (cons x xoff) (cons y yoff))
                                 (list (ceiling ntok 64) (ceiling nr 64) 1) 128 :shared 16384)))
                (dotimes (k ne)
                  (let ((nt (aref counts k)) (b (aref bases k)) (s0 (* k sb)))
                    (mm "d4_mm_iq2_xxs" s0 d grb grows (d4gpu-x st) (* 4 b d) (d4gpu-g st) (* 4 b ff) nt)
                    (mm "d4_mm_iq2_xxs" (+ s0 gb) d urb grows (d4gpu-x st) (* 4 b d) (d4gpu-u st) (* 4 b ff) nt)))
                (dispatch gpu e "d4_swiglu_w" (list rows ff clamp)
                          (list (d4gpu-g st) (d4gpu-u st) (d4gpu-w st) (d4gpu-mid st))
                          (ceiling (* rows ff) 256) 256)
                (dotimes (k ne)
                  (let ((nt (aref counts k)) (b (aref bases k)) (s0 (* k sb)))
                    (mm "d4_mm_q2_k" (+ s0 (* 2 gb)) ff drb drows (d4gpu-mid st) (* 4 b ff) (d4gpu-down st) (* 4 b d) nt))))))
          ;; back to (token, slot) order
          (let ((ds (gbuf-sap (d4gpu-down st))))
            (sb-sys:with-pinned-objects (down-out)
              (let ((osap (sb-sys:vector-sap down-out)))
                (dotimes (k ne)
                  (loop for (tk . slot) across (svref users k) for i from 0
                        do (sap-copy (sb-sys:sap+ osap (* 4 (+ (* tk nu) slot) d))
                                     (sb-sys:sap+ ds (* 4 (+ (aref bases k) i) d)) (* 4 d))))))))))
    down-out))

;;; --- the attention and shared expert's Q8_0 products, for a chunk ----------------------
;;; Their raw Q8_0 blocks (6 GB) are read into one GPU buffer the first time;
;;; a product copies the chunk's rows in, runs D4_MM_Q8_0 and copies the
;;; result out (unified memory: these are memcpys, not transfers).

(defparameter +ds4f-gpu-q8-tensors+
  '("attn_q_a.weight" "attn_q_b.weight" "attn_kv.weight" "attn_output_a.weight" "attn_output_b.weight"
    "ffn_gate_shexp.weight" "ffn_up_shexp.weight" "ffn_down_shexp.weight"))

(defun ds4f-gpu-weights (m)
  "(values buffer table): the Q8_0 tensors of +DS4F-GPU-Q8-TENSORS+ for every
   layer, in one GPU buffer; TABLE maps a name to (offset row-bytes rows cols)."
  (let ((st (ds4f-gpu-state m)))
    (or (getf (ds4f-gpu-plist m) :weights)
        (let* ((g (ds4f-gguf m)) (table (make-hash-table :test 'equal)) (total 0) (jobs '()))
          (dotimes (li (ds4f-n-layer m))
            (dolist (short +ds4f-gpu-q8-tensors+)
              (let* ((name (format nil "blk.~D.~A" li short)) (ti (gguf-tensor g name))
                     (dims (tensor-info-dims ti)) (bytes (tensor-bytes ti)) (rows (reduce #'* (rest dims))))
                (setf (gethash name table) (list total (floor bytes rows) rows (first dims)))
                (push (list total (+ (gguf-data-offset g) (tensor-info-offset ti)) bytes) jobs)
                (incf total (* 64 (ceiling bytes 64))))))
          (let ((buf (gpu-buffer (d4gpu-gpu st) total)) (fd (ds4f-fd m)) (jobs (coerce jobs 'simple-vector)))
            (parallel-for (k (length jobs) (ds4f-pool m))
              (destructuring-bind (at off bytes) (svref jobs k)
                (pread-into fd (sb-sys:sap+ (gbuf-sap buf) at) off bytes)))
            (setf (getf (ds4f-gpu-plist m) :weights) (cons buf table))
            (getf (ds4f-gpu-plist m) :weights))))))

(defun d4gpu-io (st bytes-x bytes-y)
  "Two scratch buffers of at least BYTES-X and BYTES-Y."
  (let ((gpu (d4gpu-gpu st)) (x (getf (d4gpu-plist* st) :x)) (y (getf (d4gpu-plist* st) :y)))
    (unless (and x (>= (gbuf-bytes x) bytes-x)) (setf x (gpu-buffer gpu bytes-x) (getf (d4gpu-plist* st) :x) x))
    (unless (and y (>= (gbuf-bytes y) bytes-y)) (setf y (gpu-buffer gpu bytes-y) (getf (d4gpu-plist* st) :y) y))
    (values x y)))

(defun gpu-q8mm! (m li short xs n out &key groups)
  "OUT = W XS for N tokens, W layer LI's Q8_0 tensor SHORT, on the GPU.  With
   GROUPS (count gdim rank): row r of W against group r/RANK of each token,
   as GROUPED-Q8MM!."
  (destructuring-bind (buf . table) (ds4f-gpu-weights m)
    (destructuring-bind (woff rbytes rows cols) (gethash (format nil "blk.~D.~A" li short) table)
      (let* ((st (ds4f-gpu-state m)) (gpu (d4gpu-gpu st))
             (xw (if groups (* (first groups) (second groups)) cols)))
        (declare (type f32vec xs out))
        (multiple-value-bind (x y) (d4gpu-io st (* 4 n xw) (* 4 n rows))
          (sb-sys:with-pinned-objects (xs out)
            (sap-copy (gbuf-sap x) (sb-sys:vector-sap xs) (* 4 n xw))
            (with-gpu-work (e gpu)
              (if groups
                  (destructuring-bind (count gdim rank) groups
                    (dotimes (g count)
                      (dispatch gpu e "d4_mm_q8_0" (list gdim rank n rbytes xw rows)
                                (list (cons buf (+ woff (* g rank rbytes))) (cons x (* 4 g gdim)) (cons y (* 4 g rank)))
                                (list (ceiling n 64) (ceiling rank 64) 1) 128 :shared 16384)))
                  (dispatch gpu e "d4_mm_q8_0" (list cols rows n rbytes cols rows)
                            (list (cons buf woff) x y)
                            (list (ceiling n 64) (ceiling rows 64) 1) 128 :shared 16384)))
            (sap-copy (sb-sys:vector-sap out) (gbuf-sap y) (* 4 n rows))))
        out))))

;;; --- attention for a chunk ------------------------------------------------------------

(defun d4gpu-buf (st key bytes)
  "Scratch buffer KEY of ST, at least BYTES."
  (let ((b (getf (d4gpu-plist* st) key)))
    (unless (and b (>= (gbuf-bytes b) bytes))
      (setf b (gpu-buffer (d4gpu-gpu st) (max bytes 16)) (getf (d4gpu-plist* st) key) b))
    b))

(defun copy-floats-in (buf v n)
  (declare (type f32vec v))
  (sb-sys:with-pinned-objects (v) (sap-copy (gbuf-sap buf) (sb-sys:vector-sap v) (* 4 n))))

(defun copy-u32-in (buf v)
  (dotimes (i (length v)) (setf (sb-sys:sap-ref-32 (gbuf-sap buf) (* 4 i)) (aref v i))))

(defun gpu-attend! (m l n q kr nkr kc nkc raws nraw ncomp idx maxc heads)
  "D4_ATTEND for N tokens: Q (N x heads x 512, RoPE'd), raw rows KR (NKR),
   compressed rows KC (NKC); per token its raw window (RAWS start, NRAW
   rows) and its NCOMP compressed rows, listed in IDX (MAXC a token).  The
   attention output into HEADS."
  (let* ((st (ds4f-gpu-state m)) (gpu (d4gpu-gpu st))
         (nh (ds4f-n-head m)) (hd (ds4f-head-dim m)) (qn (* n nh hd))
         (bq (d4gpu-buf st :aq (* 4 qn))) (bkr (d4gpu-buf st :akr (* 4 (max 1 nkr) hd)))
         (bkc (d4gpu-buf st :akc (* 4 (max 1 nkc) hd)))
         (bra (d4gpu-buf st :araws (* 4 n))) (bnr (d4gpu-buf st :anraw (* 4 n)))
         (bnc (d4gpu-buf st :ancomp (* 4 n))) (bix (d4gpu-buf st :aidx (* 4 (max 1 (* n maxc)))))
         (bsk (d4gpu-buf st :asinks (* 4 nh))) (bo (d4gpu-buf st :aout (* 4 qn))))
    (assert (= hd 512) () "d4_attend is written for 512-wide heads")
    (assert (= nh 64) () "d4_attend is written for 64 heads")
    (copy-floats-in bq q qn)
    (copy-floats-in bkr kr (* nkr hd))
    (when (plusp nkc) (copy-floats-in bkc kc (* nkc hd)))
    (copy-u32-in bra raws) (copy-u32-in bnr nraw) (copy-u32-in bnc ncomp)
    (when (plusp maxc) (copy-u32-in bix idx))
    (copy-floats-in bsk (dl-sinks l) nh)
    (with-gpu-work (e gpu)
      (dispatch gpu e "d4_attend" (list nh hd (/ 1f0 (sqrt (float hd 1f0))) (max 1 maxc))
                (list bq bkr bkc bra bnr bnc bix bsk bo)
                n 512 :shared (* 4 (+ (* 8 512) (* 64 8) (* 8 64) 128))))
    (sb-sys:with-pinned-objects (heads)
      (sap-copy (sb-sys:vector-sap heads) (gbuf-sap bo) (* 4 qn)))
    heads))
