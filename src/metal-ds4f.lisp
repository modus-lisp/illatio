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

(defun stage-buffer (st parity)
  "Staging buffer PARITY (0 or 1): two, so a layer's experts can be read
   while the one before it computes."
  (if (zerop parity)
      (d4gpu-stage st)
      (or (getf (d4gpu-plist* st) :stage1)
          (setf (getf (d4gpu-plist* st) :stage1) (gpu-buffer (d4gpu-gpu st) (gbuf-bytes (d4gpu-stage st)))))))

(defun read-layer-experts (m l parity &key (threads 4))
  "Start reading every expert of layer L into staging buffer PARITY, slot =
   expert id, on THREADS threads of their own; the threads, to join."
  (let* ((st (ds4f-gpu-state m)) (buf (stage-buffer st parity)) (fd (ds4f-fd m))
         (ne (ds4f-n-expert m)) (sb (d4gpu-slot-bytes st)) (gb (d4gpu-gate-bytes st)))
    (destructuring-bind (goff grb grows gcols) (dl-gate-exps l)
      (declare (ignore gcols))
      (destructuring-bind (uoff urb urows ucols) (dl-up-exps l)
        (declare (ignore urows ucols))
        (destructuring-bind (doff drb drows dcols) (dl-down-exps l)
          (declare (ignore dcols))
          (loop for k below threads
                collect (let ((k k))
                          (sb-thread:make-thread
                           (lambda ()
                             (multiple-value-bind (s e) (share ne k threads)
                               (loop for ex from s below e
                                     do (let ((slot (sb-sys:sap+ (gbuf-sap buf) (* ex sb))))
                                          (pread-into fd slot (+ goff (* ex grows grb)) gb)
                                          (pread-into fd (sb-sys:sap+ slot gb) (+ uoff (* ex grows urb)) gb)
                                          (pread-into fd (sb-sys:sap+ slot (* 2 gb)) (+ doff (* ex drows drb)) (* drows drb))))))
                           :name "illatio expert read"))))))))

(defun ds4f-gpu-experts! (m l li ev users norms ws n down-out &key preloaded (parity 0))
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
            (parallel-for (k (if preloaded 0 ne) pool)   ; read ahead already: slot = expert id
              (let ((e (svref ev k)) (slot (sb-sys:sap+ sbase (* k sb))))
                (pread-into fd slot (+ goff (* e grows grb)) gb)
                (pread-into fd (sb-sys:sap+ slot gb) (+ uoff (* e grows urb)) gb)
                (pread-into fd (sb-sys:sap+ slot (* 2 gb)) (+ doff (* e drows drb)) (* drows drb)))))
          ;; each expert's tokens gathered, with their routing weights
          (let ((xs (gbuf-sap (d4gpu-x st))) (wsap (gbuf-sap (d4gpu-w st))))
            (sb-sys:with-pinned-objects (norms)
              (let ((nsap (sb-sys:vector-sap norms)))
                (parallel-for (k ne pool)
                  (loop for (tk . slot) across (svref users k) for i from 0
                        do (let ((row (+ (aref bases k) i)))
                             (sap-copy (sb-sys:sap+ xs (* 4 row d)) (sb-sys:sap+ nsap (* 4 tk d)) (* 4 d))
                             (setf (sb-sys:sap-ref-single wsap (* 4 row)) (aref (svref ws tk) slot))))))))
          (let ((stage (stage-buffer st parity)) (sb (d4gpu-slot-bytes st)) (gb (d4gpu-gate-bytes st))
                (slot-of (lambda (k) (if preloaded (svref ev k) k))))
            (with-gpu-work (e gpu)
              (flet ((mm (kernel woff cols rbytes nr x xoff y yoff ntok)
                       (dispatch gpu e kernel (list cols nr ntok rbytes cols nr)
                                 (list (cons stage woff) (cons x xoff) (cons y yoff))
                                 (list (ceiling ntok 64) (ceiling nr 64) 1) 128 :shared 16384)))
                (dotimes (k ne)
                  (let ((nt (aref counts k)) (b (aref bases k)) (s0 (* (funcall slot-of k) sb)))
                    (mm "d4_mm_iq2_xxs" s0 d grb grows (d4gpu-x st) (* 4 b d) (d4gpu-g st) (* 4 b ff) nt)
                    (mm "d4_mm_iq2_xxs" (+ s0 gb) d urb grows (d4gpu-x st) (* 4 b d) (d4gpu-u st) (* 4 b ff) nt)))
                (dispatch gpu e "d4_swiglu_w" (list rows ff clamp)
                          (list (d4gpu-g st) (d4gpu-u st) (d4gpu-w st) (d4gpu-mid st))
                          (ceiling (* rows ff) 256) 256)
                (dotimes (k ne)
                  (let ((nt (aref counts k)) (b (aref bases k)) (s0 (* (funcall slot-of k) sb)))
                    (mm "d4_mm_q2_k" (+ s0 (* 2 gb)) ff drb drows (d4gpu-mid st) (* 4 b ff) (d4gpu-down st) (* 4 b d) nt))))))
          ;; back to (token, slot) order
          (let ((ds (gbuf-sap (d4gpu-down st))))
            (sb-sys:with-pinned-objects (down-out)
              (let ((osap (sb-sys:vector-sap down-out)))
                (parallel-for (k ne pool)
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

(defun gpu-ix-scores! (m n q w k ncomp vis scores)
  "D4_IX_SCORES for N tokens: their indexer queries Q (N x heads x 128,
   RoPE'd and FP4-rounded) and scaled head weights W (N x heads) against the
   NCOMP indexer rows K; token t sees VIS[t] of them.  SCORES, N x NCOMP,
   -inf past what a token sees."
  (let* ((st (ds4f-gpu-state m)) (gpu (d4gpu-gpu st))
         (nh (ds4f-ix-heads m)) (hd (ds4f-ix-dim m)) (qn (* n nh hd))
         (bq (d4gpu-buf st :iq (* 4 qn))) (bw (d4gpu-buf st :iw (* 4 n nh)))
         (bk (d4gpu-buf st :ik (* 4 ncomp hd))) (bv (d4gpu-buf st :ivis (* 4 n)))
         (bs (d4gpu-buf st :iscores (* 4 n ncomp))))
    (assert (= hd 128) () "d4_ix_scores is written for 128-wide indexer heads")
    (copy-floats-in bq q qn) (copy-floats-in bw w (* n nh))
    (copy-floats-in bk k (* ncomp hd)) (copy-u32-in bv vis)
    (with-gpu-work (e gpu)
      (dispatch gpu e "d4_ix_scores" (list n ncomp nh) (list bq bw bk bv bs)
                (list (ceiling ncomp 32) (ceiling n 8)) 128
                :shared (* 4 (+ (* 8 128) (* 32 128) (* 8 32)))))
    (sb-sys:with-pinned-objects (scores)
      (sap-copy (sb-sys:vector-sap scores) (gbuf-sap bs) (* 4 n ncomp)))
    scores))

;;; --- decode: an expert cache on the GPU ------------------------------------------------
;;; A token's six experts a layer come from a cache of expert slots in one GPU
;;; buffer (as ds4's streaming expert cache), the least recently used given up
;;; for a miss, a miss read by pread; then one command buffer: gate and up for
;;; the six (D4_MV_IQ2_XXS, the six at once), the clamped SwiGLU times their
;;; routing weights, down (D4_MV_Q2_K).  The CPU sums the six in slot order.

(defvar *ds4f-expert-cache-gb* 16
  "The decode expert cache's size, in GB (an expert is about 7 MB).")

(defstruct (d4xcache (:conc-name dx-)) buf (n 0) slot-of owner used (tick 0) (free 0) (hits 0) (misses 0) (read-time 0))

(defun ds4f-expert-cache (m)
  (let ((st (ds4f-gpu-state m)))
    (or (getf (d4gpu-plist* st) :xcache)
        (let* ((sb (d4gpu-slot-bytes st))
               (n (max 6 (floor (* *ds4f-expert-cache-gb* 1000000000) sb)))
               (keys (* (ds4f-n-layer m) (ds4f-n-expert m))))
          (setf (getf (d4gpu-plist* st) :xcache)
                (seed-expert-cache
                 m (make-d4xcache :buf (gpu-buffer (d4gpu-gpu st) (* n sb)) :n n
                                  :slot-of (make-array keys :element-type 'fixnum :initial-element -1)
                                  :owner (make-array n :element-type 'fixnum :initial-element -1)
                                  :used (make-array n :element-type 'fixnum :initial-element 0))))))))

(defvar *ds4f-seed-experts* t
  "Whether the decode expert cache starts with ds4's most used experts.")

(defun seed-expert-cache (m xc)
  "XC with as many of +DS4F-HOTLIST+'s experts as it holds read in, the
   most used first, if *DS4F-SEED-EXPERTS*."
  (when (and *ds4f-seed-experts* (= (ds4f-n-layer m) 43) (= (ds4f-n-expert m) 256))
    (let* ((ne (ds4f-n-expert m)) (sb (d4gpu-slot-bytes (ds4f-gpu-state m)))
           (k (min (dx-n xc) (floor (length +ds4f-hotlist+) 2)))
           (by-layer (make-array (ds4f-n-layer m) :initial-element '())))
      (dotimes (s k)
        (let ((li (aref +ds4f-hotlist+ (* 2 s))) (e (aref +ds4f-hotlist+ (1+ (* 2 s)))))
          (push (cons e s) (svref by-layer li))
          (setf (aref (dx-owner xc) s) (+ (* li ne) e) (aref (dx-slot-of xc) (+ (* li ne) e)) s)))
      (dotimes (li (length by-layer))
        (let ((v (svref by-layer li)))
          (when v
            (read-experts-pieces m (svref (ds4f-layers m) li) (mapcar #'car v)
                                 (mapcar (lambda (x) (* (cdr x) sb)) v) (dx-buf xc)))))
      (setf (dx-free xc) k)))
  xc)

(defun cache-experts (m l li sel)
  "The cache slots of layer LI's experts SEL, the missing ones read in."
  (let* ((xc (ds4f-expert-cache m)) (ne (ds4f-n-expert m)) (k (length sel))
         (slots (make-array k :element-type 'fixnum)) (missing '())
         (slot-of (dx-slot-of xc)) (owner (dx-owner xc)) (used (dx-used xc))
         (tick (incf (dx-tick xc))))
    (declare (type (simple-array fixnum (*)) slot-of owner used))
    (dotimes (i k)
      (let ((s (aref slot-of (+ (* li ne) (aref sel i)))))
        (if (>= s 0)
            (progn (setf (aref used s) tick (aref slots i) s) (incf (dx-hits xc)))
            (setf (aref slots i) -1))))
    (dotimes (i k)
      (when (minusp (aref slots i))
        (let ((s (if (< (dx-free xc) (dx-n xc))
                     (prog1 (dx-free xc) (incf (dx-free xc)))
                     ;; the least recently used, never one this token holds
                     (let ((best -1) (bu most-positive-fixnum))
                       (dotimes (j (dx-n xc) best)
                         (when (< (aref used j) bu) (setf best j bu (aref used j))))))))
          (when (>= (aref owner s) 0) (setf (aref slot-of (aref owner s)) -1))
          (setf (aref owner s) (+ (* li ne) (aref sel i)) (aref slot-of (aref owner s)) s
                (aref used s) tick (aref slots i) s)
          (push (cons (aref sel i) s) missing)
          (incf (dx-misses xc)))))
    (when missing
      (let ((t0 (get-internal-real-time)))
        (read-experts-pieces m l (mapcar #'car missing)
                             (mapcar (lambda (x) (* (cdr x) (d4gpu-slot-bytes (ds4f-gpu-state m)))) missing)
                             (dx-buf xc))
        (incf (dx-read-time xc) (- (get-internal-real-time) t0))))
    slots))

(defparameter *ds4f-read-piece* (* 1024 1024)
  "Expert reads go in pieces of this many bytes, on the pool's threads: the
   SSD reads many at once.")

(defun read-experts-pieces (m l experts offsets buf)
  "Layer L's EXPERTS (gate, up, down) from the file into BUF at OFFSETS,
   in pieces of *DS4F-READ-PIECE* across the pool."
  (let* ((st (ds4f-gpu-state m)) (fd (ds4f-fd m)) (gb (d4gpu-gate-bytes st)) (piece *ds4f-read-piece*)
         (jobs '()))
    (destructuring-bind (goff grb grows gcols) (dl-gate-exps l)
      (declare (ignore gcols))
      (destructuring-bind (uoff urb urows ucols) (dl-up-exps l)
        (declare (ignore urows ucols))
        (destructuring-bind (doff drb drows dcols) (dl-down-exps l)
          (declare (ignore dcols))
          (loop for e in experts for at in offsets
                do (loop for (src dst bytes) in (list (list (+ goff (* e grows grb)) at gb)
                                                      (list (+ uoff (* e grows urb)) (+ at gb) gb)
                                                      (list (+ doff (* e drows drb)) (+ at (* 2 gb)) (* drows drb)))
                         do (loop for o from 0 below bytes by piece
                                  do (push (list (+ src o) (+ dst o) (min piece (- bytes o))) jobs)))))))
    (let ((v (coerce jobs 'simple-vector)) (base (gbuf-sap buf)))
      (parallel-for (j (length v) (ds4f-pool m))
        (destructuring-bind (src dst bytes) (svref v j)
          (pread-into fd (sb-sys:sap+ base dst) src bytes))))))

(defun ds4f-gpu-ffn-1! (m l li norm sel w moe &optional meanwhile)
  "The routed experts SEL (weights W) of layer LI for one token's NORM, on
   the GPU from the expert cache; their sum, in slot order, into MOE.
   MEANWHILE, a function, runs on the CPU while the GPU does."
  (let* ((st (ds4f-gpu-state m)) (gpu (d4gpu-gpu st)) (xc (ds4f-expert-cache m))
         (d (ds4f-d m)) (ff (ds4f-ff-exp m)) (k (length sel)) (clamp (svref (ds4f-clamps m) li))
         (sb (d4gpu-slot-bytes st)) (gb (d4gpu-gate-bytes st))
         (slots (cache-experts m l li sel))
         (bx (d4gpu-buf st :dx (* 4 d))) (bo (d4gpu-buf st :doffs (* 8 k)))
         (bw (d4gpu-buf st :dw (* 4 k))) (bg (d4gpu-buf st :dg (* 4 k ff)))
         (bu (d4gpu-buf st :du (* 4 k ff))) (bm (d4gpu-buf st :dmid (* 4 k ff)))
         (bd (d4gpu-buf st :ddown (* 4 k d))))
    (declare (type f32vec norm w moe))
    (copy-floats-in bx norm d) (copy-floats-in bw w k)
    (dotimes (i k) (setf (sb-sys:sap-ref-64 (gbuf-sap bo) (* 8 i)) (* (aref slots i) sb)))
    (let ((grb (second (dl-gate-exps l))) (drb (second (dl-down-exps l))) (buf (dx-buf xc)))
      (with-gpu-work-while (e gpu)
          (flet ((mv (kernel ncols nrows rb add x xs y)
                   (dispatch gpu e kernel (list ncols nrows rb add xs nrows) (list buf bo x y)
                             (list (ceiling nrows 8) 1 k) 64
                             :shared (if (string= kernel "d4_mv_iq2_xxs") (+ (* 8 256) 128) 0))))
            (mv "d4_mv_iq2_xxs" d ff grb 0 bx 0 bg)
            (mv "d4_mv_iq2_xxs" d ff grb gb bx 0 bu)
            (dispatch gpu e "d4_swiglu_w" (list k ff clamp) (list bg bu bw bm) (ceiling (* k ff) 256) 256)
            (mv "d4_mv_q2_k" ff d drb (* 2 gb) bm ff bd))
        (when meanwhile (funcall meanwhile))))
    (let ((ds (gbuf-sap bd)))
      (dotimes (r d moe)
        (let ((acc 0f0))
          (declare (type single-float acc))
          (dotimes (i k) (incf acc (sb-sys:sap-ref-single ds (* 4 (+ (* i d) r)))))
          (setf (aref moe r) acc))))))
