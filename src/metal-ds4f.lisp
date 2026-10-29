;;;; metal-ds4f.lisp — DeepSeek V4 Flash on the GPU: its weights, its expert
;;;; pool, the reads from the SSD.
;;;
;;; The GPU can't read the 86.7 GB mapped file in place -- Metal's buffers stop
;;; at 30 GB, and a buffer over a file mapping would have its pages made
;;; resident while a command runs -- so what it computes with is copied in:
;;; the non-routed weights once (their raw blocks, the file's own format),
;;; the routed experts by pread into slots of one pool as they are wanted, as
;;; ds4 streams them.  The passes that use them are metal-ds4f-decode.lisp
;;; (a token) and metal-ds4f-prefill.lisp (a prompt chunk).

(in-package #:illatio)

(defstruct (d4gpu (:constructor %make-d4gpu))
  gpu (n-expert 0) (slot-bytes 0) (gate-bytes 0) (down-bytes 0) (plist* '()))

(defun ds4f-gpu-state (m)
  "The GPU state of DS4F M: the device, an expert slot's layout, buffers by name."
  (or (getf (ds4f-gpu-plist m) :gpu)
      (setf (getf (ds4f-gpu-plist m) :gpu)
            (let* ((gpu (open-gpu)) (l (svref (ds4f-layers m) 0))
                   (gate-bytes (* (third (dl-gate-exps l)) (second (dl-gate-exps l))))
                   (down-bytes (* (third (dl-down-exps l)) (second (dl-down-exps l))))
                   (slot (+ (* 2 gate-bytes) down-bytes)))
              (%make-d4gpu :gpu gpu :n-expert (ds4f-n-expert m) :slot-bytes slot
                           :gate-bytes gate-bytes :down-bytes down-bytes)))))

(defvar *ds4f-nocache* t
  "Whether PREAD-INTO's reads bypass the page cache (F_NOCACHE): the experts
   they read have the GPU's caches, and the 86.7 GB file can't fit in the
   page cache anyway, so a copy there only presses on everything else.")

(defun ds4f-fd (m)
  "A descriptor open on DS4F M's file, for PREAD-INTO."
  (or (getf (ds4f-gpu-plist m) :fd)
      (setf (getf (ds4f-gpu-plist m) :fd)
            (let ((fd (or (sb-unix:unix-open (namestring (gguf-path (ds4f-gguf m))) sb-unix:o_rdonly 0)
                          (error "cannot open ~A" (gguf-path (ds4f-gguf m))))))
              (when *ds4f-nocache*
                (sb-alien:alien-funcall
                 (sb-alien:extern-alien "fcntl" (function sb-alien:int sb-alien:int sb-alien:int sb-alien:int))
                 fd 48 1))                ; F_NOCACHE
              fd))))

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

;;; --- the GPU's weights ---------------------------------------------------------------
;;; The attention's and shared expert's Q8_0 tensors, and the output head's,
;;; as the file's raw blocks in one GPU buffer (6.3 GB), locked in memory; the
;;; F16 ones in another (metal-ds4f-prefill.lisp).

(defparameter +ds4f-gpu-q8-tensors+
  '("attn_q_a.weight" "attn_q_b.weight" "attn_kv.weight" "attn_output_a.weight" "attn_output_b.weight"
    "ffn_gate_shexp.weight" "ffn_up_shexp.weight" "ffn_down_shexp.weight"))

(defun ds4f-gpu-weights (m)
  "(values buffer table): the Q8_0 tensors of +DS4F-GPU-Q8-TENSORS+ for every
   layer, in one GPU buffer; TABLE maps a name to (offset row-bytes rows cols)."
  (let ((st (ds4f-gpu-state m)))
    (or (getf (ds4f-gpu-plist m) :weights)
        (let* ((g (ds4f-gguf m)) (table (make-hash-table :test 'equal)) (total 0) (jobs '()))
          (dolist (name (cons "output.weight"
                              (loop for li below (ds4f-n-layer m)
                                    append (mapcar (lambda (short) (format nil "blk.~D.~A" li short)) +ds4f-gpu-q8-tensors+))))
              (let* ((ti (gguf-tensor g name))
                     (dims (tensor-info-dims ti)) (bytes (tensor-bytes ti)) (rows (reduce #'* (rest dims))))
                (setf (gethash name table) (list total (floor bytes rows) rows (first dims)))
                (push (list total (+ (gguf-data-offset g) (tensor-info-offset ti)) bytes) jobs)
                (incf total (* 64 (ceiling bytes 64)))))
          (let ((buf (lock-buffer (gpu-buffer (d4gpu-gpu st) total) "Q8_0 weights")) (fd (ds4f-fd m))
                (jobs (coerce jobs 'simple-vector)))
            (parallel-for (k (length jobs) (ds4f-pool m))
              (destructuring-bind (at off bytes) (svref jobs k)
                (pread-into fd (sb-sys:sap+ (gbuf-sap buf) at) off bytes)))
            (setf (getf (ds4f-gpu-plist m) :weights) (cons buf table))
            (getf (ds4f-gpu-plist m) :weights))))))

;;; --- scratch buffers by name ----------------------------------------------------------

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

;;; --- the expert pool --------------------------------------------------------------

(defvar *ds4f-expert-cache-gb* nil
  "The expert pool's size, in GB (an expert is about 7 MB); NIL: what the
   GPU's working set leaves (EXPERT-CACHE-GB).")

(defun expert-cache-gb (gpu)
  "The pool's size: as ds4 sizes its cache, 7/8 of the GPU's recommended
   working set (40.2 GB of a 48 GB M4 Pro: past it a command buffer fails
   for want of memory), less the GPU's weights (7.4 GB), a prefill chunk's
   activations (about 0.9 MB a token) and a GB for the rest; one Metal
   buffer, so no more than its longest (30 GB) allows.  22 GB here."
  (or *ds4f-expert-cache-gb*
      (let* ((dev (gpu-device gpu))
             (wss (/ (msg sb-alien:unsigned-long dev "recommendedMaxWorkingSetSize") 1d9))
             (most (/ (msg sb-alien:unsigned-long dev "maxBufferLength") 1d9)))
        (max 4 (floor (min (- most 2) (- (* 7/8 wss) 7.5 (* *ds4f-chunk* 0.0009) 1)))))))

;;; The expert pool: one GPU buffer of expert slots that prefill and decode
;;; share.  Work in flight comes first, new loads go around it: a slot a
;;; layer is computing with is PINned and a slot being read is PENDING, and
;;; neither is given up; demand reads (what the layer at hand needs) go
;;; ahead of reads ahead (what the next layer will want).  A prefill reads a
;;; layer's experts into slots of their own (FRESH), and gives them back
;;; (FREES, taken first) once the layer is done -- but for the ones its last
;;; tokens chose, which decode will most likely want next -- so a long
;;; prompt passes through a layer's worth of slots, not the whole pool.

(defstruct (d4xcache (:conc-name dx-)) buf (n 0) slot-of owner used pending pin fresh (frees '())
  (tick 0) (free 0) (decode-pins '())
  (hits 0) (misses 0) (read-time 0) (prefetched 0) (waited 0))

(defun ds4f-expert-cache (m)
  (let ((st (ds4f-gpu-state m)))
    (or (getf (d4gpu-plist* st) :xcache)
        (let* ((sb (d4gpu-slot-bytes st))
               (n (max 6 (floor (* (expert-cache-gb (d4gpu-gpu st)) 1000000000) sb)))
               (keys (* (ds4f-n-layer m) (ds4f-n-expert m))))
          (setf (getf (d4gpu-plist* st) :xcache)
                (seed-expert-cache
                 m (make-d4xcache :buf (lock-buffer (gpu-buffer (d4gpu-gpu st) (* n sb)) "expert pool") :n n
                                  :slot-of (make-array keys :element-type 'fixnum :initial-element -1)
                                  :owner (make-array n :element-type 'fixnum :initial-element -1)
                                  :used (make-array n :element-type 'fixnum :initial-element 0)
                                  :pending (make-array n :element-type 'fixnum :initial-element 0)
                                  :pin (make-array n :element-type 'fixnum :initial-element 0)
                                  :fresh (make-array n :element-type 'bit :initial-element 0))))))))

(defvar *ds4f-lock-memory* t
  "Whether the expert pool and the GPU's weights are locked in memory
   (mlock), as ds4 locks its resident weights: a command buffer makes every
   buffer it uses resident, and a pool the system had compressed in part
   (most of it is cold while a prefill runs) would be decompressed each time.")

(defun lock-buffer (b what)
  (when (and *ds4f-lock-memory* (not (gpu-mlock b)))
    (warn "illatio: could not lock the ~A (~,1F GB) in memory" what (/ (gbuf-bytes b) 1d9)))
  b)

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

(defparameter *ds4f-read-piece* (* 1024 1024)
  "Expert reads go in pieces of this many bytes, on several threads: the
   SSD reads many at once.")

(defun expert-pieces (m l experts offsets)
  "The reads of layer L's EXPERTS (gate, up, down) to OFFSETS in the cache:
   (file-offset buffer-offset bytes), pieces of *DS4F-READ-PIECE*."
  (let* ((st (ds4f-gpu-state m)) (gb (d4gpu-gate-bytes st)) (piece *ds4f-read-piece*) (jobs '()))
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
    (nreverse jobs)))

(defun read-experts-pieces (m l experts offsets buf)
  "Layer L's EXPERTS from the file into BUF at OFFSETS, across the pool."
  (let ((v (coerce (expert-pieces m l experts offsets) 'simple-vector)) (base (gbuf-sap buf)) (fd (ds4f-fd m)))
    (parallel-for (j (length v) (ds4f-pool m))
      (destructuring-bind (src dst bytes) (svref v j)
        (pread-into fd (sb-sys:sap+ base dst) src bytes)))))

;;; Reads ahead.  While layer L's experts run, the next layer's are guessed
;;; -- its router on this layer's FFN input (the stream changes little from a
;;; layer to the next; the top 8 so found hold about 82% of the six it picks)
;;; -- and the missing ones read into the cache by threads of their own.  A
;;; slot being read is PENDING (pieces left); the next layer waits for the
;;; ones it picked, and none is given up while read.  A wrong guess costs
;;; only a read: the routing is the layer's own.

(defstruct (d4reader (:conc-name dr-))
  (mutex (sb-thread:make-mutex :name "illatio expert reads")) (cv (sb-thread:make-waitqueue))
  (urgent '()) (urgent-tail '()) (ahead '()) (ahead-tail '()) threads)

(defun enqueue-reads (r jobs urgent)
  "JOBS ((src dst bytes slot)...) onto R's urgent queue (demand) or its
   queue ahead; a reader takes an urgent job before any other."
  (when jobs
    (sb-thread:with-mutex ((dr-mutex r))
      (if urgent
          (progn (if (dr-urgent-tail r) (setf (cdr (dr-urgent-tail r)) jobs) (setf (dr-urgent r) jobs))
                 (setf (dr-urgent-tail r) (last jobs)))
          (progn (if (dr-ahead-tail r) (setf (cdr (dr-ahead-tail r)) jobs) (setf (dr-ahead r) jobs))
                 (setf (dr-ahead-tail r) (last jobs))))
      (sb-thread:condition-broadcast (dr-cv r)))))

(defun ds4f-reader (m xc)
  "XC's reading-ahead threads (started the first time)."
  (let ((st (ds4f-gpu-state m)))
    (or (getf (d4gpu-plist* st) :reader)
        (let ((r (make-d4reader)) (fd (ds4f-fd m)) (base (gbuf-sap (dx-buf xc))) (pending (dx-pending xc)))
          (declare (type (simple-array fixnum (*)) pending))
          (setf (dr-threads r)
                (loop repeat 8
                      collect (sb-thread:make-thread
                               (lambda ()
                                 (loop
                                   (let ((job nil))
                                     (sb-thread:with-mutex ((dr-mutex r))
                                       (loop until (or (dr-urgent r) (dr-ahead r))
                                             do (sb-thread:condition-wait (dr-cv r) (dr-mutex r)))
                                       (if (dr-urgent r)
                                           (progn (setf job (pop (dr-urgent r)))
                                                  (unless (dr-urgent r) (setf (dr-urgent-tail r) '())))
                                           (progn (setf job (pop (dr-ahead r)))
                                                  (unless (dr-ahead r) (setf (dr-ahead-tail r) '())))))
                                     (destructuring-bind (src dst bytes slot) job
                                       (pread-into fd (sb-sys:sap+ base dst) src bytes)
                                       (sb-thread:with-mutex ((dr-mutex r))
                                         (when (zerop (decf (aref pending slot)))
                                           (sb-thread:condition-broadcast (dr-cv r))))))))
                               :name "illatio expert reads")))
          (setf (getf (d4gpu-plist* st) :reader) r)))))

(defun cache-victim (xc)
  "A slot for a new expert: one given back, else one never used, else the
   least recently used that no work holds (pinned) or is reading."
  (cond ((dx-frees xc) (pop (dx-frees xc)))
        ((< (dx-free xc) (dx-n xc)) (prog1 (dx-free xc) (incf (dx-free xc))))
        (t (let ((used (dx-used xc)) (pending (dx-pending xc)) (pin (dx-pin xc))
                 (best -1) (bu most-positive-fixnum))
             (declare (type (simple-array fixnum (*)) used pending pin) (type fixnum best bu))
             (dotimes (j (dx-n xc))
               (when (and (< (aref used j) bu) (zerop (aref pending j)) (zerop (aref pin j)))
                 (setf best j bu (aref used j))))
             (assert (>= best 0) () "every expert slot is held by work in flight")
             best))))

(defun cache-claim (xc key tick)
  "A slot of XC for KEY, its old expert forgotten."
  (let ((s (cache-victim xc)) (owner (dx-owner xc)) (slot-of (dx-slot-of xc)))
    (declare (type (simple-array fixnum (*)) owner slot-of))
    (when (>= (aref owner s) 0) (setf (aref slot-of (aref owner s)) -1))
    (setf (aref owner s) key (aref slot-of key) s (aref (dx-used xc) s) tick (aref (dx-fresh xc) s) 0)
    s))

(defun cache-release (xc s)
  "Give slot S back: its expert forgotten, the slot first to be taken."
  (let ((owner (dx-owner xc)))
    (declare (type (simple-array fixnum (*)) owner))
    (when (>= (aref owner s) 0) (setf (aref (dx-slot-of xc) (aref owner s)) -1))
    (setf (aref owner s) -1 (aref (dx-used xc) s) 0 (aref (dx-fresh xc) s) 0)
    (push s (dx-frees xc))))

(defun cache-read (m xc li slots experts urgent)
  "Start reading layer LI's EXPERTS into SLOTS (claimed), urgent or ahead."
  (let* ((r (ds4f-reader m xc)) (sb (d4gpu-slot-bytes (ds4f-gpu-state m))) (l (svref (ds4f-layers m) li))
         (jobs '()))
    (loop for s in slots for e in experts
          do (let ((pieces (expert-pieces m l (list e) (list (* s sb)))))
               (sb-thread:with-mutex ((dr-mutex r)) (setf (aref (dx-pending xc) s) (length pieces)))
               (dolist (p pieces) (push (append p (list s)) jobs))))
    (enqueue-reads r (nreverse jobs) urgent)))

(defun cache-wait (m xc slots)
  "Wait until none of SLOTS is being read."
  (let ((pending (dx-pending xc)))
    (declare (type (simple-array fixnum (*)) pending))
    (when (some (lambda (s) (plusp (aref pending s))) slots)
      (let ((r (ds4f-reader m xc)))
        (incf (dx-waited xc))
        (sb-thread:with-mutex ((dr-mutex r))
          (loop while (some (lambda (s) (plusp (aref pending s))) slots)
                do (sb-thread:condition-wait (dr-cv r) (dr-mutex r))))))))

(defun cache-experts (m l li sel)
  "The cache slots of layer LI's experts SEL for decode: the missing ones
   read in (urgently: ahead of any read ahead), the ones being read waited
   for; pinned until the next layer's lookup (their layer runs between)."
  (declare (ignore l))
  (let* ((xc (ds4f-expert-cache m)) (ne (ds4f-n-expert m)) (k (length sel))
         (slots (make-array k :element-type 'fixnum)) (miss-slots '()) (miss-experts '())
         (slot-of (dx-slot-of xc)) (used (dx-used xc)) (pin (dx-pin xc))
         (tick (incf (dx-tick xc))))
    (declare (type (simple-array fixnum (*)) slot-of used pin))
    (dolist (s (dx-decode-pins xc)) (decf (aref pin s)))
    (setf (dx-decode-pins xc) '())
    (dotimes (i k)
      (let ((s (aref slot-of (+ (* li ne) (aref sel i)))))
        (cond ((>= s 0) (setf (aref used s) tick (aref slots i) s) (incf (dx-hits xc)))
              (t (let ((s (cache-claim xc (+ (* li ne) (aref sel i)) tick)))
                   (setf (aref slots i) s)
                   (push s miss-slots) (push (aref sel i) miss-experts)
                   (incf (dx-misses xc)))))
        (incf (aref pin (aref slots i))) (push (aref slots i) (dx-decode-pins xc))))
    (let ((t0 (get-internal-real-time)))
      (cache-read m xc li miss-slots miss-experts t)
      (cache-wait m xc (coerce slots 'list))
      (incf (dx-read-time xc) (- (get-internal-real-time) t0)))
    slots))

(defun prefetch-cache-experts (m li experts)
  "Start reading layer LI's EXPERTS the pool lacks, ahead (behind any demand)."
  (let* ((xc (ds4f-expert-cache m)) (ne (ds4f-n-expert m)) (slot-of (dx-slot-of xc))
         (tick (dx-tick xc)) (slots '()) (missing '()))
    (declare (type (simple-array fixnum (*)) slot-of))
    (dolist (e experts)
      (let ((key (+ (* li ne) e)))
        (if (>= (aref slot-of key) 0)
            (setf (aref (dx-used xc) (aref slot-of key)) tick)
            (progn (push (cache-claim xc key tick) slots) (push e missing) (incf (dx-prefetched xc))))))
    (cache-read m xc li (nreverse slots) (nreverse missing) nil)))

;;; --- prefill's side of the pool ------------------------------------------------------

(defun pool-take-layer (m li experts urgent)
  "Slots for layer LI's EXPERTS for a prefill, pinned: the resident ones as
   they are, the others claimed (fresh) and their reads started.  A vector
   of the slots, by position in EXPERTS."
  (let* ((xc (ds4f-expert-cache m)) (ne (ds4f-n-expert m)) (slot-of (dx-slot-of xc)) (pin (dx-pin xc))
         (tick (dx-tick xc)) (out (make-array (length experts) :element-type 'fixnum))
         (rs '()) (re '()))
    (declare (type (simple-array fixnum (*)) slot-of pin))
    (loop for e in experts for i from 0
          do (let ((s (aref slot-of (+ (* li ne) e))))
               (when (minusp s)
                 (setf s (cache-claim xc (+ (* li ne) e) tick))
                 (setf (aref (dx-fresh xc) s) 1)
                 (push s rs) (push e re))
               (incf (aref pin s))
               (setf (aref out i) s)))
    (cache-read m xc li (nreverse rs) (nreverse re) urgent)
    out))

(defun pool-done-layer (m slots keep)
  "A prefill's layer done with SLOTS: unpinned; the fresh ones given back
   but for those holding experts KEEP says to (the chunk's last tokens'),
   which stay as recently used."
  (let* ((xc (ds4f-expert-cache m)) (pin (dx-pin xc)) (tick (incf (dx-tick xc))))
    (declare (type (simple-array fixnum (*)) pin))
    ;; a slot read ahead for an expert no token chose may be reading yet: never given back mid-read
    (cache-wait m xc (coerce slots 'list))
    (loop for s across slots
          do (decf (aref pin s))
             (cond ((funcall keep s) (setf (aref (dx-used xc) s) tick (aref (dx-fresh xc) s) 0))
                   ((and (= 1 (aref (dx-fresh xc) s)) (zerop (aref pin s))) (cache-release xc s))))))

(defun predict-experts (m li token k router-logits)
  "A guess at layer LI's experts for a token: its hash table's, else the top
   K of its ROUTER-LOGITS (the GPU's, on the previous layer's FFN input)."
  (let* ((l (svref (ds4f-layers m) li)) (nu (ds4f-n-used m)))
    (if (dl-tid2eid l)
        (loop for i below nu collect (aref (dl-tid2eid l) (+ (* token nu) i)))
        (let* ((ne (ds4f-n-expert m)) (logits (f32vec ne)) (b (dl-probs-b l)))
          (declare (type f32vec logits))
          (replace logits router-logits)
          (dotimes (i ne) (setf (aref logits i) (+ (sqrt (d4-softplus (aref logits i))) (if b (aref b i) 0f0))))
          (loop repeat k
                collect (let ((best (let ((bi 0)) (dotimes (i ne bi) (when (> (aref logits i) (aref logits bi)) (setf bi i))))))
                          (setf (aref logits best) most-negative-single-float)
                          best))))))

;;; --- small things the GPU paths share --------------------------------------------------

(defun q8mv-enc (m gpu e li short x y &key (group-rows nil) (group-stride 0))
  "Encode into E layer LI's Q8_0 tensor SHORT times buffer X into buffer Y
   (D4_MV_Q8_0: X in f32), row r of a grouped one against group r/GROUP-ROWS."
  (destructuring-bind (buf . table) (ds4f-gpu-weights m)
    (destructuring-bind (woff rbytes rows cols) (gethash (format nil "blk.~D.~A" li short) table)
      (dispatch gpu e "d4_mv_q8_0" (list cols rows rbytes (or group-rows rows) group-stride)
                (list (cons buf woff) x y) (ceiling rows 8) 64))))

(defun copy-floats-out (v buf n)
  (declare (type f32vec v))
  (sb-sys:with-pinned-objects (v) (sap-copy (sb-sys:vector-sap v) (gbuf-sap buf) (* 4 n))))

