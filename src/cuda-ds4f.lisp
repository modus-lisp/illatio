;;;; cuda-ds4f.lisp — DeepSeek V4 Flash's decode on an NVIDIA GPU, its
;;;; routed experts in host memory.
;;;
;;; metal-ds4f-decode.lisp's layers with src/cuda/ds4f.cu's kernels.  The
;;; difference is where the weights live.  An M4's GPU shares the Mac's
;;; memory and reads the SSD; here the GPU has 24 GB of its own, behind PCIe,
;;; and the host has a great deal more.  So the file is read once into host
;;; memory, pinned (86.7 GB); the non-routed weights go to the device (the
;;; Q8_0 ones repacked as dense.cu multiplies them, the F16 ones as they
;;; are: 7.4 GB); the routed experts (78 GB) stay in host memory and are
;;; copied by DMA, on a stream of their own, into a pool of device slots as
;;; they are wanted -- the pool seeded with ds4's most used experts, kept by
;;; least recent use, the next layer's guessed experts copied ahead.
;;;
;;; A layer goes as on Metal: the previous layer's routed experts, this
;;; layer's attention, its FFN up to the router's logits and the shared
;;; expert; then a wait for the logits, the routing on the CPU (D4-ROUTE),
;;; the slots' copies queued, and the compute stream made to wait for them.
;;; The caches (the context's static vectors) live on the device: a copy
;;; of each is made when decode first uses it, and decode alone uses it
;;; from then on (prefill here is decode, a token at a time).

(in-package #:illatio)

(defvar *ds4f-cuda-device* 0 "The device DS4F's decode runs on.")

(defvar *ds4f-cuda-reserve-gb* 1.5
  "Device memory left free when the expert pool is sized: the scratch
   buffers, the caches of a context, the driver's own.")

(defvar *ds4f-route-trace* nil "A stream each layer's routing is written to (layer, experts), or NIL.")

(defvar *ds4f-cuda-pool-gb* nil "The expert pool's size in GB; NIL: what the device has left.")

(defstruct (d4cuda (:conc-name dk-))
  dev copy-stream event
  (q8 (make-hash-table :test 'equal))   ; tensor name -> repacked CMAT
  (f16 (make-hash-table :test 'equal))  ; tensor name -> (dbuf rows cols)
  (vecs (make-hash-table :test 'eq))    ; host f32vec -> its device copy
  (bufs '())                            ; scratch: key -> dbuf
  arena-host arena-dev (arena-at 0)     ; small per-layer inputs, staged in pinned memory
  rb                                    ; pinned memory the router's logits come back to
  host (host-bytes 0)                   ; the file, in pinned host memory
  pool (n-slots 0) (slot-bytes 0) (gate-bytes 0)
  slot-of owner used pin (tick 0) (free 0)
  tier                                  ; the second device's experts (D4TIER), or NIL
  (hits 0) (misses 0) (prefetched 0) (copied 0))

(defvar *ds4f-cuda-tier-device* 1
  "A second device whose memory holds more of the experts -- the next most
   used after the first device's, fixed -- and computes them, or NIL.")

;;; The second device's experts: a pool seeded once from the hotlist (past
;;; what the first device's holds) and never changed (its link is the
;;; slower), its experts computed there.  A layer's experts there get the
;;; FFN input, their pointers and weights in one copy from pinned memory;
;;; their down rows come back to pinned memory, and the first device's
;;; stream waits for that (an event of the second's) before copying them
;;; in beside its own rows -- the CPU waits for neither.

(defstruct (d4tier (:conc-name dt-))
  dev event pool (n 0) slot-of
  host-in in host-out                   ; pinned staging, and the device's copy of the inputs
  gu mid down (hits 0))

(defconstant +dt-ptrs+ (* 4 4096) "Where the experts' pointers are in a tier's inputs (after X).")
(defconstant +dt-w+ (+ +dt-ptrs+ (* 8 32)) "Where their weights are.")

(defvar *cuda-device-count* nil)

(defun ds4f-cuda-p ()
  "Whether DS4F's decode runs on a CUDA device: *DS4F-GPU* says so and
   there is one."
  (and (ds4f-gpu-p :decode)
       (plusp (or *cuda-device-count* (setf *cuda-device-count* (or (ignore-errors (cuda-device-count)) 0))))))

;;; --- the file in host memory ------------------------------------------------------

(defun d4-pread (fd sap offset bytes)
  (loop with done = 0
        while (< done bytes)
        do (let ((got (sb-alien:alien-funcall
                       (sb-alien:extern-alien "pread" (function sb-alien:long sb-alien:int sb-sys:system-area-pointer
                                                                sb-alien:unsigned-long sb-alien:long))
                       fd (sb-sys:sap+ sap done) (- bytes done) (+ offset done))))
             (when (<= got 0) (error "pread of ~:D bytes at ~:D failed" bytes offset))
             (incf done got))))

(defun d4-memcpy (dst src bytes)
  (sb-alien:alien-funcall
   (sb-alien:extern-alien "memcpy" (function sb-sys:system-area-pointer sb-sys:system-area-pointer
                                             sb-sys:system-area-pointer sb-alien:unsigned-long))
   dst src bytes))

(defun d4-host-copy (m)
  "(values sap bytes): the whole of DS4F M's file in host memory, read by
   the model's threads in 64 MB pieces, then pinned."
  (let* ((path (namestring (truename (gguf-path (ds4f-gguf m)))))
         (bytes (with-open-file (s path :element-type '(unsigned-byte 8)) (file-length s)))
         (span (* 2097152 (ceiling bytes 2097152)))
         (base (%mmap-anon span)) (sap (sb-sys:int-sap base))
         (fd (or (sb-unix:unix-open path sb-unix:o_rdonly 0) (error "cannot open ~A" path)))
         (piece (* 64 1024 1024)) (t0 (get-internal-real-time)))
    ;; huge pages: fewer for the kernel to walk, and for the driver to pin
    (sb-alien:alien-funcall (sb-alien:extern-alien "madvise" (function sb-alien:int sb-alien:unsigned-long
                                                                       sb-alien:unsigned-long sb-alien:int))
                            base span 14)  ; MADV_HUGEPAGE
    (unwind-protect
         (parallel-for (k (ceiling bytes piece) (ds4f-pool m))
           (let ((o (* k piece)))
             (d4-pread fd (sb-sys:sap+ sap o) o (min piece (- bytes o)))))
      (sb-unix:unix-close fd))
    (let ((t1 (get-internal-real-time)))
      (cuda-host-register sap span)
      (format *error-output* "illatio: ~,1F GB read in ~,1Fs, pinned in ~,1Fs~%" (/ bytes 1d9)
              (/ (- t1 t0) internal-time-units-per-second)
              (/ (- (get-internal-real-time) t1) internal-time-units-per-second)))
    (values sap bytes)))

;;; --- the device's weights ------------------------------------------------------------

(defparameter +ds4f-cuda-q8-tensors+
  '("attn_q_a.weight" "attn_q_b.weight" "attn_kv.weight" "attn_output_a.weight" "attn_output_b.weight"
    "ffn_gate_shexp.weight" "ffn_up_shexp.weight" "ffn_down_shexp.weight"))

(defparameter +ds4f-cuda-f16-tensors+
  '("hc_attn_fn.weight" "hc_ffn_fn.weight" "attn_compressor_kv.weight" "attn_compressor_gate.weight"
    "indexer.attn_q_b.weight" "indexer.proj.weight" "indexer_compressor_kv.weight"
    "indexer_compressor_gate.weight" "ffn_gate_inp.weight"))

(defun d4-tensor-names (per-layer m &optional extra)
  (append extra
          (loop for li below (ds4f-n-layer m)
                append (mapcar (lambda (s) (format nil "blk.~D.~A" li s)) per-layer))))

(defun d4-upload-weights (m st)
  "The Q8_0 tensors repacked on the device, the F16 ones as they are."
  (let* ((g (ds4f-gguf m)) (dev (dk-dev st)) (host (dk-host st)) (raw nil))
    (unwind-protect
         (progn
           (dolist (name (d4-tensor-names +ds4f-cuda-q8-tensors+ m '("output.weight")))
             (let* ((ti (gguf-tensor g name)) (dims (tensor-info-dims ti))
                    (cols (first dims)) (rows (reduce #'* (rest dims))) (bytes (tensor-bytes ti))
                    (nb (floor (* rows cols) 32)))
               (assert (eq (tensor-info-type ti) :q8_0))
               (unless (and raw (>= (dbuf-bytes raw) bytes))
                 (cuda-free raw)
                 (setf raw (cuda-alloc dev bytes :zero nil)))
               (let ((qs (cuda-alloc dev (* rows cols) :zero nil)) (d (cuda-alloc dev (* 2 nb) :zero nil)))
                 (cuda-upload raw (sb-sys:sap+ host (+ (gguf-data-offset g) (tensor-info-offset ti))) bytes)
                 (launch dev "repack_q8" (ceiling nb 8) '(32 8) (list raw qs d (list :i64 nb)))
                 (setf (gethash name (dk-q8 st))
                       (%make-cmat :name name :type :q8_0 :rows rows :cols cols :qs qs :d d)))))
           (dolist (name (d4-tensor-names +ds4f-cuda-f16-tensors+ m '("output_hc_fn.weight")))
             (let ((ti (gguf-tensor g name nil)))
               (when ti
                 (assert (eq (tensor-info-type ti) :f16))
                 (let* ((dims (tensor-info-dims ti)) (bytes (tensor-bytes ti))
                        (b (cuda-alloc dev bytes :zero nil)))
                   (cuda-upload b (sb-sys:sap+ host (+ (gguf-data-offset g) (tensor-info-offset ti))) bytes)
                   (setf (gethash name (dk-f16 st)) (list b (reduce #'* (rest dims)) (first dims))))))))
      (cuda-free raw))))

(defun ds4f-cuda-state (m)
  "DS4F M's device state, made the first time: the file in pinned memory,
   the weights on the device, the expert pool."
  (or (getf (ds4f-gpu-plist m) :cuda)
      (let* ((dev (open-cuda *ds4f-cuda-device*)) (l0 (svref (ds4f-layers m) 0))
             (gb (* (third (dl-gate-exps l0)) (second (dl-gate-exps l0))))
             (db (* (third (dl-down-exps l0)) (second (dl-down-exps l0))))
             (st (make-d4cuda :dev dev :slot-bytes (* 256 (ceiling (+ (* 2 gb) db) 256)) :gate-bytes gb)))
        (with-cuda
          (cuda-current dev)
          (setf (dk-copy-stream st) (cuda-stream-create dev) (dk-event st) (cuda-event-create dev)
                (dk-arena-host st) (cuda-host-alloc (* 1024 1024)) (dk-arena-dev st) (cuda-alloc dev (* 1024 1024))
                (dk-rb st) (cuda-host-alloc (* 64 1024)))
          (multiple-value-bind (sap bytes) (d4-host-copy m)
            (setf (dk-host st) sap (dk-host-bytes st) bytes))
          (d4-upload-weights m st)
          (d4-make-pool m st)
          (when (and *ds4f-cuda-tier-device* (> (cuda-device-count) *ds4f-cuda-tier-device*)
                     (/= *ds4f-cuda-tier-device* *ds4f-cuda-device*))
            (setf (dk-tier st) (d4-make-tier m st (open-cuda *ds4f-cuda-tier-device*)))
            (cuda-current dev)))
        (setf (getf (ds4f-gpu-plist m) :cuda) st))))

;;; --- device copies of host vectors, scratch, the arena --------------------------------

(defun dvec (st v)
  "The device's copy of f32 vector V (made, from V's contents, the first time)."
  (declare (type f32vec v))
  (or (gethash v (dk-vecs st))
      (setf (gethash v (dk-vecs st))
            (let ((b (cuda-alloc (dk-dev st) (* 4 (max 1 (length v))) :zero nil)))
              (cuda-upload-vector b v)
              b))))

(defun forget-gpu-vector (m v)
  "Free the device's copy of V (before V is freed)."
  (let ((st (getf (ds4f-gpu-plist m) :cuda)))
    (when st
      (let ((b (gethash v (dk-vecs st))))
        (when b (cuda-free b) (remhash v (dk-vecs st)))))))

(defun dbuf-of (st key bytes)
  "Scratch buffer KEY of ST, at least BYTES."
  (let ((b (getf (dk-bufs st) key)))
    (unless (and b (>= (dbuf-bytes b) bytes))
      (cuda-free b)
      (setf b (cuda-alloc (dk-dev st) (max bytes 256)) (getf (dk-bufs st) key) b))
    b))

(defun dk-arena-reset (st) (setf (dk-arena-at st) 0))

(defun dk-put (st words &key (type :u32))
  "WORDS (floats for TYPE :F32, else integers) to the device by way of the
   arena: (dbuf . offset), its copy queued on the compute stream."
  (let* ((size (if (eq type :u64) 8 4)) (n (length words)) (at (dk-arena-at st)) (sap (dk-arena-host st)))
    (assert (< (+ at (* size n)) (* 1024 1024)))
    (let ((i 0))
      (map nil (lambda (x)
                 (ecase type
                   (:f32 (setf (sb-sys:sap-ref-single sap (+ at (* 4 i))) x))
                   (:u32 (setf (sb-sys:sap-ref-32 sap (+ at (* 4 i))) x))
                   (:u64 (setf (sb-sys:sap-ref-64 sap (+ at (* 8 i))) x)))
                 (incf i))
           words))
    (cuda-copy-in (dk-arena-dev st) at (sb-sys:sap+ sap at) (* size n))
    (setf (dk-arena-at st) (* 16 (ceiling (+ at (* size n)) 16)))
    (cons (dk-arena-dev st) at)))

(defun dk-cs (m st pos il &optional inverse)
  (dk-put st (d4-rope-cs! m pos il inverse (f32vec (ds4f-n-rot m))) :type :f32))

;;; --- the expert pool --------------------------------------------------------------

(defun d4-make-pool (m st)
  "The pool: as many slots as the device has room for, seeded."
  (let* ((dev (dk-dev st)) (sb (dk-slot-bytes st))
         (free (cuda-mem-info dev))
         (bytes (if *ds4f-cuda-pool-gb*
                    (* *ds4f-cuda-pool-gb* 1d9)
                    (- free (* *ds4f-cuda-reserve-gb* 1d9))))
         (n (max 8 (floor bytes sb)))
         (keys (* (ds4f-n-layer m) (ds4f-n-expert m))))
    (setf (dk-pool st) (cuda-alloc dev (* n sb) :zero nil) (dk-n-slots st) n
          (dk-slot-of st) (make-array keys :element-type 'fixnum :initial-element -1)
          (dk-owner st) (make-array n :element-type 'fixnum :initial-element -1)
          (dk-used st) (make-array n :element-type 'fixnum :initial-element 0)
          (dk-pin st) (make-array n :element-type 'fixnum :initial-element 0))
    (format *error-output* "illatio: expert pool of ~D slots (~,1F GB) on ~A~%" n (/ (* n sb) 1d9) (cudev-name dev))
    ;; ds4's most used experts first
    (when (and (= (ds4f-n-layer m) 43) (= (ds4f-n-expert m) 256))
      (let ((k (min n (floor (length +ds4f-hotlist+) 2))))
        (dotimes (s k)
          (let ((li (aref +ds4f-hotlist+ (* 2 s))) (e (aref +ds4f-hotlist+ (1+ (* 2 s)))))
            (setf (aref (dk-owner st) s) (+ (* li (ds4f-n-expert m)) e)
                  (aref (dk-slot-of st) (+ (* li (ds4f-n-expert m)) e)) s)
            (d4-copy-expert m st li e s)))
        (setf (dk-free st) k)
        (cuda-stream-sync (dk-copy-stream st))))
    st))

(defun d4-make-tier (m st dev)
  "The second device's pool: as many of the hotlist's experts after the
   first device's as it has room for."
  (cuda-current dev)
  (let* ((sb (dk-slot-bytes st)) (d (ds4f-d m)) (ff (ds4f-ff-exp m)) (k (ds4f-n-used m))
         (stream (cuda-stream-create dev))
         (scratch (+ (* 4 3 k ff) (* 4 k d) (* 2 +dt-w+)))
         (free (- (cuda-mem-info dev) scratch (* 0.6d9)))
         (first (dk-free st))
         (n (max 0 (min (floor free sb) (- (floor (length +ds4f-hotlist+) 2) first))))
         (keys (* (ds4f-n-layer m) (ds4f-n-expert m)))
         (tr (make-d4tier :dev dev :event (cuda-event-create dev) :n n
                          :slot-of (make-array keys :element-type 'fixnum :initial-element -1)
                          :host-in (cuda-host-alloc (* 2 +dt-w+)) :in (cuda-alloc dev (* 2 +dt-w+))
                          :host-out (cuda-host-alloc (* 4 k d))
                          :gu (cuda-alloc dev (* 4 2 k ff)) :mid (cuda-alloc dev (* 4 k ff))
                          :down (cuda-alloc dev (* 4 k d)))))
    (assert (= d 4096) () "a tier's inputs are laid out for 4096 wide")
    ;; the device's own stream is its compute stream; copies go on another
    (setf (dt-pool tr) (cuda-alloc dev (* (max n 1) sb) :zero nil))
    (when (plusp n)
      (dotimes (s n)
        (let* ((h (+ first s)) (li (aref +ds4f-hotlist+ (* 2 h))) (e (aref +ds4f-hotlist+ (1+ (* 2 h))))
               (l (svref (ds4f-layers m) li)) (host (dk-host st)) (gb (dk-gate-bytes st)) (at (* s sb)))
          (setf (aref (dt-slot-of tr) (+ (* li (ds4f-n-expert m)) e)) s)
          (destructuring-bind (goff grb grows gcols) (dl-gate-exps l)
            (declare (ignore gcols))
            (destructuring-bind (uoff urb urows ucols) (dl-up-exps l)
              (declare (ignore urows ucols))
              (destructuring-bind (doff drb drows dcols) (dl-down-exps l)
                (declare (ignore dcols))
                (cuda-copy-in (dt-pool tr) at (sb-sys:sap+ host (+ goff (* e grows grb))) gb stream)
                (cuda-copy-in (dt-pool tr) (+ at gb) (sb-sys:sap+ host (+ uoff (* e grows urb))) gb stream)
                (cuda-copy-in (dt-pool tr) (+ at (* 2 gb)) (sb-sys:sap+ host (+ doff (* e drows drb))) (* drows drb) stream))))))
      (cuda-stream-sync stream))
    (format *error-output* "illatio: and ~D more (~,1F GB) on ~A~%" n (/ (* n sb) 1d9) (cudev-name dev))
    tr))

(defun d4-copy-expert (m st li e s)
  "Queue layer LI's expert E's copy into slot S, on the copy stream."
  (let* ((l (svref (ds4f-layers m) li)) (host (dk-host st)) (gb (dk-gate-bytes st))
         (at (* s (dk-slot-bytes st))) (cs (dk-copy-stream st)))
    (destructuring-bind (goff grb grows gcols) (dl-gate-exps l)
      (declare (ignore gcols))
      (destructuring-bind (uoff urb urows ucols) (dl-up-exps l)
        (declare (ignore urows ucols))
        (destructuring-bind (doff drb drows dcols) (dl-down-exps l)
          (declare (ignore dcols))
          (cuda-copy-in (dk-pool st) at (sb-sys:sap+ host (+ goff (* e grows grb))) gb cs)
          (cuda-copy-in (dk-pool st) (+ at gb) (sb-sys:sap+ host (+ uoff (* e grows urb))) gb cs)
          (cuda-copy-in (dk-pool st) (+ at (* 2 gb)) (sb-sys:sap+ host (+ doff (* e drows drb))) (* drows drb) cs)
          (incf (dk-copied st)))))))

(defun d4-victim (st)
  "A slot for a new expert: one never used, else the least recently used
   that no layer in flight holds."
  (if (< (dk-free st) (dk-n-slots st))
      (prog1 (dk-free st) (incf (dk-free st)))
      (let ((used (dk-used st)) (pin (dk-pin st)) (best -1) (bu most-positive-fixnum))
        (declare (type (simple-array fixnum (*)) used pin) (type fixnum best bu))
        (dotimes (j (dk-n-slots st))
          (when (and (< (aref used j) bu) (zerop (aref pin j)))
            (setf best j bu (aref used j))))
        (assert (>= best 0) () "every expert slot is held")
        best)))

(defun d4-claim (m st li e)
  "A slot for layer LI's expert E, its copy queued."
  (let* ((key (+ (* li (ds4f-n-expert m)) e)) (s (d4-victim st))
         (owner (dk-owner st)) (slot-of (dk-slot-of st)))
    (declare (type (simple-array fixnum (*)) owner slot-of))
    (when (>= (aref owner s) 0) (setf (aref slot-of (aref owner s)) -1))
    (setf (aref owner s) key (aref slot-of key) s (aref (dk-used st) s) (dk-tick st))
    (d4-copy-expert m st li e s)
    s))

(defstruct (d4plan (:conc-name dp-))
  a-slots a-idx                         ; the first device's experts: its slots, their places in SEL
  b-slots b-idx)                        ; the second's

(defun d4-plan (m st li sel prev)
  "Where layer LI's experts SEL are computed: the first device's slots
   (pinned, PREV's let go; the missing ones claimed and their copies
   queued, and the compute stream made to wait for every copy so far), and
   the second's."
  (let* ((ne (ds4f-n-expert m)) (k (length sel)) (tr (dk-tier st))
         (slot-of (dk-slot-of st)) (used (dk-used st)) (pin (dk-pin st)) (tick (incf (dk-tick st)))
         (as '()) (ai '()) (bs '()) (bi '()))
    (declare (type (simple-array fixnum (*)) slot-of used pin))
    (when prev (loop for s across (dp-a-slots prev) do (decf (aref pin s))))
    (dotimes (i k)
      (let* ((key (+ (* li ne) (aref sel i))) (s (aref slot-of key))
             (s2 (if (and tr (minusp s)) (aref (dt-slot-of tr) key) -1)))
        (cond ((>= s 0) (setf (aref used s) tick) (incf (dk-hits st)) (push s as) (push i ai))
              ((>= s2 0) (incf (dt-hits tr)) (push s2 bs) (push i bi))
              (t (setf s (d4-claim m st li (aref sel i))) (incf (dk-misses st)) (push s as) (push i ai)))
        (when (>= s 0) (incf (aref pin s)))))
    (cuda-event-record (dk-event st) (dk-copy-stream st))
    (cuda-stream-wait (cudev-stream (dk-dev st)) (dk-event st))
    (flet ((v (l) (coerce (nreverse l) '(simple-array fixnum (*)))))
      (make-d4plan :a-slots (v as) :a-idx (v ai) :b-slots (v bs) :b-idx (v bi)))))

(defun d4-tier-launch (m st l plan w)
  "The second device's share of a layer's experts, queued on it: the FFN
   input (the first device's, come back in RB) and the experts' pointers
   and weights copied in, gate/up, SWIGLU, down, the down rows copied out,
   the event recorded."
  (let* ((tr (dk-tier st)) (dev (dt-dev tr)) (k (length (dp-b-slots plan)))
         (d (ds4f-d m)) (ff (ds4f-ff-exp m)) (h (dt-host-in tr)) (sb (dk-slot-bytes st)) (gb (dk-gate-bytes st))
         (base (dbuf-ptr (dt-pool tr))) (in (dt-in tr)) (li (position l (ds4f-layers m))))
    (cuda-current dev)
    (d4-memcpy h (sb-sys:sap+ (dk-rb st) 4096) (* 4 d))
    (loop for s across (dp-b-slots plan) for i from 0
          do (setf (sb-sys:sap-ref-64 h (+ +dt-ptrs+ (* 8 i))) (+ base (* s sb))
                   (sb-sys:sap-ref-64 h (+ +dt-ptrs+ (* 8 (+ k i)))) (+ base (* s sb) gb)))
    (loop for i across (dp-b-idx plan) for j from 0
          do (setf (sb-sys:sap-ref-single h (+ +dt-w+ (* 4 j))) (aref w i)))
    (cuda-copy-in in 0 h (+ +dt-w+ (* 4 k)) (cudev-stream dev))
    (launch dev "d4_mv_iq2_xxs" (list (ceiling ff 8) 1 (* 2 k)) 64
            (list d ff (second (dl-gate-exps l)) (list :i64 0) 0 ff (cons in +dt-ptrs+) in (dt-gu tr)))
    (launch dev "d4_swiglu_w" (ceiling (* k ff) 256) 256
            (list k ff (svref (ds4f-clamps m) li) (dt-gu tr) (cons (dt-gu tr) (* 4 k ff)) (cons in +dt-w+) (dt-mid tr)))
    (launch dev "d4_mv_q2_k" (list (ceiling d 8) 1 k) 64
            (list ff d (second (dl-down-exps l)) (list :i64 (* 2 gb)) ff d (cons in +dt-ptrs+) (dt-mid tr) (dt-down tr)))
    (cuda-copy-out (dt-host-out tr) (dt-down tr) 0 (* 4 k d))
    (cuda-event-record (dt-event tr) (cudev-stream dev))
    (cuda-current (dk-dev st))))

(defvar *ds4f-prefetch-max* 0
  "At most this many of the guessed experts a layer are copied ahead: about
   what the link moves while the device computes a layer (a copy ahead
   that is still running holds up the next layer's own).  0 here: a layer
   computes in about the time the link takes to move one expert, and a
   wrong guess costs a copy, so copying ahead loses more than it saves.")

(defun d4-prefetch (m st li experts)
  "Queue copies of the first *DS4F-PREFETCH-MAX* of layer LI's EXPERTS (the
   likeliest first) that neither device holds (behind the demand)."
  (let ((ne (ds4f-n-expert m)) (slot-of (dk-slot-of st)) (tr (dk-tier st)) (budget *ds4f-prefetch-max*))
    (declare (type (simple-array fixnum (*)) slot-of))
    (dolist (e experts)
      (let* ((key (+ (* li ne) e)) (s (aref slot-of key)))
        (cond ((>= s 0) (setf (aref (dk-used st) s) (dk-tick st)))
              ((and tr (>= (aref (dt-slot-of tr) key) 0)))
              ((plusp budget) (decf budget) (d4-claim m st li e) (incf (dk-prefetched st))))))))

(defun d4-predict (m li token k router-logits)
  "A guess at layer LI's experts: its hash table's, else the top K of
   ROUTER-LOGITS (its router on the previous layer's FFN input)."
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

;;; --- the ops ------------------------------------------------------------------------

(defun dk-rms (dev x w out n groups eps)
  (launch dev "d4_rms" groups (min 1024 n) (list n eps (if w 1 0) x (or w x) out)))

(defun dk-rope (dev x cs nh hd nrot)
  (launch dev "d4_rope" (ceiling (* nh (floor nrot 2)) 256) 256 (list nh hd nrot 1 x cs)))

(defun dk-mv16 (st name x y)
  (destructuring-bind (w rows cols) (or (gethash name (dk-f16 st)) (error "no F16 tensor ~A" name))
    (launch (dk-dev st) "d4_mv_f16" (ceiling rows 4) 256 (list cols rows w x y))))

(defun dk-q8mv (st name x y &key group-rows)
  "Y = W X for Q8_0 tensor NAME, X quantized first (ds4's Q8_0
   activations); row r against X's group r/GROUP-ROWS when grouped."
  (let* ((w (or (gethash name (dk-q8 st)) (error "no Q8_0 tensor ~A" name)))
         (dev (dk-dev st)) (rows (cmat-rows w)) (cols (cmat-cols w))
         (nx (if group-rows (* cols (floor rows group-rows)) cols))
         (xq (dbuf-of st :xq nx)) (xs (dbuf-of st :xs (* 4 (floor nx 32)))))
    (launch dev "d4_quant" (ceiling nx 256) 256 (list x xq xs (floor nx 32)))
    (launch dev "d4_mvq8" (ceiling rows 4) 128 (list rows cols (or group-rows rows) (cmat-qs w) (cmat-d w) xq xs y))))

(defun lname (li short) (format nil "blk.~D.~A" li short))

(defun dk-hc-pre (m st fn scale base streams out pc)
  (assert (= (ds4f-n-hc m) 4) () "d4_hc_pre_fin is written for 4 streams")
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (n (ds4f-n-hc m)) (hcd (* n d))
         (flat (dbuf-of st :flat (* 4 hcd))) (mix (dbuf-of st :mix (* 4 64))))
    (dk-rms dev streams nil flat hcd 1 (ds4f-eps m))
    (dk-mv16 st fn flat mix)
    (launch dev "d4_hc_pre_fin" (ceiling d 256) 256
            (list d n (ds4f-sinkhorn m) 1f-6 mix (dvec st scale) (dvec st base) streams out pc))))

(defun dk-hc-post (m st block streams pc out)
  (let ((d (ds4f-d m)) (n (ds4f-n-hc m)))
    (launch (dk-dev st) "d4_hc_post" (ceiling (* n d) 256) 256 (list d n 1 1f-6 block streams pc out))))

(defun dk-compressor (m st li c norm pos ratio wkv wgate ape normw rows-of n-of hd kind)
  "One compressor's step for the token at POS, and on a window's last
   token its row pushed: whether one was."
  (let* ((dev (dk-dev st)) (coff (if (= ratio 4) 2 1)) (width (* coff hd)) (pos-mod (mod pos ratio))
         (row (if (= ratio 4) (+ ratio pos-mod) pos-mod))
         (kv (dbuf-of st (if (= kind 0) :ckv :ikv) (* 4 width)))
         (sc (dbuf-of st (if (= kind 0) :csc :isc) (* 4 width)))
         (st-kv (if (= kind 0) (dc-st-kv c) (dc-ix-kv c))) (st-sc (if (= kind 0) (dc-st-score c) (dc-ix-score c))))
    (dk-mv16 st (lname li wkv) norm kv)
    (dk-mv16 st (lname li wgate) norm sc)
    (launch dev "d4_comp_store" (ceiling width 256) 256
            (list width row pos-mod (f32mat-cols ape) kv sc (dvec st (f32mat-data ape)) (dvec st st-kv) (dvec st st-sc)))
    (when (zerop (mod (1+ pos) ratio))
      (launch dev "d4_comp_pool" 1 hd
              (list hd ratio width (ds4f-eps m) kind (ds4f-n-rot m) (funcall n-of c)
                    (dvec st st-kv) (dvec st st-sc) (dvec st normw)
                    (dk-cs m st (- (1+ pos) ratio) li) (dvec st (funcall rows-of c)))
              :shared (* 4 (+ hd 32)))
      t)))

(defun dk-attention (m st l li c pos sa sb)
  "Layer LI's attention block for the token at POS: streams SA into SB."
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (hd (ds4f-head-dim m)) (nh (ds4f-n-head m)) (lq (ds4f-lora-q m))
         (nrot (ds4f-n-rot m)) (eps (ds4f-eps m)) (ratio (dl-ratio l)) (swa (ds4f-swa m))
         (cur (dbuf-of st :cur (* 4 d))) (norm (dbuf-of st :norm (* 4 d)))
         (pca (dbuf-of st :pca (* 4 32))) (qr (dbuf-of st :qr (* 4 lq)))
         (qrn (dbuf-of st :qrn (* 4 lq))) (q (dbuf-of st :q (* 4 nh hd)))
         (kv0 (dbuf-of st :kv0 (* 4 hd))) (heads (dbuf-of st :heads (* 4 nh hd)))
         (low (dbuf-of st :low (* 4 (ds4f-out-groups m) (ds4f-lora-o m))))
         (aout (dbuf-of st :aout (* 4 d)))
         (cs (dk-cs m st pos li)) (ics (dk-cs m st pos li t)))
    (dk-hc-pre m st (lname li "hc_attn_fn.weight") (dl-hc-attn-scale l) (dl-hc-attn-base l) sa cur pca)
    (dk-rms dev cur (dvec st (dl-attn-norm l)) norm d 1 eps)
    (dk-q8mv st (lname li "attn_q_a.weight") norm qr)
    (dk-q8mv st (lname li "attn_kv.weight") norm kv0)
    (dk-rms dev qr (dvec st (dl-q-a-norm l)) qrn lq 1 eps)
    (dk-q8mv st (lname li "attn_q_b.weight") qrn q)
    (dk-rms dev q nil q hd nh eps)
    (dk-rope dev q cs nh hd nrot)
    ;; the KV row into the ring (its counts kept on the CPU, as PUSH-RAW! keeps them)
    (launch dev "d4_kv_fin" 1 hd (list hd nrot eps (mod (dc-raw-total c) swa)
                                       kv0 (dvec st (dl-kv-norm l)) cs (dvec st (dc-raw c)))
            :shared (* 4 (+ hd 32)))
    (incf (dc-raw-total c))
    (setf (dc-n-raw c) (min swa (1+ (dc-n-raw c))))
    (when (plusp ratio)
      (when (dk-compressor m st li c norm pos ratio "attn_compressor_kv.weight" "attn_compressor_gate.weight"
                           (dl-comp-ape l) (dl-comp-norm l) #'dc-comp #'dc-n-comp hd 0)
        (incf (dc-n-comp c)))
      (when (= ratio 4)
        (when (dk-compressor m st li c norm pos ratio "indexer_compressor_kv.weight" "indexer_compressor_gate.weight"
                             (dl-ixc-ape l) (dl-ixc-norm l) #'dc-ix-comp #'dc-n-ix (ds4f-ix-dim m) 1)
          (incf (dc-n-ix c)))))
    (let* ((nc (if (dc-comp c) (dc-n-comp c) 0)) (top-k (ds4f-ix-top-k m))
           (indexed (and (= ratio 4) (> (dc-n-ix c) top-k)))
           (idx (dbuf-of st :ixidx (* 4 top-k))))
      (when indexed
        (let* ((ixh (ds4f-ix-heads m)) (ixd (ds4f-ix-dim m)) (nix (dc-n-ix c))
               (ixq (dbuf-of st :ixq (* 4 ixh ixd))) (ixw (dbuf-of st :ixw (* 4 ixh)))
               (scores (dbuf-of st :ixs (* 4 nix))))
          (assert (and (= ixd 128) (<= ixh 64)) () "d4_ix_scores1 is written for 128-wide heads, 64 at most")
          (dk-mv16 st (lname li "indexer.attn_q_b.weight") qrn ixq)
          (dk-mv16 st (lname li "indexer.proj.weight") norm ixw)
          (launch dev "d4_ixq_fin" ixh 128 (list ixh nrot (/ 1f0 (sqrt (float (* ixh ixd) 1f0))) ixq cs ixw))
          (launch dev "d4_ix_scores1" (ceiling nix 32) 256 (list ixh nix ixq ixw (dvec st (dc-ix-comp c)) scores))
          (launch dev "d4_topk" 1 1024 (list nix top-k scores idx))))
      (let ((kc (if (dc-comp c) (dvec st (dc-comp c)) (dvec st (dc-raw c))))
            (ncomp (if indexed top-k nc)))
        (assert (= hd 512) () "d4_attend1 is written for 512-wide heads (128 threads x 4)")
        (launch dev "d4_attend1" (floor nh 4) 128
                (list hd (/ 1f0 (sqrt (float hd 1f0))) swa (mod (- (dc-raw-total c) (dc-n-raw c)) swa)
                      (dc-n-raw c) ncomp (if indexed 0 1)
                      q (dvec st (dc-raw c)) kc idx (dvec st (dl-sinks l)) heads)
                :shared (* 4 (+ (* 4 hd) (* 4 (+ (dc-n-raw c) ncomp)) 8)))))
    (dk-rope dev heads ics nh hd nrot)
    (dk-q8mv st (lname li "attn_output_a.weight") heads low :group-rows (ds4f-lora-o m))
    (dk-q8mv st (lname li "attn_output_b.weight") low aout)
    (dk-hc-post m st aout sa pca sb)))

(defun dk-ffn-head (m st l li sb)
  "Layer LI's FFN up to the routing: HC pre of SB, the norm, the router's
   logits (and the next layer's router on the same input, for the guess),
   the shared expert."
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (ff (ds4f-ff-exp m)) (ne (ds4f-n-expert m))
         (cur (dbuf-of st :cur (* 4 d))) (norm2 (dbuf-of st :norm2 (* 4 d)))
         (pcf (dbuf-of st :pcf (* 4 32))) (logits (dbuf-of st :logits (* 4 2 ne)))
         (sg (dbuf-of st :sg (* 4 ff))) (su (dbuf-of st :su (* 4 ff)))
         (smid (dbuf-of st :smid (* 4 ff))) (sout (dbuf-of st :sout (* 4 d)))
         (one (dk-put st (list 1f0) :type :f32)))
    (dk-hc-pre m st (lname li "hc_ffn_fn.weight") (dl-hc-ffn-scale l) (dl-hc-ffn-base l) sb cur pcf)
    (dk-rms dev cur (dvec st (dl-ffn-norm l)) norm2 d 1 (ds4f-eps m))
    (dk-mv16 st (lname li "ffn_gate_inp.weight") norm2 logits)
    (when (and (plusp *ds4f-prefetch-k*) (plusp *ds4f-prefetch-max*) (< (1+ li) (ds4f-n-layer m))
               (null (dl-tid2eid (svref (ds4f-layers m) (1+ li)))))
      (dk-mv16 st (lname (1+ li) "ffn_gate_inp.weight") norm2 (cons logits (* 4 ne))))
    (dk-q8mv st (lname li "ffn_gate_shexp.weight") norm2 sg)
    (dk-q8mv st (lname li "ffn_up_shexp.weight") norm2 su)
    (launch dev "d4_swiglu_w" (ceiling ff 256) 256 (list 1 ff (svref (ds4f-clamps m) li) sg su one smid))
    (dk-q8mv st (lname li "ffn_down_shexp.weight") smid sout)))

(defun dk-ffn-tail (m st l li plan w sb sa)
  "Layer LI's routed experts (PLAN's, weights W) on its FFN input, summed
   with the shared expert, and its FFN's HC post: streams SB into SA.  The
   first device's experts' down rows first, then the second's (copied in
   when it is done), summed in the experts' order."
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (ff (ds4f-ff-exp m))
         (slots (dp-a-slots plan)) (k1 (length slots)) (k2 (length (dp-b-slots plan))) (k (+ k1 k2))
         (clamp (svref (ds4f-clamps m) li)) (sbytes (dk-slot-bytes st)) (gb (dk-gate-bytes st))
         (base (dbuf-ptr (dk-pool st)))
         (norm2 (dbuf-of st :norm2 (* 4 d)))
         (bgu (dbuf-of st :dgu (* 4 2 k ff))) (bm (dbuf-of st :dmid (* 4 k ff)))
         (bd (dbuf-of st :ddown (* 4 k d))) (moe (dbuf-of st :moe (* 4 d)))
         (perm (let ((p (make-list k)))
                 (loop for i across (dp-a-idx plan) for j from 0 do (setf (nth i p) j))
                 (loop for i across (dp-b-idx plan) for j from k1 do (setf (nth i p) j))
                 (dk-put st p :type :u32))))
    (when (plusp k1)
      (let ((ptrs (dk-put st (append (loop for s across slots collect (+ base (* s sbytes)))
                                     (loop for s across slots collect (+ base (* s sbytes) gb)))
                          :type :u64))
            (bw (dk-put st (loop for i across (dp-a-idx plan) collect (aref w i)) :type :f32)))
        (launch dev "d4_mv_iq2_xxs" (list (ceiling ff 8) 1 (* 2 k1)) 64
                (list d ff (second (dl-gate-exps l)) (list :i64 0) 0 ff ptrs norm2 bgu))
        (launch dev "d4_swiglu_w" (ceiling (* k1 ff) 256) 256 (list k1 ff clamp bgu (cons bgu (* 4 k1 ff)) bw bm))
        (launch dev "d4_mv_q2_k" (list (ceiling d 8) 1 k1) 64
                (list ff d (second (dl-down-exps l)) (list :i64 (* 2 gb)) ff d ptrs bm bd))))
    (when (plusp k2)
      (let ((tr (dk-tier st)))
        (cuda-stream-wait (cudev-stream dev) (dt-event tr))
        (cuda-copy-in bd (* 4 k1 d) (dt-host-out tr) (* 4 k2 d))))
    (launch dev "d4_moe_sum" (ceiling d 256) 256 (list d k bd (dbuf-of st :sout (* 4 d)) moe perm))
    (dk-hc-post m st moe sb (dbuf-of st :pcf (* 4 32)) sa)))

(defun dk-head (m st streams)
  "output_logits_one from STREAMS: the logits into :OUT."
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (n (ds4f-n-hc m)) (hcd (* n d)) (eps (ds4f-eps m))
         (flat (dbuf-of st :hflat (* 4 hcd))) (pre (dbuf-of st :hpre (* 4 16)))
         (embd (dbuf-of st :hembd (* 4 d))) (norm (dbuf-of st :hnorm (* 4 d)))
         (out (dbuf-of st :out (* 4 (ds4f-n-vocab m)))))
    (dk-rms dev streams nil flat hcd 1 eps)
    (dk-mv16 st "output_hc_fn.weight" flat pre)
    (launch dev "d4_head_mix" (ceiling d 256) 256
            (list d n (ds4f-hc-eps m) pre (dvec st (ds4f-out-hc-scale m)) (dvec st (ds4f-out-hc-base m)) streams embd))
    (dk-rms dev embd (dvec st (ds4f-output-norm m)) norm d 1 eps)
    (dk-q8mv st "output.weight" norm out)
    out))

;;; --- a token ---------------------------------------------------------------------------

(defun ds4f-cuda-forward (ctx token logits)
  "DS4F-FORWARD-1 on the GPU: TOKEN at the next position."
  (let* ((m (d4ctx-model ctx)) (st (ds4f-cuda-state m)) (dev (dk-dev st))
         (d (ds4f-d m)) (n (ds4f-n-hc m)) (hcd (* n d)) (pos (d4ctx-pos ctx)) (wf (ds4f-file m))
         (ne (ds4f-n-expert m)) (nl (ds4f-n-layer m)) (rb (dk-rb st))
         (sa (dbuf-of st :sa (* 4 hcd))) (sb (dbuf-of st :sb (* 4 hcd)))
         (lg (f32vec ne)) (pl (f32vec ne)) (pending nil) (prev nil))
    (declare (type u8vec wf))
    (with-cuda
      (cuda-current dev)
      ;; the token's embedding into each stream
      (let ((e (f32vec hcd)) (o (+ (ds4f-embd-off m) (* token d 2))))
        (dotimes (j d)
          (let ((v (f16-at wf (+ o (* 2 j)))))
            (dotimes (h n) (setf (aref e (+ (* h d) j)) v))))
        (cuda-upload-vector sa e))
      (dotimes (li nl)
        (let ((l (svref (ds4f-layers m) li)) (c (svref (d4ctx-caches ctx) li)))
          (dk-arena-reset st)
          (when pending (apply #'dk-ffn-tail m st (append pending (list sb sa))))
          (dk-attention m st l li c pos sa sb)
          (dk-ffn-head m st l li sb)
          ;; the routing, on the CPU
          (cuda-copy-out rb (dbuf-of st :logits 0) 0 (* 4 2 ne))
          (when (dk-tier st) (cuda-copy-out (sb-sys:sap+ rb 4096) (dbuf-of st :norm2 0) 0 (* 4 d)))
          (cuda-sync dev)
          (dotimes (i ne)
            (setf (aref lg i) (sb-sys:sap-ref-single rb (* 4 i))
                  (aref pl i) (sb-sys:sap-ref-single rb (* 4 (+ ne i)))))
          (multiple-value-bind (sel w) (d4-route m l li nil token lg)
            (when *ds4f-route-trace* (format *ds4f-route-trace* "~D~{ ~D~}~%" li (coerce sel 'list)))
            (let ((plan (d4-plan m st li sel prev)))
              (when (plusp (length (dp-b-slots plan))) (d4-tier-launch m st l plan w))
              (when (and (plusp *ds4f-prefetch-k*) (plusp *ds4f-prefetch-max*) (< (1+ li) nl))
                (d4-prefetch m st (1+ li) (d4-predict m (1+ li) token *ds4f-prefetch-k* pl)))
              (setf pending (list l li plan w) prev plan)))))
      (dk-arena-reset st)
      (apply #'dk-ffn-tail m st (append pending (list sb sa)))
      (when logits
        (let ((out (dk-head m st sa)) (v (d4ctx-logits ctx)))
          (sb-sys:with-pinned-objects (v)
            (cuda-download out (sb-sys:vector-sap v) (* 4 (ds4f-n-vocab m))))))
      (loop for s across (dp-a-slots prev) do (decf (aref (dk-pin st) s)))
      (cuda-sync dev))
    (incf (d4ctx-pos ctx))
    ctx))

(defun ds4f-cuda-stats (m)
  "The expert pool's counts so far, a string."
  (let ((st (getf (ds4f-gpu-plist m) :cuda)))
    (if st
        (let ((h2 (if (dk-tier st) (dt-hits (dk-tier st)) 0)))
          (format nil "experts: ~D hits, ~D on the second device, ~D misses (~,1F%), ~D prefetched, ~D copied"
                  (dk-hits st) h2 (dk-misses st)
                  (* 100 (/ (+ h2 (dk-hits st)) (max 1 (+ h2 (dk-hits st) (dk-misses st))))) (dk-prefetched st) (dk-copied st)))
        "no device state")))
