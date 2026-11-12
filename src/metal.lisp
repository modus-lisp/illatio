;;;; metal.lisp — the GPU, on macOS, from Lisp: Metal through the Objective-C
;;;; runtime.
;;;
;;; No Objective-C or C of ours: objc_msgSend is called through SBCL's alien
;;; interface with each message's own signature, and the kernels are Metal
;;; Shading Language source (src/metal/*.metal) compiled when the device is
;;; opened.  Two things about calling Apple's code from SBCL:
;;;   - its float code expects the IEEE exceptions masked, as C runs, so every
;;;     entry here masks SBCL's traps (WITH-METAL);
;;;   - a struct over 16 bytes (MTLSize is three NSUIntegers) is passed by
;;;     reference on arm64 (AAPCS64), so it goes as a pointer to its words.
;;; Objects made by convenience constructors are autoreleased, so each piece
;;; of work runs inside an autorelease pool.

(in-package #:illatio)

(defvar *metal-loaded* nil)

(defun load-metal-frameworks ()
  (unless *metal-loaded*
    (sb-alien:load-shared-object "/usr/lib/libobjc.A.dylib")
    (sb-alien:load-shared-object "/System/Library/Frameworks/Foundation.framework/Foundation")
    (sb-alien:load-shared-object "/System/Library/Frameworks/Metal.framework/Metal")
    (setf *metal-loaded* t)))

(sb-alien:define-alien-type ptr sb-sys:system-area-pointer)

;; at load, so the foreign functions below resolve; SBCL reopens these when a
;; saved core starts
(load-metal-frameworks)

(defmacro with-metal (&body body)
  `(sb-int:with-float-traps-masked (:overflow :invalid :inexact :divide-by-zero :underflow)
     ,@body))

(defun sel (name)
  (sb-alien:alien-funcall
   (sb-alien:extern-alien "sel_registerName" (function sb-sys:system-area-pointer sb-alien:c-string))
   name))

(defun objc-class (name)
  (sb-alien:alien-funcall
   (sb-alien:extern-alien "objc_getClass" (function sb-sys:system-area-pointer sb-alien:c-string))
   name))

(defmacro msg (ret obj selector &rest args)
  "[OBJ SELECTOR args...]: each arg is (alien-type value).  The selector is
   looked up every time (a hash lookup in libobjc): an address kept from one
   process would be wrong in a core image's next one."
  (let ((types (mapcar #'first args)))
    `(sb-alien:alien-funcall
      (sb-alien:extern-alien "objc_msgSend"
                             (function ,ret sb-sys:system-area-pointer sb-sys:system-area-pointer ,@types))
      ,obj (sel ,selector)
      ,@(mapcar #'second args))))

(defun nullp (p) (zerop (sb-sys:sap-int p)))

(defun autorelease-push ()
  (sb-alien:alien-funcall (sb-alien:extern-alien "objc_autoreleasePoolPush" (function sb-sys:system-area-pointer))))
(defun autorelease-pop (pool)
  (sb-alien:alien-funcall (sb-alien:extern-alien "objc_autoreleasePoolPop" (function sb-alien:void sb-sys:system-area-pointer)) pool))

(defmacro with-autorelease (&body body)
  (let ((p (gensym)))
    `(let ((,p (autorelease-push)))
       (unwind-protect (progn ,@body) (autorelease-pop ,p)))))

(defun ns-string (s)
  (msg ptr (objc-class "NSString") "stringWithUTF8String:" (sb-alien:c-string s)))

(defun ns-string-lisp (ns)
  (if (nullp ns) "" (msg sb-alien:c-string ns "UTF8String")))

(defun ns-error-text (err)
  (if (nullp err) "(no error object)"
      (ns-string-lisp (msg ptr err "localizedDescription"))))

;;; --- the device -----------------------------------------------------------------

(defstruct (gpu (:constructor %make-gpu))
  device queue library
  (pipelines (make-hash-table :test 'equal)))

(defun metal-source ()
  "The kernels' source: every .metal file in src/metal/, concatenated."
  (let ((dir (merge-pathnames "metal/" (asdf:system-relative-pathname "illatio" "src/"))))
    (with-output-to-string (o)
      (dolist (f (sort (directory (merge-pathnames "*.metal" dir)) #'string< :key #'namestring))
        (with-open-file (s f)
          (loop for line = (read-line s nil) while line do (write-line line o)))))))

(defun open-gpu (&optional (source (metal-source)))
  "The system's GPU with its command queue and SOURCE compiled."
  (load-metal-frameworks)
  (with-metal
    (with-autorelease
      (let ((dev (sb-alien:alien-funcall
                  (sb-alien:extern-alien "MTLCreateSystemDefaultDevice" (function sb-sys:system-area-pointer)))))
        (when (nullp dev) (error "no Metal device"))
        (let* ((queue (msg ptr dev "newCommandQueue"))
               (errp (make-array 1 :element-type '(unsigned-byte 64) :initial-element 0))
               (lib (sb-sys:with-pinned-objects (errp)
                      (msg ptr dev "newLibraryWithSource:options:error:"
                           (ptr (ns-string source)) (ptr (sb-sys:int-sap 0)) (ptr (sb-sys:vector-sap errp))))))
          (when (nullp lib)
            (error "Metal compile failed: ~A" (ns-error-text (sb-sys:int-sap (aref errp 0)))))
          (%make-gpu :device dev :queue queue :library lib))))))

(defun gpu-name (gpu)
  (with-metal (with-autorelease (ns-string-lisp (msg ptr (gpu-device gpu) "name")))))

(defun pipeline (gpu name)
  "The compute pipeline for kernel NAME, made the first time it is asked for."
  (or (gethash name (gpu-pipelines gpu))
      (setf (gethash name (gpu-pipelines gpu))
            (with-metal
              (with-autorelease
                (let ((fn (msg ptr (gpu-library gpu) "newFunctionWithName:" (ptr (ns-string name))))
                      (errp (make-array 1 :element-type '(unsigned-byte 64) :initial-element 0)))
                  (when (nullp fn) (error "no Metal kernel ~S" name))
                  (let ((pso (sb-sys:with-pinned-objects (errp)
                               (msg ptr (gpu-device gpu) "newComputePipelineStateWithFunction:error:"
                                    (ptr fn) (ptr (sb-sys:vector-sap errp))))))
                    (when (nullp pso)
                      (error "Metal pipeline ~A: ~A" name (ns-error-text (sb-sys:int-sap (aref errp 0)))))
                    pso)))))))

;;; --- buffers --------------------------------------------------------------------

(defstruct (gbuf (:constructor %make-gbuf))
  obj sap (bytes 0 :type (unsigned-byte 62))
  device                                ; the MTLDevice it was made on (parking makes it again there)
  wrapped                               ; T: memory not its own (GPU-WRAP's), which it never frees
  locked)                               ; T: its pages locked (GPU-MLOCK), to be locked again if remade

(defconstant +shared+ 0 "MTLResourceStorageModeShared: CPU and GPU see the same memory.")

(defvar *gpu-alloc-hook* nil
  "A function called with each buffer GPU-BUFFER or GPU-WRAP makes (a
   parking's record, WITH-PARKING-RECORD), or NIL.")

(defun new-buffer-object (device bytes)
  (with-metal
    (let ((obj (msg ptr device "newBufferWithLength:options:"
                    (sb-alien:unsigned-long (max bytes 16)) (sb-alien:unsigned-long +shared+))))
      (when (nullp obj) (error "Metal: no buffer of ~:D bytes" bytes))
      obj)))

(defun wrap-object (device sap bytes)
  (with-metal
    (let ((obj (msg ptr device "newBufferWithBytesNoCopy:length:options:deallocator:"
                    (ptr sap) (sb-alien:unsigned-long bytes) (sb-alien:unsigned-long +shared+)
                    (ptr (sb-sys:int-sap 0)))))
      (when (nullp obj) (error "Metal: could not wrap ~:D bytes" bytes))
      obj)))

(defun gpu-buffer (gpu bytes)
  "A new zeroed shared buffer of BYTES."
  (let* ((obj (new-buffer-object (gpu-device gpu) bytes))
         (b (%make-gbuf :obj obj :sap (with-metal (msg ptr obj "contents")) :bytes bytes :device (gpu-device gpu))))
    (when *gpu-alloc-hook* (funcall *gpu-alloc-hook* b))
    b))

(defun mlock-bytes (sap bytes)
  (zerop (sb-alien:alien-funcall
          (sb-alien:extern-alien "mlock" (function sb-alien:int sb-sys:system-area-pointer sb-alien:unsigned-long))
          sap bytes)))

(defun gpu-mlock (b)
  "Lock buffer B's pages in memory (mlock): the system neither compresses nor
   pages them out, so a command buffer using B doesn't wait on making it
   resident.  Whether it could."
  (setf (gbuf-locked b) (mlock-bytes (gbuf-sap b) (gbuf-bytes b))))

(defun gpu-free (b)
  "Give back buffer B (GPU-BUFFER's or GPU-WRAP's; a wrapped one's memory
   stays its owner's)."
  (when (gbuf-obj b)
    (with-metal (msg sb-alien:void (gbuf-obj b) "release"))
    (setf (gbuf-obj b) nil))
  nil)

(defun gpu-wrap (gpu sap bytes)
  "BYTES at SAP (page-aligned, BYTES a whole number of pages) as a buffer,
   without copying: unified memory, the GPU reads it where it is."
  (let ((b (%make-gbuf :obj (wrap-object (gpu-device gpu) sap bytes) :sap sap :bytes bytes
                       :device (gpu-device gpu) :wrapped t)))
    (when *gpu-alloc-hook* (funcall *gpu-alloc-hook* b))
    b))

;;; --- parking: one program's GPU memory given up for another's ----------------------
;;; cuda.lisp's parking on unified memory.  The memory is the same memory, but
;;; Metal's working set is not: a buffer counts against it until released,
;;; and a command buffer past it fails.  So a PARKING keeps the buffers its
;;; owner made (while WITH-PARKING-RECORD binds the hook) and can release
;;; them all (GPU-PARK) and make them again (GPU-UNPARK), in the same GBUFs,
;;; so whatever holds them holds them still.  A wrapped buffer (a mapped
;;; file's) is just released and wrapped again; one of its own is copied out
;;; to host memory first and back after, every time: kept, the copy would be
;;; the same RAM held twice (CUDA keeps a CONST one's, in host memory to
;;; spare).  The owner must not run while parked.

(defstruct (parking (:constructor make-parking (&optional name)))
  name
  (bufs '())                                  ; every GBUF recorded, newest first
  (const (make-hash-table :test 'eq))         ; GBUF -> T: its contents never change
  (host (make-hash-table :test 'eq))          ; GBUF -> its host copy (a SAP, malloc's)
  (saved (make-hash-table :test 'eq))         ; GBUF -> T: the host copy is current
  (parked '())                                ; the GBUFs released, to be made again
  (bytes 0))

(defun parking-recorder (parking const)
  "A *GPU-ALLOC-HOOK* recording new buffers in PARKING (as CONST: weights).
   What their owner frees is dropped from the record now and then."
  (let ((n 0))
    (lambda (b)
      (when (> (incf n) 4096)
        (setf n 0
              (parking-bufs parking)
              (delete-if (lambda (x) (and (null (gbuf-obj x)) (not (member x (parking-parked parking)))))
                         (parking-bufs parking))))
      (push b (parking-bufs parking))
      (when const (setf (gethash b (parking-const parking)) t)))))

(defmacro with-parking-record ((parking &key const) &body body)
  "BODY's GPU buffers recorded in PARKING (as CONST: weights), and in no
   other: an outer record (a global one) is shadowed."
  `(let ((*gpu-alloc-hook* (parking-recorder ,parking ,const))) ,@body))

(defun host-alloc (bytes)
  (let ((p (sb-alien:alien-funcall (sb-alien:extern-alien "malloc" (function sb-sys:system-area-pointer sb-alien:unsigned-long))
                                   bytes)))
    (when (nullp p) (error "no host memory for ~:D bytes" bytes))
    p))

(defun host-free (sap)
  (sb-alien:alien-funcall (sb-alien:extern-alien "free" (function sb-alien:void sb-sys:system-area-pointer)) sap))

(defun copy-bytes-sap (dst src bytes)
  (sb-alien:alien-funcall
   (sb-alien:extern-alien "memcpy" (function sb-sys:system-area-pointer sb-sys:system-area-pointer
                                             sb-sys:system-area-pointer sb-alien:unsigned-long))
   dst src bytes))

(defun gpu-park (parking)
  "PARKING's buffers released (their contents, but a wrapped one's, copied to
   host memory first).  Buffers their owner freed are forgotten.  Returns
   the bytes given up.  A buffer is in PARKING-PARKED as soon as it is
   released, so a failure part way leaves what was released to come back."
  (let ((live '()) (out (make-hash-table :test 'eq)))
    (dolist (b (parking-parked parking)) (setf (gethash b out) t))
    (dolist (b (parking-bufs parking))
      (cond ((gbuf-obj b) (push b live))
            ((gethash b out))
            (t (let ((h (gethash b (parking-host parking))))    ; freed by its owner
                 (when h (host-free h) (remhash b (parking-host parking)))
                 (remhash b (parking-const parking)) (remhash b (parking-saved parking))))))
    (setf live (nreverse live)
          (parking-bufs parking) (append live (parking-parked parking))
          (parking-bytes parking) (reduce #'+ (parking-parked parking) :key #'gbuf-bytes))
    (dolist (b live)
      (unless (or (gbuf-wrapped b)
                  (and (gethash b (parking-const parking)) (gethash b (parking-saved parking))))
        (let ((h (or (gethash b (parking-host parking))
                     (setf (gethash b (parking-host parking)) (host-alloc (gbuf-bytes b))))))
          (copy-bytes-sap h (gbuf-sap b) (gbuf-bytes b))
          (setf (gethash b (parking-saved parking)) t)))
      (gpu-free b)
      (unless (gbuf-wrapped b) (setf (gbuf-sap b) nil))
      (push b (parking-parked parking))
      (incf (parking-bytes parking) (gbuf-bytes b)))
    (parking-bytes parking)))

(defun prefetch-pages (ranges &key (threads 8))
  "The pages of RANGES, ((sap . bytes) ...), read into memory now: a mapped
   file's, which the system may have dropped while its buffer was parked,
   would otherwise come back a fault at a time under the GPU (after a
   DeepSeek model's turn, 20 GB of a picture's were).  The kernel is asked to read
   the ranges ahead, and THREADS threads touch a page each in turn."
  (let ((page 16384) (total 0))
    (dolist (r ranges)
      (sb-alien:alien-funcall
       (sb-alien:extern-alien "madvise" (function sb-alien:int sb-sys:system-area-pointer sb-alien:unsigned-long sb-alien:int))
       (car r) (cdr r) 3)                                                     ; MADV_WILLNEED
      (incf total (cdr r)))
    (when (plusp total)
      (let ((workers
              (loop for k below threads
                    collect (let ((k k))
                              (sb-thread:make-thread
                               (lambda ()
                                 (let ((sum 0))
                                   (dolist (r ranges sum)
                                     (loop for off from (* k page) below (cdr r) by (* threads page)
                                           do (incf sum (sb-sys:sap-ref-8 (car r) off))))))
                               :name "illatio page prefetch")))))
        (mapc #'sb-thread:join-thread workers)))
    total))

(defun gpu-unpark (parking)
  "PARKING's buffers made again, as they were (a wrapped one's pages read in
   first, PREFETCH-PAGES)."
  (let ((done '()))
    (handler-bind ((error (lambda (e) (declare (ignore e))     ; out of memory: leave it all parked
                            (dolist (b done) (gpu-free b) (unless (gbuf-wrapped b) (setf (gbuf-sap b) nil))))))
      (dolist (b (parking-parked parking))
        (if (gbuf-wrapped b)
            (setf (gbuf-obj b) (wrap-object (gbuf-device b) (gbuf-sap b) (gbuf-bytes b)))
            (let ((obj (new-buffer-object (gbuf-device b) (gbuf-bytes b))))
              (setf (gbuf-obj b) obj (gbuf-sap b) (with-metal (msg ptr obj "contents")))
              (copy-bytes-sap (gbuf-sap b) (gethash b (parking-host parking)) (gbuf-bytes b))
              ;; the copy not kept, a CONST one's either: the same RAM as the buffer, twice
              (host-free (gethash b (parking-host parking)))
              (remhash b (parking-host parking)) (remhash b (parking-saved parking))))
        (when (gbuf-locked b) (mlock-bytes (gbuf-sap b) (gbuf-bytes b)))
        (push b done))
      (prefetch-pages (loop for b in (parking-parked parking)
                            when (gbuf-wrapped b) collect (cons (gbuf-sap b) (gbuf-bytes b))))))
  (setf (parking-parked parking) '())
  parking)

(defun gbuf-floats (b n &optional (start 0))
  "Floats START..START+N-1 of buffer B, as a vector."
  (let ((v (make-array n :element-type 'single-float)))
    (dotimes (i n v) (setf (aref v i) (sb-sys:sap-ref-single (gbuf-sap b) (* 4 (+ start i)))))))

(defun gbuf-set-floats (b v &optional (start 0))
  (declare (type (simple-array single-float (*)) v))
  (dotimes (i (length v) b) (setf (sb-sys:sap-ref-single (gbuf-sap b) (* 4 (+ start i))) (aref v i))))

;;; --- encoding work --------------------------------------------------------------

;;; A command buffer's one encoder dispatches concurrently, with a barrier
;;; before each dispatch -- as a serial encoder runs, at the same cost -- but
;;; for the ones in a TOGETHER, which wait for what came before and run side
;;; by side: a dependent chain of small dispatches costs some microseconds a
;;; step however little each does, so the independent ones (projections of
;;; the same input) are one step, not several.  (ds4's concurrent FFN.)

(defstruct (batch-enc (:conc-name enc-) (:constructor %make-enc))
  cb enc
  (group nil)             ; NIL: a barrier before each dispatch; :FIRST, :IN: in a TOGETHER
  (started nil)           ; whether anything is dispatched yet (no barrier before the first)
  (gate nil)              ; GATE-WORK's (src . dst) grid tables, or NIL
  (ngated 0)              ; dispatches through them so far
  (args (make-array 64 :element-type '(unsigned-byte 32)))
  (sizes (make-array 6 :element-type '(unsigned-byte 64))))

(defun begin-work (gpu)
  (let* ((cb (msg ptr (gpu-queue gpu) "commandBuffer"))
         (enc (msg ptr cb "computeCommandEncoderWithDispatchType:" (sb-alien:unsigned-long 1)))) ; concurrent
    (%make-enc :cb cb :enc enc)))

(defmacro together ((e) &body body)
  "BODY's dispatches into E, independent of each other: after what came
   before them, side by side, and before what comes after."
  (let ((was (gensym)))
    `(let ((,was (enc-group ,e)))
       (unless ,was (setf (enc-group ,e) :first))
       (multiple-value-prog1 (progn ,@body)
         (setf (enc-group ,e) ,was)))))

;;; A gated command buffer runs only if a flag the GPU wrote before it says
;;; so: its dispatches are indirect, their grids copied from a table the CPU
;;; fills (SRC) into the one they read (DST) by a first dispatch, d4_gate,
;;; as they are or as zeros -- so work can be queued before it is known to
;;; be wanted, and cost nothing when it is not.

(defconstant +gate-max+ 256 "Dispatches a gated command buffer may hold.")

(defun gate-work (gpu e ok src dst)
  "Gate E on the u32 at OK ((buffer . offset)): its dispatches from here run
   if it is not zero.  SRC and DST, buffers of +GATE-MAX+ grids (12 bytes
   each), are E's own while it runs."
  (dispatch gpu e "d4_gate" (list (* 3 +gate-max+)) (list ok src dst) (ceiling (* 3 +gate-max+) 256) 256)
  (setf (enc-gate e) (cons src dst) (enc-ngated e) 0))

(defun dispatch-barrier (e)
  "Before a dispatch into E: wait for the ones before it, unless it joins a TOGETHER."
  (case (enc-group e)
    ((nil) (when (enc-started e) (msg sb-alien:void (enc-enc e) "memoryBarrierWithScope:" (sb-alien:unsigned-long 1))))
    (:first (when (enc-started e) (msg sb-alien:void (enc-enc e) "memoryBarrierWithScope:" (sb-alien:unsigned-long 1)))
     (setf (enc-group e) :in)))
  (setf (enc-started e) t))

(defun commit-work (e)
  "Finish encoding and start it running."
  (msg sb-alien:void (enc-enc e) "endEncoding")
  (msg sb-alien:void (enc-cb e) "commit"))

(defun wait-work (e)
  "Wait for E, committed, to finish."
  (msg sb-alien:void (enc-cb e) "waitUntilCompleted")
  (let ((err (msg ptr (enc-cb e) "error")))
    (unless (nullp err) (error "Metal: ~A" (ns-error-text err)))))

(defun end-work (e)
  "Finish encoding, run it, and wait."
  (commit-work e)
  (wait-work e))

(defun arg-word (x)
  (etypecase x
    (single-float (ldb (byte 32 0) (sb-kernel:single-float-bits x)))
    (integer (ldb (byte 32 0) x))))

(defun dispatch (gpu e kernel args buffers groups threads &key (shared 0))
  "Run KERNEL over GROUPS threadgroups of THREADS (each an integer or a list
   of up to three).  ARGS, 32-bit words (integers or floats), go in buffer 0;
   BUFFERS are (gbuf . byte-offset) or gbufs, from buffer 1.  SHARED bytes of
   threadgroup memory at index 0."
  (dispatch-barrier e)
  (let ((enc (enc-enc e)) (a (enc-args e)) (sz (enc-sizes e)))
    (msg sb-alien:void enc "setComputePipelineState:" (ptr (pipeline gpu kernel)))
    (loop for x in args for i from 0 do (setf (aref a i) (arg-word x)))
    (sb-sys:with-pinned-objects (a sz)
      (msg sb-alien:void enc "setBytes:length:atIndex:"
           (ptr (sb-sys:vector-sap a)) (sb-alien:unsigned-long (* 4 (max 1 (length args))))
           (sb-alien:unsigned-long 0))
      (loop for b in buffers for i from 1
            do (multiple-value-bind (buf off) (if (consp b) (values (car b) (cdr b)) (values b 0))
                 (msg sb-alien:void enc "setBuffer:offset:atIndex:"
                      (ptr (gbuf-obj buf)) (sb-alien:unsigned-long off) (sb-alien:unsigned-long i))))
      (when (plusp shared)
        (msg sb-alien:void enc "setThreadgroupMemoryLength:atIndex:"
             (sb-alien:unsigned-long shared) (sb-alien:unsigned-long 0)))
      (flet ((size3 (x o)
               (let ((l (if (listp x) x (list x))))
                 (dotimes (k 3) (setf (aref sz (+ o k)) (or (nth k l) 1))))))
        (size3 groups 0) (size3 threads 3))
      (if (enc-gate e)
          (let ((i (enc-ngated e)) (src (gbuf-sap (car (enc-gate e)))))
            (assert (< i +gate-max+) () "a gated command buffer holds at most ~D dispatches" +gate-max+)
            (dotimes (k 3) (setf (sb-sys:sap-ref-32 src (+ (* 12 i) (* 4 k))) (aref sz k)))
            (setf (enc-ngated e) (1+ i))
            (msg sb-alien:void enc "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:"
                 (ptr (gbuf-obj (cdr (enc-gate e)))) (sb-alien:unsigned-long (* 12 i))
                 (ptr (sb-sys:sap+ (sb-sys:vector-sap sz) 24))))
          (msg sb-alien:void enc "dispatchThreadgroups:threadsPerThreadgroup:"
               (ptr (sb-sys:vector-sap sz)) (ptr (sb-sys:sap+ (sb-sys:vector-sap sz) 24)))))
    (when *gpu-profile* (profile-dispatch gpu e kernel))))

(defvar *gpu-profile* nil
  "When a hash table, each dispatch runs as its own command buffer and its
   GPU time (seconds) is added under its kernel's name.")

(defun profile-dispatch (gpu e kernel)
  "Close E's command buffer, run it, charge its GPU time to KERNEL, and open
   a fresh one in E."
  (end-work e)
  (let ((t0 (msg double-float (enc-cb e) "GPUStartTime"))
        (t1 (msg double-float (enc-cb e) "GPUEndTime")))
    (incf (gethash kernel *gpu-profile* 0d0) (- t1 t0)))
  (let ((fresh (begin-work gpu)))
    (setf (enc-cb e) (enc-cb fresh) (enc-enc e) (enc-enc fresh) (enc-started e) nil)))

(defmacro with-gpu-work ((e gpu) &body body)
  "Encode BODY's dispatches into one command buffer, then run it and wait."
  `(with-metal
     (with-autorelease
       (let ((,e (begin-work ,gpu)))
         ,@body
         (end-work ,e)))))

(defmacro with-gpu-work-while ((e gpu) encode &body meanwhile)
  "Encode ENCODE's dispatches into one command buffer and start it; run
   MEANWHILE on the CPU while it runs, then wait for it.  MEANWHILE's values."
  `(with-metal
     (with-autorelease
       (let ((,e (begin-work ,gpu)))
         ,encode
         (commit-work ,e)
         (multiple-value-prog1 (progn ,@meanwhile) (wait-work ,e))))))
