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

(defstruct (gbuf (:constructor %make-gbuf)) obj sap (bytes 0 :type (unsigned-byte 62)))

(defconstant +shared+ 0 "MTLResourceStorageModeShared: CPU and GPU see the same memory.")

(defun gpu-buffer (gpu bytes)
  "A new zeroed shared buffer of BYTES."
  (with-metal
    (let ((obj (msg ptr (gpu-device gpu) "newBufferWithLength:options:"
                    (sb-alien:unsigned-long (max bytes 16)) (sb-alien:unsigned-long +shared+))))
      (when (nullp obj) (error "Metal: no buffer of ~:D bytes" bytes))
      (let ((sap (msg ptr obj "contents")))
        (%make-gbuf :obj obj :sap sap :bytes bytes)))))

(defun gpu-mlock (b)
  "Lock buffer B's pages in memory (mlock): the system neither compresses nor
   pages them out, so a command buffer using B doesn't wait on making it
   resident.  Whether it could."
  (zerop (sb-alien:alien-funcall
          (sb-alien:extern-alien "mlock" (function sb-alien:int sb-sys:system-area-pointer sb-alien:unsigned-long))
          (gbuf-sap b) (gbuf-bytes b))))

(defun gpu-free (b)
  "Give back buffer B (GPU-BUFFER's or GPU-WRAP's; a wrapped one's memory
   stays its owner's)."
  (with-metal (msg sb-alien:void (gbuf-obj b) "release"))
  nil)

(defun gpu-wrap (gpu sap bytes)
  "BYTES at SAP (page-aligned, BYTES a whole number of pages) as a buffer,
   without copying: unified memory, the GPU reads it where it is."
  (with-metal
    (let ((obj (msg ptr (gpu-device gpu) "newBufferWithBytesNoCopy:length:options:deallocator:"
                    (ptr sap) (sb-alien:unsigned-long bytes) (sb-alien:unsigned-long +shared+)
                    (ptr (sb-sys:int-sap 0)))))
      (when (nullp obj) (error "Metal: could not wrap ~:D bytes" bytes))
      (%make-gbuf :obj obj :sap sap :bytes bytes))))

(defun gbuf-floats (b n &optional (start 0))
  "Floats START..START+N-1 of buffer B, as a vector."
  (let ((v (make-array n :element-type 'single-float)))
    (dotimes (i n v) (setf (aref v i) (sb-sys:sap-ref-single (gbuf-sap b) (* 4 (+ start i)))))))

(defun gbuf-set-floats (b v &optional (start 0))
  (declare (type (simple-array single-float (*)) v))
  (dotimes (i (length v) b) (setf (sb-sys:sap-ref-single (gbuf-sap b) (* 4 (+ start i))) (aref v i))))

;;; --- encoding work --------------------------------------------------------------

(defstruct (batch-enc (:conc-name enc-) (:constructor %make-enc))
  cb enc
  (args (make-array 64 :element-type '(unsigned-byte 32)))
  (sizes (make-array 6 :element-type '(unsigned-byte 64))))

(defun begin-work (gpu)
  (let* ((cb (msg ptr (gpu-queue gpu) "commandBuffer"))
         (enc (msg ptr cb "computeCommandEncoder")))
    (%make-enc :cb cb :enc enc)))

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
      (msg sb-alien:void enc "dispatchThreadgroups:threadsPerThreadgroup:"
           (ptr (sb-sys:vector-sap sz)) (ptr (sb-sys:sap+ (sb-sys:vector-sap sz) 24))))
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
    (setf (enc-cb e) (enc-cb fresh) (enc-enc e) (enc-enc fresh))))

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
