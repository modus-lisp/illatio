;;;; cuda.lisp — NVIDIA GPUs, on Linux, from Lisp: the CUDA driver API and
;;;; NVRTC through SBCL's alien interface.
;;;
;;; No C of ours: libcuda's entry points are called with their own
;;; signatures, and the kernels are CUDA C source (src/cuda/*.cu) that NVRTC
;;; compiles for each device's architecture when it opens -- as metal.lisp
;;; compiles src/metal/ -- the cubin cached under ~/.cache/illatio/cuda.
;;; The libraries are loaded when a device is first opened, not when this
;;; file is, so an image built here runs where there is no CUDA at all.
;;; Like Metal, the driver expects C's floating-point environment, so every
;;; entry masks SBCL's traps (WITH-CUDA).

(in-package #:illatio)

(defvar *cuda-loaded* nil)

(defparameter *nvrtc-candidates*
  '("/usr/local/cuda/lib64/libnvrtc.so.12" "/usr/local/cuda/lib64/libnvrtc.so"
    "libnvrtc.so.12" "libnvrtc.so" "libnvrtc.so.11.2")
  "Where NVRTC may be, in order: the newest toolkit's first.")

(defun load-cuda-libraries ()
  (unless *cuda-loaded*
    (sb-alien:load-shared-object "libcuda.so.1" :dont-save t)
    (or (loop for p in *nvrtc-candidates*
              thereis (ignore-errors (sb-alien:load-shared-object p :dont-save t) p))
        (error "CUDA: no NVRTC library (looked for ~{~A~^, ~})" *nvrtc-candidates*))
    (setf *cuda-loaded* t)))

(defmacro with-cuda (&body body)
  `(sb-int:with-float-traps-masked (:overflow :invalid :inexact :divide-by-zero :underflow)
     ,@body))

(defun cuda-error-string (code)
  (sb-alien:with-alien ((s sb-alien:c-string))
    (sb-alien:alien-funcall
     (sb-alien:extern-alien "cuGetErrorString" (function sb-alien:int sb-alien:int (* sb-alien:c-string)))
     code (sb-alien:addr s))
    (or s (format nil "error ~D" code))))

(defmacro cu (name &rest args)
  "Call the driver's NAME with ARGS, each (alien-type value); signal unless
   it returns CUDA_SUCCESS."
  (let ((r (gensym)))
    `(let ((,r (sb-alien:alien-funcall
                (sb-alien:extern-alien ,name (function sb-alien:int ,@(mapcar #'first args)))
                ,@(mapcar #'second args))))
       (unless (zerop ,r) (error "CUDA: ~A: ~A" ,name (cuda-error-string ,r)))
       ,r)))

(defmacro with-out ((&rest vars) &body body)
  "Bind each VAR to a fresh 64-bit out-parameter cell (an alien
   unsigned-long); BODY passes (addr VAR)."
  `(sb-alien:with-alien ,(mapcar (lambda (v) `(,v sb-alien:unsigned-long 0)) vars)
     ,@body))

(defmacro out-ptr (var) `(sb-alien:alien-sap (sb-alien:addr ,var)))

;;; --- devices ------------------------------------------------------------------

(defstruct (cudev (:constructor %make-cudev))
  (index 0 :type fixnum)
  name ctx stream module arch
  (total-memory 0)
  (max-shared 49152)                    ; bytes of shared memory a block may opt in to
  (functions (make-hash-table :test 'equal))
  (shared-set (make-hash-table :test 'equal)) ; kernel -> the dynamic shared size allowed it
  ;; cuLaunchKernel's argument block: 64 values of 8 bytes, then 64 pointers to them
  (params (sb-sys:int-sap 0)))

(defvar *cuda-devices* (make-hash-table) "Index -> open CUDEV.")

(defun cuda-device-count ()
  (load-cuda-libraries)
  (with-cuda
    (cu "cuInit" (sb-alien:unsigned 0))
    (with-out (n)
      (cu "cuDeviceGetCount" (sb-sys:system-area-pointer (out-ptr n)))
      (ldb (byte 32 0) n))))

(defun cuda-source ()
  "The kernels' source: every .cu file in src/cuda/, concatenated."
  (let ((dir (merge-pathnames "cuda/" (asdf:system-relative-pathname "illatio" "src/"))))
    (with-output-to-string (o)
      (dolist (f (sort (directory (merge-pathnames "*.cu" dir)) #'string< :key #'namestring))
        (with-open-file (s f)
          (loop for line = (read-line s nil) while line do (write-line line o)))))))

(defun cuda-current (dev)
  "Make DEV's context the calling thread's (a context is per thread)."
  (cu "cuCtxSetCurrent" (sb-sys:system-area-pointer (cudev-ctx dev))))

(defun open-cuda (&optional (index 0))
  "Device INDEX, opened once: its primary context, a stream, and the
   kernels compiled for it."
  (or (gethash index *cuda-devices*)
      (setf (gethash index *cuda-devices*) (open-cuda-1 index))))

(defun open-cuda-1 (index)
  (load-cuda-libraries)
  (with-cuda
    (cu "cuInit" (sb-alien:unsigned 0))
    (with-out (dev ctx stream mem)
      (cu "cuDeviceGet" (sb-sys:system-area-pointer (out-ptr dev)) (sb-alien:int index))
      (let* ((d (ldb (byte 32 0) dev))
             (name (let ((buf (make-array 256 :element-type '(unsigned-byte 8) :initial-element 0)))
                     (sb-sys:with-pinned-objects (buf)
                       (cu "cuDeviceGetName" (sb-sys:system-area-pointer (sb-sys:vector-sap buf))
                           (sb-alien:int 256) (sb-alien:int d)))
                     (sb-ext:octets-to-string (subseq buf 0 (position 0 buf)))))
             (arch (flet ((attr (a)
                            (sb-alien:with-alien ((v sb-alien:int 0))
                              (cu "cuDeviceGetAttribute" ((* sb-alien:int) (sb-alien:addr v))
                                  (sb-alien:int a) (sb-alien:int d))
                              v)))
                     (+ (* 10 (attr 75)) (attr 76)))) ; compute capability major, minor
             (max-shared (sb-alien:with-alien ((v sb-alien:int 0))
                           (cu "cuDeviceGetAttribute" ((* sb-alien:int) (sb-alien:addr v))
                               (sb-alien:int 97) (sb-alien:int d)) ; ..._PER_BLOCK_OPTIN
                           v)))
        (cu "cuDevicePrimaryCtxRetain" (sb-sys:system-area-pointer (out-ptr ctx)) (sb-alien:int d))
        (cu "cuCtxSetCurrent" (sb-sys:system-area-pointer (sb-sys:int-sap ctx)))
        (cu "cuStreamCreate" (sb-sys:system-area-pointer (out-ptr stream)) (sb-alien:unsigned 0))
        (cu "cuDeviceTotalMem_v2" (sb-sys:system-area-pointer (out-ptr mem)) (sb-alien:int d))
        (let ((cubin (cuda-cubin (cuda-source) arch)))
          (with-out (module)
            (sb-sys:with-pinned-objects (cubin)
              (cu "cuModuleLoadData" (sb-sys:system-area-pointer (out-ptr module))
                  (sb-sys:system-area-pointer (sb-sys:vector-sap cubin))))
            (%make-cudev :index index :name name :arch arch
                         :ctx (sb-sys:int-sap ctx) :stream (sb-sys:int-sap stream)
                         :module (sb-sys:int-sap module) :total-memory mem :max-shared max-shared
                         :params (sb-sys:int-sap (%mmap-anon (* 2 64 8))))))))))

(defun cuda-function (dev name)
  (or (gethash name (cudev-functions dev))
      (setf (gethash name (cudev-functions dev))
            (with-out (f)
              (cu "cuModuleGetFunction" (sb-sys:system-area-pointer (out-ptr f))
                  (sb-sys:system-area-pointer (cudev-module dev)) (sb-alien:c-string name))
              (sb-sys:int-sap f)))))

;;; --- compiling ----------------------------------------------------------------

(defmacro nvrtc (name &rest args)
  (let ((r (gensym)))
    `(let ((,r (sb-alien:alien-funcall
                (sb-alien:extern-alien ,name (function sb-alien:int ,@(mapcar #'first args)))
                ,@(mapcar #'second args))))
       (unless (zerop ,r)
         (error "NVRTC: ~A: ~A" ,name
                (sb-alien:alien-funcall
                 (sb-alien:extern-alien "nvrtcGetErrorString" (function sb-alien:c-string sb-alien:int)) ,r)))
       ,r)))

(defun nvrtc-version ()
  (sb-alien:with-alien ((ma sb-alien:int 0) (mi sb-alien:int 0))
    (nvrtc "nvrtcVersion" ((* sb-alien:int) (sb-alien:addr ma)) ((* sb-alien:int) (sb-alien:addr mi)))
    (format nil "~D.~D" ma mi)))

(defun cuda-cache-path (source arch)
  (merge-pathnames (format nil "illatio/cuda/~(~36R~)-sm~D-nvrtc~A.cubin"
                           (sxhash source) arch (nvrtc-version))
                   (uiop:xdg-cache-home)))

(defun cuda-cubin (source arch)
  "SOURCE compiled for sm_ARCH: from the cache when it has been before."
  (let ((path (cuda-cache-path source arch)))
    (if (probe-file path)
        (with-open-file (s path :element-type '(unsigned-byte 8))
          (let ((v (make-array (file-length s) :element-type '(unsigned-byte 8))))
            (read-sequence v s) v))
        (let ((cubin (nvrtc-compile source arch)))
          (ensure-directories-exist path)
          (with-open-file (s path :element-type '(unsigned-byte 8) :direction :output :if-exists :supersede)
            (write-sequence cubin s))
          cubin))))

(defun nvrtc-compile (source arch)
  (with-cuda
    (with-out (prog)
      (nvrtc "nvrtcCreateProgram" (sb-sys:system-area-pointer (out-ptr prog))
             (sb-alien:c-string source) (sb-alien:c-string "illatio.cu")
             (sb-alien:int 0) (sb-sys:system-area-pointer (sb-sys:int-sap 0))
             (sb-sys:system-area-pointer (sb-sys:int-sap 0)))
      (let* ((p (sb-sys:int-sap prog))
             (opts (list (format nil "--gpu-architecture=sm_~D" arch) "--std=c++17" "-lineinfo"))
             (cstrs (mapcar (lambda (o) (sb-alien:make-alien-string o)) opts))
             (arr (sb-alien:make-alien sb-sys:system-area-pointer (length opts))))
        (unwind-protect
             (progn
               (loop for c in cstrs for i from 0
                     do (setf (sb-alien:deref arr i) (sb-alien:alien-sap c)))
               (let ((r (sb-alien:alien-funcall
                         (sb-alien:extern-alien "nvrtcCompileProgram"
                                                (function sb-alien:int sb-sys:system-area-pointer sb-alien:int
                                                          sb-sys:system-area-pointer))
                         p (length opts) (sb-alien:alien-sap arr))))
                 (unless (zerop r)
                   (error "NVRTC: the kernels did not compile:~%~A" (nvrtc-log p))))
               (with-out (n)
                 (nvrtc "nvrtcGetCUBINSize" (sb-sys:system-area-pointer p) (sb-sys:system-area-pointer (out-ptr n)))
                 (let ((v (make-array n :element-type '(unsigned-byte 8))))
                   (sb-sys:with-pinned-objects (v)
                     (nvrtc "nvrtcGetCUBIN" (sb-sys:system-area-pointer p)
                            (sb-sys:system-area-pointer (sb-sys:vector-sap v))))
                   v)))
          (mapc #'sb-alien:free-alien cstrs)
          (sb-alien:free-alien arr)
          (sb-alien:with-alien ((pp sb-sys:system-area-pointer p))
            (nvrtc "nvrtcDestroyProgram" ((* sb-sys:system-area-pointer) (sb-alien:addr pp)))))))))

(defun nvrtc-log (p)
  (with-out (n)
    (nvrtc "nvrtcGetProgramLogSize" (sb-sys:system-area-pointer p) (sb-sys:system-area-pointer (out-ptr n)))
    (let ((v (make-array n :element-type '(unsigned-byte 8))))
      (sb-sys:with-pinned-objects (v)
        (nvrtc "nvrtcGetProgramLog" (sb-sys:system-area-pointer p) (sb-sys:system-area-pointer (sb-sys:vector-sap v))))
      (sb-ext:octets-to-string (subseq v 0 (max 0 (1- n)))))))

;;; --- memory -------------------------------------------------------------------

(defstruct (dbuf (:constructor %make-dbuf)) (ptr 0 :type (unsigned-byte 64)) (bytes 0 :type (unsigned-byte 62)) dev)

(defun cuda-alloc (dev bytes &key (zero t))
  "BYTES of DEV's memory (zeroed unless ZERO is NIL)."
  (with-cuda
    (cuda-current dev)
    (with-out (p)
      (cu "cuMemAlloc_v2" (sb-sys:system-area-pointer (out-ptr p)) (sb-alien:unsigned-long (max bytes 256)))
      (let ((b (%make-dbuf :ptr p :bytes bytes :dev dev)))
        (when zero (cuda-zero b))
        b))))

(defun cuda-free (b)
  (when (and b (plusp (dbuf-ptr b)))
    (with-cuda
      (cuda-current (dbuf-dev b))
      (cu "cuMemFree_v2" (sb-alien:unsigned-long (dbuf-ptr b))))
    (setf (dbuf-ptr b) 0))
  nil)

(defun cuda-zero (b)
  (with-cuda
    (cu "cuMemsetD8Async" (sb-alien:unsigned-long (dbuf-ptr b)) (sb-alien:unsigned-char 0)
        (sb-alien:unsigned-long (dbuf-bytes b)) (sb-sys:system-area-pointer (cudev-stream (dbuf-dev b)))))
  b)

(defun cuda-mem-info (dev)
  "(VALUES free total) bytes on DEV."
  (with-cuda
    (cuda-current dev)
    (with-out (f tot)
      (cu "cuMemGetInfo_v2" (sb-sys:system-area-pointer (out-ptr f)) (sb-sys:system-area-pointer (out-ptr tot)))
      (values f tot))))

(defun cuda-upload (b sap bytes &optional (offset 0))
  "BYTES at host address SAP into B at OFFSET, in DEV's stream order; returns
   when the copy has been made (the host memory may be reused)."
  (with-cuda
    (let ((dev (dbuf-dev b)))
      (cu "cuMemcpyHtoDAsync_v2" (sb-alien:unsigned-long (+ (dbuf-ptr b) offset))
          (sb-sys:system-area-pointer sap) (sb-alien:unsigned-long bytes)
          (sb-sys:system-area-pointer (cudev-stream dev)))
      (cuda-sync dev))))

(defun cuda-upload-vector (b v &optional (offset 0))
  "The specialized vector V's bytes into B at OFFSET."
  (let ((bytes (* (length v) (etypecase v
                               ((simple-array single-float (*)) 4) ((simple-array (signed-byte 32) (*)) 4)
                               ((simple-array (unsigned-byte 8) (*)) 1) ((simple-array (signed-byte 8) (*)) 1)))))
    (sb-sys:with-pinned-objects (v)
      (cuda-upload b (sb-sys:vector-sap v) bytes offset))
    b))

(defun cuda-download (b sap bytes &optional (offset 0))
  "BYTES of B at OFFSET to host address SAP, after the work queued before it."
  (with-cuda
    (let ((dev (dbuf-dev b)))
      (cu "cuMemcpyDtoHAsync_v2" (sb-sys:system-area-pointer sap)
          (sb-alien:unsigned-long (+ (dbuf-ptr b) offset)) (sb-alien:unsigned-long bytes)
          (sb-sys:system-area-pointer (cudev-stream dev)))
      (cuda-sync dev))))

(defun cuda-floats (b n &optional (start 0))
  "Floats START..START+N-1 of B, as a vector."
  (let ((v (make-array n :element-type 'single-float)))
    (sb-sys:with-pinned-objects (v)
      (cuda-download b (sb-sys:vector-sap v) (* 4 n) (* 4 start)))
    v))

(defun cuda-sync (dev)
  (with-cuda (cu "cuStreamSynchronize" (sb-sys:system-area-pointer (cudev-stream dev)))))

;;; --- launching ----------------------------------------------------------------

(defun launch (dev kernel grid block args &key (shared 0))
  "Queue KERNEL over GRID blocks of BLOCK threads (each an integer or a list
   of up to three) on DEV's stream.  ARGS are, in the kernel's order: DBUFs
   or (DBUF . byte-offset) for pointers, NIL for a null pointer, integers
   for ints, single-floats for floats, and (:I64 n) for a long long."
  (let ((pv (cudev-params dev)))
    (loop for a in args for i from 0
          for slot = (sb-sys:sap+ pv (* 8 i))
          do (etypecase a
               (null (setf (sb-sys:sap-ref-64 slot 0) 0))
               (dbuf (setf (sb-sys:sap-ref-64 slot 0) (dbuf-ptr a)))
               ((cons dbuf integer) (setf (sb-sys:sap-ref-64 slot 0) (+ (dbuf-ptr (car a)) (cdr a))))
               ((cons (eql :i64)) (setf (sb-sys:sap-ref-64 slot 0) (ldb (byte 64 0) (second a))))
               (integer (setf (sb-sys:sap-ref-32 slot 0) (ldb (byte 32 0) a)))
               (single-float (setf (sb-sys:sap-ref-single slot 0) a)))
             (setf (sb-sys:sap-ref-sap pv (* 8 (+ 64 i))) slot))
    (when (> shared (gethash kernel (cudev-shared-set dev) 49152))
      ;; past 48 KB a kernel must be allowed its dynamic shared memory
      (with-cuda
        (cu "cuFuncSetAttribute" (sb-sys:system-area-pointer (cuda-function dev kernel))
            (sb-alien:int 8) (sb-alien:int shared))) ; CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
      (setf (gethash kernel (cudev-shared-set dev)) shared))
    (flet ((dim (x k) (let ((l (if (listp x) x (list x)))) (or (nth k l) 1))))
      (with-cuda
        (cu "cuLaunchKernel" (sb-sys:system-area-pointer (cuda-function dev kernel))
            (sb-alien:unsigned (dim grid 0)) (sb-alien:unsigned (dim grid 1)) (sb-alien:unsigned (dim grid 2))
            (sb-alien:unsigned (dim block 0)) (sb-alien:unsigned (dim block 1)) (sb-alien:unsigned (dim block 2))
            (sb-alien:unsigned shared) (sb-sys:system-area-pointer (cudev-stream dev))
            (sb-sys:system-area-pointer (sb-sys:sap+ pv (* 8 64)))
            (sb-sys:system-area-pointer (sb-sys:int-sap 0)))))))
