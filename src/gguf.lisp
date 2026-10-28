;;;; gguf.lisp — the GGUF container: header, metadata, tensor directory.
;;;
;;; Spec: ggml/docs/gguf.md.  All integers little-endian.  We read the
;;; header and directory through a stream and leave tensor data on disk;
;;; LOAD-TENSOR seeks to one tensor and reads only it, so a model is never
;;; held twice in memory.

(in-package #:illatio)

(defstruct (gguf (:constructor %make-gguf))
  path
  (version 0)
  (metadata (make-hash-table :test 'equal))
  (tensors (make-hash-table :test 'equal)) ; name -> tensor-info
  (tensor-order '())                       ; names in file order
  (data-offset 0)
  (alignment 32))

(defstruct tensor-info
  name
  dims          ; list, ggml order: (ne0 ne1 ...), ne0 is the contiguous axis
  type          ; ggml type keyword
  offset)       ; relative to the data section

(defun meta (gguf key &optional default)
  (multiple-value-bind (v found) (gethash key (gguf-metadata gguf))
    (if found v default)))

;;; ggml types we know the layout of.  BLOCK is elements per block, BYTES is
;;; bytes per block.
(defparameter +ggml-types+
  '((0 :f32 1 4) (1 :f16 1 2) (2 :q4_0 32 18) (3 :q4_1 32 20)
    (6 :q5_0 32 22) (7 :q5_1 32 24) (8 :q8_0 32 34) (9 :q8_1 32 36)
    (10 :q2_k 256 84) (11 :q3_k 256 110) (12 :q4_k 256 144)
    (13 :q5_k 256 176) (14 :q6_k 256 210) (15 :q8_k 256 292)
    (24 :i8 1 1) (25 :i16 1 2) (26 :i32 1 4) (27 :i64 1 8) (28 :f64 1 8)
    (30 :bf16 1 2)))

(defun ggml-type (code)
  (or (second (assoc code +ggml-types+))
      (error "GGUF: unknown ggml tensor type ~D" code)))

(defun ggml-type-layout (type)
  (let ((e (find type +ggml-types+ :key #'second)))
    (values (third e) (fourth e))))

(defun tensor-elements (ti)
  (reduce #'* (tensor-info-dims ti)))

(defun tensor-bytes (ti)
  (multiple-value-bind (block bytes) (ggml-type-layout (tensor-info-type ti))
    (* (/ (tensor-elements ti) block) bytes)))

;;; --- primitive readers ------------------------------------------------------

(defun read-uint (s n)
  (let ((v 0))
    (dotimes (i n v)
      (let ((b (read-byte s nil)))
        (unless b (error "GGUF: unexpected end of file"))
        (setf v (logior v (ash b (* 8 i))))))))

(defun read-sint (s n)
  (let ((v (read-uint s n)))
    (if (logbitp (1- (* 8 n)) v) (- v (ash 1 (* 8 n))) v)))

(defun read-f32 (s) (bits-single (read-uint s 4)))

(defun read-f64 (s)
  (let ((b (read-uint s 8)))
    (sb-kernel:make-double-float (let ((hi (ldb (byte 32 32) b)))
                                   (if (logbitp 31 hi) (- hi (ash 1 32)) hi))
                                 (ldb (byte 32 0) b))))

(defun utf8-decode (octets)
  (sb-ext:octets-to-string octets :external-format `(:utf-8 :replacement ,(code-char #xfffd))))

(defun read-gguf-string (s)
  (let* ((n (read-uint s 8))
         (buf (make-array n :element-type '(unsigned-byte 8))))
    (unless (= (read-sequence buf s) n)
      (error "GGUF: truncated string"))
    (utf8-decode buf)))

(defun read-value (s type)
  (ecase type
    (0 (read-uint s 1)) (1 (read-sint s 1))
    (2 (read-uint s 2)) (3 (read-sint s 2))
    (4 (read-uint s 4)) (5 (read-sint s 4))
    (6 (read-f32 s))
    (7 (/= 0 (read-uint s 1)))
    (8 (read-gguf-string s))
    (9 (let* ((et (read-uint s 4))
              (n (read-uint s 8))
              (v (make-array n)))
         (dotimes (i n v) (setf (aref v i) (read-value s et)))))
    (10 (read-uint s 8)) (11 (read-sint s 8))
    (12 (read-f64 s))))

;;; --- the file ---------------------------------------------------------------

(defun read-gguf (path)
  "Read the header, metadata and tensor directory of the GGUF file at PATH."
  (with-open-file (s path :element-type '(unsigned-byte 8))
    (let ((magic (read-uint s 4)))
      (unless (= magic #x46554747)      ; "GGUF"
        (error "~A is not a GGUF file" path)))
    (let* ((g (%make-gguf :path (pathname path)))
           (version (read-uint s 4))
           (n-tensors (read-uint s 8))
           (n-kv (read-uint s 8)))
      (unless (member version '(2 3))
        (error "GGUF version ~D is not supported (2 and 3 are)" version))
      (setf (gguf-version g) version)
      (dotimes (i n-kv)
        (let* ((key (read-gguf-string s))
               (type (read-uint s 4)))
          (setf (gethash key (gguf-metadata g)) (read-value s type))))
      (setf (gguf-alignment g) (meta g "general.alignment" 32))
      (let ((order '()))
        (dotimes (i n-tensors)
          (let* ((name (read-gguf-string s))
                 (nd (read-uint s 4))
                 (dims (loop repeat nd collect (read-uint s 8)))
                 (type (ggml-type (read-uint s 4)))
                 (offset (read-uint s 8)))
            (push name order)
            (setf (gethash name (gguf-tensors g))
                  (make-tensor-info :name name :dims dims :type type
                                    :offset offset))))
        (setf (gguf-tensor-order g) (nreverse order)))
      (let ((a (gguf-alignment g))
            (p (file-position s)))
        (setf (gguf-data-offset g) (* a (ceiling p a))))
      g)))

(defun gguf-tensor (g name &optional (errorp t))
  (or (gethash name (gguf-tensors g))
      (and errorp (error "GGUF: no tensor named ~S" name))))

(defun call-with-tensor-bytes (g ti fn &key (chunk (* 4 1024 1024)))
  "Stream the raw bytes of tensor TI to FN in chunks: (FN buffer start end),
   START being the byte position within the tensor."
  (with-open-file (s (gguf-path g) :element-type '(unsigned-byte 8))
    (file-position s (+ (gguf-data-offset g) (tensor-info-offset ti)))
    (let* ((total (tensor-bytes ti))
           (buf (make-array (min chunk total) :element-type '(unsigned-byte 8))))
      (loop with pos = 0
            while (< pos total)
            do (let* ((want (min (length buf) (- total pos)))
                      (got (read-sequence buf s :end want)))
                 (unless (= got want) (error "GGUF: truncated tensor ~A" (tensor-info-name ti)))
                 (funcall fn buf pos want)
                 (incf pos want))))))
