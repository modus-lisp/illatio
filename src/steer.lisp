;;;; steer.lisp — directional steering: a direction per layer taken out of
;;;; (or put into) the layer's attention output and its FFN output, as the
;;;; token passes:
;;;;
;;;;     y <- y - scale * u (u . y)        u: the layer's direction, a unit vector
;;;;
;;;; ds4's (antirez/ds4, dir-steering/), and its file: one row of D f32s per
;;;; layer, the rows in order, nothing else (a ds4 .f32 file loads as it is).
;;;; A positive scale removes the direction, 1 entirely; a negative one
;;;; amplifies it; 0 leaves the pass as it was.  The FFN output is where ds4
;;;; steers by default (scale 1), the attention's is off (0).
;;;;
;;;; A direction comes from two sets of prompts (BUILD-DIRECTION): each
;;;; layer's output for the last token of each, the mean over the target
;;;; set less the mean over the contrast set, made a unit vector.  The
;;;; outputs are taken where the steering is applied (*STEER-CAPTURE*).
;;;;
;;;; Where it applies: the dense models on the CPU, CUDA and Metal, DeepSeek
;;;; V4, V4.1 and GLM-5.3 on CUDA (their attention and FFN outputs as they
;;;; join the hyper-connection streams).  Not yet: the DeepSeek models on
;;;; Metal or the CPU, Qwen3.5.

(in-package #:illatio)

(defstruct (steering (:constructor %make-steering))
  path
  (n-layer 0 :type fixnum)              ; rows in the file: layers past them are left alone
  (d 0 :type fixnum)
  dirs                                  ; f32vec, N-LAYER rows of D, each a unit vector
  (ffn 1f0 :type single-float)          ; the scale after the FFN
  (attn 0f0 :type single-float)         ; after the attention
  (device (make-hash-table :test 'eq))) ; CUDA device -> its copy of DIRS

(defvar *steering* nil "The steering in force, or NIL.")

(defvar *steer-capture* nil
  "A function called with (LAYER KIND ROW) for each layer's attention
   (KIND :ATTN) and FFN (:FFN) output of the last token of each pass, ROW a
   fresh f32vec, taken before any steering; or NIL.  BUILD-DIRECTION's.")

(defun read-f32-file (path)
  (with-open-file (s path :element-type '(unsigned-byte 32))
    (let* ((n (file-length s)) (u (make-array n :element-type '(unsigned-byte 32))) (v (f32vec n)))
      (read-sequence u s)
      (dotimes (i n v) (setf (aref v i) (sb-kernel:make-single-float (if (logbitp 31 (aref u i)) (- (aref u i) (expt 2 32)) (aref u i))))))))

(defun write-f32-file (path v)
  (with-open-file (s path :element-type '(unsigned-byte 32) :direction :output :if-exists :supersede)
    (write-sequence (map '(vector (unsigned-byte 32)) (lambda (x) (ldb (byte 32 0) (sb-kernel:single-float-bits x))) v) s))
  path)

(defun steer-shape (m)
  "(values layers width) of model M as steering sees it."
  (let ((impl (model-impl m)))
    (if impl (values (ds4f-n-layer impl) (ds4f-d impl)) (values (model-n-layer m) (model-d m)))))

(defun load-steering (path model &key (ffn 1f0) (attn 0f0))
  "The steering file at PATH for MODEL: one row of the model's width per
   layer, as many rows as it has layers or fewer (the rest left alone).
   Each row is made a unit vector."
  (multiple-value-bind (layers d) (steer-shape model)
    (let* ((v (read-f32-file path)) (rows (floor (length v) d)))
      (unless (and (plusp rows) (zerop (mod (length v) d)) (<= rows layers))
        (error "~A: ~D floats is not rows of ~D, at most ~D of them (this model's width and layers)"
               path (length v) d layers))
      (dotimes (r rows)
        (let ((norm (sqrt (loop for i from (* r d) below (* (1+ r) d) sum (expt (aref v i) 2)))))
          (when (plusp norm)
            (loop for i from (* r d) below (* (1+ r) d) do (setf (aref v i) (/ (aref v i) norm))))))
      (%make-steering :path path :n-layer rows :d d :dirs v
                      :ffn (coerce ffn 'single-float) :attn (coerce attn 'single-float)))))

(defun steer-scale (li kind)
  "The scale for layer LI's KIND (:FFN, :ATTN) output: 0 for none."
  (let ((s *steering*))
    (if (and s (< li (steering-n-layer s)))
        (if (eq kind :ffn) (steering-ffn s) (steering-attn s))
        0f0)))

(defun steer-wants-p (li kind)
  "Whether layer LI's KIND output is to be steered or captured."
  (or *steer-capture* (/= 0f0 (steer-scale li kind))))

(defun steer-key ()
  "What a context's cache depends on: a change starts the prompt over."
  (let ((s *steering*))
    (and s (or (/= 0 (steering-ffn s)) (/= 0 (steering-attn s)))
         (list (steering-path s) (steering-ffn s) (steering-attn s)))))

(defun steer-rows! (v li kind n d &optional (start 0))
  "N rows of D in V from START: the last captured, then all steered, in place."
  (declare (type f32vec v) (type fixnum n d start))
  (when *steer-capture*
    (let ((at (+ start (* (1- n) d))))
      (funcall *steer-capture* li kind (subseq v at (+ at d)))))
  (let ((scale (steer-scale li kind)))
    (unless (zerop scale)
      (let ((u (steering-dirs *steering*)) (u0 (* li d)))
        (declare (type f32vec u))
        (dotimes (r n)
          (let* ((o (+ start (* r d)))
                 (k (* scale (loop for i of-type fixnum below d
                                   sum (* (aref u (+ u0 i)) (aref v (+ o i))) of-type single-float))))
            (dotimes (i d) (decf (aref v (+ o i)) (* k (aref u (+ u0 i)))))))))
    v))

;;; --- building a direction ----------------------------------------------------------

(defun last-token-outputs (model text kind)
  "Each layer's KIND output for TEXT's last token: a vector of f32vecs."
  (let ((rows (make-array (steer-shape model) :initial-element nil))
        (ctx (make-context model :n-ctx (+ 16 (length (tokenize model text))))))
    (unwind-protect
         (let ((*steer-capture* (lambda (li k row) (when (eq k kind) (setf (aref rows li) row))))
               (*steering* nil))
           (eval-tokens ctx (tokenize model text)))
      (free-context ctx))
    rows))

(defun build-direction (model targets contrasts &key (kind :ffn) out (log *error-output*))
  "A steering file from two lists of texts (already in the model's template
   if they should be): per layer, the mean of the targets' last-token KIND
   outputs less the contrasts', a unit vector.  Written to OUT (with OUT's
   .json beside it) when given; returned as an f32vec."
  (multiple-value-bind (layers d) (steer-shape model)
    (let ((sum (f32vec (* layers d))) (n 0))
      (flet ((add (texts sign)
               (dolist (text texts)
                 (let ((rows (last-token-outputs model text kind)))
                   (dotimes (li layers)
                     (let ((r (aref rows li)))
                       (when r (dotimes (i d) (incf (aref sum (+ (* li d) i)) (* sign (/ (aref r i) (length texts)))))))))
                 (when log (format log "~&illatio: ~D/~D prompts~%" (incf n) (+ (length targets) (length contrasts)))))))
        (add targets 1f0)
        (add contrasts -1f0))
      (dotimes (li layers)
        (let ((norm (sqrt (loop for i from (* li d) below (* (1+ li) d) sum (expt (aref sum i) 2)))))
          (when (plusp norm) (loop for i from (* li d) below (* (1+ li) d) do (setf (aref sum i) (/ (aref sum i) norm))))))
      (when out
        (write-f32-file out sum)
        (with-open-file (s (make-pathname :type "json" :defaults out) :direction :output :if-exists :supersede)
          (format s "{\"model\": ~S, \"layers\": ~D, \"width\": ~D, \"component\": \"~(~A~)_out\", \"targets\": ~D, \"contrasts\": ~D}~%"
                  (model-name model) layers d kind (length targets) (length contrasts))))
      sum)))
