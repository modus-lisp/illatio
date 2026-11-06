;;;; ngpt-ablate.lisp — a trained model taken apart: components switched off,
;;;; directions found and removed, all by editing weights, judged by forward
;;;; passes.
;;;
;;; The weight-space reading of "ablation": instead of training again without
;;; a component (an hour a question in the Mac speedrun), switch it off in the
;;; trained weights and measure what the loss loses.  Not the same question --
;;; a model trained without a part grows around its absence -- but an answer
;;; in seconds, good for deciding which retrainings are worth an hour.
;;;
;;; A component is switched off by zeroing what it writes into the residual
;;; stream (a head's columns of the attention projection, a layer's MLP
;;; projection) or the scalar that lets it in (a U-net skip, x0's mixing, the
;;; value residual).  A direction D in the residual stream is removed from
;;; every matrix that writes into it, W <- (I - D D^T) W, as abliteration
;;; removes a refusal direction; the stream is RMS-normed before every read,
;;; so what is removed is the direction, not a scale.  Directions come from
;;; contrasts: the difference of a stream's mean over two sets of inputs.
;;;
;;; An edit is a function of the parameters (a host copy) that changes them
;;; in place and returns the ranges it changed; those ranges alone go back to
;;; the GPU, and are restored after.

(in-package #:illatio)

;;; --- edits --------------------------------------------------------------------

(defun ngpt-head-columns (m li h)
  "The ranges of the attention projection's columns that head H of layer LI writes through."
  (let* ((c (ngpt-c m)) (o (lof (ngpt-params m) :projw li)))
    (loop for r below c collect (cons (+ o (* r c) (* h 64)) (+ o (* r c) (* h 64) 64)))))

(defun zero-ranges (ranges)
  (lambda (p) (dolist (r ranges ranges) (fill p 0f0 :start (car r) :end (cdr r)))))

(defun ngpt-tensor-range (m name &optional (li 0) (plus 0) (count (lsize (ngpt-params m) name)))
  (let ((o (+ plus (lof (ngpt-params m) name li)))) (cons o (+ o count))))

(defun ngpt-component-edit (m component)
  "The edit switching COMPONENT off:
     (:head LI H)  head H of layer LI      (:attn LI)  layer LI's attention
     (:mlp LI)     layer LI's MLP          (:skip J)   the J-th U-net skip
     (:x0 LI)      layer LI's mix of x0    (:vres LI)  layer LI's value residual (V its own)"
  (destructuring-bind (kind &optional (i 0) (h 0)) component
    (ecase kind
      (:head (zero-ranges (ngpt-head-columns m i h)))
      (:attn (zero-ranges (list (ngpt-tensor-range m :projw i))))
      (:mlp (zero-ranges (list (ngpt-tensor-range m :fcprojw i))))
      (:skip (zero-ranges (list (ngpt-tensor-range m :skipw 0 (* 4 i) 1))))
      (:x0 (zero-ranges (list (ngpt-tensor-range m :lambdas i 1 1))))
      (:vres (zero-ranges (list (ngpt-tensor-range m :vlamb i 0 1)))))))

(defun ngpt-components (m)
  "Every component NGPT-COMPONENT-EDIT knows, in the model's order."
  (let ((l (ngpt-l m)))
    (append (loop for li below l
                  append (append (loop for h below (ngpt-nh m) collect (list :head li h))
                                 (list (list :attn li) (list :mlp li) (list :x0 li))
                                 (when (plusp li) (list (list :vres li)))))
            (loop for j below (- l (ngpt-encoders m)) collect (list :skip j)))))

(defun unit (d)
  (let ((n (sqrt (loop for x across d sum (* (coerce x 'double-float) x)))))
    (map-into (f32vec (length d)) (lambda (x) (coerce (/ x n) 'single-float)) d)))

(defun ngpt-direction-edit (m d &key (layers :all) embedding)
  "The edit removing direction D (C floats, made a unit) from what LAYERS'
   attention and MLP write into the residual stream (and with EMBEDDING, from
   every token's embedding): W <- (I - D D^T) W."
  (let* ((d (unit d)) (c (ngpt-c m)) (pl (ngpt-params m))
         (layers (if (eq layers :all) (loop for li below (ngpt-l m) collect li) layers)))
    (lambda (p)
      (declare (type f32vec p d))
      (let ((ranges '()))
        (flet ((project-columns (o cin)
                 ;; W is C rows (the stream) of CIN: each column loses its D component
                 (let ((dot (make-array cin :element-type 'double-float :initial-element 0d0)))
                   (dotimes (r c)
                     (let ((dr (aref d r)) (row (+ o (* r cin))))
                       (dotimes (k cin) (incf (aref dot k) (* dr (aref p (+ row k)))))))
                   (dotimes (r c)
                     (let ((dr (aref d r)) (row (+ o (* r cin))))
                       (dotimes (k cin)
                         (decf (aref p (+ row k)) (coerce (* dr (aref dot k)) 'single-float)))))
                   (push (cons o (+ o (* c cin))) ranges))))
          (dolist (li layers)
            (project-columns (lof pl :projw li) c)
            (project-columns (lof pl :fcprojw li) (* 4 c)))
          (when embedding
            (let ((o (lof pl :wte)))
              (dotimes (tok (ngpt-v m))
                (let ((row (+ o (* tok c))) (dot 0d0))
                  (dotimes (k c) (incf dot (* (aref d k) (aref p (+ row k)))))
                  (dotimes (k c) (decf (aref p (+ row k)) (coerce (* dot (aref d k)) 'single-float)))))
              (push (cons o (+ o (* (ngpt-v m) c))) ranges))))
        ranges))))

(defun ngpt-apply-ranges (m p ranges)
  (dolist (r ranges)
    (copy-to-gbuf (ngpt-pbuf m) (subseq p (car r) (cdr r)) (car r))))

(defun call-with-ngpt-edits (m edits fn &optional (p0 (ngpt-params-vector m)))
  "FN called with the EDITS made to M's weights; the weights restored after.
   P0 is the parameters as they are (passed in, to share one copy)."
  (let ((p (copy-seq p0)) (ranges '()))
    (dolist (e edits) (setf ranges (append (funcall e p) ranges)))
    (ngpt-apply-ranges m p ranges)
    (unwind-protect (funcall fn)
      (ngpt-apply-ranges m p0 ranges))))

(defmacro with-ngpt-edits ((m edits &optional p0) &body body)
  `(call-with-ngpt-edits ,m ,edits (lambda () ,@body) ,@(when p0 (list p0))))

;;; --- reading the stream --------------------------------------------------------

(defun ngpt-batches (files m batches)
  "BATCHES batches of M's shape from FILES' first tokens, as (inputs . targets)."
  (let ((ld (make-shard-loader files (ngpt-b m) (ngpt-tt m))))
    (loop repeat batches
          collect (multiple-value-bind (x y) (shard-next ld) (cons (copy-seq x) (copy-seq y))))))

(defun ngpt-batches-loss (m batches)
  (/ (loop for (x . y) in batches sum (ngpt-loss m x y)) (length batches)))

(defun ngpt-mean-activation (m batches name &optional (li 0) (rows nil))
  "The mean over BATCHES' positions (those ROWS x positions allow, a function
   of row and position, if given) of activation NAME (layer LI): C floats."
  (let* ((c (ngpt-c m)) (n (* (ngpt-b m) (ngpt-tt m))) (sum (make-array c :element-type 'double-float :initial-element 0d0))
         (count 0) (buf (f32vec (* n c))))
    (loop for (x . y) in batches
          do (ngpt-loss m x y)
             (copy-from-gbuf (ngpt-abuf m) buf (lof (ngpt-acts m) name li))
             (dotimes (i n)
               (when (or (null rows) (funcall rows (floor i (ngpt-tt m)) (mod i (ngpt-tt m))))
                 (incf count)
                 (dotimes (k c) (incf (aref sum k) (aref buf (+ (* i c) k)))))))
    (map-into (f32vec c) (lambda (s) (coerce (/ s (max 1 count)) 'single-float)) sum)))

(defun ngpt-contrast-direction (m batches-a batches-b name &optional (li 0))
  "The difference of the means of activation NAME (layer LI) over BATCHES-A and BATCHES-B."
  (map 'f32vec #'- (ngpt-mean-activation m batches-a name li) (ngpt-mean-activation m batches-b name li)))

;;; --- the table ------------------------------------------------------------------

(defun ngpt-ablation-table (m batches &key (components (ngpt-components m)) (out *standard-output*))
  "For each of COMPONENTS, the loss over BATCHES with it switched off, less
   the loss with it on.  Printed, largest first; the list of (component . delta)."
  (let* ((p0 (ngpt-params-vector m)) (base (ngpt-batches-loss m batches))
         (rows (loop for comp in components
                     collect (cons comp (- (with-ngpt-edits (m (list (ngpt-component-edit m comp)) p0)
                                             (ngpt-batches-loss m batches))
                                           base)))))
    (format out "loss ~,4F over ~D batches~%" base (length batches))
    (dolist (r (sort (copy-list rows) #'> :key #'cdr) rows)
      (format out "  ~22A ~,4@F~%" (format nil "~{~(~A~)~^ ~}" (car r)) (cdr r)))))
