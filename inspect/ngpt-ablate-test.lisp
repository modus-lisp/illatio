;;;; ngpt-ablate-test.lisp — ngpt-ablate.lisp's edits on a tiny random model.
;;;
;;;   sbcl --dynamic-space-size 4096 --non-interactive --load inspect/ngpt-ablate-test.lisp
;;;
;;; Every weight random (as in nanogpt-gradcheck.lisp), in f32.  Checks:
;;; edits are undone exactly; every head off is the attention off; a
;;; checkpoint round-trips; a removed direction is gone from what each layer
;;; writes into the stream, and an edited layer's output is the old one less
;;; its component along the direction.

(defvar cl-user::*system* "illatio/train")
(load (merge-pathnames "load.lisp" (or *load-truename* *default-pathname-defaults*)))
(in-package #:illatio)

(defvar *failures* 0)
(defun check (name ok &optional (fmt "") &rest args)
  (format t "~:[FAIL~;ok  ~]  ~A ~?~%" ok name fmt args)
  (unless ok (incf *failures*)))

(let* ((*g2-precision* '((:head . "") (:attention . "") (:linear . "")))
       (b 2) (tt 128) (v 256) (c 128)
       (m (make-nanogpt :l 4 :c c :v v :vp v :b b :tt tt :cap 30f0))
       (np (ngpt-n-params m)) (gauss (gaussian-filler 11)) (p (f32vec np))
       (batches (loop for s below 2
                      collect (let ((x (make-array (* b tt) :element-type '(signed-byte 32)))
                                    (y (make-array (* b tt) :element-type '(signed-byte 32))))
                                (dotimes (i (* b tt)) (setf (aref x i) (random v) (aref y i) (random v)))
                                (cons x y)))))
  (dotimes (i np) (setf (aref p i) (* 0.05f0 (funcall gauss))))
  (dolist (name '(:lambdas :vlamb))
    (dotimes (li 4) (setf (aref p (lof (ngpt-params m) name li)) 0.7f0
                          (aref p (+ 1 (lof (ngpt-params m) name li))) 0.4f0)))
  (dotimes (j 2) (setf (aref p (+ (lof (ngpt-params m) :skipw) (* 4 j))) 0.8f0))
  (copy-to-gbuf (ngpt-pbuf m) p)
  (let ((base (ngpt-batches-loss m batches)))
    ;; every component changes the loss, and is undone
    (let ((moved (loop for comp in (ngpt-components m)
                       count (/= base (with-ngpt-edits (m (list (ngpt-component-edit m comp)) p)
                                        (ngpt-batches-loss m batches))))))
      (check "every component matters" (= moved (length (ngpt-components m))) "~D of ~D" moved (length (ngpt-components m))))
    (check "edits undone" (and (equalp p (ngpt-params-vector m)) (= base (ngpt-batches-loss m batches))))
    ;; all heads = the attention
    (let ((heads (with-ngpt-edits (m (loop for h below (ngpt-nh m) collect (ngpt-component-edit m (list :head 1 h))) p)
                   (ngpt-batches-loss m batches)))
          (attn (with-ngpt-edits (m (list (ngpt-component-edit m '(:attn 1))) p) (ngpt-batches-loss m batches))))
      (check "all heads = attention" (= heads attn) "~,6F ~,6F" heads attn)))
  ;; checkpoint
  (ngpt-save m "/tmp/ngpt-ablate-test.ckpt")
  (let ((m2 (ngpt-load "/tmp/ngpt-ablate-test.ckpt" :b b :tt tt :gpu (ngpt-gpu m))))
    (check "checkpoint round trip" (equalp p (ngpt-params-vector m2))))
  ;; a direction removed from every layer: what each layer writes has none of it
  (let* ((c (ngpt-c m)) (n (* b tt)) (buf (f32vec (* n c)))
         (d (unit (let ((r (f32vec c))) (dotimes (k c r) (setf (aref r k) (funcall gauss)))))))
    (with-ngpt-edits (m (list (ngpt-direction-edit m d)) p)
      (ngpt-loss m (car (first batches)) (cdr (first batches)))
      (dolist (name '(:proj :fcproj))
        (dotimes (li 4)
          (copy-from-gbuf (ngpt-abuf m) buf (lof (ngpt-acts m) name li))
          (let ((along 0d0) (size 0d0))
            (dotimes (i n)
              (setf along (max along (abs (loop for k below c sum (* (aref d k) (aref buf (+ (* i c) k))))))))
            (loop for x across buf do (setf size (max size (abs x))))
            (check (format nil "~(~A~) ~D has no D" name li) (< along (* 1e-5 size)) "max |.D| ~,2E of ~,2E" along size))))))
  ;; layer 0 alone: its attention's output loses its D component and nothing else
  (let* ((c (ngpt-c m)) (n (* b tt)) (buf (f32vec (* n c)))
         (d (unit (let ((r (f32vec c))) (dotimes (k c r) (setf (aref r k) (funcall gauss)))))))
    (flet ((out0 ()
             (ngpt-loss m (car (first batches)) (cdr (first batches)))
             (copy-from-gbuf (ngpt-abuf m) buf (lof (ngpt-acts m) :proj 0))
             (copy-seq buf)))
      (let* ((before (out0))
             (after (with-ngpt-edits (m (list (ngpt-direction-edit m d :layers '(0))) p) (out0)))
             (worst 0d0) (size 0d0))
        ;; layer 0's attention reads x0 alone, which no edit touches, so AFTER = (I - DD^T) BEFORE exactly
        (dotimes (i n)
          (let ((dot (loop for k below c sum (* (aref d k) (aref before (+ (* i c) k))))))
            (dotimes (k c)
              (let ((want (- (aref before (+ (* i c) k)) (* dot (aref d k)))))
                (setf worst (max worst (abs (- want (aref after (+ (* i c) k)))))
                      size (max size (abs want)))))))
        (check "layer 0: proj = (I - DD^T) proj" (< worst (* 1e-5 size)) "max err ~,2E of ~,2E" worst size)))))

(format t "~:[~D FAILED~;all passed~*~]~%" (zerop *failures*) *failures*)
(sb-ext:exit :code (if (zerop *failures*) 0 1))
