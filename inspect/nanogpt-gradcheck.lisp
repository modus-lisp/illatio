;;;; nanogpt-gradcheck.lisp — nanogpt.lisp's backward against finite differences.
;;;
;;;   sbcl --dynamic-space-size 4096 --non-interactive --load inspect/nanogpt-gradcheck.lisp
;;;
;;; A tiny model (2 layers, 128 channels: 2 heads, a 256-token vocabulary,
;;; 160 positions: past one 64-row tile, not a multiple of it) with every
;;; weight random (the zero-initialized ones would hide most gradients), in
;;; f32.  The step is 4e-2 along a matrix's direction, 1e-2 along the scalars'
;;; (their curvature is larger; Richardson's extrapolation over e and e/2 cancels
;;; its e^2 error), the loss summed in double.  For each tensor, a random unit direction U in it:
;;; the gradient's component along U against (L(p + eU) - L(p - eU)) / 2e.
;;; ILLATIO_NGPT_TIED=1: the head tied to the embedding.  ILLATIO_NGPT_WEIGHTS=1:
;;; a loss weight per position, half of them 0.

(defvar cl-user::*system* "illatio/train")
(load (merge-pathnames "load.lisp" (or *load-truename* *default-pathname-defaults*)))
(in-package #:illatio)

(let* ((*random-state* (sb-ext:seed-random-state 1337))  ; compiling uses the default one
       (*g2-precision* '((:head . "") (:attention . "") (:linear . "")))
       (b 2) (tt 160) (v 256)
       (tied (equal (uiop:getenv "ILLATIO_NGPT_TIED") "1"))
       (m (make-nanogpt :l 2 :c 128 :v v :vp v :b b :tt tt :cap 30f0 :tied tied))
       (np (ngpt-n-params m)) (gauss (gaussian-filler 7))
       (p (f32vec np)) (x (make-array (* b tt) :element-type '(signed-byte 32)))
       (y (make-array (* b tt) :element-type '(signed-byte 32)))
       (allok t))
  (dotimes (i np) (setf (aref p i) (* 0.1f0 (funcall gauss))))
  ;; the scalars near their trained values, not near 0, where the stream's
  ;; RMS norms make the loss sharply curved along them
  (let ((pl (ngpt-params m)))
    (flet ((near (name li k mean)
             (setf (aref p (+ k (lof pl name li))) (+ mean (* 0.1f0 (funcall gauss))))))
      (dotimes (li (ngpt-l m))
        (near :lambdas li 0 1f0) (near :lambdas li 1 0.5f0) (near :vlamb li 0 0.5f0))
      (dotimes (j (- (ngpt-l m) (ngpt-encoders m))) (near :skipw 0 (* 4 j) 1f0))))
  (dotimes (i (* b tt)) (setf (aref x i) (random v) (aref y i) (random v)))
  (copy-to-gbuf (ngpt-pbuf m) p)
  (when (equal (uiop:getenv "ILLATIO_NGPT_WEIGHTS") "1")
    (ngpt-set-weights m (let ((w (f32vec (* b tt))))
                          (dotimes (i (* b tt) w) (when (zerop (random 2)) (setf (aref w i) (+ 0.5f0 (random 2f0))))))))
  (let* ((loss (ngpt-grad m x y)) (g (copy-from-gbuf (ngpt-gbuf m) (f32vec np))))
    (format t "loss ~,6F~%" loss)
    (dolist (name (layout-order (ngpt-params m)))
      (let* ((o (lof (ngpt-params m) name)) (n (lsize (ngpt-params m) name (ngpt-l m)))
             (eps (if (member name *ngpt-scalars*) 1d-2 4d-2))
             (u (make-array n :element-type 'double-float)) (norm 0d0) (an 0d0))
        (dotimes (k n) (let ((r (coerce (funcall gauss) (quote double-float)))) (setf (aref u k) r) (incf norm (* r r))))
        (setf norm (sqrt norm))
        (dotimes (k n) (setf (aref u k) (/ (aref u k) norm)) (incf an (* (aref g (+ o k)) (aref u k))))
        (flet ((at (s)
                 (let ((q (copy-seq p)))
                   (dotimes (k n) (setf (aref q (+ o k)) (coerce (+ (aref p (+ o k)) (* s eps (aref u k))) 'single-float)))
                   (copy-to-gbuf (ngpt-pbuf m) q)
                   (ngpt-loss m x y))))
          (let* ((fd (flet ((d (k) (/ (- (at k) (at (- k))) (* 2 k eps))))
                       ;; the scalars' curvature is large: Richardson's extrapolation cancels its eps^2 term
                       (if (member name *ngpt-scalars*) (/ (- (* 4 (d 0.5d0)) (d 1)) 3) (d 1))))
                 (rel (/ (abs (- fd an)) (max (abs fd) (abs an) 1d-6)))
                 ;; or within the differences' noise: a position's f32 loss is good to
                 ;; about 1e-7 of 6, their mean over 320 to about 3e-8, so a difference
                 ;; over e = 1e-2 is good to about 3e-6 (Richardson's, to about 1e-5)
                 (ok (or (< rel 1d-2) (< (abs (- fd an)) 1d-5))))
            (setf allok (and allok ok))
            (format t "~:[NOT OK~;ok    ~] ~10A analytic ~12,6E  numeric ~12,6E  rel ~,2E~%" ok name an fd rel)))))
    (copy-to-gbuf (ngpt-pbuf m) p))
  (format t "~:[FAIL~;PASS~]~%" allok)
  (uiop:quit (if allok 0 1)))
