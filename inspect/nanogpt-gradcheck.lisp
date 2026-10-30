;;;; nanogpt-gradcheck.lisp — nanogpt.lisp's backward against finite differences.
;;;
;;;   sbcl --dynamic-space-size 4096 --non-interactive --load inspect/nanogpt-gradcheck.lisp
;;;
;;; A tiny model (2 layers, 128 channels: 2 heads, a 256-token vocabulary)
;;; with every weight random (the zero-initialized ones would hide most
;;; gradients), in f32.  For each tensor, a random unit direction U in it:
;;; the gradient's component along U against (L(p + eU) - L(p - eU)) / 2e.

(defvar cl-user::*system* "illatio/train")
(load (merge-pathnames "load.lisp" (or *load-truename* *default-pathname-defaults*)))
(in-package #:illatio)

(let* ((*g2-precision* '((:head . "") (:attention . "") (:linear . "")))
       (b 2) (tt 32) (v 256)
       (m (make-nanogpt :l 2 :c 128 :v v :vp v :b b :tt tt :cap 30f0))
       (np (ngpt-n-params m)) (gauss (gaussian-filler 7))
       (p (f32vec np)) (x (make-array (* b tt) :element-type '(signed-byte 32)))
       (y (make-array (* b tt) :element-type '(signed-byte 32)))
       (eps 4d-2) (allok t))
  (dotimes (i np) (setf (aref p i) (* 0.1f0 (funcall gauss))))
  (dotimes (i (* b tt)) (setf (aref x i) (random v) (aref y i) (random v)))
  (copy-to-gbuf (ngpt-pbuf m) p)
  (let* ((loss (ngpt-grad m x y)) (g (copy-from-gbuf (ngpt-gbuf m) (f32vec np))))
    (format t "loss ~,6F~%" loss)
    (dolist (name (layout-order (ngpt-params m)))
      (let* ((o (lof (ngpt-params m) name)) (n (lsize (ngpt-params m) name (ngpt-l m)))
             (u (make-array n :element-type 'double-float)) (norm 0d0) (an 0d0))
        (dotimes (k n) (let ((r (coerce (funcall gauss) (quote double-float)))) (setf (aref u k) r) (incf norm (* r r))))
        (setf norm (sqrt norm))
        (dotimes (k n) (setf (aref u k) (/ (aref u k) norm)) (incf an (* (aref g (+ o k)) (aref u k))))
        (flet ((at (s)
                 (let ((q (copy-seq p)))
                   (dotimes (k n) (setf (aref q (+ o k)) (coerce (+ (aref p (+ o k)) (* s eps (aref u k))) 'single-float)))
                   (copy-to-gbuf (ngpt-pbuf m) q)
                   (ngpt-loss m x y))))
          (let* ((fd (/ (- (at 1) (at -1)) (* 2 eps)))
                 (rel (/ (abs (- fd an)) (max (abs fd) (abs an) 1d-6)))
                 (ok (< rel 1d-2)))
            (setf allok (and allok ok))
            (format t "~:[NOT OK~;ok    ~] ~10A analytic ~12,6E  numeric ~12,6E  rel ~,2E~%" ok name an fd rel)))))
    (copy-to-gbuf (ngpt-pbuf m) p))
  (format t "~:[FAIL~;PASS~]~%" allok)
  (uiop:quit (if allok 0 1)))
