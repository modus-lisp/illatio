;;;; image.lisp — pictures for a vision encoder: an RGB picture (decoded by
;;;; pigment, in the server: openai.lisp) resized as ds4 resizes it and cut
;;;; into the patches DeepSeek V4.1's encoder takes; and a picture's tokens.

(in-package #:illatio)

(defstruct (rgb-image (:constructor make-rgb-image (width height pixels)))
  (width 0 :type fixnum) (height 0 :type fixnum)
  (pixels nil :type u8vec))             ; R G B, row after row

(define-condition image-error (error) ((what :initarg :what :reader image-error-what))
  (:report (lambda (c s) (format s "image: ~A" (image-error-what c)))))

(defun image-fail (fmt &rest args) (error 'image-error :what (apply #'format nil fmt args)))

;;; --- ds4's resize, and V4.1's patches -----------------------------------------------

(defun cubic (x a)
  (declare (type double-float x a))
  (let ((x (abs x)))
    (cond ((< x 1d0) (+ (* (- (* (+ a 2d0) x) (+ a 3d0)) x x) 1d0))
          ((< x 2d0) (- (* (+ (* (- (* a x) (* 5d0 a)) x) (* 8d0 a)) x) (* 4d0 a)))
          (t 0d0))))

(defun resize-weights (src dst)
  "For each of DST positions: (values lo hi weights) over SRC positions, as
   ds4_resize_rgb_bicubic spans and weighs them (a = -0.5), normalized."
  (let* ((scale (/ (float src 1d0) dst)) (filter (if (>= scale 1d0) (/ scale) 1d0))
         (support (if (>= scale 1d0) (* 2d0 scale) 2d0)))
    (loop for d below dst
          collect (let* ((center (* scale (+ d 0.5d0)))
                         (lo (max 0 (truncate (+ (- center support) 0.5d0))))
                         (hi (min src (truncate (+ center support 0.5d0))))
                         (ws (make-array (max 0 (- hi lo)) :element-type 'double-float)))
                    (loop for i from lo below hi
                          do (setf (aref ws (- i lo)) (cubic (* (- (+ i 0.5d0) center) filter) -0.5d0)))
                    (let ((sum (reduce #'+ ws)))
                      (map-into ws (lambda (x) (/ x sum)) ws))
                    (list lo hi ws)))))

(defun resize-bicubic (img dw dh canvas cw ox oy)
  "IMG resized to DW x DH into float CANVAS (CW wide, RGB) at OX, OY: ds4's
   two-dimensional weighting, done one axis at a time (the weights factor),
   rounded once at the end."
  (declare (type (simple-array single-float (*)) canvas) (optimize speed (safety 1)))
  (let* ((sw (rgb-image-width img)) (sh (rgb-image-height img)) (src (rgb-image-pixels img))
         (wx (coerce (resize-weights sw dw) 'simple-vector)) (wy (coerce (resize-weights sh dh) 'simple-vector))
         (tmp (make-array (* sh dw 3) :element-type 'double-float)))
    (declare (type u8vec src) (type (simple-array double-float (*)) tmp) (type fixnum sw sh dw dh cw ox oy))
    (dotimes (y sh)
      (dotimes (dx dw)
        (destructuring-bind (lo hi ws) (svref wx dx)
          (declare (type fixnum lo hi) (type (simple-array double-float (*)) ws))
          (let ((r 0d0) (g 0d0) (b 0d0))
            (declare (type double-float r g b))
            (loop for x fixnum from lo below hi
                  do (let ((wgt (aref ws (- x lo))) (o (* 3 (+ x (* y sw)))))
                       (incf r (* wgt (aref src o))) (incf g (* wgt (aref src (+ o 1)))) (incf b (* wgt (aref src (+ o 2))))))
            (let ((o (* 3 (+ dx (* y dw)))))
              (setf (aref tmp o) r (aref tmp (+ o 1)) g (aref tmp (+ o 2)) b))))))
    (dotimes (dy dh)
      (destructuring-bind (lo hi ws) (svref wy dy)
        (declare (type fixnum lo hi) (type (simple-array double-float (*)) ws))
        (dotimes (dx dw)
          (let ((r 0d0) (g 0d0) (b 0d0))
            (declare (type double-float r g b))
            (loop for y fixnum from lo below hi
                  do (let ((wgt (aref ws (- y lo))) (o (* 3 (+ dx (* y dw)))))
                       (incf r (* wgt (aref tmp o))) (incf g (* wgt (aref tmp (+ o 1)))) (incf b (* wgt (aref tmp (+ o 2))))))
            (let ((o (* 3 (+ ox dx (* (+ oy dy) cw)))))
              (flet ((cl (v) (coerce (max 0 (min 255 (round (the double-float v)))) 'single-float)))
                (setf (aref canvas o) (cl r) (aref canvas (+ o 1)) (cl g) (aref canvas (+ o 2)) (cl b))))))))
    canvas))

(defun ds41-grid-tokens (height width)
  "ds4_deepseek4_grid_tokens for V4.1: (values tokens rows cols) of the
   language model's image grid for a HEIGHT x WIDTH picture."
  (let ((h (floor (+ (floor height 14) 2) 3)) (w (floor (+ (floor width 14) 2) 3)))
    (values (+ (* h (1+ w)) 2) h w)))

(defun ds41-solve-resize (height width budget)
  "ds4_deepseek4_solve_resize for V4.1: (values best-height best-width) or NIL."
  (let* ((ratio (/ (float height 1d0) width))
         (max-w-f (- (sqrt (+ (/ (- budget 2d0) ratio) 0.25d0)) 0.5d0))
         (max-h-f (* max-w-f ratio)))
    (cond ((< max-w-f 1d0)
           (let ((mh (floor (- budget 2) 2)))
             (and (plusp mh) (values (* mh 42) 42))))
          ((< max-h-f 1d0)
           (let ((mw (1- (- budget 2))))
             (and (> mw 1) (values 42 (* mw 42)))))
          (t (let ((mw (floor max-w-f)) (mh (floor max-h-f)))
               (and (plusp mw) (plusp mh)
                    (let ((scale (min (/ (* mw 42d0) width) (/ (* mh 42d0) height))))
                      (values (* 14 (floor (* height scale) 14)) (* 14 (floor (* width scale) 14))))))))))

(defun ds41-image-patches (img)
  "ds4_image_preprocess_deepseek41: (values patches grid-h grid-w llm-h llm-w),
   PATCHES each 588 values (channel, row, column of a 14 x 14 patch) of the
   picture fitted, centered, into a mid-gray canvas sized for at most 1024
   tokens, each in -1..1, padded to 592 a patch."
  (let* ((iw (rgb-image-width img)) (ih (rgb-image-height img))
         (pw iw) (ph ih) (min-pixels 295936))
    (when (< (* pw ph) min-pixels)
      (let ((s (sqrt (/ (float min-pixels 1d0) (* pw ph)))))
        (setf pw (max 1 (truncate (* pw s))) ph (max 1 (truncate (* ph s))))))
    (let ((bw (* 14 (ceiling pw 14))) (bh (* 14 (ceiling ph 14))) (budget 1024))
      (loop while (> (ds41-grid-tokens bh bw) 1024)
            do (multiple-value-bind (nh nw) (and (> budget 4) (ds41-solve-resize ph pw budget))
                 (unless (and nh (plusp nh) (plusp nw)) (image-fail "can't fit ~Dx~D in the vision token budget" iw ih))
                 (setf bh nh bw nw)
                 (decf budget)))
      (let* ((scale (min (/ (float bw 1d0) iw) (/ (float bh 1d0) ih)))
             (cw (min bw (max 1 (round (* iw scale))))) (ch (min bh (max 1 (round (* ih scale)))))
             (ox (round (* (- bw cw) 0.5d0))) (oy (round (* (- bh ch) 0.5d0)))
             (canvas (make-array (* bw bh 3) :element-type 'single-float :initial-element 127f0)))
        (resize-bicubic img cw ch canvas bw ox oy)
        (let* ((gh (floor bh 14)) (gw (floor bw 14))
               (patches (make-array (* gh gw 592) :element-type 'single-float :initial-element 0f0)))
          (dotimes (py gh)
            (dotimes (px gw)
              (let ((o (* 592 (+ px (* py gw)))))
                (dotimes (c 3)
                  (dotimes (y 14)
                    (dotimes (x 14)
                      (setf (aref patches (+ o (* c 196) (* y 14) x))
                            (- (/ (aref canvas (+ c (* 3 (+ (* px 14) x (* (+ (* py 14) y) bw))))) 127.5f0) 1f0))))))))
          (multiple-value-bind (tokens lh lw) (ds41-grid-tokens bh bw)
            (declare (ignore tokens))
            (values patches gh gw lh lw)))))))

;;; --- a picture's place in a prompt ----------------------------------------------------
;;; Its positions are tokens of their own, past any vocabulary:
;;; +IMAGE-TOKEN-BASE+ + 1024 x its number + its row in the block (start,
;;; the rows of the grid each closed by a newline, end).  A model embeds such
;;; a token as that row of *IMAGE-BLOCKS*; a prompt's tokens then name what
;;; they hold, so prompt reuse stops where one picture differs from another.
;;; Numbers come from the picture's bytes, so the same picture, in this
;;; process or a saved state's, is the same tokens.

(defvar *vision* nil "The vision encoder (cuda-vision.lisp's LOAD-VISION-CUDA), or NIL.")

(defconstant +image-token-base+ (ash 1 24))
(defconstant +image-block-max+ 1024)

(defvar *image-blocks* (make-hash-table) "number -> (key width . rows): a picture's embedding rows")
(defvar *image-order* '() "numbers, the most recently used first")
(defparameter *image-blocks-kept* 16 "How many pictures' rows stay (~21 MB each).")
(defvar *image-lock* (sb-thread:make-mutex :name "image blocks"))

(declaim (inline image-token-p))
(defun image-token-p (token) (>= token +image-token-base+))

(defun image-key (bytes)
  "Two FNV-1a hashes (64 bits, different offsets) of BYTES and its length."
  (declare (type u8vec bytes) (optimize speed))
  (let ((a #xcbf29ce484222325) (b #x84222325cbf29ce4) (p #x100000001b3))
    (declare (type (unsigned-byte 64) a b))
    (loop for x across bytes
          do (setf a (ldb (byte 64 0) (* (logxor a x) p))
                   b (ldb (byte 64 0) (* (logxor b (logxor x #x5a)) p))))
    (format nil "~16,'0x~16,'0x~x" a b (length bytes))))

(defun image-row (token)
  "(values rows offset width) of TOKEN's embedding, or an error."
  (multiple-value-bind (number row) (floor (- token +image-token-base+) +image-block-max+)
    (let ((block (gethash number *image-blocks*)))
      (unless block (error "image token ~D: its picture is no longer held" token))
      (destructuring-bind (key width . rows) block
        (declare (ignore key))
        (values rows (* row width) width)))))

(defun image-block-tokens (key width rows)
  "Hold ROWS (embedding rows, WIDTH each) for the picture KEY: its tokens."
  (let ((count (floor (length rows) width)))
    (assert (<= count +image-block-max+))
    (sb-thread:with-mutex (*image-lock*)
      (let ((number (ldb (byte 20 0) (sxhash key))))
        ;; another picture's number: the next free one
        (loop for b = (gethash number *image-blocks*)
              while (and b (string/= (car b) key))
              do (setf number (ldb (byte 20 0) (1+ number))))
        (setf (gethash number *image-blocks*) (list* key width rows))
        (setf *image-order* (cons number (remove number *image-order*)))
        (loop while (> (length *image-order*) *image-blocks-kept*)
              do (remhash (car (last *image-order*)) *image-blocks*)
                 (setf *image-order* (butlast *image-order*)))
        (loop for i below count collect (+ +image-token-base+ (* number +image-block-max+) i))))))

(defun image-tokens-held (key)
  "KEY's tokens if its rows are held still."
  (sb-thread:with-mutex (*image-lock*)
    (loop for number being the hash-keys of *image-blocks* using (hash-value b)
          when (string= (car b) key)
            do (setf *image-order* (cons number (remove number *image-order*)))
               (return (loop for i below (floor (length (cddr b)) (cadr b))
                             collect (+ +image-token-base+ (* number +image-block-max+) i))))))
