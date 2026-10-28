;;;; half.lisp — IEEE binary16 <-> single-float, bit for bit.
;;;
;;; ggml stores block scales as fp16, and the rounding of a scale into fp16
;;; is part of the arithmetic we have to reproduce, not a storage detail.

(in-package #:illatio)

(declaim (inline half-to-single single-to-half bits-single single-bits))

(defun bits-single (bits)
  (declare (type (unsigned-byte 32) bits))
  (sb-kernel:make-single-float (if (logbitp 31 bits)
                                   (- bits (ash 1 32))
                                   bits)))

(defun single-bits (x)
  (declare (type single-float x))
  (ldb (byte 32 0) (sb-kernel:single-float-bits x)))

(defun half-to-single (h)
  "Decode the binary16 bit pattern H."
  (declare (type (unsigned-byte 16) h))
  (let ((sign (ash (ldb (byte 1 15) h) 31))
        (exp (ldb (byte 5 10) h))
        (man (ldb (byte 10 0) h)))
    (bits-single
     (cond ((= exp 31)                  ; inf / nan
            (logior sign #x7f800000 (ash man 13)))
           ((/= exp 0)                  ; normal
            (logior sign (ash (+ exp 112) 23) (ash man 13)))
           ((= man 0) sign)             ; +-0
           (t                           ; subnormal: renormalise
            (let ((e 113))
              (loop until (logbitp 10 man)
                    do (setf man (ash man 1)) (decf e))
              (logior sign (ash e 23) (ash (ldb (byte 10 0) man) 13))))))))

(defun single-to-half (x)
  "Round X to the nearest binary16, ties to even (what vcvt and F16C do)."
  (declare (type single-float x))
  (let* ((b (single-bits x))
         (sign (ash (ldb (byte 1 31) b) 15))
         (exp (ldb (byte 8 23) b))
         (man (ldb (byte 23 0) b)))
    (cond ((= exp 255)
           (logior sign #x7c00 (if (zerop man) 0 #x200)))
          (t
           ;; Work on the full significand with an implicit bit, then shift
           ;; it into place with round-to-nearest-even.
           (let* ((e (- exp 127))         ; unbiased
                  (sig (if (zerop exp) man (logior man #x800000))))
             (cond
               ((> e 15) (logior sign #x7c00))
               (t
                ;; target: half exponent field he = e+15; if he<=0 subnormal
                (let* ((he (+ e 15))
                       (shift (if (> he 0) 13 (+ 13 (- 1 he))))
                       (shift (if (zerop exp) (max shift 14) shift)))
                  (if (> shift 24)
                      sign
                      (let* ((q (ash sig (- shift)))
                             (rem (ldb (byte shift 0) sig))
                             (half (ash 1 (1- shift))))
                        (when (or (> rem half)
                                  (and (= rem half) (logbitp 0 q)))
                          (incf q))
                        ;; q includes the implicit bit for normals; adding the
                        ;; exponent field by addition lets a mantissa carry
                        ;; ripple into the exponent, as it must.
                        (logior sign
                                (if (> he 0)
                                    (+ (ash (1- he) 10) q)
                                    q))))))))))))
