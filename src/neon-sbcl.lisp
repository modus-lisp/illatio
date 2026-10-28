;;;; neon-sbcl.lisp — the hot loops as NEON, for SBCL on arm64.
;;;
;;; Three kernels, each the portable one in tensor.lisp written as a NEON
;;; loop that gives the same bits:
;;;
;;;   %Q8-DOT-4ROWS  four rows of a Q8_0 matrix against one quantized vector.
;;;                  SDOT does sixteen int8 x int8 products per instruction,
;;;                  four lanes of four, added into int32s; a 32-element block
;;;                  is two SDOTs per row.  Three ADDPs fold four rows' sums
;;;                  into one vector, lane r = row r's exact block sum, and the
;;;                  convert, scale and accumulate run on all four lanes.  Each
;;;                  lane is Q8-ROW-DOT-AT's arithmetic for its row, in its
;;;                  order: float of the integer sum, times (dw*dx), added to
;;;                  the row total.
;;;   %F32-DOT16     attention's q.k, summed in F32-DOT16's order (sixteen
;;;                  strided partial sums and a fixed tree).
;;;   %F32-AXPY4     attention's out += w*v, lane-parallel.
;;;
;;; Never FMLA: a fused multiply-add rounds once where the portable code rounds
;;; twice, and then the two would not agree.  (mill keeps the same rule, for
;;; the same reason.)  consistency-test checks NEON against portable to the bit
;;; on whole forward passes; the portable code is the oracle.
;;;
;;; SBCL's arm64 assembler has no SDOT, and LD1's register pairs must be
;;; consecutive, which TN allocation does not promise, so the loops are
;;; instruction words, emitted four BYTEs at a time (its DWORD is 64 bits wide,
;;; and a word followed by four zero bytes is an illegal instruction).  A loop
;;; body is collected as a list and its backward branch is computed from the
;;; list's length, never counted by hand.  Every encoding was checked against
;;; the system assembler (as -arch arm64; objdump -d):
;;;
;;;   ldp   qA, qB, [xN], #32     acc10400 | B<<10 | N<<5 | A
;;;   ldr   qT, [xN], #16         3cc10400 | N<<5 | T
;;;   ldr   qT, [xN]              3dc00000 | N<<5 | T
;;;   str   qT, [xN], #16         3c810400 | N<<5 | T
;;;   str   qT, [xN]              3d800000 | N<<5 | T
;;;   ldr   sT, [xN], #4          bc404400 | N<<5 | T
;;;   movi  vD.2d, #0             6f00e400 | D            (bits 5-9 are the immediate)
;;;   dup   vD.4s, vN.s[0]        4e040400 | N<<5 | D
;;;   sdot  vD.4s, vN.16b, vM.16b 4e809400 | M<<16 | N<<5 | D
;;;   addp  vD.4s, vN.4s, vM.4s   4ea0bc00 | M<<16 | N<<5 | D
;;;   scvtf vD.4s, vN.4s          4e21d800 | N<<5 | D
;;;   fmul  vD.4s, vN.4s, vM.4s   6e20dc00 | M<<16 | N<<5 | D
;;;   fadd  vD.4s, vN.4s, vM.4s   4e20d400 | M<<16 | N<<5 | D
;;;   faddp vD.4s, vN.4s, vM.4s   6e20d400 | M<<16 | N<<5 | D
;;;   faddp sD, vN.2s             7e30d800 | N<<5 | D
;;;   subs  xD, xN, #1            f1000400 | N<<5 | D
;;;   b.ne  -K instructions       54000001 | (-K mod 2^19)<<5
;;;
;;; SDOT is FEAT_DotProd: every Apple M-series and the Pi 5's A76 have it, the
;;; Pi Zero 2's A53 does not.  NEON-DOT-P asks the machine; without it every
;;; caller runs the portable kernels.  This file defines VOPs that tensor.lisp
;;; and model.lisp use: SBCL inlines a VOP only into code compiled after it
;;; exists, so they are separate files, ordered by the .asd (as in
;;; secp256k1-fast).

(in-package #:illatio)

(sb-c:defknown %q8-dot-4rows
    (sb-sys:system-area-pointer (unsigned-byte 32) sb-sys:system-area-pointer
     sb-sys:system-area-pointer sb-sys:system-area-pointer (unsigned-byte 32)
     sb-sys:system-area-pointer)
    (values) ())

(sb-c:defknown %f32-dot16
    (sb-sys:system-area-pointer sb-sys:system-area-pointer (unsigned-byte 32))
    single-float (sb-c:flushable))

(sb-c:defknown %f32-axpy4
    (sb-sys:system-area-pointer sb-sys:system-area-pointer single-float (unsigned-byte 32))
    (values) ())

;;; Instruction words, from register numbers.
(defun a64 (base d &optional n m)
  (logior base d (if n (ash n 5) 0) (if m (ash m 16) 0)))
(defun a64-ldp-q (a b ptr) (logior #xacc10400 (ash b 10) (ash ptr 5) a))
(defun a64-bne-back (k) (logior #x54000001 (ash (ldb (byte 19 0) (- k)) 5)))

(in-package #:sb-vm)

(defmacro illatio::with-a64 ((&rest tns) &body body)
  "Bind each TN's register number over BODY, where (WORD x) emits an
   instruction word and (LOOP-WHILE-COUNT count words...) emits a loop body
   that runs until COUNT (decremented once per pass) reaches zero."
  `(let ,(mapcar (lambda (tn) `(,tn (tn-offset ,tn))) tns)
     (flet ((illatio::word (w) (dotimes (i 4) (inst byte (ldb (byte 8 (* 8 i)) w)))))
       (flet ((illatio::loop-while-count (count words)
                (dolist (w words) (illatio::word w))
                (illatio::word (illatio::a64 #xf1000400 count count))
                (illatio::word (illatio::a64-bne-back (1+ (length words))))))
         ,@body))))

(define-vop (illatio::%q8-dot-4rows)
  (:translate illatio::%q8-dot-4rows)
  (:policy :fast-safe)
  ;; Seven arguments and eight walking pointers do not fit in the integer
  ;; registers, so each argument that is only a starting value is TARGETed at
  ;; the temporary that walks from it, and may share its register.
  (:args (w :scs (sap-reg) :target p0) (stride :scs (unsigned-reg) :target p1)
         (x :scs (sap-reg) :target px) (ws4 :scs (sap-reg) :target pws)
         (xs :scs (sap-reg) :target pxs) (n :scs (unsigned-reg) :target count)
         (out :scs (sap-reg)))
  (:arg-types system-area-pointer unsigned-num system-area-pointer
              system-area-pointer system-area-pointer unsigned-num system-area-pointer)
  (:temporary (:sc unsigned-reg :from (:argument 0)) p0)
  (:temporary (:sc unsigned-reg :from (:argument 1)) p1)
  (:temporary (:sc unsigned-reg :from (:argument 2)) px)
  (:temporary (:sc unsigned-reg :from (:argument 3)) pws)
  (:temporary (:sc unsigned-reg :from (:argument 4)) pxs)
  (:temporary (:sc unsigned-reg :from (:argument 5)) count)
  (:temporary (:sc unsigned-reg) p2) (:temporary (:sc unsigned-reg) p3)
  (:temporary (:sc double-reg) xa) (:temporary (:sc double-reg) xb)
  (:temporary (:sc double-reg) a0) (:temporary (:sc double-reg) b0)
  (:temporary (:sc double-reg) a1) (:temporary (:sc double-reg) b1)
  (:temporary (:sc double-reg) a2) (:temporary (:sc double-reg) b2)
  (:temporary (:sc double-reg) a3) (:temporary (:sc double-reg) b3)
  (:temporary (:sc double-reg) c0) (:temporary (:sc double-reg) c1)
  (:temporary (:sc double-reg) c2) (:temporary (:sc double-reg) c3)
  (:temporary (:sc double-reg) t0) (:temporary (:sc double-reg) t1)
  (:temporary (:sc double-reg) dw) (:temporary (:sc double-reg) dx)
  (:temporary (:sc double-reg) sum)
  (:generator 60
    (move p0 w) (move px x) (move pws ws4) (move pxs xs) (move count n)
    (move p1 stride)
    (inst add p2 p1 p1) (inst add p3 p2 p1)       ; 2*stride, 3*stride
    (inst add p1 p1 p0) (inst add p2 p2 p0) (inst add p3 p3 p0)
    (illatio::with-a64 (p0 p1 p2 p3 px pws pxs count out xa xb a0 b0 a1 b1 a2 b2 a3 b3
                        c0 c1 c2 c3 t0 t1 dw dx sum)
      (illatio::word (illatio::a64 #x6f00e400 sum))                ; totals = 0
      (illatio::loop-while-count
       count
       (list (illatio::a64-ldp-q xa xb px)                          ; 32 activations, once
             (illatio::a64-ldp-q a0 b0 p0) (illatio::a64-ldp-q a1 b1 p1)
             (illatio::a64-ldp-q a2 b2 p2) (illatio::a64-ldp-q a3 b3 p3)
             (illatio::a64 #x6f00e400 c0) (illatio::a64 #x6f00e400 c1)
             (illatio::a64 #x6f00e400 c2) (illatio::a64 #x6f00e400 c3)
             (illatio::a64 #x4e809400 c0 a0 xa) (illatio::a64 #x4e809400 c0 b0 xb)
             (illatio::a64 #x4e809400 c1 a1 xa) (illatio::a64 #x4e809400 c1 b1 xb)
             (illatio::a64 #x4e809400 c2 a2 xa) (illatio::a64 #x4e809400 c2 b2 xb)
             (illatio::a64 #x4e809400 c3 a3 xa) (illatio::a64 #x4e809400 c3 b3 xb)
             (illatio::a64 #x4ea0bc00 t0 c0 c1) (illatio::a64 #x4ea0bc00 t1 c2 c3)
             (illatio::a64 #x4ea0bc00 t0 t0 t1)                     ; lane r: row r's int sum
             (illatio::a64 #x4e21d800 t0 t0)                        ; to float
             (illatio::a64 #x3cc10400 dw pws)                       ; the four row scales
             (illatio::a64 #xbc404400 dx pxs)                       ; the activation scale
             (illatio::a64 #x4e040400 dx dx)                        ; ... in every lane
             (illatio::a64 #x6e20dc00 dw dw dx)                     ; dw*dx
             (illatio::a64 #x6e20dc00 t0 t0 dw)                     ; float(sum) * (dw*dx)
             (illatio::a64 #x4e20d400 sum sum t0)))                 ; row totals
      (illatio::word (illatio::a64 #x3d800000 sum out)))))         ; four results out

(define-vop (illatio::%f32-dot16)
  (:translate illatio::%f32-dot16)
  (:policy :fast-safe)
  (:args (a :scs (sap-reg)) (b :scs (sap-reg)) (n :scs (unsigned-reg)))
  (:arg-types system-area-pointer system-area-pointer unsigned-num)
  (:results (sum :scs (single-reg)))
  (:result-types single-float)
  (:temporary (:sc unsigned-reg) pa) (:temporary (:sc unsigned-reg) pb)
  (:temporary (:sc unsigned-reg) count)
  (:temporary (:sc double-reg) a0) (:temporary (:sc double-reg) a1)
  (:temporary (:sc double-reg) a2) (:temporary (:sc double-reg) a3)
  (:temporary (:sc double-reg) b0) (:temporary (:sc double-reg) b1)
  (:temporary (:sc double-reg) b2) (:temporary (:sc double-reg) b3)
  (:temporary (:sc double-reg) k0) (:temporary (:sc double-reg) k1)
  (:temporary (:sc double-reg) k2) (:temporary (:sc double-reg) k3)
  (:generator 40
    (inst mov pa a) (inst mov pb b) (inst mov count n)
    (illatio::with-a64 (pa pb count sum a0 a1 a2 a3 b0 b1 b2 b3 k0 k1 k2 k3)
      (dolist (k (list k0 k1 k2 k3)) (illatio::word (illatio::a64 #x6f00e400 k)))
      (illatio::loop-while-count                                    ; 16 elements a pass
       count
       (list (illatio::a64-ldp-q a0 a1 pa) (illatio::a64-ldp-q a2 a3 pa)
             (illatio::a64-ldp-q b0 b1 pb) (illatio::a64-ldp-q b2 b3 pb)
             (illatio::a64 #x6e20dc00 a0 a0 b0) (illatio::a64 #x6e20dc00 a1 a1 b1)
             (illatio::a64 #x6e20dc00 a2 a2 b2) (illatio::a64 #x6e20dc00 a3 a3 b3)
             (illatio::a64 #x4e20d400 k0 k0 a0) (illatio::a64 #x4e20d400 k1 k1 a1)
             (illatio::a64 #x4e20d400 k2 k2 a2) (illatio::a64 #x4e20d400 k3 k3 a3)))
      ;; ((k0+k1)+(k2+k3)) lane-wise, then (l0+l1)+(l2+l3)
      (illatio::word (illatio::a64 #x4e20d400 k0 k0 k1))
      (illatio::word (illatio::a64 #x4e20d400 k2 k2 k3))
      (illatio::word (illatio::a64 #x4e20d400 k0 k0 k2))
      (illatio::word (illatio::a64 #x6e20d400 k0 k0 k0))
      (illatio::word (illatio::a64 #x7e30d800 sum k0)))))

(define-vop (illatio::%f32-axpy4)
  (:translate illatio::%f32-axpy4)
  (:policy :fast-safe)
  (:args (y :scs (sap-reg)) (x :scs (sap-reg)) (w :scs (single-reg)) (n :scs (unsigned-reg)))
  (:arg-types system-area-pointer system-area-pointer single-float unsigned-num)
  (:temporary (:sc unsigned-reg) py) (:temporary (:sc unsigned-reg) px)
  (:temporary (:sc unsigned-reg) count)
  (:temporary (:sc double-reg) wv) (:temporary (:sc double-reg) xv)
  (:temporary (:sc double-reg) yv)
  (:generator 20
    (inst mov py y) (inst mov px x) (inst mov count n)
    (illatio::with-a64 (py px count w wv xv yv)
      (illatio::word (illatio::a64 #x4e040400 wv w))                ; w in every lane
      (illatio::loop-while-count                                    ; 4 elements a pass
       count
       (list (illatio::a64 #x3cc10400 xv px)                        ; xv <- x, x += 16
             (illatio::a64 #x3dc00000 yv py)                        ; yv <- y
             (illatio::a64 #x6e20dc00 xv xv wv)                     ; w*x
             (illatio::a64 #x4e20d400 yv yv xv)                     ; y + w*x
             (illatio::a64 #x3c810400 yv py))))))                   ; y <- yv, y += 16

(in-package #:illatio)

(defun cpu-has-dotprod-p ()
  "FEAT_DotProd, asked of the OS: sysctl on macOS, /proc/cpuinfo on Linux."
  (or (ignore-errors
       (string= "1" (string-trim '(#\Newline #\Space)
                                 (with-output-to-string (o)
                                   (sb-ext:run-program "/usr/sbin/sysctl"
                                                       '("-n" "hw.optional.arm.FEAT_DotProd")
                                                       :output o :error nil)))))
      (ignore-errors
       (with-open-file (s "/proc/cpuinfo")
         (loop for line = (read-line s nil) while line
               thereis (and (eql 0 (search "Features" line))
                            (search " asimddp" line)))))))

(defvar *neon-dot* :unknown
  "T when the SDOT kernel can run here.  Decided on first use, not at load
   time, so a core saved on one machine asks again on the next.")

(declaim (inline neon-dot-p))
(defun neon-dot-p ()
  (if (eq *neon-dot* :unknown)
      (setf *neon-dot* (and (cpu-has-dotprod-p) t))
      *neon-dot*))
