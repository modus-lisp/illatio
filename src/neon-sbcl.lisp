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
;;;   ldp   qA, qB, [xN], #32     acc10000 | B<<10 | N<<5 | A
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
;;; and for the K-quant kernels:
;;;   fmul / fsub / fadd sD,sN,sM 1e200800 / 1e203800 / 1e202800 | M<<16 | N<<5 | D
;;;   uxtl  vD.8h, vN.8b          2f08a400 | N<<5 | D     (uxtl2 .16b: 6f08a400)
;;;   uxtl  vD.4s, vN.4h          2f10a400 | N<<5 | D     (uxtl2 .8h:  6f10a400)
;;;   sxtl  vD.8h, vN.8b          0f08a400 | N<<5 | D     (sxtl2 .16b: 4f08a400)
;;;   sxtl  vD.4s, vN.4h          0f10a400 | N<<5 | D     (sxtl2 .8h:  4f10a400)
;;;   mul / mla vD.4s,vN.4s,vM.4s 4ea09c00 / 4ea09400 | M<<16 | N<<5 | D
;;;   mla vD.4s, vN.4s, vM.s[i]   6f800000 | M<<16 | N<<5 | D, +L (bit 21) for i odd, +H (bit 11) for i >= 2
;;;   sub vD.4s, vN.4s, vM.4s     6ea08400 | M<<16 | N<<5 | D
;;;   shl vD.4s, vN.4s, #5        4f255400 | N<<5 | D
;;;   addv  sD, vN.4s             4eb1b800 | N<<5 | D
;;;   scvtf sD, sN                5e21d800 | N<<5 | D
;;;   and / orr vD.16b            4e201c00 / 4ea01c00 | M<<16 | N<<5 | D
;;;   ushr vD.16b, vN.16b, #2/4/6 6f0e0400 / 6f0c0400 / 6f0a0400 | N<<5 | D
;;;   shl  vD.16b, vN.16b, #4     4f0c5400 | N<<5 | D
;;;   movi vD.16b, #15 / #3       4f00e5e0 / 4f00e460 | D
;;;   ldr  xT, [xN, #8k]          f9400000 | k<<10 | N<<5 | T
;;; and for the batched Q6_K kernel (FEAT_I8MM):
;;;   smmla vD.4s, vN.16b, vM.16b 4e80a400 | M<<16 | N<<5 | D
;;;   zip1 / zip2 vD.2d           4ec03800 / 4ec07800 | M<<16 | N<<5 | D
;;;   mov  vD.d[1], vN.d[1]       6e184400 | N<<5 | D
;;;   sub  vD.16b, vN.16b, vM.16b 6e208400 | M<<16 | N<<5 | D
;;;   movi vD.16b, #32            4f01e400 | D
;;;   str  qT, [xN, #16]          3d800400 | N<<5 | T
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

;;; The K-quant kernels walk six arrays, more pointers than SBCL can keep in
;;; argument registers beside six walking copies, so they take one: DESC, six
;;; words holding the arrays' addresses, loaded into the walkers at entry.
(sb-c:defknown %q4k-dot (sb-sys:system-area-pointer (unsigned-byte 32))
    single-float (sb-c:flushable))

(sb-c:defknown %q6k-dot (sb-sys:system-area-pointer (unsigned-byte 32))
    single-float (sb-c:flushable))

;;; Two Q6_K rows against four tokens (two interleaved pairs, see PACK-QUADS!):
;;; DESC holds both rows' quants, scales and d, the packed activations and
;;; their scales, and where the eight results go.
(sb-c:defknown %q6k-dot-2x4 (sb-sys:system-area-pointer (unsigned-byte 32))
    (values) ())

(sb-c:defknown %f32-axpy4
    (sb-sys:system-area-pointer sb-sys:system-area-pointer single-float (unsigned-byte 32))
    (values) ())

;;; Instruction words, from register numbers.
(defun a64 (base d &optional n m)
  (logior base d (if n (ash n 5) 0) (if m (ash m 16) 0)))
(defun a64-ldp-q (a b ptr)
  ;; acc10000, not acc10400: bit 10 is the low bit of Rt2, so the old base
  ;; loaded an even B into B+1 -- harmless only while every pair's second
  ;; register happened to be odd (it was, until the Q6_K kernel)
  (logior #xacc10000 (ash b 10) (ash ptr 5) a))
(defun a64-bne-back (k) (logior #x54000001 (ash (ldb (byte 19 0) (- k)) 5)))
(defun a64-mla-lane (d n m i)
  "mla vD.4s, vN.4s, vM.s[I]"
  (logior #x6f800000 (if (oddp i) (ash 1 21) 0) (if (>= i 2) (ash 1 11) 0) (ash m 16) (ash n 5) d))

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

;;; Q4_K against Q8_K, one row: Q4K-ROW-DOT's arithmetic, in its order.  Per
;;; super-block: the eight mins times the activation's eight 32-sums
;;; (SUM -= dmin*that), then each 64 weights' low and high nibbles SDOTted with
;;; their 32 activations, each group's dot multiplied by its scale into one
;;; int32x4 (SUM += d*its lanes' total).  QS, SC (8 scales, 8 mins a block), DD
;;; (d, dmin), YQ, YD (one per block) and YB (eight 32-sums a block) all walk.
(define-vop (illatio::%q4k-dot)
  (:translate illatio::%q4k-dot)
  (:policy :fast-safe)
  (:args (desc :scs (sap-reg) :target pd) (nb :scs (unsigned-reg) :target count))
  (:arg-types system-area-pointer unsigned-num)
  (:results (sum :scs (single-reg)))
  (:result-types single-float)
  (:temporary (:sc unsigned-reg) pqs)
  (:temporary (:sc unsigned-reg) psc)
  (:temporary (:sc unsigned-reg) pdd)
  (:temporary (:sc unsigned-reg) pyq)
  (:temporary (:sc unsigned-reg) pyd)
  (:temporary (:sc unsigned-reg) pyb)
  ;; DESC and NB live in these for the whole VOP: an argument's own register
  ;; is free for a temporary once SBCL has loaded it
  (:temporary (:sc unsigned-reg :from (:argument 0)) pd)
  (:temporary (:sc unsigned-reg :from (:argument 1)) count)
  (:temporary (:sc double-reg) mask) (:temporary (:sc double-reg) sm)
  (:temporary (:sc double-reg) wa) (:temporary (:sc double-reg) wb)
  (:temporary (:sc double-reg) s0) (:temporary (:sc double-reg) s1)
  (:temporary (:sc double-reg) m0) (:temporary (:sc double-reg) m1)
  (:temporary (:sc double-reg) b0) (:temporary (:sc double-reg) b1)
  (:temporary (:sc double-reg) tt) (:temporary (:sc double-reg) acc)
  (:temporary (:sc double-reg) qa) (:temporary (:sc double-reg) qb)
  (:temporary (:sc double-reg) y0) (:temporary (:sc double-reg) y1)
  (:temporary (:sc double-reg) y2) (:temporary (:sc double-reg) y3)
  (:temporary (:sc double-reg) l0) (:temporary (:sc double-reg) l1)
  (:temporary (:sc double-reg) dl) (:temporary (:sc double-reg) dh)
  (:temporary (:sc double-reg) ydi) (:temporary (:sc double-reg) dx)
  (:temporary (:sc double-reg) dmx)
  (:generator 80
    (move pd desc) (move count nb)
    (illatio::with-a64 (pqs psc pdd pyq pyd pyb count sum mask sm wa wb s0 s1 m0 m1 b0 b1
                        tt acc qa qb y0 y1 y2 y3 l0 l1 dl dh ydi dx dmx pd)
      (illatio::word (illatio::a64 (logior #xf9400000 (ash 0 10)) pqs pd))
      (illatio::word (illatio::a64 (logior #xf9400000 (ash 1 10)) psc pd))
      (illatio::word (illatio::a64 (logior #xf9400000 (ash 2 10)) pdd pd))
      (illatio::word (illatio::a64 (logior #xf9400000 (ash 3 10)) pyq pd))
      (illatio::word (illatio::a64 (logior #xf9400000 (ash 4 10)) pyd pd))
      (illatio::word (illatio::a64 (logior #xf9400000 (ash 5 10)) pyb pd))
      (illatio::word (illatio::a64 #x6f00e400 sum))                 ; sum = 0
      (illatio::word (illatio::a64 #x4f00e5e0 mask))                ; 0x0f in every byte
      (illatio::loop-while-count
       count
       (append
        (list (illatio::a64 #xbc404400 ydi pyd)                      ; y.d
              (illatio::a64 #xbc404400 dx pdd)                       ; d
              (illatio::a64 #xbc404400 dmx pdd)                      ; dmin
              (illatio::a64 #x1e200800 dx ydi dx)                    ; y.d*d
              (illatio::a64 #x1e200800 dmx ydi dmx)                  ; y.d*dmin
              (illatio::a64 #x3cc10400 sm psc)                       ; 8 scales, 8 mins
              (illatio::a64 #x2f08a400 wa sm) (illatio::a64 #x6f08a400 wb sm)
              (illatio::a64 #x2f10a400 s0 wa) (illatio::a64 #x6f10a400 s1 wa)
              (illatio::a64 #x2f10a400 m0 wb) (illatio::a64 #x6f10a400 m1 wb)
              (illatio::a64-ldp-q b0 b1 pyb)                         ; eight 32-sums
              (illatio::a64 #x4ea09c00 tt m0 b0) (illatio::a64 #x4ea09400 tt m1 b1)
              (illatio::a64 #x4eb1b800 tt tt)                        ; imin
              (illatio::a64 #x5e21d800 tt tt)
              (illatio::a64 #x1e200800 tt dmx tt)
              (illatio::a64 #x1e203800 sum sum tt)                   ; sum -= dmin*imin
              (illatio::a64 #x6f00e400 acc))
        (loop for k below 4
              for sv = (if (< k 2) s0 s1)
              for lane = (* 2 (mod k 2))
              append (list (illatio::a64-ldp-q qa qb pqs)            ; 32 bytes: 64 weights
                           (illatio::a64-ldp-q y0 y1 pyq) (illatio::a64-ldp-q y2 y3 pyq)
                           (illatio::a64 #x4e201c00 l0 qa mask) (illatio::a64 #x4e201c00 l1 qb mask)
                           (illatio::a64 #x6f0c0400 qa qa) (illatio::a64 #x6f0c0400 qb qb)
                           (illatio::a64 #x6f00e400 dl)
                           (illatio::a64 #x4e809400 dl l0 y0) (illatio::a64 #x4e809400 dl l1 y1)
                           (illatio::a64 #x6f00e400 dh)
                           (illatio::a64 #x4e809400 dh qa y2) (illatio::a64 #x4e809400 dh qb y3)
                           (illatio::a64-mla-lane acc dl sv lane)
                           (illatio::a64-mla-lane acc dh sv (1+ lane))))
        (list (illatio::a64 #x4eb1b800 acc acc)                      ; isum
              (illatio::a64 #x5e21d800 acc acc)
              (illatio::a64 #x1e200800 acc dx acc)
              (illatio::a64 #x1e202800 sum sum acc)))))))            ; sum += d*isum

;;; Q6_K against Q8_K, one row: Q6K-ROW-DOT's arithmetic.  The 6-bit quants
;;; are rebuilt 0..63 (four low nibbles, two high bits from QH), SDOTted in
;;; groups of 16, each times its signed scale; the activation's sixteen
;;; 16-sums times the scales give the -32 offset, taken off lane-wise before
;;; the one across-lanes add.  SUM += (d*y.d) * that.
(define-vop (illatio::%q6k-dot)
  (:translate illatio::%q6k-dot)
  (:policy :fast-safe)
  (:args (desc :scs (sap-reg) :target pd) (nb :scs (unsigned-reg) :target count))
  (:arg-types system-area-pointer unsigned-num)
  (:results (sum :scs (single-reg)))
  (:result-types single-float)
  (:temporary (:sc unsigned-reg) pq)
  (:temporary (:sc unsigned-reg) psc)
  (:temporary (:sc unsigned-reg) pdd)
  (:temporary (:sc unsigned-reg) pyq)
  (:temporary (:sc unsigned-reg) pyd)
  (:temporary (:sc unsigned-reg) pyb)
  ;; DESC and NB live in these for the whole VOP: an argument's own register
  ;; is free for a temporary once SBCL has loaded it
  (:temporary (:sc unsigned-reg :from (:argument 0)) pd)
  (:temporary (:sc unsigned-reg :from (:argument 1)) count)
  (:temporary (:sc double-reg) m4) (:temporary (:sc double-reg) m3)
  (:temporary (:sc double-reg) scb) (:temporary (:sc double-reg) wa)
  (:temporary (:sc double-reg) wb)
  (:temporary (:sc double-reg) sc0) (:temporary (:sc double-reg) sc1)
  (:temporary (:sc double-reg) sc2) (:temporary (:sc double-reg) sc3)
  (:temporary (:sc double-reg) tt) (:temporary (:sc double-reg) acc)
  (:temporary (:sc double-reg) h0) (:temporary (:sc double-reg) h1)
  (:temporary (:sc double-reg) q0) (:temporary (:sc double-reg) q1)
  (:temporary (:sc double-reg) q2) (:temporary (:sc double-reg) q3)
  (:temporary (:sc double-reg) hb) (:temporary (:sc double-reg) v)
  (:temporary (:sc double-reg) y0) (:temporary (:sc double-reg) y1)
  (:temporary (:sc double-reg) y2) (:temporary (:sc double-reg) y3)
  (:temporary (:sc double-reg) dt) (:temporary (:sc double-reg) ydi)
  (:temporary (:sc double-reg) dq)
  (:generator 100
    (move pd desc) (move count nb)
    (illatio::with-a64 (pq psc pdd pyq pyd pyb count sum m4 m3 scb wa wb sc0 sc1 sc2 sc3
                        tt acc h0 h1 q0 q1 q2 q3 hb v y0 y1 y2 y3 dt ydi dq pd)
      (illatio::word (illatio::a64 (logior #xf9400000 (ash 0 10)) pq pd))
      (illatio::word (illatio::a64 (logior #xf9400000 (ash 1 10)) psc pd))
      (illatio::word (illatio::a64 (logior #xf9400000 (ash 2 10)) pdd pd))
      (illatio::word (illatio::a64 (logior #xf9400000 (ash 3 10)) pyq pd))
      (illatio::word (illatio::a64 (logior #xf9400000 (ash 4 10)) pyd pd))
      (illatio::word (illatio::a64 (logior #xf9400000 (ash 5 10)) pyb pd))
      (illatio::word (illatio::a64 #x6f00e400 sum))
      (illatio::word (illatio::a64 #x4f00e5e0 m4))                  ; 0x0f
      (illatio::word (illatio::a64 #x4f00e460 m3))                  ; 0x03
      (flet ((group (lowfn qreg hreg hshift scv lane)
               ;; one 16-weight group: its quants 0..63 into V, dotted with Y, times its scale
               (append
                (if (zerop hshift)
                    (list (illatio::a64 #x4e201c00 hb hreg m3))
                    (list (illatio::a64 (ecase hshift (2 #x6f0e0400) (4 #x6f0c0400) (6 #x6f0a0400)) hb hreg)
                          (illatio::a64 #x4e201c00 hb hb m3)))
                (list (illatio::a64 #x4f0c5400 hb hb))               ; << 4
                (if (eq lowfn :lo)
                    (list (illatio::a64 #x4e201c00 v qreg m4))
                    (list (illatio::a64 #x6f0c0400 v qreg)))
                (list (illatio::a64 #x4ea01c00 v v hb))))
             (dot (yreg scv lane)
               (list (illatio::a64 #x6f00e400 dt)
                     (illatio::a64 #x4e809400 dt v yreg)
                     (illatio::a64-mla-lane acc dt scv lane))))
        (illatio::loop-while-count
         count
         (append
          (list (illatio::a64 #xbc404400 ydi pyd)
                (illatio::a64 #xbc404400 dq pdd)
                (illatio::a64 #x1e200800 dq dq ydi)                  ; d*y.d
                (illatio::a64 #x3cc10400 scb psc)                    ; 16 signed scales
                (illatio::a64 #x0f08a400 wa scb) (illatio::a64 #x4f08a400 wb scb)
                (illatio::a64 #x0f10a400 sc0 wa) (illatio::a64 #x4f10a400 sc1 wa)
                (illatio::a64 #x0f10a400 sc2 wb) (illatio::a64 #x4f10a400 sc3 wb)
                (illatio::a64-ldp-q y0 y1 pyb) (illatio::a64-ldp-q y2 y3 pyb) ; sixteen 16-sums
                (illatio::a64 #x4ea09c00 tt sc0 y0) (illatio::a64 #x4ea09400 tt sc1 y1)
                (illatio::a64 #x4ea09400 tt sc2 y2) (illatio::a64 #x4ea09400 tt sc3 y3)

                (illatio::a64 #x6f00e400 acc))
          (loop for c below 2
                for sa = (if (zerop c) sc0 sc2)
                for sb = (if (zerop c) sc1 sc3)
                append (append
                        (list (illatio::a64-ldp-q h0 h1 pq)           ; 32 bytes of high bits
                              (illatio::a64-ldp-q q0 q1 pq) (illatio::a64-ldp-q q2 q3 pq)
                              (illatio::a64-ldp-q y0 y1 pyq) (illatio::a64-ldp-q y2 y3 pyq))
                        ;; low nibbles: weights 0-15 16-31 32-47 48-63 of this half
                        (group :lo q0 h0 0 sa 0) (dot y0 sa 0)
                        (group :lo q1 h1 0 sa 1) (dot y1 sa 1)
                        (group :lo q2 h0 2 sa 2) (dot y2 sa 2)
                        (group :lo q3 h1 2 sa 3) (dot y3 sa 3)
                        (list (illatio::a64-ldp-q y0 y1 pyq) (illatio::a64-ldp-q y2 y3 pyq))
                        ;; high nibbles: weights 64-127
                        (group :hi q0 h0 4 sb 0) (dot y0 sb 0)
                        (group :hi q1 h1 4 sb 1) (dot y1 sb 1)
                        (group :hi q2 h0 6 sb 2) (dot y2 sb 2)
                        (group :hi q3 h1 6 sb 3) (dot y3 sb 3)))
          (list (illatio::a64 #x4f255400 tt tt)                      ; 32*imins, lane-wise
                (illatio::a64 #x6ea08400 acc acc tt)                 ; isum - 32*imins
                (illatio::a64 #x4eb1b800 acc acc)
                (illatio::a64 #x5e21d800 acc acc)
                (illatio::a64 #x1e200800 acc dq acc)
                (illatio::a64 #x1e202800 sum sum acc))))))))

;;; Q6_K, two rows (A, B) against four tokens, for a batch: each 16-weight
;;; group is rebuilt once per row and used for all four tokens.  The quants
;;; are rebuilt as q-32 (signed), so no activation sums are needed: the
;;; group's integer dot is the same integer the one-row kernel gets by
;;; subtracting 32 times them.  ZIP puts A's and B's halves side by side, and
;;; SMMLA multiplies that 2x8 by a token pair's 2x8 (PACK-QUADS! lays them
;;; out so), giving [A.t0 A.t1 B.t0 B.t1] for the group.  Each row's scale
;;; multiplies its two lanes (an accumulator for A's scales, one for B's, the
;;; halves rejoined at the end).  Per super-block and (row, token) it is then
;;; the one-row kernel's float arithmetic: SUM += (d*y.d) * float(isum).
(define-vop (illatio::%q6k-dot-2x4)
  (:translate illatio::%q6k-dot-2x4)
  (:policy :fast-safe)
  (:args (desc :scs (sap-reg) :target pd) (nb :scs (unsigned-reg) :target count))
  (:arg-types system-area-pointer unsigned-num)
  (:temporary (:sc unsigned-reg) pqa) (:temporary (:sc unsigned-reg) pqb)
  (:temporary (:sc unsigned-reg) psa) (:temporary (:sc unsigned-reg) psb)
  (:temporary (:sc unsigned-reg) pda) (:temporary (:sc unsigned-reg) pdb)
  (:temporary (:sc unsigned-reg) px) (:temporary (:sc unsigned-reg) py)
  (:temporary (:sc unsigned-reg :from (:argument 0)) pd)
  (:temporary (:sc unsigned-reg :from (:argument 1)) count)
  (:temporary (:sc double-reg) m4) (:temporary (:sc double-reg) m3)
  (:temporary (:sc double-reg) c32)
  (:temporary (:sc double-reg) ha0) (:temporary (:sc double-reg) ha1)
  (:temporary (:sc double-reg) hb0) (:temporary (:sc double-reg) hb1)
  (:temporary (:sc double-reg) qa) (:temporary (:sc double-reg) qb)
  (:temporary (:sc double-reg) sba) (:temporary (:sc double-reg) sbb)
  (:temporary (:sc double-reg) saa) (:temporary (:sc double-reg) sab)
  (:temporary (:sc double-reg) sbaa) (:temporary (:sc double-reg) sbab)
  (:temporary (:sc double-reg) a0a) (:temporary (:sc double-reg) a0b)
  (:temporary (:sc double-reg) a1a) (:temporary (:sc double-reg) a1b)
  (:temporary (:sc double-reg) s0) (:temporary (:sc double-reg) s1)
  (:temporary (:sc double-reg) va) (:temporary (:sc double-reg) vb)
  (:temporary (:sc double-reg) hb)
  (:temporary (:sc double-reg) w1) (:temporary (:sc double-reg) w2)
  (:temporary (:sc double-reg) x1) (:temporary (:sc double-reg) x2)
  (:temporary (:sc double-reg) dd)
  (:generator 300
    (move pd desc) (move count nb)
    (illatio::with-a64 (pqa pqb psa psb pda pdb px py pd count m4 m3 c32 ha0 ha1 hb0 hb1 qa qb
                        sba sbb saa sab sbaa sbab a0a a0b a1a a1b s0 s1 va vb hb w1 w2 x1 x2 dd)
      (loop for r in (list pqa pqb psa psb pda pdb px py) for k from 0
            do (illatio::word (illatio::a64 (logior #xf9400000 (ash k 10)) r pd)))
      (illatio::word (illatio::a64 #x6f00e400 s0))
      (illatio::word (illatio::a64 #x6f00e400 s1))
      (illatio::word (illatio::a64 #x4f00e5e0 m4))                  ; 0x0f
      (illatio::word (illatio::a64 #x4f00e460 m3))                  ; 0x03
      (illatio::word (illatio::a64 #x4f01e400 c32))                 ; 32
      (flet ((rebuild (v q h shift hi)
               ;; V = (low or high nibble of Q) | (2 bits of H at SHIFT) << 4, minus 32
               (append
                (if (zerop shift)
                    (list (illatio::a64 #x4e201c00 hb h m3))
                    (list (illatio::a64 (ecase shift (2 #x6f0e0400) (4 #x6f0c0400) (6 #x6f0a0400)) hb h)
                          (illatio::a64 #x4e201c00 hb hb m3)))
                (list (illatio::a64 #x4f0c5400 hb hb)
                      (if hi (illatio::a64 #x6f0c0400 v q) (illatio::a64 #x4e201c00 v q m4))
                      (illatio::a64 #x4ea01c00 v v hb)
                      (illatio::a64 #x6e208400 v v c32)))))
        (illatio::loop-while-count
         count
         (append
          (list (illatio::a64 #x3cc10400 sba psa)                    ; 16 signed scales, each row
                (illatio::a64 #x3cc10400 sbb psb)
                (illatio::a64 #x6f00e400 a0a) (illatio::a64 #x6f00e400 a0b)
                (illatio::a64 #x6f00e400 a1a) (illatio::a64 #x6f00e400 a1b))
          (loop for c below 2
                append (append
                        ;; this half's scales as int32: groups 0-3 (low nibbles), 4-7 (high)
                        (list (illatio::a64 (if (zerop c) #x0f08a400 #x4f08a400) va sba)
                              (illatio::a64 #x0f10a400 saa va) (illatio::a64 #x4f10a400 sab va)
                              (illatio::a64 (if (zerop c) #x0f08a400 #x4f08a400) vb sbb)
                              (illatio::a64 #x0f10a400 sbaa vb) (illatio::a64 #x4f10a400 sbab vb)
                              (illatio::a64-ldp-q ha0 ha1 pqa) (illatio::a64-ldp-q hb0 hb1 pqb))
                        (loop for x below 4
                              append (append
                                      (list (illatio::a64 #x3cc10400 qa pqa) (illatio::a64 #x3cc10400 qb pqb))
                                      (loop for hi in '(nil t)
                                            for shift = (+ (* 2 (floor x 2)) (if hi 4 0))
                                            append (append
                                                    (rebuild va qa (if (evenp x) ha0 ha1) shift hi)
                                                    (rebuild vb qb (if (evenp x) hb0 hb1) shift hi)
                                                    (list (illatio::a64 #x4ec03800 w1 va vb)
                                                          (illatio::a64 #x4ec07800 w2 va vb))
                                                    (loop for (acca accb) in (list (list a0a a0b) (list a1a a1b))
                                                          append (list (illatio::a64-ldp-q x1 x2 px)
                                                                       (illatio::a64 #x6f00e400 dd)
                                                                       (illatio::a64 #x4e80a400 dd w1 x1)
                                                                       (illatio::a64 #x4e80a400 dd w2 x2)
                                                                       (illatio::a64-mla-lane acca dd (if hi sab saa) x)
                                                                       (illatio::a64-mla-lane accb dd (if hi sbab sbaa) x)))))))))
          ;; the super-block's floats: [dA dA dB dB] * [y.d of the pair], then per lane
          (list (illatio::a64 #xbc404400 va pda) (illatio::a64 #xbc404400 vb pdb)
                (illatio::a64 #x4e040400 va va) (illatio::a64 #x4e040400 vb vb)
                (illatio::a64 #x4ec03800 hb va vb)
                (illatio::a64-ldp-q x1 x2 py)
                (illatio::a64 #x6e20dc00 w1 hb x1) (illatio::a64 #x6e20dc00 w2 hb x2)
                (illatio::a64 #x6e184400 a0a a0b) (illatio::a64 #x6e184400 a1a a1b)
                (illatio::a64 #x4e21d800 a0a a0a) (illatio::a64 #x4e21d800 a1a a1a)
                (illatio::a64 #x6e20dc00 a0a w1 a0a) (illatio::a64 #x6e20dc00 a1a w2 a1a)
                (illatio::a64 #x4e20d400 s0 s0 a0a) (illatio::a64 #x4e20d400 s1 s1 a1a)))))
      (illatio::word (illatio::a64 (logior #xf9400000 (ash 8 10)) pqa pd)) ; the results' address
      (illatio::word (illatio::a64 #x3d800000 s0 pqa))
      (illatio::word (illatio::a64 #x3d800400 s1 pqa)))))

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

(defvar *neon-mmla* :unknown
  "T when SMMLA (FEAT_I8MM: Apple M2 and later) can run here; asked on first use.")

(defun neon-mmla-p ()
  "Whether the batched Q6_K kernel can run: SDOT's kernels are on, and SMMLA is here."
  (and (neon-dot-p)
       (if (eq *neon-mmla* :unknown)
           (setf *neon-mmla*
                 (and (or (ignore-errors
                           (string= "1" (string-trim '(#\Newline #\Space)
                                                     (with-output-to-string (o)
                                                       (sb-ext:run-program "/usr/sbin/sysctl"
                                                                           '("-n" "hw.optional.arm.FEAT_I8MM")
                                                                           :output o :error nil)))))
                          (ignore-errors
                           (with-open-file (s "/proc/cpuinfo")
                             (loop for line = (read-line s nil) while line
                                   thereis (and (eql 0 (search "Features" line))
                                                (search " i8mm" line))))))
                      t))
           *neon-mmla*)))
