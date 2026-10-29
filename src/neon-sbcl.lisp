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
;;;   mov  xD, xN                 aa0003e0 | N<<16 | D
;;;   stp  qA, qB, [xN], #32      ac810000 | B<<10 | N<<5 | A
;;;   str  qT, [xN], #16          3c810400 | N<<5 | T
;;;   dup  vD.4s, vN.s[i]         4e040400 | i<<19 | N<<5 | D
;;;   mla  vD.4s, vN.4s, vM.4s    4ea09400 | M<<16 | N<<5 | D
;;;   movi vD.16b, #0x30          4f01e600 | D
;;; and for the delta rule:
;;;   ldr  sT, [xN, #4k]          bd400000 | k<<10 | N<<5 | T
;;;   ldr  sT, [xN, xM]           bc606800 | M<<16 | N<<5 | T
;;;   add  xD, xN, #4             91001000 | N<<5 | D
;;;   add  xD, xN, xM             8b000000 | M<<16 | N<<5 | D
;;;   ldp / stp qA, qB, [xN, #16k] ad400000 / ad000000 | k<<15 | B<<10 | N<<5 | A
;;;   fsub vD.4s, vN.4s, vM.4s    4ea0d400 | M<<16 | N<<5 | D
;;; and for the IQ2_XXS / Q2_K expert rows:
;;;   ldr  hT, [xN], #2           7c402400 | N<<5 | T
;;;   fcvt sD, hN                 1ee24000 | N<<5 | D
;;;   ldr  wT, [xN], #4           b8404400 | N<<5 | T
;;;   ubfx xD, xN, #L, #W         d3400000 | L<<16 | (L+W-1)<<10 | N<<5 | D
;;;   orr  xD, xN, xM, lsl #S     aa000000 | M<<16 | S<<10 | N<<5 | D
;;;   ldr  dT, [xN, xM, lsl #3]   fc607800 | M<<16 | N<<5 | T
;;;   mov  vD.d[1], vN.d[0]       6e180400 | N<<5 | D
;;;   lsr  wD, wN, #28            531c7c00 | N<<5 | D
;;;   scvtf sD, wN                1e220000 | N<<5 | D
;;;   fmov sD, #0.5 / #0.25       1e2c1000 / 1e2a1000 | D
;;;   movi dD, #0                 2f00e400 | D
;;;   fneg sD, sN                 1e214000 | N<<5 | D
;;;   shl  vD.16b, vN.16b, #2     4f0a5400 | N<<5 | D     (ushr #2: 6f0e0400)
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

;;; A Q6_K row pair rebuilt once for a whole batch (see %Q6K-UNPACK-2), then
;;; two such pairs against every token quad (see %Q6K-GEMM-4X4).
(sb-c:defknown %q6k-unpack-2 (sb-sys:system-area-pointer (unsigned-byte 32))
    (values) ())
(sb-c:defknown %q6k-gemm-4x4 (sb-sys:system-area-pointer (unsigned-byte 32))
    (values) ())

;;; The delta rule for sixteen values of one head, one token (see Q35-DELTA!).
(sb-c:defknown %delta-block (sb-sys:system-area-pointer (unsigned-byte 32))
    (values) ())

;;; DeepSeek V4 Flash's expert rows (iquant.lisp): an IQ2_XXS or Q2_K row in
;;; the GGUF against Q8_K activations.  DESC: the row, then the activations'
;;; quants and scales (IQ2_XXS: and the signed-grid table; Q2_K: and their
;;; 16-sums).
(sb-c:defknown %iq2xxs-dot (sb-sys:system-area-pointer (unsigned-byte 32))
    single-float (sb-c:flushable))
(sb-c:defknown %q2k-dot (sb-sys:system-area-pointer (unsigned-byte 32))
    single-float (sb-c:flushable))

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

(defun q6k-rebuild-words (v q h shift hi hb m4 m30 c32)
  "Instruction words rebuilding a 16-weight Q6_K group as int8 q-32 into V:
   (low or high nibble of Q) | (2 bits of H at SHIFT) << 4, minus 32.  The two
   bits are moved to bits 4-5 and masked there (M30 is 0x30 a byte)."
  (append
   (ecase shift
     (0 (list (a64 #x4f0c5400 hb h) (a64 #x4e201c00 hb hb m30)))
     (2 (list (a64 #x4f0a5400 hb h) (a64 #x4e201c00 hb hb m30)))
     (4 (list (a64 #x4e201c00 hb h m30)))
     (6 (list (a64 #x6f0e0400 hb h) (a64 #x4e201c00 hb hb m30))))
   (list (if hi (a64 #x6f0c0400 v q) (a64 #x4e201c00 v q m4))
         (a64 #x4ea01c00 v v hb)
         (a64 #x6e208400 v v c32))))

(defun a64-dup-lane (d n i) (logior #x4e040400 (ash i 19) (ash n 5) d))
(defun a64-mov-x (d n) (logior #xaa0003e0 (ash n 16) d))
(defun a64-loop (count words)
  "WORDS, then decrement COUNT and branch back to them while it is not zero."
  (append words (list (a64 #xf1000400 count count) (a64-bne-back (1+ (length words))))))

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
      (illatio::word (illatio::a64 #x4f01e600 m3))                  ; 0x30
      (illatio::word (illatio::a64 #x4f01e400 c32))                 ; 32
      (flet ((rebuild (v q h shift hi)
               ;; V = (low or high nibble of Q) | (2 bits of H at SHIFT) << 4, minus 32;
               ;; the two bits are moved to bits 4-5 and masked there (0x30)
               (append
                (ecase shift
                  (0 (list (illatio::a64 #x4f0c5400 hb h) (illatio::a64 #x4e201c00 hb hb m3)))
                  (2 (list (illatio::a64 #x4f0a5400 hb h) (illatio::a64 #x4e201c00 hb hb m3)))
                  (4 (list (illatio::a64 #x4e201c00 hb h m3)))
                  (6 (list (illatio::a64 #x6f0e0400 hb h) (illatio::a64 #x4e201c00 hb hb m3))))
                (list (if hi (illatio::a64 #x6f0c0400 v q) (illatio::a64 #x4e201c00 v q m4))
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

;;; Rebuild rows A and B of a Q6_K matrix for a batch, into U: per super-block,
;;; per group in the order %Q6K-DOT-2X4 takes them, 48 bytes -- the two SMMLA
;;; operands (A's and B's halves zipped) and the scale vector [sA sA sB sB] as
;;; int32 -- then 16 bytes, [dA dA dB dB].  The rebuilding is %Q6K-DOT-2X4's,
;;; done once a batch instead of once every four tokens.
(define-vop (illatio::%q6k-unpack-2)
  (:translate illatio::%q6k-unpack-2)
  (:policy :fast-safe)
  (:args (desc :scs (sap-reg) :target pd) (nb :scs (unsigned-reg) :target count))
  (:arg-types system-area-pointer unsigned-num)
  (:temporary (:sc unsigned-reg) pqa) (:temporary (:sc unsigned-reg) pqb)
  (:temporary (:sc unsigned-reg) psa) (:temporary (:sc unsigned-reg) psb)
  (:temporary (:sc unsigned-reg) pda) (:temporary (:sc unsigned-reg) pdb)
  (:temporary (:sc unsigned-reg) pu)
  (:temporary (:sc unsigned-reg :from (:argument 0)) pd)
  (:temporary (:sc unsigned-reg :from (:argument 1)) count)
  (:temporary (:sc double-reg) m4) (:temporary (:sc double-reg) m30)
  (:temporary (:sc double-reg) c32)
  (:temporary (:sc double-reg) ha0) (:temporary (:sc double-reg) ha1)
  (:temporary (:sc double-reg) hb0) (:temporary (:sc double-reg) hb1)
  (:temporary (:sc double-reg) qa) (:temporary (:sc double-reg) qb)
  (:temporary (:sc double-reg) sba) (:temporary (:sc double-reg) sbb)
  (:temporary (:sc double-reg) saa) (:temporary (:sc double-reg) sab)
  (:temporary (:sc double-reg) sbaa) (:temporary (:sc double-reg) sbab)
  (:temporary (:sc double-reg) va) (:temporary (:sc double-reg) vb)
  (:temporary (:sc double-reg) hb)
  (:temporary (:sc double-reg) w1) (:temporary (:sc double-reg) w2)
  (:generator 300
    (move pd desc) (move count nb)
    (illatio::with-a64 (pqa pqb psa psb pda pdb pu pd count m4 m30 c32 ha0 ha1 hb0 hb1 qa qb
                        sba sbb saa sab sbaa sbab va vb hb w1 w2)
      (loop for r in (list pqa pqb psa psb pda pdb pu) for k from 0
            do (illatio::word (illatio::a64 (logior #xf9400000 (ash k 10)) r pd)))
      (illatio::word (illatio::a64 #x4f00e5e0 m4))                  ; 0x0f
      (illatio::word (illatio::a64 #x4f01e600 m30))                 ; 0x30
      (illatio::word (illatio::a64 #x4f01e400 c32))                 ; 32
      (illatio::loop-while-count
       count
       (append
        (list (illatio::a64 #x3cc10400 sba psa) (illatio::a64 #x3cc10400 sbb psb))
        (loop for c below 2
              append (append
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
                                                  (illatio::q6k-rebuild-words va qa (if (evenp x) ha0 ha1) shift hi hb m4 m30 c32)
                                                  (illatio::q6k-rebuild-words vb qb (if (evenp x) hb0 hb1) shift hi hb m4 m30 c32)
                                                  (list (illatio::a64 #x4ec03800 w1 va vb)
                                                        (illatio::a64 #x4ec07800 w2 va vb)
                                                        (logior #xac810000 (ash w2 10) (ash pu 5) w1)
                                                        (illatio::a64-dup-lane va (if hi sab saa) x)
                                                        (illatio::a64-dup-lane vb (if hi sbab sbaa) x)
                                                        (illatio::a64 #x4ec03800 w1 va vb)
                                                        (illatio::a64 #x3c810400 w1 pu))))))))
        (list (illatio::a64 #xbc404400 va pda) (illatio::a64 #xbc404400 vb pdb)
              (illatio::a64 #x4e040400 va va) (illatio::a64 #x4e040400 vb vb)
              (illatio::a64 #x4ec03800 w1 va vb)
              (illatio::a64 #x3c810400 w1 pu)))))))

;;; Two rebuilt row pairs (U0, U1) against NQ token quads: per group, each
;;; pair's operands once, SMMLA against both token pairs, the scale vector
;;; times the four integers.  Per super-block and (row, token) the one-row
;;; kernel's floats, as in %Q6K-DOT-2X4.  Results: per quad, sixteen floats --
;;; for U0 then U1, for token pair 0 then 1, [A.t A.t' B.t B.t'].
(define-vop (illatio::%q6k-gemm-4x4)
  (:translate illatio::%q6k-gemm-4x4)
  (:policy :fast-safe)
  (:args (desc :scs (sap-reg) :target pd) (nb :scs (unsigned-reg) :target nbs))
  (:arg-types system-area-pointer unsigned-num)
  (:temporary (:sc unsigned-reg) pu0) (:temporary (:sc unsigned-reg) pu1)
  (:temporary (:sc unsigned-reg) pb0) (:temporary (:sc unsigned-reg) pb1)
  (:temporary (:sc unsigned-reg) px) (:temporary (:sc unsigned-reg) py)
  (:temporary (:sc unsigned-reg) pr) (:temporary (:sc unsigned-reg) cq)
  (:temporary (:sc unsigned-reg) count)
  (:temporary (:sc unsigned-reg :from (:argument 0)) pd)
  (:temporary (:sc unsigned-reg :from (:argument 1)) nbs)
  (:temporary (:sc double-reg) s00) (:temporary (:sc double-reg) s01)
  (:temporary (:sc double-reg) s10) (:temporary (:sc double-reg) s11)
  (:temporary (:sc double-reg) a00) (:temporary (:sc double-reg) a01)
  (:temporary (:sc double-reg) a10) (:temporary (:sc double-reg) a11)
  (:temporary (:sc double-reg) w1a) (:temporary (:sc double-reg) w2a)
  (:temporary (:sc double-reg) sva)
  (:temporary (:sc double-reg) w1b) (:temporary (:sc double-reg) w2b)
  (:temporary (:sc double-reg) svb)
  (:temporary (:sc double-reg) x1) (:temporary (:sc double-reg) x2)
  (:temporary (:sc double-reg) dda) (:temporary (:sc double-reg) ddb)
  (:generator 300
    (move pd desc) (move nbs nb)
    (illatio::with-a64 (pu0 pu1 pb0 pb1 px py pr cq count pd nbs s00 s01 s10 s11 a00 a01 a10 a11
                        w1a w2a sva w1b w2b svb x1 x2 dda ddb)
      (loop for r in (list pb0 pb1 px py pr cq) for k from 0
            do (illatio::word (illatio::a64 (logior #xf9400000 (ash k 10)) r pd)))
      (illatio::loop-while-count
       cq
       (append
        (list (illatio::a64-mov-x pu0 pb0) (illatio::a64-mov-x pu1 pb1) (illatio::a64-mov-x count nbs)
              (illatio::a64 #x6f00e400 s00) (illatio::a64 #x6f00e400 s01)
              (illatio::a64 #x6f00e400 s10) (illatio::a64 #x6f00e400 s11))
        (illatio::a64-loop
         count
         (append
          (list (illatio::a64 #x6f00e400 a00) (illatio::a64 #x6f00e400 a01)
                (illatio::a64 #x6f00e400 a10) (illatio::a64 #x6f00e400 a11))
          (loop repeat 16
                append (append
                        (list (illatio::a64-ldp-q w1a w2a pu0) (illatio::a64 #x3cc10400 sva pu0)
                              (illatio::a64-ldp-q w1b w2b pu1) (illatio::a64 #x3cc10400 svb pu1))
                        (loop for (acca accb) in (list (list a00 a10) (list a01 a11))
                              append (list (illatio::a64-ldp-q x1 x2 px)
                                           (illatio::a64 #x6f00e400 dda) (illatio::a64 #x6f00e400 ddb)
                                           (illatio::a64 #x4e80a400 dda w1a x1) (illatio::a64 #x4e80a400 ddb w1b x1)
                                           (illatio::a64 #x4e80a400 dda w2a x2) (illatio::a64 #x4e80a400 ddb w2b x2)
                                           (illatio::a64 #x4ea09400 acca dda sva)
                                           (illatio::a64 #x4ea09400 accb ddb svb)))))
          ;; [dA dA dB dB] of each pair of rows, times the token pairs' y.d
          (list (illatio::a64 #x3cc10400 w1a pu0) (illatio::a64 #x3cc10400 w1b pu1)
                (illatio::a64-ldp-q x1 x2 py)
                (illatio::a64 #x6e20dc00 w2a w1a x1) (illatio::a64 #x6e20dc00 sva w1a x2)
                (illatio::a64 #x6e20dc00 w2b w1b x1) (illatio::a64 #x6e20dc00 svb w1b x2))
          (loop for (acc dq sum) in (list (list a00 w2a s00) (list a01 sva s01)
                                          (list a10 w2b s10) (list a11 svb s11))
                append (list (illatio::a64 #x4e21d800 acc acc)
                             (illatio::a64 #x6e20dc00 acc dq acc)
                             (illatio::a64 #x4e20d400 sum sum acc)))))
        (list (illatio::a64 #x3c810400 s00 pr) (illatio::a64 #x3c810400 s01 pr)
              (illatio::a64 #x3c810400 s10 pr) (illatio::a64 #x3c810400 s11 pr)))))))

;;; Q35-DELTA!'s two passes for values v0..v0+15 of a head, a lane each: the
;;; same operations as its portable loop, in its order, never fused.  DESC:
;;; S at (k=0, v0), the head's q in C (k at +XOFF bytes), XOFF, v's values at
;;; v0, where o's go, the bytes from one k's row of S to the next, then
;;; decay, beta and the q scale as floats.
(define-vop (illatio::%delta-block)
  (:translate illatio::%delta-block)
  (:policy :fast-safe)
  (:args (desc :scs (sap-reg) :target pd) (kd :scs (unsigned-reg) :target kdr))
  (:arg-types system-area-pointer unsigned-num)
  (:temporary (:sc unsigned-reg) ps) (:temporary (:sc unsigned-reg) psb)
  (:temporary (:sc unsigned-reg) pq) (:temporary (:sc unsigned-reg) pqb)
  (:temporary (:sc unsigned-reg) xoff) (:temporary (:sc unsigned-reg) pcv)
  (:temporary (:sc unsigned-reg) po) (:temporary (:sc unsigned-reg) stride)
  (:temporary (:sc unsigned-reg) count)
  (:temporary (:sc unsigned-reg :from (:argument 0)) pd)
  (:temporary (:sc unsigned-reg :from (:argument 1)) kdr)
  (:temporary (:sc double-reg) s0) (:temporary (:sc double-reg) s1)
  (:temporary (:sc double-reg) s2) (:temporary (:sc double-reg) s3)
  (:temporary (:sc double-reg) k0) (:temporary (:sc double-reg) k1)
  (:temporary (:sc double-reg) k2) (:temporary (:sc double-reg) k3)
  (:temporary (:sc double-reg) d0) (:temporary (:sc double-reg) d1)
  (:temporary (:sc double-reg) d2) (:temporary (:sc double-reg) d3)
  (:temporary (:sc double-reg) o0) (:temporary (:sc double-reg) o1)
  (:temporary (:sc double-reg) o2) (:temporary (:sc double-reg) o3)
  (:temporary (:sc double-reg) t0) (:temporary (:sc double-reg) t1)
  (:temporary (:sc double-reg) t2) (:temporary (:sc double-reg) t3)
  (:temporary (:sc double-reg) dec) (:temporary (:sc double-reg) bb)
  (:temporary (:sc double-reg) sc) (:temporary (:sc double-reg) ck)
  (:temporary (:sc double-reg) qs)
  (:generator 100
    (move pd desc) (move kdr kd)
    (illatio::with-a64 (ps psb pq pqb xoff pcv po stride count pd kdr s0 s1 s2 s3 k0 k1 k2 k3
                        d0 d1 d2 d3 o0 o1 o2 o3 t0 t1 t2 t3 dec bb sc ck qs)
      (let ((ss (list s0 s1 s2 s3)) (ks (list k0 k1 k2 k3)) (ds (list d0 d1 d2 d3))
            (os (list o0 o1 o2 o3)) (ts (list t0 t1 t2 t3)))
        (flet ((each (fn &rest lists) (apply #'mapcar fn lists))
               (ldp-off (a b n k) (logior #xad400000 (ash k 15) (ash b 10) (ash n 5) a))
               (stp-off (a b n k) (logior #xad000000 (ash k 15) (ash b 10) (ash n 5) a)))
          (flet ((load-s () (list (ldp-off s0 s1 ps 0) (ldp-off s2 s3 ps 2)))
                 (store-s () (list (stp-off s0 s1 ps 0) (stp-off s2 s3 ps 2))))
            (loop for r in (list psb pqb xoff pcv po stride) for k from 0
                  do (illatio::word (illatio::a64 (logior #xf9400000 (ash k 10)) r pd)))
            (loop for r in (list dec bb sc) for k from 6
                  do (illatio::word (illatio::a64 (logior #xbd400000 (ash (* 2 k) 10)) r pd))
                     (illatio::word (illatio::a64 #x4e040400 r r)))
            (dolist (r ks) (illatio::word (illatio::a64 #x6f00e400 r)))
            (dolist (w (list (illatio::a64-mov-x ps psb) (illatio::a64-mov-x pq pqb)
                             (illatio::a64-mov-x count kdr)))
              (illatio::word w))
            ;; pass 1, each k: S *= decay; sk += S * k[k]
            (illatio::loop-while-count
             count
             (append
              (list (illatio::a64 #xbc606800 ck pq xoff) (illatio::a64 #x91001000 pq pq)
                    (illatio::a64 #x4e040400 ck ck))
              (load-s)
              (each (lambda (sr) (illatio::a64 #x6e20dc00 sr sr dec)) ss)
              (store-s)
              (each (lambda (tr sr) (illatio::a64 #x6e20dc00 tr sr ck)) ts ss)
              (each (lambda (kr tr) (illatio::a64 #x4e20d400 kr kr tr)) ks ts)
              (list (illatio::a64 #x8b000000 ps ps stride))))
            ;; d = (v - sk) * beta
            (dolist (w (list (illatio::a64-ldp-q t0 t1 pcv) (illatio::a64-ldp-q t2 t3 pcv)))
              (illatio::word w))
            (each (lambda (dr tr kr) (illatio::word (illatio::a64 #x4ea0d400 dr tr kr))) ds ts ks)
            (each (lambda (dr) (illatio::word (illatio::a64 #x6e20dc00 dr dr bb))) ds)
            (dolist (r os) (illatio::word (illatio::a64 #x6f00e400 r)))
            (dolist (w (list (illatio::a64-mov-x ps psb) (illatio::a64-mov-x pq pqb)
                             (illatio::a64-mov-x count kdr)))
              (illatio::word w))
            ;; pass 2, each k: S += k[k] * d; o += S * (q[k] * scale)
            (illatio::loop-while-count
             count
             (append
              (list (illatio::a64 #xbc606800 ck pq xoff) (illatio::a64 #xbc404400 qs pq)
                    (illatio::a64 #x1e200800 qs qs sc)
                    (illatio::a64 #x4e040400 ck ck) (illatio::a64 #x4e040400 qs qs))
              (load-s)
              (each (lambda (tr dr) (illatio::a64 #x6e20dc00 tr ck dr)) ts ds)
              (each (lambda (sr tr) (illatio::a64 #x4e20d400 sr sr tr)) ss ts)
              (store-s)
              (each (lambda (tr sr) (illatio::a64 #x6e20dc00 tr sr qs)) ts ss)
              (each (lambda (ov tr) (illatio::a64 #x4e20d400 ov ov tr)) os ts)
              (list (illatio::a64 #x8b000000 ps ps stride))))
            (dolist (w (list (logior #xac810000 (ash o1 10) (ash po 5) o0)
                             (logior #xac810000 (ash o3 10) (ash po 5) o2)))
              (illatio::word w))))))))

;;; IQ2XXS-ROW-DOT's arithmetic: per sub-block of 32, four 8-byte rows of the
;;; signed grid (+IQ2XXS-SIGNED+, indexed by grid*128 + sign pattern) SDOTted
;;; with the activations, times (0.5 + LS), the even and odd sub-blocks
;;; summed apart; per block TOTAL += (d * y.d) * (their sum); a quarter of it.
(define-vop (illatio::%iq2xxs-dot)
  (:translate illatio::%iq2xxs-dot)
  (:policy :fast-safe)
  (:args (desc :scs (sap-reg) :target pd) (nb :scs (unsigned-reg) :target count))
  (:arg-types system-area-pointer unsigned-num)
  (:results (sum :scs (single-reg)))
  (:result-types single-float)
  (:temporary (:sc unsigned-reg) pw) (:temporary (:sc unsigned-reg) ptab)
  (:temporary (:sc unsigned-reg) pyq) (:temporary (:sc unsigned-reg) pyd)
  (:temporary (:sc unsigned-reg) wg) (:temporary (:sc unsigned-reg) ws)
  (:temporary (:sc unsigned-reg) t1) (:temporary (:sc unsigned-reg) t2)
  (:temporary (:sc unsigned-reg :from (:argument 0)) pd)
  (:temporary (:sc unsigned-reg :from (:argument 1)) count)
  (:temporary (:sc double-reg) dw) (:temporary (:sc double-reg) ydv)
  (:temporary (:sc double-reg) dd) (:temporary (:sc double-reg) suma)
  (:temporary (:sc double-reg) sumb) (:temporary (:sc double-reg) half)
  (:temporary (:sc double-reg) va) (:temporary (:sc double-reg) vb)
  (:temporary (:sc double-reg) vt) (:temporary (:sc double-reg) ya)
  (:temporary (:sc double-reg) yb) (:temporary (:sc double-reg) acc)
  (:temporary (:sc double-reg) lsv) (:temporary (:sc double-reg) tt)
  (:generator 200
    (move pd desc) (move count nb)
    (illatio::with-a64 (pw ptab pyq pyd wg ws t1 t2 pd count sum dw ydv dd suma sumb half
                        va vb vt ya yb acc lsv tt)
      (loop for r in (list pw ptab pyq pyd) for k from 0
            do (illatio::word (illatio::a64 (logior #xf9400000 (ash k 10)) r pd)))
      (illatio::word (illatio::a64 #x2f00e400 sum))                 ; total = 0
      (illatio::word (illatio::a64 #x1e2c1000 half))                ; 0.5
      (flet ((ubfx (d n lsb width)
               (logior #xd3400000 (ash lsb 16) (ash (+ lsb width -1) 10) (ash n 5) d)))
        (illatio::loop-while-count
         count
         (append
          (list (illatio::a64 #x7c402400 dw pw) (illatio::a64 #x1ee24000 dw dw)   ; d
                (illatio::a64 #xbc404400 ydv pyd)                                ; y.d
                (illatio::a64 #x1e200800 dd dw ydv)                              ; d * y.d
                (illatio::a64 #x2f00e400 suma) (illatio::a64 #x2f00e400 sumb))
          (loop for ib below 8
                append (append
                        (list (illatio::a64 #xb8404400 wg pw) (illatio::a64 #xb8404400 ws pw))
                        (loop for l below 4
                              append (list (ubfx t1 wg (* 8 l) 8)
                                           (ubfx t2 ws (* 7 l) 7)
                                           (logior #xaa000000 (ash t1 16) (ash 7 10) (ash t2 5) t2)
                                           (logior #xfc607800 (ash t2 16) (ash ptab 5)
                                                   (case l (0 va) (2 vb) (t vt)))
                                           (case l
                                             (1 (illatio::a64 #x6e180400 va vt))
                                             (3 (illatio::a64 #x6e180400 vb vt))
                                             (t #xd503201f))))                   ; nop
                        (list (illatio::a64-ldp-q ya yb pyq)
                              (illatio::a64 #x6f00e400 acc)
                              (illatio::a64 #x4e809400 acc va ya)
                              (illatio::a64 #x4e809400 acc vb yb)
                              (illatio::a64 #x4eb1b800 acc acc)                  ; the sub-block's integer dot
                              (illatio::a64 #x5e21d800 acc acc)
                              (illatio::a64 #x531c7c00 t1 ws)                    ; LS
                              (illatio::a64 #x1e220000 lsv t1)
                              (illatio::a64 #x1e202800 lsv half lsv)             ; 0.5 + LS
                              (illatio::a64 #x1e200800 tt acc lsv)
                              (if (evenp ib)
                                  (illatio::a64 #x1e202800 suma suma tt)
                                  (illatio::a64 #x1e202800 sumb sumb tt)))))
          (list (illatio::a64 #x1e202800 tt suma sumb)
                (illatio::a64 #x1e200800 tt dd tt)
                (illatio::a64 #x1e202800 sum sum tt)))))
      (illatio::word (illatio::a64 #x1e2a1000 tt))                  ; 0.25
      (illatio::word (illatio::a64 #x1e200800 sum tt sum)))))

;;; Q2K-ROW-DOT's arithmetic: per block, SUM += (-y.d * dmin) * (the mins
;;; against the activations' 16-sums), then SUM += (y.d * d) * (each
;;; 16-weight group's SDOT times its scale, summed).
(define-vop (illatio::%q2k-dot)
  (:translate illatio::%q2k-dot)
  (:policy :fast-safe)
  (:args (desc :scs (sap-reg) :target pd) (nb :scs (unsigned-reg) :target count))
  (:arg-types system-area-pointer unsigned-num)
  (:results (sum :scs (single-reg)))
  (:result-types single-float)
  (:temporary (:sc unsigned-reg) pw) (:temporary (:sc unsigned-reg) pyq)
  (:temporary (:sc unsigned-reg) pyd) (:temporary (:sc unsigned-reg) pbs)
  (:temporary (:sc unsigned-reg :from (:argument 0)) pd)
  (:temporary (:sc unsigned-reg :from (:argument 1)) count)
  (:temporary (:sc double-reg) sc) (:temporary (:sc double-reg) qa0)
  (:temporary (:sc double-reg) qa1) (:temporary (:sc double-reg) qb0)
  (:temporary (:sc double-reg) qb1) (:temporary (:sc double-reg) m3)
  (:temporary (:sc double-reg) m4) (:temporary (:sc double-reg) w0)
  (:temporary (:sc double-reg) w1) (:temporary (:sc double-reg) s0)
  (:temporary (:sc double-reg) s1) (:temporary (:sc double-reg) s2)
  (:temporary (:sc double-reg) s3) (:temporary (:sc double-reg) v)
  (:temporary (:sc double-reg) yv) (:temporary (:sc double-reg) tq)
  (:temporary (:sc double-reg) acc) (:temporary (:sc double-reg) dh)
  (:temporary (:sc double-reg) dmh) (:temporary (:sc double-reg) ydb)
  (:temporary (:sc double-reg) tt)
  (:generator 200
    (move pd desc) (move count nb)
    (illatio::with-a64 (pw pyq pyd pbs pd count sum sc qa0 qa1 qb0 qb1 m3 m4 w0 w1 s0 s1 s2 s3
                        v yv tq acc dh dmh ydb tt)
      (loop for r in (list pw pyq pyd pbs) for k from 0
            do (illatio::word (illatio::a64 (logior #xf9400000 (ash k 10)) r pd)))
      (illatio::word (illatio::a64 #x2f00e400 sum))
      (illatio::word (illatio::a64 #x4f00e460 m3))                  ; 0x03
      (illatio::word (illatio::a64 #x4f00e5e0 m4))                  ; 0x0f
      (flet ((widen4 (src)
               ;; 16 bytes of SRC as four int32x4 in s0..s3 (w0, w1 the halves)
               (list (illatio::a64 #x2f08a400 w0 src) (illatio::a64 #x6f08a400 w1 src)
                     (illatio::a64 #x2f10a400 s0 w0) (illatio::a64 #x6f10a400 s1 w0)
                     (illatio::a64 #x2f10a400 s2 w1) (illatio::a64 #x6f10a400 s3 w1))))
        (illatio::loop-while-count
         count
         (append
          (list (illatio::a64 #x3cc10400 sc pw)                          ; 16 scale/min bytes
                (illatio::a64-ldp-q qa0 qa1 pw) (illatio::a64-ldp-q qb0 qb1 pw)
                (illatio::a64 #x7c402400 dh pw) (illatio::a64 #x1ee24000 dh dh)
                (illatio::a64 #x7c402400 dmh pw) (illatio::a64 #x1ee24000 dmh dmh)
                (illatio::a64 #xbc404400 ydb pyd)
                (illatio::a64 #x1e200800 dh ydb dh)                      ; d = y.d * d
                (illatio::a64 #x1e214000 tt ydb)
                (illatio::a64 #x1e200800 dmh tt dmh)                     ; dmin = -y.d * dmin
                (illatio::a64 #x6f0c0400 v sc))                          ; the mins
          (widen4 v)
          (list (illatio::a64-ldp-q yv tq pbs)
                (illatio::a64 #x4ea09c00 acc s0 yv) (illatio::a64 #x4ea09400 acc s1 tq)
                (illatio::a64-ldp-q yv tq pbs)
                (illatio::a64 #x4ea09400 acc s2 yv) (illatio::a64 #x4ea09400 acc s3 tq)
                (illatio::a64 #x4eb1b800 acc acc) (illatio::a64 #x5e21d800 acc acc)
                (illatio::a64 #x1e200800 tt dmh acc)
                (illatio::a64 #x1e202800 sum sum tt)                     ; sum += dmin * mins
                (illatio::a64 #x4e201c00 v sc m4))                       ; the scales
          (widen4 v)
          (list (illatio::a64 #x6f00e400 acc))
          (loop for h below 2
                append (loop for k below 4
                             append (loop for half below 2
                                          for q = (if (zerop h) (if (zerop half) qa0 qa1) (if (zerop half) qb0 qb1))
                                          for g = (+ (* h 8) (* 2 k) half)
                                          append (append
                                                  (if (zerop k)
                                                      (list (illatio::a64 #x4e201c00 v q m3))
                                                      (list (illatio::a64 (ecase k (1 #x6f0e0400) (2 #x6f0c0400) (3 #x6f0a0400)) v q)
                                                            (illatio::a64 #x4e201c00 v v m3)))
                                                  (list (illatio::a64 #x3cc10400 yv pyq)
                                                        (illatio::a64 #x6f00e400 tq)
                                                        (illatio::a64 #x4e809400 tq v yv)
                                                        (illatio::a64-mla-lane acc tq (nth (floor g 4) (list s0 s1 s2 s3)) (mod g 4)))))))
          (list (illatio::a64 #x4eb1b800 acc acc) (illatio::a64 #x5e21d800 acc acc)
                (illatio::a64 #x1e200800 tt dh acc)
                (illatio::a64 #x1e202800 sum sum tt))))))))

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

(declaim (inline delta-block-p))
(defun delta-block-p (vd)
  "Whether Q35-DELTA! runs its heads sixteen values a call (%DELTA-BLOCK)."
  (and (neon-dot-p) (zerop (mod vd 16))))

(defun delta-blocks! (S base c kq kk vv o oo vd kd decay b scale)
  "Q35-DELTA!'s two passes for one head and token, sixteen values a call."
  (declare (type f32vec S c o) (type fixnum base kq kk vv oo vd kd)
           (type single-float decay b scale) (optimize speed (safety 0)))
  (let ((desc (make-array 9 :element-type '(unsigned-byte 64))))
    (declare (dynamic-extent desc))
    (sb-sys:with-pinned-objects (S c o desc)
      (let ((s0 (sb-sys:sap-int (sb-sys:vector-sap S)))
            (c0 (sb-sys:sap-int (sb-sys:vector-sap c)))
            (o0 (sb-sys:sap-int (sb-sys:vector-sap o)))
            (dsap (sb-sys:vector-sap desc)))
        (declare (type (unsigned-byte 64) s0 c0 o0))
        (setf (aref desc 1) (+ c0 (* 4 kq))
              (aref desc 2) (* 4 (- kk kq))
              (aref desc 5) (* 4 vd)
              (aref desc 6) (ldb (byte 32 0) (sb-kernel:single-float-bits decay))
              (aref desc 7) (ldb (byte 32 0) (sb-kernel:single-float-bits b))
              (aref desc 8) (ldb (byte 32 0) (sb-kernel:single-float-bits scale)))
        (loop for v0 of-type fixnum from 0 below vd by 16
              do (setf (aref desc 0) (+ s0 (* 4 (+ base v0)))
                       (aref desc 3) (+ c0 (* 4 (+ vv v0)))
                       (aref desc 4) (+ o0 (* 4 (+ oo v0))))
                 (%delta-block dsap kd))))))
