;;;; gpt2-train.lisp — GPT-2 trained: forward, backward, AdamW, in f32.
;;;
;;; Ported from llm.c's train_gpt2.c (Andrej Karpathy, MIT), its clean CPU
;;; reference, layer for layer and in its memory layout: the parameters in
;;; one vector (wte, wpe, then each per-layer tensor for all layers), the
;;; activations in another, their gradients in two more of the same shapes.
;;; Every backward ADDS into gradients the step zeroes first, as there.
;;; test_gpt2.c's oracle (inspect/gpt2-train-test.lisp) checks the logits,
;;; the loss and every parameter's gradient against PyTorch's, then ten
;;; AdamW steps' losses.
;;;
;;; The matrix products are the work.  Forward is a dot of each weight row
;;; with each token's row (F32-DOT16, NEON where there is one), the weight
;;; row in cache across the tokens; backward to the input is axpys of weight
;;; rows into a block of token rows held in cache, and to the weight, axpys
;;; of the token rows into each weight row's gradient.

(in-package #:illatio)

;;; ---------------------------------------------------------------------------
;;; files: llm.c's headers are 256 int32s, then the payload

(defun raw-read (fd vector bytes &optional (start 0))
  "Read BYTES bytes from FD into VECTOR (a specialized simple array) at byte START."
  (sb-sys:with-pinned-objects (vector)
    (let ((sap (sb-sys:vector-sap vector)) (done 0))
      (loop while (< done bytes)
            do (multiple-value-bind (n err)
                   (sb-unix:unix-read fd (sb-sys:sap+ sap (+ start done)) (min (- bytes done) (ash 1 30)))
                 (unless (and n (plusp n))
                   (error "read failed after ~D of ~D bytes (errno ~A)" done bytes err))
                 (incf done n)))))
  vector)

(defmacro with-raw-file ((fd path) &body body)
  `(let ((,fd (sb-unix:unix-open (namestring (truename ,path)) sb-unix:o_rdonly 0)))
     (unless ,fd (error "cannot open ~A" ,path))
     (unwind-protect (progn ,@body) (sb-unix:unix-close ,fd))))

(defun read-header (fd magic)
  (let ((h (make-array 256 :element-type '(signed-byte 32))))
    (raw-read fd h 1024)
    (unless (= (aref h 0) magic) (error "bad magic ~D (want ~D)" (aref h 0) magic))
    h))

;;; ---------------------------------------------------------------------------
;;; the model

(defconstant +gpt2-tensors+ 16)
(defconstant +gpt2-acts+ 23)

;; parameter tensors, in the file's order
(defconstant +wte+ 0) (defconstant +wpe+ 1) (defconstant +ln1w+ 2) (defconstant +ln1b+ 3)
(defconstant +qkvw+ 4) (defconstant +qkvb+ 5) (defconstant +attprojw+ 6) (defconstant +attprojb+ 7)
(defconstant +ln2w+ 8) (defconstant +ln2b+ 9) (defconstant +fcw+ 10) (defconstant +fcb+ 11)
(defconstant +fcprojw+ 12) (defconstant +fcprojb+ 13) (defconstant +lnfw+ 14) (defconstant +lnfb+ 15)
;; activations
(defconstant +encoded+ 0) (defconstant +ln1+ 1) (defconstant +ln1-mean+ 2) (defconstant +ln1-rstd+ 3)
(defconstant +qkv+ 4) (defconstant +atty+ 5) (defconstant +preatt+ 6) (defconstant +att+ 7)
(defconstant +attproj+ 8) (defconstant +residual2+ 9) (defconstant +ln2+ 10) (defconstant +ln2-mean+ 11)
(defconstant +ln2-rstd+ 12) (defconstant +fch+ 13) (defconstant +fch-gelu+ 14) (defconstant +fcproj+ 15)
(defconstant +residual3+ 16) (defconstant +lnf+ 17) (defconstant +lnf-mean+ 18) (defconstant +lnf-rstd+ 19)
(defconstant +logits+ 20) (defconstant +probs+ 21) (defconstant +losses+ 22)

(defstruct (gpt2 (:constructor %make-gpt2))
  (max-t 0 :type fixnum) (v 0 :type fixnum) (vp 0 :type fixnum)
  (l 0 :type fixnum) (nh 0 :type fixnum) (c 0 :type fixnum)
  (param-sizes nil) (param-offsets nil) (n-params 0 :type fixnum)
  (params nil) (grads nil) (m nil) (vv nil)       ; f32vecs of N-PARAMS
  (act-sizes nil) (act-offsets nil) (n-acts 0 :type fixnum)
  (acts nil) (dacts nil)
  (b 0 :type fixnum) (tt 0 :type fixnum)
  (inputs nil) (targets nil)
  (mean-loss -1f0 :type single-float)
  (pool nil))

(defun offsets (sizes)
  (let ((o 0))
    (map 'simple-vector (lambda (s) (prog1 o (incf o s))) sizes)))

(defun param-sizes-for (vp c max-t l)
  (vector (* vp c) (* max-t c) (* l c) (* l c) (* l 3 c c) (* l 3 c) (* l c c) (* l c)
          (* l c) (* l c) (* l 4 c c) (* l 4 c) (* l c 4 c) (* l c) c c))

(defun act-sizes-for (m b tt)
  (let ((c (gpt2-c m)) (nh (gpt2-nh m)) (l (gpt2-l m)) (vp (gpt2-vp m)))
    (vector (* b tt c) (* l b tt c) (* l b tt) (* l b tt) (* l b tt 3 c) (* l b tt c)
            (* l b nh tt tt) (* l b nh tt tt) (* l b tt c) (* l b tt c) (* l b tt c)
            (* l b tt) (* l b tt) (* l b tt 4 c) (* l b tt 4 c) (* l b tt c) (* l b tt c)
            (* b tt c) (* b tt) (* b tt) (* b tt vp) (* b tt vp) (* b tt))))

(declaim (inline poff aoff))
(defun poff (m i &optional (layer 0))
  "Offset of parameter tensor I (layer LAYER's slice of it) in the parameter vectors."
  (+ (the fixnum (svref (gpt2-param-offsets m) i))
     (* layer (floor (the fixnum (svref (gpt2-param-sizes m) i)) (max 1 (gpt2-l m))))))
(defun aoff (m i &optional (layer 0))
  "Offset of activation I (layer LAYER's slice of it) in the activation vectors."
  (+ (the fixnum (svref (gpt2-act-offsets m) i))
     (* layer (floor (the fixnum (svref (gpt2-act-sizes m) i)) (max 1 (gpt2-l m))))))

(defun load-gpt2 (path &key (pool (make-pool)))
  "GPT-2 from llm.c's checkpoint (gpt2_124M.bin: header, then f32 parameters)."
  (with-raw-file (fd path)
    (let* ((h (read-header fd 20240326)))
      (unless (= (aref h 1) 3) (error "checkpoint version ~D, want 3" (aref h 1)))
      (let* ((max-t (aref h 2)) (v (aref h 3)) (l (aref h 4)) (nh (aref h 5)) (c (aref h 6)) (vp (aref h 7))
             (sizes (param-sizes-for vp c max-t l))
             (n (reduce #'+ sizes))
             (m (%make-gpt2 :max-t max-t :v v :vp vp :l l :nh nh :c c
                            :param-sizes sizes :param-offsets (offsets sizes) :n-params n
                            :params (f32vec n) :pool pool)))
        (raw-read fd (gpt2-params m) (* 4 n))
        m))))

;;; ---------------------------------------------------------------------------
;;; the layers.  X XO: a vector and the offset its tensor starts at.

(defmacro fast (&body body)
  `(locally (declare (optimize speed (safety 0) (debug 0))) ,@body))

(defun encoder-forward (out oo inp wte wteo wpeo b tt c)
  (declare (type f32vec out wte) (type (simple-array (signed-byte 32) (*)) inp)
           (type fixnum oo wteo wpeo b tt c))
  (fast
    (dotimes (bt (* b tt))
      (let ((ix (aref inp bt)) (tpos (mod bt tt)))
        (dotimes (i c)
          (setf (aref out (+ oo (* bt c) i))
                (+ (aref wte (+ wteo (* ix c) i)) (aref wte (+ wpeo (* tpos c) i)))))))))

(defun encoder-backward (dwte dwteo dwpeo dout doo inp b tt c)
  (declare (type f32vec dwte dout) (type (simple-array (signed-byte 32) (*)) inp)
           (type fixnum dwteo dwpeo doo b tt c))
  (fast
    (dotimes (bt (* b tt))
      (let ((ix (aref inp bt)) (tpos (mod bt tt)))
        (dotimes (i c)
          (let ((d (aref dout (+ doo (* bt c) i))))
            (incf (aref dwte (+ dwteo (* ix c) i)) d)
            (incf (aref dwte (+ dwpeo (* tpos c) i)) d)))))))

(defun layernorm-forward (pool out oo st mo ro inp io w wo bo n c)
  "N rows of C: OUT = (INP - mean) * rstd * W + B (W's and B's at WO, BO in W),
   each row's mean and rstd kept in ST at MO and RO."
  (declare (type f32vec out st inp w) (type fixnum oo mo ro io wo bo n c))
  (parallel-for (r n pool)
    (fast
      (let ((x (+ io (* r c))) (m 0f0) (v 0f0))
        (declare (type fixnum x) (type single-float m v))
        (dotimes (i c) (incf m (aref inp (+ x i))))
        (setf m (/ m c))
        (dotimes (i c) (let ((d (- (aref inp (+ x i)) m))) (incf v (* d d))))
        (setf v (/ v c))
        (let ((s (/ 1f0 (sqrt (+ v 1f-5)))) (o (+ oo (* r c))))
          (declare (type single-float s) (type fixnum o))
          (dotimes (i c)
            (setf (aref out (+ o i))
                  (+ (* (* s (- (aref inp (+ x i)) m)) (aref w (+ wo i))) (aref w (+ bo i)))))
          (setf (aref st (+ mo r)) m (aref st (+ ro r)) s))))))

(defun layernorm-backward (pool dinp dio dw dwo dbo dout doo inp io w wo st mo ro n c)
  (declare (type f32vec dinp dw dout inp w st) (type fixnum dio dwo dbo doo io wo mo ro n c))
  ;; the input's gradient, row by row in parallel
  (parallel-for (r n pool)
    (fast
      (let ((x (+ io (* r c))) (d (+ doo (* r c))) (di (+ dio (* r c)))
            (mu (aref st (+ mo r))) (rs (aref st (+ ro r)))
            (dm 0f0) (dnm 0f0))
        (declare (type fixnum x d di) (type single-float mu rs dm dnm))
        (dotimes (i c)
          (let ((nrm (* (- (aref inp (+ x i)) mu) rs)) (dn (* (aref w (+ wo i)) (aref dout (+ d i)))))
            (incf dm dn) (incf dnm (* dn nrm))))
        (setf dm (/ dm c) dnm (/ dnm c))
        (dotimes (i c)
          (let ((nrm (* (- (aref inp (+ x i)) mu) rs)) (dn (* (aref w (+ wo i)) (aref dout (+ d i)))))
            (incf (aref dinp (+ di i)) (* (- (- dn dm) (* nrm dnm)) rs)))))))
  ;; the weight's and bias's, column by column in parallel
  (parallel-for (i c pool)
    (fast
      (let ((gw 0f0) (gb 0f0))
        (declare (type single-float gw gb))
        (dotimes (r n)
          (let ((dv (aref dout (+ doo (* r c) i))))
            (incf gb dv)
            (incf gw (* (* (- (aref inp (+ io (* r c) i)) (aref st (+ mo r))) (aref st (+ ro r))) dv))))
        (incf (aref dw (+ dbo i)) gb)
        (incf (aref dw (+ dwo i)) gw)))))

(defun matmul-forward (pool out oo inp io w wo bias bo n c oc)
  "OUT[r,o] = BIAS[o] + INP[r,:] . W[o,:] for N rows of C, OC outputs (W is OC x C)."
  (declare (type f32vec out inp w) (type fixnum oo io wo bo n c oc))
  (assert (zerop (mod c 16)))
  (pool-run pool
            (lambda (k cnt)
              (multiple-value-bind (s e) (share oc k cnt)
                (declare (type fixnum s e))
                (with-attention-kernels (dot axpy) (w w inp out)
                  (fast
                    (loop for o of-type fixnum from s below e
                          do (let ((b0 (if bias (aref (the f32vec bias) (+ bo o)) 0f0))
                                   (wr (+ wo (* o c))))
                               (declare (type single-float b0) (type fixnum wr))
                               (dotimes (r n)
                                 (setf (aref out (+ oo (* r oc) o))
                                       (+ b0 (the single-float (dot wr (+ io (* r c)) c))))))))))))
  out)

(defconstant +mm-rows+ 16 "Token rows the input's gradient accumulates while a weight row is in cache.")

(defun matmul-backward (pool dinp dio dw dwo dbias dbo dout doo inp io w wo n c oc)
  "Gradients of OUT = INP W^T + BIAS: DINP += DOUT W, DW += DOUT^T INP, DBIAS += sum DOUT."
  (declare (type f32vec dinp dw dout inp w) (type fixnum dio dwo dbo doo io wo n c oc))
  ;; into the input: blocks of rows in parallel, each weight row axpied into them
  (let ((blocks (ceiling n +mm-rows+)))
    (pool-run pool
              (lambda (k cnt)
                (multiple-value-bind (s e) (share blocks k cnt)
                  (declare (type fixnum s e))
                  (with-attention-kernels (dot axpy) (w w dout dinp)
                    (fast
                      (loop for blk of-type fixnum from s below e
                            do (let ((r0 (* blk +mm-rows+)) (r1 (min n (* (1+ blk) +mm-rows+))))
                                 (declare (type fixnum r0 r1))
                                 (dotimes (o oc)
                                   (loop for r of-type fixnum from r0 below r1
                                         do (let ((d (aref dout (+ doo (* r oc) o))))
                                              (unless (zerop d)
                                                (axpy d (+ wo (* o c)) (+ dio (* r c)) c)))))))))))))
  ;; into the weight and bias: output rows in parallel
  (pool-run pool
            (lambda (k cnt)
              (multiple-value-bind (s e) (share oc k cnt)
                (declare (type fixnum s e))
                (with-attention-kernels (dot axpy) (inp inp dout dw)
                  (fast
                    (loop for o of-type fixnum from s below e
                          do (let ((sum 0f0))
                               (declare (type single-float sum))
                               (dotimes (r n)
                                 (let ((d (aref dout (+ doo (* r oc) o))))
                                   (incf sum d)
                                   (unless (zerop d)
                                     (axpy d (+ io (* r c)) (+ dwo (* o c)) c))))
                               (when dbias (incf (aref (the f32vec dbias) (+ dbo o)) sum)))))))))
  nil)

(defun attention-forward (pool out oo st po ao inp io b tt c nh)
  "Causal self-attention over INP's (B,T,3C) Q K V: OUT (B,T,C); the scores
   before and after the softmax, (B,NH,T,T) each, kept in ST at PO and AO."
  (declare (type f32vec out st inp) (type fixnum oo po ao io b tt c nh))
  (let* ((c3 (* 3 c)) (hs (floor c nh)) (scale (/ 1f0 (sqrt (coerce hs 'single-float)))))
    (declare (type fixnum c3 hs) (type single-float scale))
    (parallel-for (bth (* b tt nh) pool)
      (fast
        (multiple-value-bind (bt h) (floor bth nh)
          (multiple-value-bind (bb tq) (floor bt tt)
            (let* ((q (+ io (* bb tt c3) (* tq c3) (* h hs)))
                   (row (+ (* bb nh tt tt) (* h tt tt) (* tq tt)))
                   (pre (+ po row)) (pa (+ ao row))
                   (mx -10000f0) (sum 0f0))
              (declare (type fixnum q row pre pa) (type single-float mx sum))
              (loop for t2 of-type fixnum from 0 to tq
                    do (let ((k (+ io (* bb tt c3) (* t2 c3) (* h hs) c)) (v 0f0))
                         (declare (type fixnum k) (type single-float v))
                         (dotimes (i hs) (incf v (* (aref inp (+ q i)) (aref inp (+ k i)))))
                         (setf v (* v scale))
                         (when (> v mx) (setf mx v))
                         (setf (aref st (+ pre t2)) v)))
              (loop for t2 of-type fixnum from 0 to tq
                    do (let ((e (exp (- (aref st (+ pre t2)) mx))))
                         (incf sum e) (setf (aref st (+ pa t2)) e)))
              (let ((inv (if (zerop sum) 0f0 (/ 1f0 sum))))
                (dotimes (t2 tt)
                  (setf (aref st (+ pa t2)) (if (<= t2 tq) (* (aref st (+ pa t2)) inv) 0f0))))
              (let ((o (+ oo (* bb tt c) (* tq c) (* h hs))))
                (declare (type fixnum o))
                (dotimes (i hs) (setf (aref out (+ o i)) 0f0))
                (loop for t2 of-type fixnum from 0 to tq
                      do (let ((vv (+ io (* bb tt c3) (* t2 c3) (* h hs) (* 2 c))) (a (aref st (+ pa t2))))
                           (declare (type fixnum vv))
                           (dotimes (i hs) (incf (aref out (+ o i)) (* a (aref inp (+ vv i)))))))))))))))

(defun attention-backward (pool dinp dio dst dpo dao dout doo inp io att ao b tt c nh)
  "ATTENTION-FORWARD's gradients: into DINP's Q K V, by way of the scores'
   gradients (in DST at DPO before the softmax, DAO after)."
  (declare (type f32vec dinp dst dout inp att) (type fixnum dio dpo dao doo io ao b tt c nh))
  (let* ((c3 (* 3 c)) (hs (floor c nh)) (scale (/ 1f0 (sqrt (coerce hs 'single-float)))))
    (declare (type fixnum c3 hs) (type single-float scale))
    ;; one (batch, head) a job: the keys' and values' gradients sum over queries
    (parallel-for (bh (* b nh) pool)
      (fast
        (multiple-value-bind (bb h) (floor bh nh)
          (dotimes (tq tt)
            (let* ((row (+ (* bb nh tt tt) (* h tt tt) (* tq tt)))
                   (pa (+ ao row)) (dpa (+ dao row)) (dpre (+ dpo row))
                   (q (+ io (* bb tt c3) (* tq c3) (* h hs)))
                   (dq (+ dio (* bb tt c3) (* tq c3) (* h hs)))
                   (d (+ doo (* bb tt c) (* tq c) (* h hs))))
              (declare (type fixnum row pa dpa dpre q dq d))
              (loop for t2 of-type fixnum from 0 to tq
                    do (let ((vv (+ io (* bb tt c3) (* t2 c3) (* h hs) (* 2 c)))
                             (dv (+ dio (* bb tt c3) (* t2 c3) (* h hs) (* 2 c)))
                             (a (aref att (+ pa t2))) (s 0f0))
                         (declare (type fixnum vv dv) (type single-float a s))
                         (dotimes (i hs)
                           (incf s (* (aref inp (+ vv i)) (aref dout (+ d i))))
                           (incf (aref dinp (+ dv i)) (* a (aref dout (+ d i)))))
                         (incf (aref dst (+ dpa t2)) s)))
              (loop for t2 of-type fixnum from 0 to tq
                    do (let ((a2 (aref att (+ pa t2))) (g (aref dst (+ dpa t2))))
                         (declare (type single-float a2 g))
                         (loop for t3 of-type fixnum from 0 to tq
                               do (incf (aref dst (+ dpre t3))
                                        (* (* a2 (- (if (= t2 t3) 1f0 0f0) (aref att (+ pa t3)))) g)))))
              (loop for t2 of-type fixnum from 0 to tq
                    do (let ((k (+ io (* bb tt c3) (* t2 c3) (* h hs) c))
                             (dk (+ dio (* bb tt c3) (* t2 c3) (* h hs) c))
                             (g (* (aref dst (+ dpre t2)) scale)))
                         (declare (type fixnum k dk) (type single-float g))
                         (dotimes (i hs)
                           (incf (aref dinp (+ dq i)) (* (aref inp (+ k i)) g))
                           (incf (aref dinp (+ dk i)) (* (aref inp (+ q i)) g))))))))))))

(defconstant +gelu-scale+ (coerce (sqrt (/ 2d0 pi)) 'single-float))

(defun gelu-forward (pool out oo inp io n)
  (declare (type f32vec out inp) (type fixnum oo io n))
  (parallel-for (i n pool)
    (fast
      (let* ((x (aref inp (+ io i))) (cube (* 0.044715f0 x x x)))
        (setf (aref out (+ oo i)) (* 0.5f0 x (+ 1f0 (tanh (* +gelu-scale+ (+ x cube))))))))))

(defun gelu-backward (pool dinp dio inp io dout doo n)
  (declare (type f32vec dinp inp dout) (type fixnum dio io doo n))
  (parallel-for (i n pool)
    (fast
      (let* ((x (aref inp (+ io i))) (cube (* 0.044715f0 x x x))
             (arg (* +gelu-scale+ (+ x cube)))
             (th (tanh arg)) (ch (cosh arg)) (sech (/ 1f0 (* ch ch)))
             (g (+ (* 0.5f0 (+ 1f0 th))
                   (* x 0.5f0 sech +gelu-scale+ (+ 1f0 (* 3f0 0.044715f0 x x))))))
        (incf (aref dinp (+ dio i)) (* g (aref dout (+ doo i))))))))

(defun residual-forward (out oo a ao bv bo n)
  (declare (type f32vec out a bv) (type fixnum oo ao bo n))
  (fast (dotimes (i n) (setf (aref out (+ oo i)) (+ (aref a (+ ao i)) (aref bv (+ bo i)))))))

(defun residual-backward (da dao db dbo dout doo n)
  (declare (type f32vec da db dout) (type fixnum dao dbo doo n))
  (fast (dotimes (i n)
          (let ((d (aref dout (+ doo i))))
            (incf (aref da (+ dao i)) d) (incf (aref db (+ dbo i)) d)))))

(defun softmax-forward (pool probs po logits lo n v vp)
  "Each of N rows of VP: the softmax of its first V logits, its padding 0."
  (declare (type f32vec probs logits) (type fixnum po lo n v vp))
  (parallel-for (r n pool)
    (fast
      (let ((l (+ lo (* r vp))) (p (+ po (* r vp))) (mx -10000f0) (sum 0f0))
        (declare (type fixnum l p) (type single-float mx sum))
        (dotimes (i v) (when (> (aref logits (+ l i)) mx) (setf mx (aref logits (+ l i)))))
        (dotimes (i v) (let ((e (exp (- (aref logits (+ l i)) mx)))) (setf (aref probs (+ p i)) e) (incf sum e)))
        (dotimes (i v) (setf (aref probs (+ p i)) (/ (aref probs (+ p i)) sum)))
        (loop for i of-type fixnum from v below vp do (setf (aref probs (+ p i)) 0f0))))))

(defun crossentropy-forward (losses lso probs po targets n vp)
  (declare (type f32vec losses probs) (type (simple-array (signed-byte 32) (*)) targets)
           (type fixnum lso po n vp))
  (fast (dotimes (r n) (setf (aref losses (+ lso r)) (- (log (aref probs (+ po (* r vp) (aref targets r)))))))))

(defun crossentropy-softmax-backward (pool dlogits dlo dlosses dlso probs po targets n v vp)
  (declare (type f32vec dlogits dlosses probs) (type (simple-array (signed-byte 32) (*)) targets)
           (type fixnum dlo dlso po n v vp))
  (parallel-for (r n pool)
    (fast
      (let ((dl (aref dlosses (+ dlso r))) (ix (aref targets r)) (d (+ dlo (* r vp))) (p (+ po (* r vp))))
        (declare (type single-float dl) (type fixnum d p))
        (dotimes (i v)
          (incf (aref dlogits (+ d i)) (* (- (aref probs (+ p i)) (if (= i ix) 1f0 0f0)) dl)))))))

;;; ---------------------------------------------------------------------------
;;; the passes

(defun gpt2-forward (m inputs targets b tt)
  "Forward over B rows of TT tokens (INPUTS, TARGETS: int32 vectors of B*TT,
   TARGETS optional): the activations, and with TARGETS the mean loss."
  (declare (type fixnum b tt))
  (let ((v (gpt2-v m)) (vp (gpt2-vp m)) (l (gpt2-l m)) (nh (gpt2-nh m)) (c (gpt2-c m))
        (pool (gpt2-pool m)))
    (dotimes (i (* b tt))
      (assert (< -1 (aref inputs i) v))
      (when targets (assert (< -1 (aref targets i) v))))
    (if (null (gpt2-acts m))
        (let ((sizes (act-sizes-for m b tt)))
          (setf (gpt2-b m) b (gpt2-tt m) tt
                (gpt2-act-sizes m) sizes (gpt2-act-offsets m) (offsets sizes)
                (gpt2-n-acts m) (reduce #'+ sizes)
                (gpt2-acts m) (f32vec (gpt2-n-acts m))
                (gpt2-inputs m) (make-array (* b tt) :element-type '(signed-byte 32))
                (gpt2-targets m) (make-array (* b tt) :element-type '(signed-byte 32))))
        (unless (and (= b (gpt2-b m)) (= tt (gpt2-tt m)))
          (error "model is set up for B=~D T=~D, not B=~D T=~D" (gpt2-b m) (gpt2-tt m) b tt)))
    (replace (gpt2-inputs m) inputs)
    (when targets (replace (gpt2-targets m) targets))
    (let ((p (gpt2-params m)) (a (gpt2-acts m)) (n (* b tt)))
      (sb-int:with-float-traps-masked (:overflow :inexact :underflow)
        (encoder-forward a (aoff m +encoded+) (gpt2-inputs m) p (poff m +wte+) (poff m +wpe+) b tt c)
        (dotimes (li l)
          (let ((res (if (zerop li) (aoff m +encoded+) (aoff m +residual3+ (1- li)))))
            (layernorm-forward pool a (aoff m +ln1+ li) a (aoff m +ln1-mean+ li) (aoff m +ln1-rstd+ li) a res
                               p (poff m +ln1w+ li) (poff m +ln1b+ li) n c)
            (matmul-forward pool a (aoff m +qkv+ li) a (aoff m +ln1+ li) p (poff m +qkvw+ li) p (poff m +qkvb+ li) n c (* 3 c))
            (attention-forward pool a (aoff m +atty+ li) a (aoff m +preatt+ li) (aoff m +att+ li) a (aoff m +qkv+ li) b tt c nh)
            (matmul-forward pool a (aoff m +attproj+ li) a (aoff m +atty+ li) p (poff m +attprojw+ li) p (poff m +attprojb+ li) n c c)
            (residual-forward a (aoff m +residual2+ li) a res a (aoff m +attproj+ li) (* n c))
            (layernorm-forward pool a (aoff m +ln2+ li) a (aoff m +ln2-mean+ li) (aoff m +ln2-rstd+ li) a (aoff m +residual2+ li)
                               p (poff m +ln2w+ li) (poff m +ln2b+ li) n c)
            (matmul-forward pool a (aoff m +fch+ li) a (aoff m +ln2+ li) p (poff m +fcw+ li) p (poff m +fcb+ li) n c (* 4 c))
            (gelu-forward pool a (aoff m +fch-gelu+ li) a (aoff m +fch+ li) (* n 4 c))
            (matmul-forward pool a (aoff m +fcproj+ li) a (aoff m +fch-gelu+ li) p (poff m +fcprojw+ li) p (poff m +fcprojb+ li) n (* 4 c) c)
            (residual-forward a (aoff m +residual3+ li) a (aoff m +residual2+ li) a (aoff m +fcproj+ li) (* n c))))
        (layernorm-forward pool a (aoff m +lnf+) a (aoff m +lnf-mean+) (aoff m +lnf-rstd+) a (aoff m +residual3+ (1- l))
                           p (poff m +lnfw+) (poff m +lnfb+) n c)
        (matmul-forward pool a (aoff m +logits+) a (aoff m +lnf+) p (poff m +wte+) nil 0 n c vp)
        (softmax-forward pool a (aoff m +probs+) a (aoff m +logits+) n v vp)
        (if targets
            (progn
              (crossentropy-forward a (aoff m +losses+) a (aoff m +probs+) (gpt2-targets m) n vp)
              (let ((s 0f0))
                (dotimes (i n) (incf s (aref a (+ (aoff m +losses+) i))))
                (setf (gpt2-mean-loss m) (/ s n))))
            (setf (gpt2-mean-loss m) -1f0))))
    m))

(defun gpt2-zero-grad (m)
  (when (gpt2-grads m) (fill (the f32vec (gpt2-grads m)) 0f0))
  (when (gpt2-dacts m) (fill (the f32vec (gpt2-dacts m)) 0f0))
  m)

(defun gpt2-backward (m)
  "The gradients of the mean loss of the last forward (with targets), added
   into the parameters' (GPT2-GRADS) and the activations'."
  (when (= (gpt2-mean-loss m) -1f0) (error "forward with targets first"))
  (unless (gpt2-grads m)
    (setf (gpt2-grads m) (f32vec (gpt2-n-params m))
          (gpt2-dacts m) (f32vec (gpt2-n-acts m))))
  (let* ((b (gpt2-b m)) (tt (gpt2-tt m)) (v (gpt2-v m)) (vp (gpt2-vp m)) (l (gpt2-l m))
         (nh (gpt2-nh m)) (c (gpt2-c m)) (n (* b tt)) (pool (gpt2-pool m))
         (p (gpt2-params m)) (g (gpt2-grads m)) (a (gpt2-acts m)) (da (gpt2-dacts m)))
    (sb-int:with-float-traps-masked (:overflow :inexact :underflow)
      (fill da (/ 1f0 n) :start (aoff m +losses+) :end (+ (aoff m +losses+) n))
      (crossentropy-softmax-backward pool da (aoff m +logits+) da (aoff m +losses+) a (aoff m +probs+)
                                     (gpt2-targets m) n v vp)
      (matmul-backward pool da (aoff m +lnf+) g (poff m +wte+) nil 0 da (aoff m +logits+)
                       a (aoff m +lnf+) p (poff m +wte+) n c vp)
      (layernorm-backward pool da (aoff m +residual3+ (1- l)) g (poff m +lnfw+) (poff m +lnfb+)
                          da (aoff m +lnf+) a (aoff m +residual3+ (1- l)) p (poff m +lnfw+)
                          a (aoff m +lnf-mean+) (aoff m +lnf-rstd+) n c)
      (loop for li from (1- l) downto 0
            do (let ((res (if (zerop li) (aoff m +encoded+) (aoff m +residual3+ (1- li)))))
                 (residual-backward da (aoff m +residual2+ li) da (aoff m +fcproj+ li)
                                    da (aoff m +residual3+ li) (* n c))
                 (matmul-backward pool da (aoff m +fch-gelu+ li) g (poff m +fcprojw+ li) g (poff m +fcprojb+ li)
                                  da (aoff m +fcproj+ li) a (aoff m +fch-gelu+ li) p (poff m +fcprojw+ li) n (* 4 c) c)
                 (gelu-backward pool da (aoff m +fch+ li) a (aoff m +fch+ li) da (aoff m +fch-gelu+ li) (* n 4 c))
                 (matmul-backward pool da (aoff m +ln2+ li) g (poff m +fcw+ li) g (poff m +fcb+ li)
                                  da (aoff m +fch+ li) a (aoff m +ln2+ li) p (poff m +fcw+ li) n c (* 4 c))
                 (layernorm-backward pool da (aoff m +residual2+ li) g (poff m +ln2w+ li) (poff m +ln2b+ li)
                                     da (aoff m +ln2+ li) a (aoff m +residual2+ li) p (poff m +ln2w+ li)
                                     a (aoff m +ln2-mean+ li) (aoff m +ln2-rstd+ li) n c)
                 (residual-backward da res da (aoff m +attproj+ li) da (aoff m +residual2+ li) (* n c))
                 (matmul-backward pool da (aoff m +atty+ li) g (poff m +attprojw+ li) g (poff m +attprojb+ li)
                                  da (aoff m +attproj+ li) a (aoff m +atty+ li) p (poff m +attprojw+ li) n c c)
                 (attention-backward pool da (aoff m +qkv+ li) da (aoff m +preatt+ li) (aoff m +att+ li)
                                     da (aoff m +atty+ li) a (aoff m +qkv+ li) a (aoff m +att+ li) b tt c nh)
                 (matmul-backward pool da (aoff m +ln1+ li) g (poff m +qkvw+ li) g (poff m +qkvb+ li)
                                  da (aoff m +qkv+ li) a (aoff m +ln1+ li) p (poff m +qkvw+ li) n c (* 3 c))
                 (layernorm-backward pool da res g (poff m +ln1w+ li) (poff m +ln1b+ li)
                                     da (aoff m +ln1+ li) a res p (poff m +ln1w+ li)
                                     a (aoff m +ln1-mean+ li) (aoff m +ln1-rstd+ li) n c)))
      (encoder-backward g (poff m +wte+) (poff m +wpe+) da (aoff m +encoded+) (gpt2-inputs m) b tt c))
    m))

(defun gpt2-update (m lr beta1 beta2 eps weight-decay step)
  "AdamW (PyTorch's), STEP counting from 1."
  (declare (type single-float lr beta1 beta2 eps weight-decay))
  (unless (gpt2-m m)
    (setf (gpt2-m m) (f32vec (gpt2-n-params m)) (gpt2-vv m) (f32vec (gpt2-n-params m))))
  (let ((p (gpt2-params m)) (g (gpt2-grads m)) (mm (gpt2-m m)) (vv (gpt2-vv m))
        (c1 (- 1f0 (expt beta1 step))) (c2 (- 1f0 (expt beta2 step))))
    (declare (type f32vec p g mm vv) (type single-float c1 c2))
    (sb-int:with-float-traps-masked (:overflow :inexact :underflow)
      (parallel-for (i (gpt2-n-params m) (gpt2-pool m))
        (fast
          (let* ((w (aref p i)) (gr (aref g i))
                 (mi (+ (* beta1 (aref mm i)) (* (- 1f0 beta1) gr)))
                 (vi (+ (* beta2 (aref vv i)) (* (- 1f0 beta2) gr gr))))
            (setf (aref mm i) mi (aref vv i) vi
                  (aref p i) (- w (* lr (+ (/ (/ mi c1) (+ (sqrt (the (single-float 0f0) (/ vi c2))) eps)) (* weight-decay w)))))))))
    m))

;;; ---------------------------------------------------------------------------
;;; data, tokens, samples: llm.c's token files, its tokenizer, its sampler

(defstruct (token-loader (:constructor %make-token-loader))
  (tokens nil :type (simple-array (unsigned-byte 16) (*)))
  (b 0 :type fixnum) (tt 0 :type fixnum) (pos 0 :type fixnum)
  (inputs nil) (targets nil))

(defun make-token-loader (path b tt)
  "Batches of B rows of TT tokens from llm.c's token file PATH (header, then
   uint16 GPT-2 tokens), in order, wrapping at the end -- its loader unshuffled."
  (with-raw-file (fd path)
    (let* ((h (read-header fd 20240520)) (n (aref h 2))
           (tokens (make-array n :element-type '(unsigned-byte 16))))
      (unless (= (aref h 1) 1) (error "token file version ~D, want 1" (aref h 1)))
      (raw-read fd tokens (* 2 n))
      (%make-token-loader :tokens tokens :b b :tt tt
                          :inputs (make-array (* b tt) :element-type '(signed-byte 32))
                          :targets (make-array (* b tt) :element-type '(signed-byte 32))))))

(defun loader-batches (ld) (floor (1- (length (token-loader-tokens ld))) (* (token-loader-b ld) (token-loader-tt ld))))
(defun loader-reset (ld) (setf (token-loader-pos ld) 0) ld)

(defun loader-next (ld)
  "The next batch: its inputs and targets (the inputs shifted by one)."
  (let* ((n (* (token-loader-b ld) (token-loader-tt ld))) (tk (token-loader-tokens ld)))
    (when (> (+ (token-loader-pos ld) n 1) (length tk)) (setf (token-loader-pos ld) 0))
    (let ((p (token-loader-pos ld)))
      (dotimes (i n)
        (setf (aref (token-loader-inputs ld) i) (aref tk (+ p i))
              (aref (token-loader-targets ld) i) (aref tk (+ p i 1))))
      (incf (token-loader-pos ld) n))
    (values (token-loader-inputs ld) (token-loader-targets ld))))

(defun load-gpt2-tokenizer (path)
  "llm.c's gpt2_tokenizer.bin: each token's bytes, and the end-of-text token."
  (let* ((bytes (with-raw-file (fd path)
                  (let ((v (make-array (sb-posix:stat-size (sb-posix:stat (namestring (truename path))))
                                       :element-type '(unsigned-byte 8))))
                    (raw-read fd v (length v)))))
         (h (lambda (i) (logior (aref bytes (* 4 i)) (ash (aref bytes (+ 1 (* 4 i))) 8)
                                (ash (aref bytes (+ 2 (* 4 i))) 16) (ash (aref bytes (+ 3 (* 4 i))) 24)))))
    (assert (= (funcall h 0) 20240328))
    (let* ((n (funcall h 2)) (eot (if (= (funcall h 1) 1) 50256 (funcall h 3)))
           (pieces (make-array n)) (p 1024))
      (dotimes (i n)
        (let ((len (aref bytes p)))
          (setf (aref pieces i) (subseq bytes (1+ p) (+ p 1 len)))
          (incf p (1+ len))))
      (values pieces eot))))

(defun xorshift-u32 (state)
  "llm.c's random_u32: the next state (a 64-bit word) and its 32-bit output."
  (let ((s state))
    (setf s (logxor s (ash s -12)))
    (setf s (ldb (byte 64 0) (logxor s (ash s 25))))
    (setf s (logxor s (ash s -27)))
    (values s (ash (ldb (byte 64 0) (* s #x2545F4914F6CDD1D)) -32))))

(defun sample-probs (probs po n coin)
  "The index the cumulative distribution PROBS[PO..PO+N) first exceeds COIN at."
  (let ((cdf 0f0))
    (dotimes (i n (1- n))
      (incf cdf (aref probs (+ po i)))
      (when (< coin cdf) (return i)))))

(defun gpt2-train-demo (&key (llmc (merge-pathnames "workspace/llm.c/" (user-homedir-pathname)))
                             (steps 40) (b 4) (tt 64) (lr 1f-4) (val-batches 5) (gen-t 64)
                             (out *standard-output*))
  "train_gpt2.c's main: GPT-2 124M fine-tuned on tinyshakespeare with AdamW,
   the validation loss every 10 steps, a sample every 20."
  (let* ((dir (uiop:ensure-directory-pathname llmc))
         (m (load-gpt2 (merge-pathnames "gpt2_124M.bin" dir)))
         (train (make-token-loader (merge-pathnames "dev/data/tinyshakespeare/tiny_shakespeare_train.bin" dir) b tt))
         (val (make-token-loader (merge-pathnames "dev/data/tinyshakespeare/tiny_shakespeare_val.bin" dir) b tt))
         (rng 1337) (gen (make-array (* b tt) :element-type '(signed-byte 32))))
    (multiple-value-bind (pieces eot) (load-gpt2-tokenizer (merge-pathnames "gpt2_tokenizer.bin" dir))
      (format out "~:D parameters; train ~D batches, val ~D~%" (gpt2-n-params m) (loader-batches train) (loader-batches val))
      (dotimes (step (1+ steps))
        (when (zerop (mod step 10))
          (loader-reset val)
          (let ((s 0f0))
            (dotimes (i val-batches)
              (multiple-value-bind (x y) (loader-next val) (gpt2-forward m x y b tt))
              (incf s (gpt2-mean-loss m)))
            (format out "val loss ~,6F~%" (/ s val-batches))))
        (when (and (plusp step) (zerop (mod step 20)))
          (fill gen eot)
          (let ((text (make-array 0 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0)))
            (loop for tpos from 1 below gen-t
                  do (gpt2-forward m gen nil b tt)
                     (multiple-value-bind (s u) (xorshift-u32 rng)
                       (setf rng s)
                       (let ((next (sample-probs (gpt2-acts m) (+ (aoff m +probs+) (* (1- tpos) (gpt2-vp m)))
                                                 (gpt2-v m) (/ (ash u -8) 16777216f0))))
                         (setf (aref gen tpos) next)
                         (loop for byte across (aref pieces next) do (vector-push-extend byte text)))))
            (format out "generating:~%---~%~A~%---~%"
                    (sb-ext:octets-to-string (coerce text '(simple-array (unsigned-byte 8) (*)))
                                             :external-format '(:utf-8 :replacement #\?)))))
        (when (< step steps)
          (let ((t0 (get-internal-real-time)))
            (multiple-value-bind (x y) (loader-next train) (gpt2-forward m x y b tt))
            (gpt2-zero-grad m)
            (gpt2-backward m)
            (gpt2-update m lr 0.9f0 0.999f0 1f-8 0f0 (1+ step))
            (format out "step ~D: train loss ~,6F (took ~,1F ms)~%" step (gpt2-mean-loss m)
                    (/ (- (get-internal-real-time) t0) (/ internal-time-units-per-second 1000)))
            (finish-output out))))
      m)))
