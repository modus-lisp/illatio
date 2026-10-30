;;;; metal-gpt2-train.lisp — GPT-2's training step on the GPU.
;;;
;;; gpt2-train.lisp's passes, encoded into one command buffer a step: the
;;; parameters, their gradients, AdamW's moments, the activations and their
;;; gradients in five GPU buffers laid out as the CPU's vectors are (the same
;;; offsets), so a step's results can be checked against the CPU's.  The
;;; kernels are src/metal-train/gpt2.metal, compiled on their own: inference
;;; doesn't pay for them.
;;;
;;; Attention is matrix products: each (batch, head)'s scores Q K^T (scaled),
;;; a causal softmax of each row, the output att V; backward, datt = dy V^T,
;;; dV += att^T dy, the softmax's gradient, dQ += dpre K, dK += dpre^T Q.  The
;;; scores past the diagonal are computed and masked, as the CPU's are zero.

(in-package #:illatio)

(defstruct (g2gpu (:constructor %make-g2gpu))
  gpu model                             ; the GPU; the CPU model, for shapes and offsets
  params grads mom vel acts dacts       ; f32 buffers
  inputs targets                        ; int32 buffers of B*T
  (b 0 :type fixnum) (tt 0 :type fixnum))

(defun gpt2-metal-source ()
  (uiop:read-file-string (merge-pathnames "metal-train/gpt2.metal"
                                          (asdf:system-relative-pathname "illatio" "src/"))))

(defun copy-to-gbuf (b v &optional (start 0) (n (length v)))
  "Floats V[0..N) into buffer B from float START."
  (declare (type f32vec v))
  (sb-sys:with-pinned-objects (v)
    (sb-alien:alien-funcall
     (sb-alien:extern-alien "memcpy" (function sb-sys:system-area-pointer sb-sys:system-area-pointer
                                               sb-sys:system-area-pointer sb-alien:unsigned-long))
     (sb-sys:sap+ (gbuf-sap b) (* 4 start)) (sb-sys:vector-sap v) (* 4 n)))
  b)

(defun copy-from-gbuf (b v &optional (start 0) (n (length v)))
  "Floats START..START+N of buffer B into V."
  (declare (type f32vec v))
  (sb-sys:with-pinned-objects (v)
    (sb-alien:alien-funcall
     (sb-alien:extern-alien "memcpy" (function sb-sys:system-area-pointer sb-sys:system-area-pointer
                                               sb-sys:system-area-pointer sb-alien:unsigned-long))
     (sb-sys:vector-sap v) (sb-sys:sap+ (gbuf-sap b) (* 4 start)) (* 4 n)))
  v)

(defun make-gpt2-gpu (m b tt &key (gpu (open-gpu (gpt2-metal-source))))
  "The GPU's copy of CPU model M (its parameters), for steps of B rows of TT tokens."
  (unless (gpt2-act-sizes m)
    (let ((sizes (act-sizes-for m b tt)))
      (setf (gpt2-b m) b (gpt2-tt m) tt (gpt2-act-sizes m) sizes
            (gpt2-act-offsets m) (offsets sizes) (gpt2-n-acts m) (reduce #'+ sizes))))
  (assert (and (= b (gpt2-b m)) (= tt (gpt2-tt m))))
  (let* ((np (gpt2-n-params m)) (na (gpt2-n-acts m))
         (g (%make-g2gpu :gpu gpu :model m :b b :tt tt
                         :params (gpu-buffer gpu (* 4 np)) :grads (gpu-buffer gpu (* 4 np))
                         :mom (gpu-buffer gpu (* 4 np)) :vel (gpu-buffer gpu (* 4 np))
                         :acts (gpu-buffer gpu (* 4 na)) :dacts (gpu-buffer gpu (* 4 na))
                         :inputs (gpu-buffer gpu (* 4 b tt)) :targets (gpu-buffer gpu (* 4 b tt)))))
    (copy-to-gbuf (g2gpu-params g) (gpt2-params m))
    g))

(defun set-gbuf-ints (buf v)
  (dotimes (i (length v) buf) (setf (sb-sys:signed-sap-ref-32 (gbuf-sap buf) (* 4 i)) (aref v i))))

;;; --- encoding -----------------------------------------------------------------

(defmacro with-g2 ((g) &body body)
  "BODY with GPU, M and the buffer helpers P G A DA (the parameter, gradient,
   activation and activation-gradient buffers at a float offset) bound."
  `(let* ((gpu (g2gpu-gpu ,g)) (m (g2gpu-model ,g)))
     (declare (ignorable gpu m))
     (flet ((p (o) (cons (g2gpu-params ,g) (* 4 o))) (g (o) (cons (g2gpu-grads ,g) (* 4 o)))
            (a (o) (cons (g2gpu-acts ,g) (* 4 o))) (da (o) (cons (g2gpu-dacts ,g) (* 4 o))))
       (declare (ignorable #'p #'g #'a #'da))
       ,@body)))

(defvar *g2-precision* '((:head . "_b3") (:attention . "_b") (:linear . "_b"))
  "Each kind of product's kernel: :head (the logits and their gradients), :attention,
   :linear -- \"\" f32 (exact, and on this GPU a tenth as fast), \"_b\" bf16 inputs
   with f32 sums, \"_b3\" each input as two bf16s (hi + lo) and three products, about
   f32's accuracy at a third of bf16's speed.  The head needs it: bf16 there moves the
   loss by 0.04, where everything else in bf16 moves it 0.003.")

(defun g2-gemm (gpu e ta tb mm nn kk a lda b ldb c ldc
                &key (kind :linear) (alpha 1f0) (beta 0) bias (batch 1) (nb2 1) (sa '(0 0)) (sb '(0 0)) (sc '(0 0)))
  "C = ALPHA op(A) op(B) (+ C if BETA) (+ BIAS), MMxNN by KK; BATCH of them,
   the z-th's offsets (z / NB2) and (z mod NB2) times SA's, SB's, SC's strides.
   KIND picks the precision (*G2-PRECISION*)."
  (flet ((off (b) (if (consp b) (cdr b) 0)))
    (let ((vec (and (every (lambda (x) (zerop (mod x 4))) (list* lda ldb ldc (append sa sb sc)))
                    (every (lambda (b) (zerop (mod (off b) 16))) (list a b c)))))
      (g2-dispatch-gemm gpu e ta tb (or (cdr (assoc kind *g2-precision*)) "")
                        mm nn kk a lda b ldb c ldc alpha
                        (logior (if (eql beta 0) 0 1) (if vec 2 0)) bias batch nb2 sa sb sc))))

(defun g2-dispatch-gemm (gpu e ta tb suffix mm nn kk a lda b ldb c ldc alpha beta bias batch nb2 sa sb sc)
  (dispatch gpu e (concatenate 'string
                              (cond ((and ta tb) (error "no TT product"))
                                    (ta "g2_gemm_tn") (tb "g2_gemm_nt") (t "g2_gemm_nn"))
                              suffix)
            (list mm nn kk lda ldb ldc alpha beta (if bias 1 0) nb2
                  (first sa) (second sa) (first sb) (second sb) (first sc) (second sc))
            (list a b c (or bias c))
            (list (ceiling nn 64) (ceiling mm 64) batch) 128 :shared 16384))

(defun g2-1d (gpu e kernel n args buffers)
  (dispatch gpu e kernel (cons n args) buffers (ceiling n 256) 256))

(defun g2-encode-forward (g e &key (targets t))
  (with-g2 (g)
    (let* ((b (g2gpu-b g)) (tt (g2gpu-tt g)) (c (gpt2-c m)) (nh (gpt2-nh m)) (l (gpt2-l m))
           (v (gpt2-v m)) (vp (gpt2-vp m)) (n (* b tt)) (hs (floor c nh))
           (scale (/ 1f0 (sqrt (coerce hs 'single-float)))))
      (dispatch gpu e "g2_encoder_fwd" (list n tt c)
                (list (g2gpu-inputs g) (p (poff m +wte+)) (p (poff m +wpe+)) (a (aoff m +encoded+)))
                (ceiling (* n c) 256) 256)
      (flet ((layernorm (out mean rstd inp w bias)
               (dispatch gpu e "g2_layernorm_fwd" (list n c)
                         (list (a inp) (p w) (p bias) (a out) (a mean) (a rstd)) n 32))
             (linear (out inp w bias cin cout &optional (kind :linear))
               (g2-gemm gpu e nil t n cout cin (a inp) cin (p w) cin (a out) cout :kind kind
                        :bias (and bias (p bias)))))
        (dotimes (li l)
          (let ((res (if (zerop li) (aoff m +encoded+) (aoff m +residual3+ (1- li))))
                (qkv (aoff m +qkv+ li)))
            (layernorm (aoff m +ln1+ li) (aoff m +ln1-mean+ li) (aoff m +ln1-rstd+ li) res
                       (poff m +ln1w+ li) (poff m +ln1b+ li))
            (linear qkv (aoff m +ln1+ li) (poff m +qkvw+ li) (poff m +qkvb+ li) c (* 3 c))
            ;; scores, softmax, output: one product per (batch, head)
            (g2-gemm gpu e nil t tt tt hs (a qkv) (* 3 c) (a (+ qkv c)) (* 3 c) (a (aoff m +preatt+ li)) tt
                     :alpha scale :kind :attention :batch (* b nh) :nb2 nh
                     :sa (list (* tt 3 c) hs) :sb (list (* tt 3 c) hs) :sc (list (* nh tt tt) (* tt tt)))
            (dispatch gpu e "g2_attn_softmax" (list (* b nh tt) tt)
                      (list (a (aoff m +preatt+ li)) (a (aoff m +att+ li))) (* b nh tt) 32)
            (g2-gemm gpu e nil nil tt hs tt (a (aoff m +att+ li)) tt (a (+ qkv (* 2 c))) (* 3 c)
                     (a (aoff m +atty+ li)) c
                     :kind :attention :batch (* b nh) :nb2 nh
                     :sa (list (* nh tt tt) (* tt tt)) :sb (list (* tt 3 c) hs) :sc (list (* tt c) hs))
            (linear (aoff m +attproj+ li) (aoff m +atty+ li) (poff m +attprojw+ li) (poff m +attprojb+ li) c c)
            (g2-1d gpu e "g2_add" (* n c) '() (list (a (aoff m +residual2+ li)) (a res) (a (aoff m +attproj+ li))))
            (layernorm (aoff m +ln2+ li) (aoff m +ln2-mean+ li) (aoff m +ln2-rstd+ li) (aoff m +residual2+ li)
                       (poff m +ln2w+ li) (poff m +ln2b+ li))
            (linear (aoff m +fch+ li) (aoff m +ln2+ li) (poff m +fcw+ li) (poff m +fcb+ li) c (* 4 c))
            (g2-1d gpu e "g2_gelu_fwd" (* n 4 c) '() (list (a (aoff m +fch+ li)) (a (aoff m +fch-gelu+ li))))
            (linear (aoff m +fcproj+ li) (aoff m +fch-gelu+ li) (poff m +fcprojw+ li) (poff m +fcprojb+ li) (* 4 c) c)
            (g2-1d gpu e "g2_add" (* n c) '()
                   (list (a (aoff m +residual3+ li)) (a (aoff m +residual2+ li)) (a (aoff m +fcproj+ li))))))
        (layernorm (aoff m +lnf+) (aoff m +lnf-mean+) (aoff m +lnf-rstd+) (aoff m +residual3+ (1- l))
                   (poff m +lnfw+) (poff m +lnfb+))
        (linear (aoff m +logits+) (aoff m +lnf+) (poff m +wte+) nil c vp :head))
      (dispatch gpu e "g2_softmax_ce" (list n v vp (if targets 1 0))
                (list (a (aoff m +logits+)) (a (aoff m +probs+)) (g2gpu-targets g) (a (aoff m +losses+)))
                n 1024 :shared 128))))

(defun g2-encode-backward (g e)
  "The gradients, into the zeroed gradient buffers (G2-ENCODE-ZERO-GRAD first)."
  (with-g2 (g)
    (let* ((b (g2gpu-b g)) (tt (g2gpu-tt g)) (c (gpt2-c m)) (nh (gpt2-nh m)) (l (gpt2-l m))
           (v (gpt2-v m)) (vp (gpt2-vp m)) (n (* b tt)) (hs (floor c nh))
           (scale (/ 1f0 (sqrt (coerce hs 'single-float)))))
      (dispatch gpu e "g2_ce_bwd" (list n v vp (/ 1f0 n))
                (list (da (aoff m +logits+)) (a (aoff m +probs+)) (g2gpu-targets g))
                (list (ceiling v 256) n) (list 256 1))
      (flet ((linear-back (dout inp w dw dbias cin cout &optional (kind :linear))
               ;; dinp += dout W;  dW += dout^T inp;  dbias += column sums
               (g2-gemm gpu e nil nil n cin cout (da dout) cout (p w) cin (da inp) cin :beta 1 :kind kind)
               (g2-gemm gpu e t nil cout cin n (da dout) cout (a inp) cin (g dw) cin :beta 1 :kind kind)
               (when dbias
                 (dispatch gpu e "g2_colsum" (list n cout) (list (g dbias) (da dout)) (ceiling cout 256) 256)))
             (layernorm-back (dinp dout inp w dw dbias mean rstd)
               (dispatch gpu e "g2_layernorm_bwd_x" (list n c)
                         (list (da dinp) (da dout) (a inp) (p w) (a mean) (a rstd)) n 32)
               (dispatch gpu e "g2_layernorm_bwd_w" (list n c)
                         (list (g dw) (g dbias) (da dout) (a inp) (a mean) (a rstd)) (ceiling c 256) 256)))
        (linear-back (aoff m +logits+) (aoff m +lnf+) (poff m +wte+) (poff m +wte+) nil c vp :head)
        (layernorm-back (aoff m +residual3+ (1- l)) (aoff m +lnf+) (aoff m +residual3+ (1- l))
                        (poff m +lnfw+) (poff m +lnfw+) (poff m +lnfb+) (aoff m +lnf-mean+) (aoff m +lnf-rstd+))
        (loop for li from (1- l) downto 0
              do (let ((res (if (zerop li) (aoff m +encoded+) (aoff m +residual3+ (1- li))))
                       (qkv (aoff m +qkv+ li)) (att (aoff m +att+ li)) (pre (aoff m +preatt+ li))
                       (atty (aoff m +atty+ li)))
                   (g2-1d gpu e "g2_add_to2" (* n c) '()
                          (list (da (aoff m +residual2+ li)) (da (aoff m +fcproj+ li)) (da (aoff m +residual3+ li))))
                   (linear-back (aoff m +fcproj+ li) (aoff m +fch-gelu+ li) (poff m +fcprojw+ li)
                                (poff m +fcprojw+ li) (poff m +fcprojb+ li) (* 4 c) c)
                   (g2-1d gpu e "g2_gelu_bwd" (* n 4 c) '()
                          (list (da (aoff m +fch+ li)) (a (aoff m +fch+ li)) (da (aoff m +fch-gelu+ li))))
                   (linear-back (aoff m +fch+ li) (aoff m +ln2+ li) (poff m +fcw+ li) (poff m +fcw+ li)
                                (poff m +fcb+ li) c (* 4 c))
                   (layernorm-back (aoff m +residual2+ li) (aoff m +ln2+ li) (aoff m +residual2+ li)
                                   (poff m +ln2w+ li) (poff m +ln2w+ li) (poff m +ln2b+ li)
                                   (aoff m +ln2-mean+ li) (aoff m +ln2-rstd+ li))
                   (g2-1d gpu e "g2_add_to2" (* n c) '()
                          (list (da res) (da (aoff m +attproj+ li)) (da (aoff m +residual2+ li))))
                   (linear-back (aoff m +attproj+ li) atty (poff m +attprojw+ li) (poff m +attprojw+ li)
                                (poff m +attprojb+ li) c c)
                   ;; attention: datt = dy V^T; dV += att^T dy; dpre; dQ += s dpre K; dK += s dpre^T Q
                   (g2-gemm gpu e nil t tt tt hs (da atty) c (a (+ qkv (* 2 c))) (* 3 c) (da att) tt
                            :kind :attention :batch (* b nh) :nb2 nh
                            :sa (list (* tt c) hs) :sb (list (* tt 3 c) hs) :sc (list (* nh tt tt) (* tt tt)))
                   (g2-gemm gpu e t nil tt hs tt (a att) tt (da atty) c (da (+ qkv (* 2 c))) (* 3 c)
                            :beta 1 :kind :attention :batch (* b nh) :nb2 nh
                            :sa (list (* nh tt tt) (* tt tt)) :sb (list (* tt c) hs) :sc (list (* tt 3 c) hs))
                   (dispatch gpu e "g2_attn_softmax_bwd" (list (* b nh tt) tt)
                             (list (da pre) (da att) (a att)) (* b nh tt) 32)
                   (g2-gemm gpu e nil nil tt hs tt (da pre) tt (a (+ qkv c)) (* 3 c) (da qkv) (* 3 c)
                            :alpha scale :beta 1 :kind :attention :batch (* b nh) :nb2 nh
                            :sa (list (* nh tt tt) (* tt tt)) :sb (list (* tt 3 c) hs) :sc (list (* tt 3 c) hs))
                   (g2-gemm gpu e t nil tt hs tt (da pre) tt (a qkv) (* 3 c) (da (+ qkv c)) (* 3 c)
                            :alpha scale :beta 1 :kind :attention :batch (* b nh) :nb2 nh
                            :sa (list (* nh tt tt) (* tt tt)) :sb (list (* tt 3 c) hs) :sc (list (* tt 3 c) hs))
                   (linear-back qkv (aoff m +ln1+ li) (poff m +qkvw+ li) (poff m +qkvw+ li) (poff m +qkvb+ li) c (* 3 c))
                   (layernorm-back res (aoff m +ln1+ li) res (poff m +ln1w+ li) (poff m +ln1w+ li) (poff m +ln1b+ li)
                                   (aoff m +ln1-mean+ li) (aoff m +ln1-rstd+ li)))))
      (dispatch gpu e "g2_encoder_bwd" (list n tt c)
                (list (g2gpu-inputs g) (g (poff m +wte+)) (g (poff m +wpe+)) (da (aoff m +encoded+)))
                (ceiling c 256) 256))))

(defun g2-encode-zero-grad (g e)
  (with-g2 (g)
    (g2-1d gpu e "g2_zero" (gpt2-n-params m) '() (list (g2gpu-grads g)))
    (g2-1d gpu e "g2_zero" (gpt2-n-acts m) '() (list (g2gpu-dacts g)))))

(defun g2-encode-update (g e lr beta1 beta2 eps weight-decay step)
  (with-g2 (g)
    (g2-1d gpu e "g2_adamw" (gpt2-n-params m)
           (list lr beta1 beta2 eps weight-decay (- 1f0 (expt beta1 step)) (- 1f0 (expt beta2 step)))
           (list (g2gpu-params g) (g2gpu-grads g) (g2gpu-mom g) (g2gpu-vel g)))))

(defun g2-mean-loss (g)
  (let* ((m (g2gpu-model g)) (n (* (g2gpu-b g) (g2gpu-tt g)))
         (losses (copy-from-gbuf (g2gpu-acts g) (f32vec n) (aoff m +losses+))))
    (/ (reduce #'+ losses) n)))

(defun gpt2-gpu-step (g inputs targets &key update (lr 1f-4) (beta1 0.9f0) (beta2 0.999f0)
                                            (eps 1f-8) (weight-decay 0f0) (step 1))
  "One step on the GPU, one command buffer: forward over INPUTS (int32, B*T)
   with TARGETS, zeroed gradients, backward, and with UPDATE AdamW's step
   STEP.  The mean loss."
  (set-gbuf-ints (g2gpu-inputs g) inputs)
  (set-gbuf-ints (g2gpu-targets g) targets)
  (with-gpu-work (e (g2gpu-gpu g))
    (g2-encode-forward g e)
    (g2-encode-zero-grad g e)
    (g2-encode-backward g e)
    (when update (g2-encode-update g e lr beta1 beta2 eps weight-decay step)))
  (g2-mean-loss g))
