;;;; nanogpt.lisp — the speedrun's architecture, trained on the GPU.
;;;
;;; modded-nanogpt's records 5-11 (Keller Jordan et al., MIT), from GPT-2:
;;; RMSNorm without weights before attention, the MLP and the head; merged
;;; QKV with no biases; QK-norm then rotary embeddings on each 64-wide head;
;;; ReLU^2; the output projections and an untied head initialized to zero;
;;; the logits softcapped.  Muon for the hidden matrices is nanogpt-muon.lisp.
;;;
;;; The parameters, their gradients and the optimizer's two moments are four
;;; GPU buffers laid out by one table; the activations and their gradients
;;; two more by another.  Each activation gradient is written by its first
;;; producer, as in metal-gpt2-train.lisp, so nothing is zeroed but the
;;; embedding's gradient (a scatter of += over the batch's tokens).

(in-package #:illatio)

(defstruct (layout (:constructor %make-layout))
  (table (make-hash-table)) (total 0 :type fixnum) (order '()))

(defun make-layout (specs layers)
  "SPECS: (name floats per-layer-p).  Each tensor's offset, in order; a
   per-layer tensor holds LAYERS slices of FLOATS."
  (let ((lay (%make-layout)) (o 0))
    (dolist (s specs lay)
      (destructuring-bind (name floats per-layer) s
        (setf (gethash name (layout-table lay)) (list o floats per-layer))
        (setf (layout-order lay) (append (layout-order lay) (list name)))
        (incf o (if per-layer (* layers floats) floats))
        (setf (layout-total lay) o)))))

(defun lof (lay name &optional (layer 0))
  "Offset of tensor NAME (its LAYER's slice) in LAY."
  (destructuring-bind (o floats per-layer) (or (gethash name (layout-table lay)) (error "no tensor ~S" name))
    (if per-layer (+ o (* layer floats)) o)))

(defun lsize (lay name &optional (layers 1))
  "Floats of tensor NAME: all its layers' (LAYERS, when per layer)."
  (destructuring-bind (o floats per-layer) (gethash name (layout-table lay))
    (declare (ignore o))
    (if per-layer (* layers floats) floats)))

(defstruct (ngpt (:constructor %make-ngpt))
  gpu (l 0) (nh 0) (c 0) (v 0) (vp 0) (b 0) (tt 0)
  (cap 30f0) (rope-base 10000f0)
  params acts                            ; layouts
  pbuf gbuf m1buf m2buf abuf dabuf       ; GPU buffers: parameters, gradients, two moments, acts, their gradients
  inputs targets norm)

(defun ngpt-param-specs (l c vp)
  (declare (ignore l))
  `((:wte ,(* vp c) nil)
    (:qkvw ,(* 3 c c) t) (:projw ,(* c c) t) (:fcw ,(* 4 c c) t) (:fcprojw ,(* 4 c c) t)
    (:head ,(* vp c) nil)))

(defun ngpt-act-specs (b tt c nh vp)
  (let ((n (* b tt)))
    `((:x0 ,(* n c) nil)
      (:ln1 ,(* n c) t) (:r1 ,n t) (:qkv ,(* n 3 c) t) (:qk ,(* n 2 c) t) (:qkr ,(* n 2 nh) t)
      (:pre ,(* b nh tt tt) t) (:att ,(* b nh tt tt) t) (:atty ,(* n c) t) (:proj ,(* n c) t)
      (:res2 ,(* n c) t) (:ln2 ,(* n c) t) (:r2 ,n t) (:fch ,(* n 4 c) t) (:fca ,(* n 4 c) t)
      (:fcproj ,(* n c) t) (:res3 ,(* n c) t)
      (:lnf ,(* n c) nil) (:rf ,n nil) (:logits ,(* n vp) nil) (:probs ,(* n vp) nil) (:losses ,n nil))))

(defun nanogpt-metal-source ()
  "Every kernel in src/metal-train/."
  (let ((dir (merge-pathnames "metal-train/" (asdf:system-relative-pathname "illatio" "src/"))))
    (with-output-to-string (o)
      (dolist (f (sort (directory (merge-pathnames "*.metal" dir)) #'string< :key #'namestring))
        (write-string (uiop:read-file-string f) o)))))

(defun make-nanogpt (&key (l 6) (c 384) (v 50257) (vp 50304) (b 8) (tt 1024) (cap 30f0)
                          (seed 1337) gpu)
  "A model of L layers, C channels (C/64 heads), for steps of B rows of TT
   tokens, initialized: the embedding N(0, 1)... as modded-nanogpt's: the
   embedding N(0,1), QKV and the MLP's input uniform +-sqrt(3/C) (std
   1/sqrt(C)), the projections into the residual stream and the head 0."
  (assert (zerop (mod c 64)))
  (let* ((nh (floor c 64)) (gpu (or gpu (open-gpu (nanogpt-metal-source))))
         (pl (make-layout (ngpt-param-specs l c vp) l))
         (al (make-layout (ngpt-act-specs b tt c nh vp) l))
         (np (layout-total pl)) (na (layout-total al))
         (m (%make-ngpt :gpu gpu :l l :nh nh :c c :v v :vp vp :b b :tt tt :cap cap
                        :params pl :acts al
                        :pbuf (gpu-buffer gpu (* 4 np)) :gbuf (gpu-buffer gpu (* 4 np))
                        :m1buf (gpu-buffer gpu (* 4 np)) :m2buf (gpu-buffer gpu (* 4 np))
                        :abuf (gpu-buffer gpu (* 4 na)) :dabuf (gpu-buffer gpu (* 4 na))
                        :inputs (gpu-buffer gpu (* 4 b tt)) :targets (gpu-buffer gpu (* 4 b tt))
                        :norm (gpu-buffer gpu (* 4 (+ 2 +g2-norm-groups+)))))
         (p (f32vec np)) (gauss (gaussian-filler seed)))
    (flet ((fill-normal (name std count)
             (let ((o (lof pl name)))
               (dotimes (k count) (setf (aref p (+ o k)) (* std (funcall gauss)))))))
      (fill-normal :wte 1f0 (* v c))
      (fill-normal :qkvw (/ 1f0 (sqrt (coerce c 'single-float))) (lsize pl :qkvw l))
      (fill-normal :fcw (/ 1f0 (sqrt (coerce c 'single-float))) (lsize pl :fcw l)))
    (copy-to-gbuf (ngpt-pbuf m) p)
    (setf (sb-sys:sap-ref-single (gbuf-sap (ngpt-norm m)) 0) 1f0)
    m))

(defun ngpt-n-params (m) (layout-total (ngpt-params m)))

(defun ngpt-params-vector (m)
  (copy-from-gbuf (ngpt-pbuf m) (f32vec (ngpt-n-params m))))

;;; --- encoding -----------------------------------------------------------------

(defmacro with-ngpt ((m) &body body)
  "BODY with GPU, the layouts PL AL, and P G A DA: the parameter, gradient,
   activation and activation-gradient buffers at tensor NAME (LAYER)."
  `(let* ((gpu (ngpt-gpu ,m)) (pl (ngpt-params ,m)) (al (ngpt-acts ,m)))
     (declare (ignorable gpu pl al))
     (flet ((p (name &optional (li 0)) (cons (ngpt-pbuf ,m) (* 4 (lof pl name li))))
            (g (name &optional (li 0)) (cons (ngpt-gbuf ,m) (* 4 (lof pl name li))))
            (a (name &optional (li 0) (plus 0)) (cons (ngpt-abuf ,m) (* 4 (+ plus (lof al name li)))))
            (da (name &optional (li 0) (plus 0)) (cons (ngpt-dabuf ,m) (* 4 (+ plus (lof al name li))))))
       (declare (ignorable #'p #'g #'a #'da))
       ,@body)))

(defun ngpt-encode-forward (m e &key (targets t))
  (with-ngpt (m)
    (let* ((b (ngpt-b m)) (tt (ngpt-tt m)) (c (ngpt-c m)) (nh (ngpt-nh m)) (l (ngpt-l m))
           (n (* b tt)) (hs 64) (scale (/ 1f0 (sqrt (coerce hs 'single-float)))))
      (flet ((rms (out r in) (dispatch gpu e "ng_rms_fwd" (list n c) (list in out r) n 32))
             (linear (out in w cin cout &optional (kind :linear))
               (g2-gemm gpu e nil t n cout cin in cin w cin out cout :kind kind)))
        (dispatch gpu e "ng_embed_fwd" (list n c) (list (ngpt-inputs m) (p :wte) (a :x0))
                  (ceiling (* n c) 256) 256)
        (dotimes (li l)
          (let ((xin (if (zerop li) (a :x0) (a :res3 (1- li)))))
            (rms (a :ln1 li) (a :r1 li) xin)
            (linear (a :qkv li) (a :ln1 li) (p :qkvw li) c (* 3 c))
            (dispatch gpu e "ng_qk_fwd" (list n tt c nh (ngpt-rope-base m))
                      (list (a :qkv li) (a :qk li) (a :qkr li)) (list n nh 2) 32)
            (g2-gemm gpu e nil t tt tt hs (a :qk li) (* 2 c) (a :qk li c) (* 2 c) (a :pre li) tt
                     :alpha scale :kind :attention :batch (* b nh) :nb2 nh
                     :sa (list (* tt 2 c) hs) :sb (list (* tt 2 c) hs) :sc (list (* nh tt tt) (* tt tt)))
            (dispatch gpu e "g2_attn_softmax" (list (* b nh tt) tt) (list (a :pre li) (a :att li)) (* b nh tt) 32)
            (g2-gemm gpu e nil nil tt hs tt (a :att li) tt (a :qkv li (* 2 c)) (* 3 c) (a :atty li) c
                     :kind :attention :batch (* b nh) :nb2 nh
                     :sa (list (* nh tt tt) (* tt tt)) :sb (list (* tt 3 c) hs) :sc (list (* tt c) hs))
            (linear (a :proj li) (a :atty li) (p :projw li) c c)
            (g2-1d gpu e "g2_add" (* n c) '() (list (a :res2 li) xin (a :proj li)))
            (rms (a :ln2 li) (a :r2 li) (a :res2 li))
            (linear (a :fch li) (a :ln2 li) (p :fcw li) c (* 4 c))
            (g2-1d gpu e "ng_relu2_fwd" (* n 4 c) '() (list (a :fch li) (a :fca li)))
            (linear (a :fcproj li) (a :fca li) (p :fcprojw li) (* 4 c) c)
            (g2-1d gpu e "g2_add" (* n c) '() (list (a :res3 li) (a :res2 li) (a :fcproj li)))))
        (rms (a :lnf) (a :rf) (a :res3 (1- l)))
        (linear (a :logits) (a :lnf) (p :head) c (ngpt-vp m) :head)
        (dispatch gpu e "ng_softcap_ce" (list n (ngpt-v m) (ngpt-vp m) (if targets 1 0) (ngpt-cap m))
                  (list (a :logits) (a :probs) (ngpt-targets m) (a :losses)) n 1024 :shared 128)))))

(defun ngpt-encode-backward (m e &key accumulate (dloss (/ 1f0 (* (ngpt-b m) (ngpt-tt m)))))
  "The gradients, written (or with ACCUMULATE added to the parameters')."
  (with-ngpt (m)
    (let* ((b (ngpt-b m)) (tt (ngpt-tt m)) (c (ngpt-c m)) (nh (ngpt-nh m)) (l (ngpt-l m))
           (vp (ngpt-vp m)) (n (* b tt)) (hs 64) (scale (/ 1f0 (sqrt (coerce hs 'single-float))))
           (acc (if accumulate 1 0)))
      (flet ((linear-back (dout in w dw cin cout &optional (kind :linear))
               (g2-gemm gpu e nil nil n cin cout dout cout w cin (car in) cin :kind kind)
               (g2-gemm gpu e t nil cout cin n dout cout (cdr in) cin dw cin :beta acc :kind kind))
             (rms-back (dx dy x r add)
               (dispatch gpu e "ng_rms_bwd" (list n c (if add 1 0)) (list dx dy x r) n 32)))
        (dispatch gpu e "ng_softcap_ce_bwd" (list n (ngpt-v m) vp dloss (ngpt-cap m))
                  (list (da :logits) (a :probs) (ngpt-targets m) (a :logits))
                  (list (ceiling vp 256) n) (list 256 1))
        ;; IN for LINEAR-BACK: (where the input's gradient goes . the input)
        (linear-back (da :logits) (cons (da :lnf) (a :lnf)) (p :head) (g :head) c vp :head)
        (rms-back (da :res3 (1- l)) (da :lnf) (a :res3 (1- l)) (a :rf) nil)
        (loop for li from (1- l) downto 0
              do (let ((xin (if (zerop li) (a :x0) (a :res3 (1- li))))
                       (dxin (if (zerop li) (da :x0) (da :res3 (1- li)))))
                   (g2-1d gpu e "g2_copy2" (* n c) '() (list (da :res2 li) (da :fcproj li) (da :res3 li)))
                   (linear-back (da :fcproj li) (cons (da :fca li) (a :fca li)) (p :fcprojw li) (g :fcprojw li) (* 4 c) c)
                   (g2-1d gpu e "ng_relu2_bwd" (* n 4 c) '() (list (da :fch li) (a :fch li) (da :fca li)))
                   (linear-back (da :fch li) (cons (da :ln2 li) (a :ln2 li)) (p :fcw li) (g :fcw li) c (* 4 c))
                   (rms-back (da :res2 li) (da :ln2 li) (a :res2 li) (a :r2 li) t)
                   (g2-1d gpu e "g2_copy2" (* n c) '() (list dxin (da :proj li) (da :res2 li)))
                   (linear-back (da :proj li) (cons (da :atty li) (a :atty li)) (p :projw li) (g :projw li) c c)
                   ;; datt = dy V^T; dV = att^T dy; dpre; dQ' = s dpre K'; dK' = s dpre^T Q'
                   (g2-gemm gpu e nil t tt tt hs (da :atty li) c (a :qkv li (* 2 c)) (* 3 c) (da :att li) tt
                            :kind :attention :batch (* b nh) :nb2 nh
                            :sa (list (* tt c) hs) :sb (list (* tt 3 c) hs) :sc (list (* nh tt tt) (* tt tt)))
                   (g2-gemm gpu e t nil tt hs tt (a :att li) tt (da :atty li) c (da :qkv li (* 2 c)) (* 3 c)
                            :kind :attention :batch (* b nh) :nb2 nh
                            :sa (list (* nh tt tt) (* tt tt)) :sb (list (* tt c) hs) :sc (list (* tt 3 c) hs))
                   (dispatch gpu e "g2_attn_softmax_bwd" (list (* b nh tt) tt)
                             (list (da :pre li) (da :att li) (a :att li)) (* b nh tt) 32)
                   (g2-gemm gpu e nil nil tt hs tt (da :pre li) tt (a :qk li c) (* 2 c) (da :qk li) (* 2 c)
                            :alpha scale :kind :attention :batch (* b nh) :nb2 nh
                            :sa (list (* nh tt tt) (* tt tt)) :sb (list (* tt 2 c) hs) :sc (list (* tt 2 c) hs))
                   (g2-gemm gpu e t nil tt hs tt (da :pre li) tt (a :qk li) (* 2 c) (da :qk li c) (* 2 c)
                            :alpha scale :kind :attention :batch (* b nh) :nb2 nh
                            :sa (list (* nh tt tt) (* tt tt)) :sb (list (* tt 2 c) hs) :sc (list (* tt 2 c) hs))
                   (dispatch gpu e "ng_qk_bwd" (list n tt c nh (ngpt-rope-base m))
                             (list (da :qkv li) (da :qk li) (a :qkv li) (a :qkr li)) (list n nh 2) 32)
                   (linear-back (da :qkv li) (cons (da :ln1 li) (a :ln1 li)) (p :qkvw li) (g :qkvw li) c (* 3 c))
                   (rms-back dxin (da :ln1 li) xin (a :r1 li) t)))
        (unless accumulate
          (g2-1d gpu e "g2_zero" (lsize pl :wte) '() (list (g :wte))))
        (dispatch gpu e "ng_embed_bwd" (list n c) (list (ngpt-inputs m) (g :wte) (da :x0))
                  (ceiling c 256) 256)))))

(defun ngpt-mean-loss (m)
  (let* ((n (* (ngpt-b m) (ngpt-tt m)))
         (v (copy-from-gbuf (ngpt-abuf m) (f32vec n) (lof (ngpt-acts m) :losses))))
    (/ (reduce #'+ v) n)))

(defun ngpt-set-batch (m inputs targets)
  (set-gbuf-ints (ngpt-inputs m) inputs)
  (set-gbuf-ints (ngpt-targets m) targets))

(defun ngpt-loss (m inputs targets)
  "The mean loss of a forward pass."
  (ngpt-set-batch m inputs targets)
  (with-gpu-work (e (ngpt-gpu m)) (ngpt-encode-forward m e))
  (ngpt-mean-loss m))

(defun ngpt-grad (m inputs targets)
  "Forward and backward: the mean loss, the gradients in GBUF."
  (ngpt-set-batch m inputs targets)
  (with-gpu-work (e (ngpt-gpu m))
    (ngpt-encode-forward m e)
    (ngpt-encode-backward m e))
  (ngpt-mean-loss m))
