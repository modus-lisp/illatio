;;;; gpt2-train-test.lisp — illatio's GPT-2 training against PyTorch's numbers.
;;;
;;;   sbcl --dynamic-space-size 8192 --non-interactive --load inspect/gpt2-train-test.lisp
;;;
;;; llm.c's test_gpt2.c, ported: its debug state (gpt2_124M_debug_state.bin,
;;; from train_gpt2.py) holds a batch, PyTorch's logits and loss for it, and
;;; every parameter's gradient.  Step 0's logits must agree within 1e-2, its
;;; loss within 1e-2, each gradient within 2e-2 (test_gpt2.c's bounds), and
;;; ten AdamW steps' losses within 1e-2 of PyTorch's.  Needs llm.c's starter
;;; pack ($LLMC, default ~/workspace/llm.c).
;;;
;;; With $ILLATIO_GPT2_GPU=f32 the steps run on the GPU in f32, to the same
;;; bounds.  With $ILLATIO_GPT2_GPU set otherwise they run at the GPU's
;;; precision (*G2-PRECISION*: bf16 products, the head's split), checked to
;;; test_gpt2.cu's bf16 bounds: logits 25, losses 0.05, each gradient its own.

(defvar cl-user::*system* "illatio/train")
(load (merge-pathnames "load.lisp" (or *load-truename* *default-pathname-defaults*)))
(in-package #:illatio)

(defparameter *llmc* (or (uiop:getenv "LLMC") (namestring (merge-pathnames "workspace/llm.c/" (user-homedir-pathname)))))

(defun llmc-file (name) (merge-pathnames name (uiop:ensure-directory-pathname *llmc*)))

(defparameter *expected-losses*
  '(5.270007133483887 4.059706687927246 3.3751230239868164 2.8007826805114746 2.315382242202759
    1.8490285873413086 1.3946564197540283 0.9991465210914612 0.6240804195404053 0.37651097774505615))

(defparameter *gpu-mode* (uiop:getenv "ILLATIO_GPT2_GPU"))
(defparameter *bf16* (and *gpu-mode* (string/= *gpu-mode* "f32")))
(when (equal *gpu-mode* "f32")
  (setf *g2-precision* '((:head . "") (:attention . "") (:linear . ""))))

;; test_gpt2.cu's, for bf16: wte wpe ln1w ln1b qkvw qkvb attprojw attprojb
;; ln2w ln2b fcw fcb fcprojw fcprojb lnfw lnfb
(defparameter *bf16-grad-bounds* #(5f-1 4f-3 1f-1 4f-2 5f-2 3.5f-2 2f-2 3f-2 5f-2 3f-2 3f-2 3f-2 2f-2 1f-2 1f-1 2f-2))

(defun check-tensor (got go want wo n label &optional (tol 2f-2))
  (let ((maxdiff 0f0))
    (dotimes (i n) (setf maxdiff (max maxdiff (abs (- (aref got (+ go i)) (aref want (+ wo i)))))))
    (format t "~:[NOT OK~;ok    ~] ~10A maxdiff ~,3E~%" (<= maxdiff tol) label maxdiff)
    (<= maxdiff tol)))

(let* ((m (load-gpt2 (llmc-file "gpt2_124M.bin")))
       (c (gpt2-c m)) (v (gpt2-v m)) (vp (gpt2-vp m)) (l (gpt2-l m)) (max-t (gpt2-max-t m))
       (allok t) b tt x y want-logits want-loss want-grads)
  (format t "GPT-2: ~D layers, ~D heads, ~D channels, vocab ~D (~D padded), ~:D parameters~%"
          l (gpt2-nh m) c v vp (gpt2-n-params m))
  (with-raw-file (fd (llmc-file "gpt2_124M_debug_state.bin"))
    (let ((h (read-header fd 20240327)))
      (assert (= (aref h 1) 2))
      (setf b (aref h 2) tt (aref h 3)
            x (make-array (* b tt) :element-type '(signed-byte 32))
            y (make-array (* b tt) :element-type '(signed-byte 32))
            want-logits (f32vec (* b tt v)) want-loss (f32vec 1) want-grads (f32vec (gpt2-n-params m)))
      (raw-read fd x (* 4 b tt)) (raw-read fd y (* 4 b tt))
      (raw-read fd want-logits (* 4 b tt v)) (raw-read fd want-loss 4)
      (raw-read fd want-grads (* 4 (gpt2-n-params m)))))
  (format t "batch ~D x ~D tokens~%" b tt)
  (let ((gpu (and *gpu-mode* (make-gpt2-gpu m b tt)))
        (logit-tol (if *bf16* 25f0 1f-2)) (loss-tol (if *bf16* 5f-2 1f-2)))
  (when gpu (format t "on the GPU: ~A, ~:[f32~;~:*~S~]~%" (gpu-name (g2gpu-gpu gpu))
                    (and *bf16* *g2-precision*)))
  (loop for step below 10 for want in *expected-losses*
        do (let ((t0 (get-internal-real-time)))
             (if gpu
                 (progn
                   (setf (gpt2-mean-loss m) (gpt2-gpu-step gpu x y))
                   (when (zerop step)
                     (setf (gpt2-acts m) (copy-from-gbuf (g2gpu-acts gpu) (f32vec (gpt2-n-acts m)))
                           (gpt2-grads m) (copy-from-gbuf (g2gpu-grads gpu) (f32vec (gpt2-n-params m))))))
                 (progn
                   (gpt2-forward m x y b tt)
                   (gpt2-zero-grad m)
                   (gpt2-backward m)))
             (let ((ms (/ (- (get-internal-real-time) t0) (/ internal-time-units-per-second 1000))))
               (when (zerop step)
                 (let ((a (gpt2-acts m)) (lo (aoff m +logits+)) (maxdiff 0f0))
                   (dotimes (bt (* b tt))
                     (dotimes (i v)
                       (setf maxdiff (max maxdiff (abs (- (aref a (+ lo (* bt vp) i)) (aref want-logits (+ (* bt v) i))))))))
                   (format t "~:[NOT OK~;ok    ~] logits     maxdiff ~,3E~%" (< maxdiff logit-tol) maxdiff)
                   (setf allok (and allok (< maxdiff logit-tol))))
                 (let ((ok (< (abs (- (gpt2-mean-loss m) (aref want-loss 0))) loss-tol)))
                   (format t "~:[NOT OK~;ok    ~] loss       ~,6F (PyTorch ~,6F)~%" ok (gpt2-mean-loss m) (aref want-loss 0))
                   (setf allok (and allok ok)))
                 (loop for (i name n) in `((,+wte+ "dwte" ,(* v c)) (,+wpe+ "dwpe" ,(* max-t c))
                                           (,+ln1w+ "dln1w" ,(* l c)) (,+ln1b+ "dln1b" ,(* l c))
                                           (,+qkvw+ "dqkvw" ,(* l 3 c c)) (,+qkvb+ "dqkvb" ,(* l 3 c))
                                           (,+attprojw+ "dattprojw" ,(* l c c)) (,+attprojb+ "dattprojb" ,(* l c))
                                           (,+ln2w+ "dln2w" ,(* l c)) (,+ln2b+ "dln2b" ,(* l c))
                                           (,+fcw+ "dfcw" ,(* l 4 c c)) (,+fcb+ "dfcb" ,(* l 4 c))
                                           (,+fcprojw+ "dfcprojw" ,(* l c 4 c)) (,+fcprojb+ "dfcprojb" ,(* l c))
                                           (,+lnfw+ "dlnfw" ,c) (,+lnfb+ "dlnfb" ,c))
                       do (setf allok (and (check-tensor (gpt2-grads m) (poff m i) want-grads (poff m i) n name
                                                         (if *bf16* (aref *bf16-grad-bounds* i) 2f-2))
                                           allok))))
               (if gpu
                   (with-gpu-work (e (g2gpu-gpu gpu))
                     (g2-encode-update gpu e 1f-4 0.9f0 0.999f0 1f-8 0.01f0 (1+ step) :decayed t))
                   (gpt2-update m 1f-4 0.9f0 0.999f0 1f-8 0.01f0 (1+ step)))
               (let ((ok (< (abs (- (gpt2-mean-loss m) want)) loss-tol)))
                 (setf allok (and allok ok))
                 (format t "step ~D: loss ~,6F (PyTorch ~,6F) ~:[NOT OK~;ok~]  ~,0F ms~%"
                         step (gpt2-mean-loss m) want ok ms))))))
  (format t "~:[FAIL~;PASS~]~%" allok)
  (uiop:quit (if allok 0 1)))
