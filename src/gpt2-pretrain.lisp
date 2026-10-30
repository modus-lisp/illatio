;;;; gpt2-pretrain.lisp — GPT-2 trained from scratch on FineWeb, on the GPU.
;;;
;;; llm.c's recipe for its GPT-2 (124M) reproduction (discussion #481, and
;;; the speedrun's first record): GPT-2's initialization, AdamW (betas 0.9,
;;; 0.95, weight decay 0.1 on the matrices and embeddings only), gradients
;;; clipped to norm 1, a linear warmup then a cosine to the end, steps of
;;; several micro-batches accumulated.  The data is FineWeb's GPT-2 tokens in
;;; llm.c's shards (kjj0/fineweb10B-gpt2: 100M tokens a shard), read in order;
;;; the validation loss is over a fixed slice of the validation shard.

(in-package #:illatio)

;;; --- initialization -----------------------------------------------------------

(defun gaussian-filler (seed)
  "A function of no arguments giving N(0,1) samples: xorshift64*, Box-Muller."
  (let ((s (logior 1 (ldb (byte 64 0) seed))) (spare nil))
    (lambda ()
      (if spare
          (prog1 spare (setf spare nil))
          (flet ((u ()
                   (multiple-value-bind (ns out) (xorshift-u32 s)
                     (setf s ns)
                     (/ (+ (ash out -8) 0.5d0) 16777216d0))))
            (let* ((r (sqrt (* -2d0 (log (u))))) (th (* 2d0 pi (u))))
              (setf spare (coerce (* r (sin th)) 'single-float))
              (coerce (* r (cos th)) 'single-float)))))))

(defun make-gpt2-from-scratch (&key (l 12) (nh 12) (c 768) (max-t 1024) (v 50257) (vp 50304)
                                    (seed 1337) (pool (make-pool)))
  "A GPT-2 of L layers, NH heads, C channels, initialized as GPT-2 was:
   matrices and embeddings N(0, 0.02), the two projections into the residual
   stream N(0, 0.02/sqrt(2L)), biases 0, layernorms 1 and 0.  The padded
   vocabulary's rows are 0 (their logits are masked)."
  (let* ((sizes (param-sizes-for vp c max-t l)) (n (reduce #'+ sizes))
         (m (%make-gpt2 :max-t max-t :v v :vp vp :l l :nh nh :c c
                        :param-sizes sizes :param-offsets (offsets sizes) :n-params n
                        :params (f32vec n) :pool pool))
         (p (gpt2-params m)) (gauss (gaussian-filler seed)))
    (flet ((normal (i std &optional (count (svref sizes i)))
             (let ((o (poff m i)))
               (dotimes (k count) (setf (aref p (+ o k)) (* std (funcall gauss))))))
           (ones (i) (fill p 1f0 :start (poff m i) :end (+ (poff m i) (svref sizes i)))))
      (normal +wte+ 0.02f0 (* v c))
      (normal +wpe+ 0.02f0)
      (dolist (i (list +qkvw+ +fcw+)) (normal i 0.02f0))
      (dolist (i (list +attprojw+ +fcprojw+)) (normal i (/ 0.02f0 (sqrt (* 2f0 l)))))
      (dolist (i (list +ln1w+ +ln2w+ +lnfw+)) (ones i)))
    m))

(defparameter *gpt2-decayed* (list +wte+ +wpe+ +qkvw+ +attprojw+ +fcw+ +fcprojw+)
  "The tensors weight decay applies to: the matrices and embeddings.")

;;; --- data ---------------------------------------------------------------------

(defstruct (shard-loader (:constructor %make-shard-loader))
  files (shard -1) (tokens nil) (pos 0 :type fixnum)
  (b 0 :type fixnum) (tt 0 :type fixnum) inputs targets)

(defun read-token-shard (path)
  (with-raw-file (fd path)
    (let* ((h (read-header fd 20240520)) (n (aref h 2))
           (tokens (make-array n :element-type '(unsigned-byte 16))))
      (raw-read fd tokens (* 2 n))
      tokens)))

(defun make-shard-loader (files b tt)
  "Batches of B rows of TT tokens from FILES (llm.c token shards), in order,
   each shard after the last, back to the first after them all."
  (let ((ld (%make-shard-loader :files (coerce files 'simple-vector) :b b :tt tt
                                :inputs (make-array (* b tt) :element-type '(signed-byte 32))
                                :targets (make-array (* b tt) :element-type '(signed-byte 32)))))
    (shard-advance ld)
    ld))

(defun shard-advance (ld)
  (let ((k (mod (1+ (shard-loader-shard ld)) (length (shard-loader-files ld)))))
    (setf (shard-loader-shard ld) k
          (shard-loader-tokens ld) (read-token-shard (svref (shard-loader-files ld) k))
          (shard-loader-pos ld) 0)))

(defun shard-next (ld)
  (let ((n (* (shard-loader-b ld) (shard-loader-tt ld))))
    (when (> (+ (shard-loader-pos ld) n 1) (length (shard-loader-tokens ld))) (shard-advance ld))
    (let ((tk (shard-loader-tokens ld)) (p (shard-loader-pos ld)))
      (declare (type (simple-array (unsigned-byte 16) (*)) tk) (type fixnum p))
      (dotimes (i n)
        (setf (aref (shard-loader-inputs ld) i) (aref tk (+ p i))
              (aref (shard-loader-targets ld) i) (aref tk (+ p i 1))))
      (incf (shard-loader-pos ld) n))
    (values (shard-loader-inputs ld) (shard-loader-targets ld))))

(defun fineweb-shards (dir kind)
  "DIR's fineweb_KIND_*.bin shards (KIND \"train\" or \"val\"), in order."
  (sort (mapcar #'namestring (directory (merge-pathnames (format nil "fineweb_~A_*.bin" kind)
                                                         (uiop:ensure-directory-pathname dir))))
        #'string<))

;;; --- the loop -----------------------------------------------------------------

(defun gpt2-gpu-loss (g inputs targets)
  "The mean loss of a forward pass alone."
  (set-gbuf-ints (g2gpu-inputs g) inputs)
  (set-gbuf-ints (g2gpu-targets g) targets)
  (with-gpu-work (e (g2gpu-gpu g)) (g2-encode-forward g e))
  (g2-mean-loss g))

(defun gpt2-val-loss (g files batches)
  "The mean loss over the first BATCHES batches of the validation FILES."
  (let ((ld (make-shard-loader files (g2gpu-b g) (g2gpu-tt g))) (s 0d0))
    (dotimes (i batches (/ s batches))
      (multiple-value-bind (x y) (shard-next ld) (incf s (gpt2-gpu-loss g x y))))))

(defun cosine-lr (step steps max-lr warmup min-frac)
  "Linear warmup over WARMUP steps to MAX-LR, then a cosine down to MIN-FRAC of it."
  (if (< step warmup)
      (* max-lr (/ (1+ step) warmup))
      (let ((frac (/ (- step warmup) (max 1 (- steps warmup)))))
        (* max-lr (+ min-frac (* (- 1 min-frac) 0.5 (+ 1 (cos (* pi frac)))))))))

(defun gpt2-pretrain (&key (data (merge-pathnames "workspace/modded-nanogpt/data/fineweb10B/" (user-homedir-pathname)))
                           (b 8) (tt 1024) (accum 4) (steps 100) (lr 6f-4) (warmup 10) (min-lr-frac 0f0)
                           (weight-decay 0.1f0) (clip 1f0) (val-every 25) (val-batches 20)
                           (model (make-gpt2-from-scratch :max-t tt))
                           (precision '((:head . "_b") (:attention . "_b") (:linear . "_b")))
                           (out *standard-output*))
  "Train MODEL (from scratch by default) on FineWeb for STEPS steps of ACCUM
   micro-batches of B rows of TT tokens.  The model, and the GPU's copy."
  (let* ((*g2-precision* precision)
         (g (make-gpt2-gpu model b tt))
         (train (make-shard-loader (fineweb-shards data "train") b tt))
         (val (fineweb-shards data "val"))
         (tokens (* b tt accum)))
    (format out "~:D parameters, ~D layers ~D heads ~D channels; ~:D tokens a step (~D x ~D x ~D)~%"
            (gpt2-n-params model) (gpt2-l model) (gpt2-nh model) (gpt2-c model) tokens accum b tt)
    (dotimes (step (1+ steps))
      (when (or (= step steps) (and (plusp val-every) (zerop (mod step val-every))))
        (format out "step ~D: val loss ~,4F~%" step (gpt2-val-loss g val val-batches))
        (finish-output out))
      (when (< step steps)
        (let ((t0 (get-internal-real-time)) (loss 0d0) (rate (cosine-lr step steps lr warmup min-lr-frac)))
          (dotimes (k accum)
            (multiple-value-bind (x y) (shard-next train)
              (set-gbuf-ints (g2gpu-inputs g) x)
              (set-gbuf-ints (g2gpu-targets g) y))
            (with-gpu-work (e (g2gpu-gpu g))
              (g2-encode-forward g e)
              (g2-encode-backward g e :accumulate (plusp k) :dloss (/ 1f0 (* b tt accum))))
            (incf loss (/ (g2-mean-loss g) accum)))
          (with-gpu-work (e (g2gpu-gpu g))
            (g2-encode-clip g e clip)
            (g2-encode-update g e (coerce rate 'single-float) 0.9f0 0.95f0 1f-8 weight-decay (1+ step)
                              :decayed *gpt2-decayed*))
          (let ((ms (/ (- (get-internal-real-time) t0) (/ internal-time-units-per-second 1000))))
            (format out "step ~D/~D: loss ~,4F  norm ~,3F  lr ~,2E  ~,0F ms  ~,0F tok/s~%"
                    (1+ step) steps loss (g2-grad-norm g) (coerce rate (quote single-float)) ms (/ (* tokens 1000) ms))
            (finish-output out)))))
    (values model g)))
