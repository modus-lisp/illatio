;;;; cuda-qwen35.lisp — a qwen35 model on an NVIDIA GPU (src/cuda/qwen35.cu).
;;;
;;; FORWARD-QWEN35's layers (qwen35.lisp) with the device's kernels: the
;;; projections, attention and FFN are the dense path's (cuda-dense.lisp);
;;; the gated delta net's recurrence, the gates, the convolution and the
;;; norms are qwen35.cu's.  A pass is one token or a prefill chunk; the
;;; recurrent parts walk the chunk's tokens in order on the device, so a
;;; chunk goes through the weights once.

(in-package #:illatio)

(defun q35-linear-cuda (dev m cm st l li n)
  "Linear layer LI of N tokens whose normed input is in the state's H (as
   floats) and XQ (quantized): its output added to X."
  (let* ((xq (cctx-xq st)) (xs (cctx-xs st)) (xsum (cctx-xsum st)) (h (cctx-h st))
         (ch (q35-channels m)) (kh (model-ssm-k-heads m)) (kd (model-ssm-k-dim m))
         (vh (model-ssm-v-heads m)) (vd (model-ssm-v-dim m)) (eps (model-eps m)))
    (assert (= kd 128) () "q35_delta is written for 128-wide K heads")
    (assert (zerop (mod vd 32)) () "q35_delta takes values 32 at a time")
    (flet ((mul (w y) (cmat-mul dev (cmat-of cm w) xq xs y n :xsum xsum :h h)))
      (mul (layer-wqkv l) (cctx-qkv st))
      (mul (layer-wz l) (cctx-z st))
      (mul (layer-walpha l) (cctx-alpha st))
      (mul (layer-wbeta l) (cctx-beta st)))
    (launch dev "q35_gates" (ceiling (* n vh) 256) 256
            (list (cctx-alpha st) (cctx-beta st) (cvec-of cm (layer-ssm-a l)) (cvec-of cm (layer-ssm-dt l)) vh n))
    (launch dev "q35_conv" (ceiling ch 256) 256
            (list (cctx-qkv st) (cvec-of cm (layer-conv1d l)) (svref (cctx-conv st) li) (cctx-c st) ch n))
    (launch dev "q35_l2norm" (* n 2 kh) 128 (list (cctx-c st) kd (* 2 kh) ch eps))
    (launch dev "q35_delta" (list vh (floor vd 32)) 32
            (list (cctx-c st) (cctx-beta st) (cctx-alpha st) (svref (cctx-ssm st) li) (cctx-o st)
                  kh vh vd ch n (/ 1f0 (sqrt (coerce kd 'single-float)))))
    ;; normed, gated by silu(z) and quantized for ssm_out
    (launch dev "q35_outnorm" (* n vh) 128
            (list (cctx-o st) (cvec-of cm (layer-ssm-norm l)) (cctx-z st) vd eps xq xs xsum))
    (cmat-mul dev (cmat-of cm (layer-ssm-out l)) xq xs (cctx-x st) n :acc t :xsum xsum)))

(defun q35-full-cuda (dev m cm st l li n pos0)
  "Full-attention layer LI of N tokens at POS0 whose normed input is in the
   state's XQ: its output added to X."
  (let* ((xq (cctx-xq st)) (xs (cctx-xs st)) (xsum (cctx-xsum st)) (h (cctx-h st))
         (hd (model-head-dim m)) (nh (model-n-head m)) (nkv (model-n-kv-head m)) (qd (* nh hd)))
    (assert (<= hd 512))
    (flet ((mul (w y) (cmat-mul dev (cmat-of cm w) xq xs y n :xsum xsum :h h)))
      (mul (layer-wq l) (cctx-qg st))
      (mul (layer-wk l) (cctx-k st))
      (mul (layer-wv l) (cctx-v st)))
    (launch dev "q35_qk" (list (+ nh nkv) n) 128
            (list (cctx-qg st) (cctx-k st) (cctx-v st)
                  (cvec-of cm (layer-q-norm l)) (cvec-of cm (layer-k-norm l)) (cctx-rope st)
                  (cctx-q st) (cctx-gate st) (svref (cctx-kc st) li) (svref (cctx-vc st) li)
                  nh nkv hd (model-rope-dims m) pos0 (model-eps m)))
    (attention dev m st li n pos0 (+ pos0 n) :quantize nil)
    ;; gated by sigmoid(gate), and quantized for Wo
    (launch dev "q35_gate_q8" (ceiling (* n qd) 256) 256
            (list (cctx-att st) (cctx-gate st) xq xs xsum (list :i64 (floor (* n qd) 32))))
    (cmat-mul dev (cmat-of cm (layer-wo l)) xq xs (cctx-x st) n :acc t :xsum xsum)))

(defun cuda-forward-q35 (ctx tokens logits)
  "TOKENS (at most *CUDA-CHUNK*) at the next positions of a qwen35 model;
   the logits, if LOGITS, are the last token's."
  (let* ((m (context-model ctx)) (cm (model-cuda m)) (dev (cmodel-dev cm))
         (st (cuda-state ctx)) (pos0 (context-pos ctx)) (n (length tokens))
         (eps (model-eps m)) (d (model-d m)) (f (model-ffn m))
         (x (cctx-x st)) (xq (cctx-xq st)) (xs (cctx-xs st)) (xsum (cctx-xsum st)))
    (assert (<= 1 n *cuda-chunk*))
    (when (> (+ pos0 n) (context-n-ctx ctx))
      (error "context is full (~D tokens)" (context-n-ctx ctx)))
    (with-cuda
      (cuda-current dev)
      (cuda-upload-vector (cctx-tokens st) (coerce tokens '(simple-array (signed-byte 32) (*))))
      (embed dev (cmat-of cm (model-token-embd m)) (cctx-tokens st) x n d)
      (loop for li below (model-n-layer m)
            for l = (svref (model-layers m) li)
            do ;; the normed input both quantized and as floats (the F32 matrices take those)
               (launch dev "rmsnorm_q8" (list (if (= n 1) (ceiling d 256) 1) n) 256
                       (list x (cvec-of cm (layer-attn-norm l)) (cctx-h st) xq xs xsum d eps))
               (if (eq (layer-kind l) :linear)
                   (q35-linear-cuda dev m cm st l li n)
                   (q35-full-cuda dev m cm st l li n pos0))
               (rmsnorm-q8 dev x (cvec-of cm (layer-ffn-norm l)) xq xs xsum d eps n)
               (glu-mul dev st (cmat-of cm (layer-wgate l)) (cmat-of cm (layer-wup l)) n f d)
               (launch dev "quant_q8" (ceiling (* n f) 256) 256
                       (list (cctx-fa st) xq xs xsum (list :i64 (floor (* n f) 32))))
               (cmat-mul dev (cmat-of cm (layer-wdown l)) xq xs x n :acc t :xsum xsum))
      (when logits
        (rmsnorm-q8 dev (cons x (* 4 (1- n) d)) (cvec-of cm (model-output-norm m)) xq xs xsum d eps 1)
        (cmat-mul dev (cmat-of cm (model-output m)) xq xs (cctx-logits st) 1 :xsum xsum)
        (let ((out (context-logits ctx)))
          (sb-sys:with-pinned-objects (out)
            (cuda-download (cctx-logits st) (sb-sys:vector-sap out) (* 4 (length out)))))))
    (incf (context-pos ctx) n)
    ctx))
