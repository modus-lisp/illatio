;;;; metal-inkling2.lisp — Inkling 2 Mini on the GPU (src/metal/inkling2.metal).
;;;
;;; The checkpoint is one mapped file, wrapped as one buffer: every weight
;;; is read where it lies (BF16 matrices, NVFP4 and MXFP8 experts), nothing
;;; is converted or copied.  A token is one command buffer: inkling2.lisp's
;;; INK2-TOKEN!, step for step, its 48 layer applications in order.

(in-package #:illatio)

(defstruct (ink2-gpu (:constructor %make-ink2-gpu))
  gpu file                              ; the device, and the checkpoint as a buffer
  (q8 (make-hash-table :test 'equal)))  ; tensor name -> its Q8_0 copy (*INK2-Q8*)

(defvar *ink2-q8* t
  "Whether the GPU reads the BF16 projections (and the output head) as Q8_0
   copies made at load -- half the bytes a token, the format llama.cpp's
   GGUFs of this model keep them in -- rather than as BF16 from the file.
   The router stays BF16 either way.")

(defun ink2-q8-copy (gpu bytes off rows cols pool)
  "A Q8_0 copy of the BF16 ROWS x COLS matrix at OFF in BYTES (ggml's
   quantize_row_q8_0: per 32, d = amax/127 as fp16, q = round(x/d))."
  (declare (type (simple-array (unsigned-byte 8) (*)) bytes) (type fixnum off rows cols))
  (let* ((nb (floor cols 32)) (rb (* nb 34))
         (b (gpu-buffer gpu (* rows rb))) (sap (gbuf-sap b)))
    (parallel-for (r rows pool)
      (let ((xs (make-array 32 :element-type 'single-float)))
        (dotimes (blk nb)
          (let ((amax 0f0) (o (+ off (* 2 (+ (* r cols) (* blk 32))))))
            (dotimes (i 32)
              (let ((x (bits-single (ash (logior (aref bytes (+ o (* 2 i))) (ash (aref bytes (+ o 1 (* 2 i))) 8)) 16))))
                (setf (aref xs i) x amax (max amax (abs x)))))
            (let* ((d (/ amax 127f0)) (id (if (zerop d) 0f0 (/ 1f0 d))) (dst (+ (* r rb) (* blk 34)))
                   (h (single-to-half d)))
              (setf (sb-sys:sap-ref-8 sap dst) (ldb (byte 8 0) h) (sb-sys:sap-ref-8 sap (1+ dst)) (ldb (byte 8 8) h))
              (dotimes (i 32)
                (let ((q (let ((v (* (aref xs i) id))) (if (minusp v) (- (floor (+ (- v) 0.5f0))) (floor (+ v 0.5f0))))))
                  (setf (sb-sys:sap-ref-8 sap (+ dst 2 i)) (ldb (byte 8 0) (max -127 (min 127 q)))))))))))
    b))

(defun ink2-gpu-open (m)
  "Wrap M's mapped checkpoint for the GPU, and (*INK2-Q8*) make the
   projections' Q8_0 copies."
  (let* ((st (ink2-st m)) (bytes (stfile-bytes st))
         (page 16384)
         (span (* page (ceiling (length bytes) page)))
         (gpu (open-gpu))
         (ig (%make-ink2-gpu :gpu gpu :file (gpu-wrap gpu (sb-sys:vector-sap bytes) span))))
    (when *ink2-q8*                        ; T: all of them; :HEAD: the output head only
      (flet ((q8 (name)
               (unless (or (eq *ink2-q8* t) (search "unembed" name)) (return-from q8 nil))
               (destructuring-bind (dtype shape &rest r) (st-tensor st name)
                 (declare (ignore r))
                 (assert (string= dtype "BF16"))
                 (setf (gethash name (ink2-gpu-q8 ig))
                       (ink2-q8-copy gpu bytes (st-offset st name) (first shape) (second shape) (ink2-pool m))))))
        (dotimes (l 24)
          (dolist (n '("attn.wq_du.weight" "attn.wk_dv.weight" "attn.wv_dv.weight" "attn.wo_ud.weight"
                       "attn.wr_du.weight" "attn.gate_attn.weight" "attn.attn_sink_proj.weight"))
            (q8 (format nil "model.llm.layers.~D.~A" l n)))
          (unless (ink2-layer-local (aref (ink2-layers m) l))
            (q8 (format nil "model.llm.layers.~D.attn.k_sconv.weight_proj" l))
            (q8 (format nil "model.llm.layers.~D.attn.v_sconv.weight_proj" l)))
          (when (<= 6 l 17)
            (dolist (c '("attn" "k" "v" "mlp"))
              (dolist (w '("down" "forget" "gain"))
                (q8 (format nil "model.llm.layers.~D.controllers.~A.~A_weight" l c w))))))
        (q8 "model.llm.unembed.weight")))
    ig))

;;; --- per-context state ---------------------------------------------------------
;;; Buffers for a chunk of up to NMAX tokens; a token's vector is a row, the
;;; stack's slot k the NMAX x D block at k * NMAX * D (a chunk of N uses the
;;; first N rows of each, slots N rows apart).

(defparameter *ink2-chunk* 256 "Tokens a prefill chunk takes at most.")

(defvar *ink2-variant* '()
  "Experiments with the undocumented choices, a plist: :k-norm (0 RMS, 1
   L2-clipped like Q, 2 none), :act (INK_ACT_N's modes), :rel-scale,
   :qk-scale.  Empty: the choices docs/inkling2.md settled.")
(defun ink2-v (key default) (getf *ink2-variant* key default))

(defstruct (ink2gs (:constructor %make-ink2gs))
  nmax stack
  xq xk xv xm nq nk nv nm
  q k0 v0 k v r sink gate o y y2 dk dv
  msg zin z fz gz tmp
  rl ids wts count start pairs h a ye om
  toks logits
  apps)                                 ; per application: (k-cache v-cache k-hist v-hist a-hist m-hist cap)

(defun ink2-gpu-state (m n-ctx)
  (let* ((g (ink2-gpu-gpu (ink2-metal m))) (d +ink-d+) (nm *ink2-chunk*))
    (flet ((buf (floats) (let ((b (gpu-buffer g (* 4 (max 1 floats))))) (gbuf-set-floats b (f32vec (max 1 floats))) b)))
      (%make-ink2gs
       :nmax nm :stack (buf (* 8 nm d))
       :xq (buf (* nm d)) :xk (buf (* nm d)) :xv (buf (* nm d)) :xm (buf (* nm d))
       :nq (buf (* nm d)) :nk (buf (* nm d)) :nv (buf (* nm d)) :nm (buf (* nm d))
       :q (buf (* nm 3072)) :k0 (buf (* nm 1024)) :v0 (buf (* nm 1024)) :k (buf (* nm 1024)) :v (buf (* nm 1024))
       :r (buf (* nm 384)) :sink (buf (* nm 24)) :gate (buf (* nm 3072)) :o (buf (* nm 3072))
       :y (buf (* nm d)) :y2 (buf (* nm d)) :dk (buf (* nm 32)) :dv (buf (* nm 32))
       :msg (buf (* nm d)) :zin (buf (* nm 2 d)) :z (buf (* nm 64)) :fz (buf (* nm d)) :gz (buf (* nm d)) :tmp (buf (* nm d))
       :rl (buf (* nm 128)) :ids (buf (* nm 8)) :wts (buf (* nm 8))
       :count (buf 128) :start (buf 128) :pairs (buf (* nm 8))
       :h (buf (* nm 8 2 +ink-ffn+)) :a (buf (* nm 8 +ink-ffn+)) :ye (buf (* nm 8 d)) :om (buf (* nm d))
       :toks (buf nm) :logits (buf (ink2-n-vocab m))
       :apps (coerce
              (loop for (l . nil) in (ink2-app-order m)
                    ;; a local layer's ring: its window and a chunk's worth more, so a
                    ;; chunk's later tokens don't overwrite what its earlier ones still see
                    collect (let ((cap (if (ink2-layer-local (aref (ink2-layers m) l)) (min n-ctx (+ +ink-window+ nm)) n-ctx)))
                              (list (buf (* cap 1024)) (buf (* cap 1024)) (buf (* 3 1024)) (buf (* 3 1024))
                                    (buf (* 3 d)) (buf (* 3 d)) cap)))
              'vector)))))

;;; --- a chunk of tokens -------------------------------------------------------------

(defun ink2-gpu-chunk! (c tokens)
  "TOKENS (at most *INK2-CHUNK*) at C's next positions, as one command
   buffer: INK2-TOKEN! for each, the weights read once for the chunk where
   that pays (the Q8_0 projections, the router, the experts grouped).  The
   last token's logits land in C."
  (let* ((m (ink2ctx-model c)) (ig (ink2-metal m)) (gpu (ink2-gpu-gpu ig)) (file (ink2-gpu-file ig))
         (st (ink2-st m)) (p0 (ink2ctx-pos c)) (d +ink-d+) (eps (ink2-eps m))
         (n (length tokens)) (one (= n 1))
         (s (or (ink2ctx-gpu c) (setf (ink2ctx-gpu c) (ink2-gpu-state m (ink2ctx-n-ctx c))))))
    (assert (<= 1 n (ink2gs-nmax s)))
    (assert (<= (+ p0 n) (ink2ctx-n-ctx c)) () "Inkling 2: the context holds ~D tokens" (ink2ctx-n-ctx c))
    (let ((sap (gbuf-sap (ink2gs-toks s))))
      (loop for tk in tokens for i from 0 do (setf (sb-sys:sap-ref-32 sap (* 4 i)) tk)))
    (labels ((w (name) (cons file (st-offset st (concatenate 'string "model.llm." name))))
             (lw (l name) (w (format nil "layers.~D.~A" l name)))
             (wn (l name) (format nil "model.llm.layers.~D.~A" l name))
             (slot (k) (cons (ink2gs-stack s) (* 4 k n d)))
             (n1 (e kernel cnt &rest bufs) (dispatch gpu e kernel (list cnt) bufs (ceiling cnt 256) 256))
             (mv (e wt rows cols x y)
               ;; WT: a tensor name (its Q8_0 copy, else the file's BF16), or (buffer . offset) BF16
               (let ((q (and (stringp wt) (gethash wt (ink2-gpu-q8 ig)))))
                 (cond ((and q one) (dispatch gpu e "d4_mv_q8_0" (list cols rows (* 34 (floor cols 32)) rows 0)
                                              (list q x y) (ceiling rows 8) 64))
                       (q (dispatch gpu e "d4_mm_q8_0" (list cols rows n (* 34 (floor cols 32)) cols rows)
                                    (list q x y) (list (ceiling n 64) (ceiling rows 64) 1) 128 :shared 16384))
                       ((stringp wt) (mv e (cons file (st-offset st wt)) rows cols x y))
                       (one (dispatch gpu e "ink_mv_bf16" (list cols rows) (list wt x y) (ceiling rows 4) 128))
                       (t (dispatch gpu e "ink_mv_bf16_n" (list cols rows n) (list wt x y)
                                    (list (ceiling rows 4) (ceiling n 8)) 128)))))
             (rms (e x out &key gain (len d) (groups n) (threads 1024))
               (dispatch gpu e "ink_rms" (list len (if gain 1 0) eps) (list x (or gain x) out) groups threads :shared 128))
             (depth (e q slots out)
               (dispatch gpu e "ink_depth_n"
                         (append (list d (length slots)) slots (make-list (- 8 (length slots)) :initial-element 0)
                                 (list eps n (ink2-v :depth-temp (/ 1f0 (sqrt (float d))))))
                         (list (ink2gs-stack s) q out) n 1024 :shared 128))
             (conv (e hist x taps y &key (len d) dyn kv)
               (if (and kv (eql (ink2-v :kv-conv 0) 2))
                   (n1 e "ink_copy" (* n len) x y)
                   (dispatch gpu e "ink_conv_n" (list len (if dyn (ink2-v :dyn 1) 0) n (if (if kv (eql (ink2-v :kv-conv 0) 1) (eql (ink2-v :post-conv 0) 1)) 1 0))
                             (list hist x taps (or dyn x) y) (list (ceiling len 256) n) 256))
               (dispatch gpu e "ink_hist_n" (list len 0 n 0) (list hist x) (ceiling len 256) 256))
             (mix (e l sk x out)
               (let* ((i (- l 6))
                      (u (if (member sk '(:attn :mlp)) "middle_cycle_cross_pass_queries" "middle_cycle_cross_pass_kv_queries"))
                      (row (+ (* 2 i) (if (member sk '(:mlp :v)) 1 0)))
                      (q (let ((b (w u))) (cons (car b) (+ (cdr b) (* 4 row d)))))
                      (cn (ecase sk (:attn "attn") (:k "k") (:v "v") (:mlp "mlp"))))
                 (depth e q '(6 7) (ink2gs-msg s))
                 ;; [RMSNorm(x); RMSNorm(m)] a token's row of 2d: x's at its start, m's after
                 (dispatch gpu e "ink_rms2" (list d eps) (list x (ink2gs-msg s) (ink2gs-zin s)) n 1024 :shared 128)
                 (mv e (wn l (format nil "controllers.~A.down_weight" cn)) 64 (* 2 d) (ink2gs-zin s) (ink2gs-z s))
                 (together (e)
                   (mv e (wn l (format nil "controllers.~A.forget_weight" cn)) d 64 (ink2gs-z s) (ink2gs-fz s))
                   (mv e (wn l (format nil "controllers.~A.gain_weight" cn)) d 64 (ink2gs-z s) (ink2gs-gz s)))
                 (dispatch gpu e "ink_mix" (list (* n d)) (list x (ink2gs-msg s) (ink2gs-fz s) (ink2gs-gz s) out)
                           (ceiling (* n d) 256) 256)))
             (layer (e l blocks pass)
               (let* ((ly (aref (ink2-layers m) l)) (local (ink2-layer-local ly))
                      (core (<= 6 l 17)) (mixed (and core (>= pass 2)))
                      (app (aref (ink2gs-apps s) (ink2-app-index m l pass)))
                      (cap (seventh app))
                      ;; the block's running sum is a source once it has something in it:
                      ;; for a block's first layer, its attention reads the completed blocks only
                      (srcs (if (and (zerop (mod l 6)) (eql (ink2-v :empty-part 1) 1)) blocks (append blocks '(5)))))
                 ;; the attention's inputs
                 (cond ((zerop l) (n1 e "ink_copy" (* n d) (slot 0) (ink2gs-xq s)))
                       (local (depth e (w (format nil "attn_res.attn_queries.~D" (1- l))) srcs (ink2gs-xq s)))
                       (t (loop for buf in (list (ink2gs-xq s) (ink2gs-xk s) (ink2gs-xv s)) for i from 0
                                do (depth e (let ((b (w (format nil "attn_res.attn_queries.~D" (1- l))))) (cons (car b) (+ (cdr b) (* 4 i d))))
                                          srcs buf))))
                 (when (or local (zerop l))
                   (together (e)
                     (n1 e "ink_copy" (* n d) (ink2gs-xq s) (ink2gs-xk s))
                     (n1 e "ink_copy" (* n d) (ink2gs-xq s) (ink2gs-xv s))))
                 (when mixed
                   (loop for x in (list (ink2gs-xq s) (ink2gs-xk s) (ink2gs-xv s)) for sk in '(:attn :k :v)
                         do (mix e l sk x (ink2gs-tmp s)) (n1 e "ink_copy" (* n d) (ink2gs-tmp s) x)))
                 (together (e)
                   (if local
                       (loop for x in (list (ink2gs-xq s) (ink2gs-xk s) (ink2gs-xv s)) for nx in (list (ink2gs-nq s) (ink2gs-nk s) (ink2gs-nv s))
                             do (rms e x nx :gain (lw l "attn_norm.weight")))
                       (progn (rms e (ink2gs-xq s) (ink2gs-nq s) :gain (lw l "q_input_norm.weight"))
                              (rms e (ink2gs-xk s) (ink2gs-nk s) :gain (lw l "k_input_norm.weight"))
                              (rms e (ink2gs-xv s) (ink2gs-nv s) :gain (lw l "v_input_norm.weight")))))
                 ;; projections
                 (together (e)
                   (mv e (wn l "attn.wq_du.weight") 3072 d (ink2gs-nq s) (ink2gs-q s))
                   (mv e (wn l "attn.wk_dv.weight") 1024 d (ink2gs-nk s) (ink2gs-k0 s))
                   (mv e (wn l "attn.wv_dv.weight") 1024 d (ink2gs-nv s) (ink2gs-v0 s))
                   (mv e (wn l "attn.wr_du.weight") 384 d (ink2gs-nq s) (ink2gs-r s))
                   (mv e (wn l "attn.attn_sink_proj.weight") 24 d (ink2gs-nq s) (ink2gs-sink s))
                   (mv e (wn l "attn.gate_attn.weight") 3072 d (ink2gs-nq s) (ink2gs-gate s))
                   (unless local
                     (mv e (wn l "attn.k_sconv.weight_proj") 32 d (ink2gs-nk s) (ink2gs-dk s))
                     (mv e (wn l "attn.v_sconv.weight_proj") 32 d (ink2gs-nv s) (ink2gs-dv s))))
                 (progn
                   (if local
                       (progn (conv e (third app) (ink2gs-k0 s) (lw l "attn.k_sconv.weight") (ink2gs-k s) :len 1024 :kv t)
                              (conv e (fourth app) (ink2gs-v0 s) (lw l "attn.v_sconv.weight") (ink2gs-v s) :len 1024 :kv t))
                       (progn (conv e (third app) (ink2gs-k0 s) (lw l "attn.k_sconv.static_w") (ink2gs-k s) :len 1024 :dyn (ink2gs-dk s) :kv t)
                              (conv e (fourth app) (ink2gs-v0 s) (lw l "attn.v_sconv.static_w") (ink2gs-v s) :len 1024 :dyn (ink2gs-dv s) :kv t))))
                 ;; K, V normed per head into the cache; Q and r clipped
                 (together (e)
                   (dispatch gpu e "ink_kv_n" (list p0 cap eps (ink2-v :k-norm 0) (ink2-qk-clip m)) (list (ink2gs-k s) (first app)) (list 8 n) 128 :shared 128)
                   (dispatch gpu e "ink_kv_n" (list p0 cap eps 0 0f0) (list (ink2gs-v s) (second app)) (list 8 n) 128 :shared 128)
                   (dispatch gpu e "ink_clip" (list 128 (ink2-qk-clip m) 24 (ink2-v :q-clip 0)) (list (ink2gs-q s) (lw l "attn.q_l2_clip.scale")) (* 24 n) 128 :shared 128)
                   (dispatch gpu e "ink_clip" (list 16 (ink2-rel-clip m) 24 (ink2-v :r-clip 0)) (list (ink2gs-r s) (lw l "attn.rel_l2_clip.scale")) (* 24 n) 16 :shared 128))
                 (dispatch gpu e "ink_attend_n" (list p0 cap (ink2-v :qk-scale (/ 1f0 (sqrt 128f0))) +ink-extent+ (if local +ink-window+ 0) (ink2-v :rel-scale 1f0)
                                 (ink2-v :gqa-mod 0) (ink2-v :sink 1))
                           (list (ink2gs-q s) (ink2gs-r s) (ink2gs-sink s) (first app) (second app)
                                 (lw l "attn.rel_logits_proj.proj") (ink2gs-o s))
                           (list 24 n) 256 :shared 256)
                 (dispatch gpu e "ink_gate" (list (* n 3072) (ink2-v :gate 0)) (list (ink2gs-o s) (ink2gs-gate s)) (ceiling (* n 3072) 256) 256)
                 (mv e (wn l "attn.wo_ud.weight") d 3072 (ink2gs-o s) (ink2gs-y s))
                 (conv e (fifth app) (ink2gs-y s) (lw l "attn_sconv.weight") (ink2gs-y2 s))
                 (n1 e "ink_add" (* n d) (slot 5) (ink2gs-y2 s))
                 ;; the MoE
                 (depth e (w (format nil "attn_res.mlp_queries.~D" l)) (append blocks '(5)) (ink2gs-xm s))
                 (when mixed (mix e l :mlp (ink2gs-xm s) (ink2gs-tmp s)) (n1 e "ink_copy" (* n d) (ink2gs-tmp s) (ink2gs-xm s)))
                 (rms e (ink2gs-xm s) (ink2gs-nm s) :gain (lw l "mlp_norm.weight"))
                 (mv e (lw l "mlp.gate.weight") 128 d (ecase (ink2-v :router-in 0) (0 (ink2gs-nm s)) (1 (ink2gs-xm s))) (ink2gs-rl s))
                 (dispatch gpu e "ink_route_n" (list (ink2-route-scale m) (ink2-v :bias-w 0))
                           (list (ink2gs-rl s) (lw l "mlp.gate.bias") (ink2gs-ids s) (ink2gs-wts s)) n 128)
                 (let* ((w13 (ink2-layer-w13 ly))
                        (nv (eq (ink2-experts-kind w13) :nvfp4))
                        (p (format nil "layers.~D.mlp.experts." l)))
                   (flet ((wt (wn suffix) (w (concatenate 'string p wn suffix))))
                     (if one
                         (flet ((xmv (wn rows cols x xs y)
                                  (dispatch gpu e (if nv "ink_xmv_nvfp4" "ink_xmv_mxfp8") (list cols rows xs d (if nv 2688f0 1f0))
                                            (list (wt wn "_weight") (wt wn "_weight_scale") (if nv (wt wn "_weight_amax") x)
                                                  (ink2gs-ids s) x y)
                                            (list (ceiling rows (if nv 8 4)) 8) 128)))
                           (xmv "w13" (* 2 +ink-ffn+) d (ink2gs-nm s) 0 (ink2gs-h s))
                           (dispatch gpu e "ink_act_n" (list +ink-ffn+ 8 (ink2-v :act 0) (if (and (not nv) (eql (ink2-v :mx-halves 0) 1)) 1 0)) (list (ink2gs-h s) (ink2gs-a s)) (ceiling (* 8 +ink-ffn+) 256) 256)
                           (xmv "w2" d +ink-ffn+ (ink2gs-a s) +ink-ffn+ (ink2gs-ye s)))
                         (flet ((xmm (wn rows cols x x-pair y)
                                  (dispatch gpu e (if nv "ink_xmm_nvfp4" "ink_xmm_mxfp8") (list cols rows (if x-pair 1 0) (if nv 2688f0 1f0))
                                            (list (wt wn "_weight") (wt wn "_weight_scale") (if nv (wt wn "_weight_amax") x)
                                                  (ink2gs-count s) (ink2gs-start s) (ink2gs-pairs s) x y)
                                            (list (ceiling rows 64) 128 (ceiling n 8)) 64)))
                           (dispatch gpu e "ink_group" (list n) (list (ink2gs-ids s) (ink2gs-count s) (ink2gs-start s) (ink2gs-pairs s)) 1 1)
                           (xmm "w13" (* 2 +ink-ffn+) d (ink2gs-nm s) nil (ink2gs-h s))
                           (dispatch gpu e "ink_act_n" (list +ink-ffn+ (* 8 n) (ink2-v :act 0) (if (and (not nv) (eql (ink2-v :mx-halves 0) 1)) 1 0)) (list (ink2gs-h s) (ink2gs-a s)) (ceiling (* 8 n +ink-ffn+) 256) 256)
                           (xmm "w2" d +ink-ffn+ (ink2gs-a s) t (ink2gs-ye s))))))
                 (dispatch gpu e "ink_combine_n" (list d n) (list (ink2gs-ye s) (ink2gs-wts s) (ink2gs-om s)) (list (ceiling d 256) n) 256)
                 (conv e (sixth app) (ink2gs-om s) (lw l "mlp_sconv.weight") (ink2gs-y2 s))
                 (n1 e "ink_add" (* n d) (slot 5) (ink2gs-y2 s))))
             (run-block (e from to blocks into pass)
               (dispatch gpu e "ink_zero" (list (* n d)) (list (slot 5)) (ceiling (* n d) 256) 256)
               (loop for l from from below to do (layer e l blocks pass))
               (n1 e "ink_copy" (* n d) (slot 5) (slot into))))
      (with-gpu-work (e gpu)
        (dispatch gpu e "ink_embed_n" (list d) (list (w "embed.weight") (ink2gs-toks s) (ink2gs-tmp s)) (list (ceiling d 256) n) 256)
        (rms e (ink2gs-tmp s) (slot 0) :gain (and (eql (ink2-v :embed-norm 1) 1) (w "embed_norm.weight")))
        (run-block e 0 6 '(0) 1 1)
        (loop for pass from 1 to (ink2-passes m)
              do (run-block e 6 12 '(0 1) 2 pass)
                 (run-block e 12 18 '(0 1 2) 3 pass)
                 (together (e)
                   (n1 e "ink_copy" (* n d) (slot 2) (slot 6))
                   (n1 e "ink_copy" (* n d) (slot 3) (slot 7))))
        (run-block e 18 24 '(0 1 2 3) 4 1)
        ;; the output: the last token's only
        (dispatch gpu e "ink_depth_n"
                  (list d 5 0 1 2 3 4 0 0 0 eps n (ink2-v :depth-temp (/ 1f0 (sqrt (float d)))))
                  (list (ink2gs-stack s) (w "attn_res.output_query") (ink2gs-tmp s)) n 1024 :shared 128)
        (rms e (cons (ink2gs-tmp s) (* 4 (1- n) d)) (ink2gs-xm s) :gain (w "norm.weight") :groups 1)
        (let ((one t)) (declare (ignorable one))
          (dispatch gpu e "d4_mv_q8_0" (list d (ink2-n-vocab m) (* 34 (floor d 32)) (ink2-n-vocab m) 0)
                    (list (gethash "model.llm.unembed.weight" (ink2-gpu-q8 ig)) (ink2gs-xm s) (ink2gs-logits s))
                    (ceiling (ink2-n-vocab m) 8) 64))
        (dispatch gpu e "ink_softcap" (list (ink2-n-vocab m) (ink2-softcap m)) (list (ink2gs-logits s))
                  (ceiling (ink2-n-vocab m) 256) 256))
      (replace (ink2ctx-logits c) (gbuf-floats (ink2gs-logits s) (ink2-n-vocab m)))
      (incf (ink2ctx-pos c) n)
      (ink2ctx-logits c))))

(defun ink2-gpu-token! (c token) (ink2-gpu-chunk! c (list token)))

(defun ink2-gpu-eval! (c tokens)
  "TOKENS a chunk at a time."
  (loop while tokens
        do (let ((k (min (length tokens) *ink2-chunk*)))
             (ink2-gpu-chunk! c (subseq tokens 0 k))
             (setf tokens (nthcdr k tokens))))
  (ink2ctx-logits c))
