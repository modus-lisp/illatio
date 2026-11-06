;;;; cuda-vision.lisp — DeepSeek V4.1's vision encoder on an NVIDIA GPU: the
;;;; mmproj file's ViT (32 blocks) and projector as ds4 runs them
;;;; (ds4_deepseek4_vision_gpu.cuh), a picture's patches (image.lisp) in, the
;;;; language model's rows for it out.  Kernels: src/cuda/vision.cu.
;;;;
;;;; Its weights and its scratch, sized for the largest picture, are taken
;;;; on the device before the model's expert pool sizes itself to what is
;;;; left (LOAD-VISION-CUDA first).

(in-package #:illatio)

(defstruct (vision (:constructor %make-vision))
  dev
  (w (make-hash-table :test 'equal))    ; tensor name -> its DBUF
  start newline end                     ; the block's markers (F32VECs, WIDTH)
  (width 5120)                          ; the language model's embedding width
  (buf (make-hash-table)))              ; scratch, by name

(defparameter +vision-max-rows+ 9216
  "Patches in the largest picture: V4.1's grid is at most 1024 tokens, each
   3 x 3 patches.")

(defun tensor-octets (g ti)
  (let ((v (make-array (tensor-bytes ti) :element-type '(unsigned-byte 8))))
    (call-with-tensor-bytes g ti (lambda (buf pos n) (replace v buf :start1 pos :end2 n)))
    v))

(defun load-vision-cuda (path &key (device 0))
  "The vision encoder in the mmproj file at PATH (llama.cpp's clip GGUF,
   projector deepseek41v) on CUDA device DEVICE, as *VISION*."
  (let* ((t0 (get-internal-real-time))
         (g (read-gguf path)) (dev (open-cuda device)) (v (%make-vision :dev dev)) (bytes 0))
    (unless (and (equal (meta g "general.architecture") "clip")
                 (equal (meta g "clip.projector_type") "deepseek41v"))
      (error "~A is not a DeepSeek V4.1 vision projector (clip, deepseek41v)" path))
    (unless (and (= (meta g "clip.vision.block_count" 0) 32) (= (meta g "clip.vision.embedding_length" 0) 1024)
                 (= (meta g "clip.vision.patch_size" 0) 14) (= (meta g "clip.vision.projector.scale_factor" 0) 3))
      (error "~A: not the encoder's shape (32 blocks of 1024, 14-pixel patches, 3 x 3 merged)" path))
    (setf (vision-width v) (meta g "clip.vision.projection_dim" 5120))
    (with-cuda
      (cuda-current dev)
      (dolist (name (gguf-tensor-order g))
        (let* ((ti (gguf-tensor g name)) (raw (tensor-octets g ti)))
          (cond ((member name '("v.token_embd.img_start" "v.image_newline" "v.token_embd.img_end") :test #'string=)
                 (let ((f (f32vec (vision-width v))))
                   (dotimes (i (length f))
                     (setf (aref f i) (bits-single (logior (aref raw (* 4 i)) (ash (aref raw (+ 1 (* 4 i))) 8)
                                                           (ash (aref raw (+ 2 (* 4 i))) 16) (ash (aref raw (+ 3 (* 4 i))) 24)))))
                   (cond ((search "start" name) (setf (vision-start v) f))
                         ((search "newline" name) (setf (vision-newline v) f))
                         (t (setf (vision-end v) f)))))
                ((string= name "v.patch_embd.weight")
                 ;; 1024 rows of 588 (channel, row, column), padded to 592: the product's columns go 16 at a time
                 (let ((p (make-array (* 1024 592 4) :element-type '(unsigned-byte 8) :initial-element 0)))
                   (dotimes (r 1024) (replace p raw :start1 (* r 592 4) :start2 (* r 588 4) :end2 (* (1+ r) 588 4)))
                   (setf raw p))))
          (let ((b (cuda-alloc dev (length raw) :zero nil)))
            (sb-sys:with-pinned-objects (raw) (cuda-upload b (sb-sys:vector-sap raw) (length raw)))
            (incf bytes (length raw))
            (setf (gethash name (vision-w v)) b))))
      (unless (and (vision-start v) (vision-newline v) (vision-end v))
        (error "~A: the image start, newline and end embeddings are missing" path))
      (let ((n +vision-max-rows+) (o (ceiling +vision-max-rows+ 9)) (wd (vision-width v)))
        (loop for (key floats) in `((:patches ,(* n 592)) (:x ,(* n 1024)) (:t ,(* n 1024)) (:qkv ,(* n 3072))
                                    (:q ,(* n 1024)) (:k ,(* n 1024)) (:v ,(* n 1024)) (:att ,(* n 1024))
                                    (:g ,(* n 2816)) (:u ,(* n 2816))
                                    (:merged ,(* o 9216)) (:h1 ,(* o wd)) (:h2 ,(* o wd)))
              do (setf (gethash key (vision-buf v)) (cuda-alloc dev (* 4 floats) :zero nil))
                 (incf bytes (* 4 floats)))))
    (format *error-output* "illatio: vision encoder (~,1F GB with its scratch) on ~A, ~,1Fs~%"
            (/ bytes 1d9) (cudev-name dev) (/ (- (get-internal-real-time) t0) internal-time-units-per-second))
    (setf *vision* v)))

(defun vision-run (v patches gh gw)
  "The encoder over PATCHES (GH x GW of them, 592 values each): the
   language model's rows for the picture's ceil(GH/3) x ceil(GW/3) grid, an
   F32VEC of them WIDTH apart."
  (let* ((dev (vision-dev v)) (rows (* gh gw)) (oh (ceiling gh 3)) (ow (ceiling gw 3)) (orows (* oh ow))
         (wd (vision-width v)) (out (f32vec (* orows wd))))
    (assert (<= rows +vision-max-rows+))
    (labels ((b (key) (gethash key (vision-buf v)))
             (w (name) (or (gethash name (vision-w v)) (error "vision: no tensor ~A" name)))
             (blk (il s) (w (format nil "v.blk.~D.~A" il s)))
             (mm (kernel weight ncols nrows n x y)
               (launch dev kernel (list (ceiling n 64) (ceiling nrows 64)) 256 (list ncols nrows n ncols nrows weight x y)))
             (bias (y weight n width &optional res)
               (launch dev "v_bias" (ceiling (* n width) 256) 256 (list (list :i64 (* n width)) width y weight res)))
             (rms (x weight y)
               (launch dev "v_rms" rows 256 (list 1024 1f-6 x weight y))))
      (with-cuda
        (cuda-current dev)
        (cuda-upload-vector (b :patches) patches)
        (mm "d41_mm_f32" (w "v.patch_embd.weight") 592 1024 rows (b :patches) (b :x))
        (bias (b :x) (w "v.patch_embd.bias") rows 1024)
        (let ((x (b :x)) (tt (b :t)))
          (dotimes (il 32)
            (rms x (blk il "ln1.weight") tt)
            (mm "g53_mm_bf16" (blk il "attn_qkv.weight") 1024 3072 rows tt (b :qkv))
            (bias (b :qkv) (blk il "attn_qkv.bias") rows 3072)
            (launch dev "v_rope" (list rows 16) 32 (list rows gw (b :qkv) (b :q) (b :k) (b :v)))
            (launch dev "v_attend" (list (ceiling rows 64) 16) 256 (list rows (b :q) (b :k) (b :v) (b :att)))
            (mm "g53_mm_bf16" (blk il "attn_out.weight") 1024 1024 rows (b :att) tt)
            (bias tt (blk il "attn_out.bias") rows 1024 x)
            (rotatef x tt)
            (rms x (blk il "ln2.weight") tt)
            (mm "g53_mm_bf16" (blk il "ffn_gate.weight") 1024 2816 rows tt (b :g))
            (mm "g53_mm_bf16" (blk il "ffn_up.weight") 1024 2816 rows tt (b :u))
            (launch dev "v_swiglu" (ceiling (* rows 2816) 256) 256 (list (list :i64 (* rows 2816)) (b :g) (b :u)))
            (mm "g53_mm_bf16" (blk il "ffn_down.weight") 2816 1024 rows (b :g) tt)
            (launch dev "v_add" (ceiling (* rows 1024) 256) 256 (list (list :i64 (* rows 1024)) tt x))
            (rotatef x tt))
          (rms x (w "v.post_ln.weight") tt)
          (launch dev "v_merge3" (list (ceiling 9216 256) orows) 256 (list gh gw tt (b :merged))))
        (mm "g53_mm_bf16" (w "mm.1.weight") 9216 wd orows (b :merged) (b :h1))
        (bias (b :h1) (w "mm.1.bias") orows wd)
        (launch dev "v_gelu" (ceiling (* orows wd) 256) 256 (list (list :i64 (* orows wd)) (b :h1)))
        (mm "g53_mm_bf16" (w "mm.2.weight") wd wd orows (b :h1) (b :h2))
        (bias (b :h2) (w "mm.2.bias") orows wd)
        (sb-sys:with-pinned-objects (out)
          (cuda-download (b :h2) (sb-sys:vector-sap out) (* 4 orows wd)))
        (cuda-sync dev)))
    (values out oh ow)))

(defun vision-image-tokens (bytes)
  "The prompt tokens for the picture BYTES (a PNG or JPEG): start, its grid's
   rows each closed by a newline, end -- encoded once, then held."
  (let* ((bytes (coerce bytes 'u8vec)) (key (image-key bytes)))
    (or (image-tokens-held key)
        (let ((v (or *vision* (error "no vision encoder: start the server with --mmproj"))))
          (multiple-value-bind (patches gh gw) (ds41-image-patches (decode-image bytes))
            (multiple-value-bind (emb oh ow) (vision-run v patches gh gw)
              (let* ((wd (vision-width v)) (count (+ (* oh (1+ ow)) 2)) (rows (f32vec (* count wd))) (at 0))
                (flet ((put (src start) (replace rows src :start1 (* at wd) :start2 start :end2 (+ start wd)) (incf at)))
                  (put (vision-start v) 0)
                  (dotimes (r oh)
                    (dotimes (c ow) (put emb (* (+ c (* r ow)) wd)))
                    (put (vision-newline v) 0))
                  (put (vision-end v) 0))
                (image-block-tokens key wd rows))))))))
