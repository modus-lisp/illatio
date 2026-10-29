;;;; metal-ds4f.lisp — DeepSeek V4 Flash's prefill experts on the GPU, streamed
;;;; from the SSD.
;;;
;;; The GPU can't read the 86.7 GB mapped file in place -- Metal's buffers stop
;;; at 30 GB, and a buffer over a file mapping would have its pages made
;;; resident while a command runs -- so a layer's routed experts are copied,
;;; from the mapping (the SSD, or the page cache when they are there), into a
;;; GPU-visible staging buffer, one slot an expert, as ds4 does with its expert
;;; cache.  The copies run on the pool's threads, an expert each.  Then one
;;; command buffer: for each expert, its gate and up products against the
;;; tokens that chose it (gathered into a block of rows), the clamped SwiGLU
;;; times each row's routing weight, and its down product.  The CPU sums each
;;; token's experts in slot order.

(in-package #:illatio)

(defstruct (d4gpu (:constructor %make-d4gpu))
  gpu stage (slot-bytes 0) (gate-bytes 0) (down-bytes 0)
  x g u w mid down (rows 0))            ; scratch, sized for ROWS routed rows

(defun ds4f-gpu-state (m)
  "The GPU state of DS4F M: a staging slot for each of a layer's experts."
  (or (getf (ds4f-gpu-plist m) :gpu)
      (setf (getf (ds4f-gpu-plist m) :gpu)
            (let* ((gpu (open-gpu)) (l (svref (ds4f-layers m) 0))
                   (gate-bytes (* (third (dl-gate-exps l)) (second (dl-gate-exps l))))
                   (down-bytes (* (third (dl-down-exps l)) (second (dl-down-exps l))))
                   (slot (+ (* 2 gate-bytes) down-bytes)))
              (%make-d4gpu :gpu gpu :stage (gpu-buffer gpu (* (ds4f-n-expert m) slot))
                           :slot-bytes slot :gate-bytes gate-bytes :down-bytes down-bytes)))))

(defun d4gpu-scratch (st m rows)
  "ST's row buffers, grown to hold ROWS routed rows."
  (when (< (d4gpu-rows st) rows)
    (let ((gpu (d4gpu-gpu st)) (d (ds4f-d m)) (ff (ds4f-ff-exp m)))
      (setf (d4gpu-x st) (gpu-buffer gpu (* 4 rows d)) (d4gpu-g st) (gpu-buffer gpu (* 4 rows ff))
            (d4gpu-u st) (gpu-buffer gpu (* 4 rows ff)) (d4gpu-w st) (gpu-buffer gpu (* 4 rows))
            (d4gpu-mid st) (gpu-buffer gpu (* 4 rows ff)) (d4gpu-down st) (gpu-buffer gpu (* 4 rows d))
            (d4gpu-rows st) rows)))
  st)

(defun ds4f-fd (m)
  "A descriptor open on DS4F M's file, for PREAD-INTO."
  (or (getf (ds4f-gpu-plist m) :fd)
      (setf (getf (ds4f-gpu-plist m) :fd)
            (or (sb-unix:unix-open (namestring (gguf-path (ds4f-gguf m))) sb-unix:o_rdonly 0)
                (error "cannot open ~A" (gguf-path (ds4f-gguf m)))))))

(defun pread-into (fd dst offset bytes)
  "BYTES of FD at OFFSET into DST: straight from the disk (or the page
   cache), large reads, no faults on a mapping."
  (loop with done = 0
        while (< done bytes)
        do (let ((got (sb-alien:alien-funcall
                       (sb-alien:extern-alien "pread" (function sb-alien:long sb-alien:int sb-sys:system-area-pointer
                                                                sb-alien:unsigned-long sb-alien:long))
                       fd (sb-sys:sap+ dst done) (- bytes done) (+ offset done))))
             (when (<= got 0) (error "pread of ~:D bytes at ~:D failed" bytes offset))
             (incf done got))))

(defun sap-copy (dst src bytes)
  (sb-alien:alien-funcall
   (sb-alien:extern-alien "memcpy" (function sb-sys:system-area-pointer sb-sys:system-area-pointer
                                             sb-sys:system-area-pointer sb-alien:unsigned-long))
   dst src bytes))

(defun ds4f-gpu-experts! (m l li ev users norms ws n down-out)
  "The routed FFN of a chunk's layer LI on the GPU.  EV: the distinct experts;
   USERS: for each, its (token . slot)s; NORMS: the tokens' normed inputs;
   WS: their routing weights.  DOWN-OUT gets each (token, slot)'s down
   product at (token*N-USED + slot)*D."
  (let* ((st (ds4f-gpu-state m)) (gpu (d4gpu-gpu st)) (pool (ds4f-pool m))
         (d (ds4f-d m)) (ff (ds4f-ff-exp m)) (nu (ds4f-n-used m)) (ne (length ev))
         (clamp (svref (ds4f-clamps m) li))
         (counts (map 'vector #'length users))
         (bases (let ((b 0)) (map 'vector (lambda (k) (prog1 b (incf b k))) counts)))
         (rows (reduce #'+ counts)))
    (declare (type f32vec norms down-out) (ignore n))
    (d4gpu-scratch st m rows)
    ;; the experts into their slots (the reads that stream from the SSD)
    (destructuring-bind (goff grb grows gcols) (dl-gate-exps l)
      (declare (ignore gcols))
      (destructuring-bind (uoff urb urows ucols) (dl-up-exps l)
        (declare (ignore ucols urows))
        (destructuring-bind (doff drb drows dcols) (dl-down-exps l)
          (declare (ignore dcols))
          (let ((fd (ds4f-fd m)) (sbase (gbuf-sap (d4gpu-stage st)))
                (sb (d4gpu-slot-bytes st)) (gb (d4gpu-gate-bytes st)))
            (parallel-for (k ne pool)
              (let ((e (svref ev k)) (slot (sb-sys:sap+ sbase (* k sb))))
                (pread-into fd slot (+ goff (* e grows grb)) gb)
                (pread-into fd (sb-sys:sap+ slot gb) (+ uoff (* e grows urb)) gb)
                (pread-into fd (sb-sys:sap+ slot (* 2 gb)) (+ doff (* e drows drb)) (* drows drb)))))
          ;; each expert's tokens gathered, with their routing weights
          (let ((xs (gbuf-sap (d4gpu-x st))) (wsap (gbuf-sap (d4gpu-w st))))
            (sb-sys:with-pinned-objects (norms)
              (let ((nsap (sb-sys:vector-sap norms)))
                (dotimes (k ne)
                  (loop for (tk . slot) across (svref users k) for i from 0
                        do (let ((row (+ (aref bases k) i)))
                             (sap-copy (sb-sys:sap+ xs (* 4 row d)) (sb-sys:sap+ nsap (* 4 tk d)) (* 4 d))
                             (setf (sb-sys:sap-ref-single wsap (* 4 row)) (aref (svref ws tk) slot))))))))
          (let ((stage (d4gpu-stage st)) (sb (d4gpu-slot-bytes st)) (gb (d4gpu-gate-bytes st)))
            (with-gpu-work (e gpu)
              (flet ((mm (kernel woff cols rbytes nr x xoff y yoff ntok)
                       (dispatch gpu e kernel (list cols nr ntok rbytes)
                                 (list (cons stage woff) (cons x xoff) (cons y yoff))
                                 (list (ceiling ntok 64) (ceiling nr 64) 1) 128 :shared 16384)))
                (dotimes (k ne)
                  (let ((nt (aref counts k)) (b (aref bases k)) (s0 (* k sb)))
                    (mm "d4_mm_iq2_xxs" s0 d grb grows (d4gpu-x st) (* 4 b d) (d4gpu-g st) (* 4 b ff) nt)
                    (mm "d4_mm_iq2_xxs" (+ s0 gb) d urb grows (d4gpu-x st) (* 4 b d) (d4gpu-u st) (* 4 b ff) nt)))
                (dispatch gpu e "d4_swiglu_w" (list rows ff clamp)
                          (list (d4gpu-g st) (d4gpu-u st) (d4gpu-w st) (d4gpu-mid st))
                          (ceiling (* rows ff) 256) 256)
                (dotimes (k ne)
                  (let ((nt (aref counts k)) (b (aref bases k)) (s0 (* k sb)))
                    (mm "d4_mm_q2_k" (+ s0 (* 2 gb)) ff drb drows (d4gpu-mid st) (* 4 b ff) (d4gpu-down st) (* 4 b d) nt))))))
          ;; back to (token, slot) order
          (let ((ds (gbuf-sap (d4gpu-down st))))
            (sb-sys:with-pinned-objects (down-out)
              (let ((osap (sb-sys:vector-sap down-out)))
                (dotimes (k ne)
                  (loop for (tk . slot) across (svref users k) for i from 0
                        do (sap-copy (sb-sys:sap+ osap (* 4 (+ (* tk nu) slot) d))
                                     (sb-sys:sap+ ds (* 4 (+ (aref bases k) i) d)) (* 4 d))))))))))
    down-out))
