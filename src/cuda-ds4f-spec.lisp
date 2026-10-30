;;;; cuda-ds4f-spec.lisp — DS4F's speculative steps on the device.
;;;
;;; DS4F-SPECULATE's verifying pass (the token and its guesses, a few tokens
;;; through every layer at once) for CUDA.  It is the prefill's layer-major
;;; pass (cuda-ds4f-prefill.lisp) with decode's products: each dense weight
;;; read once for all the tokens by matvecs that sum as decode's do
;;; (D4_MVQ8N, D4_MV_F16N), and the routed experts as decode has them -- the
;;; routing on the CPU, a layer at a time, the union of the tokens' experts
;;; from the first device's pool (those it lacks claimed and read through
;;; from host memory), and the second device's tier (computed there) -- each
;;; expert's rows once for the tokens that chose it.  What the pass wrote past
;;; the tokens kept is undone on the device: the raw ring's rows put back,
;;; the compressors' windows restored and carried over the kept tokens alone,
;;; the counts wound back.

(in-package #:illatio)

(defconstant +dv-nt+ 9 "Tokens a verifying pass takes at most (D4_NT in ds4f.cu).")

(defun dv-host (st key bytes)
  "Pinned host buffer KEY of ST (made the first time, BYTES)."
  (or (getf (dk-bufs st) key) (setf (getf (dk-bufs st) key) (cuda-host-alloc bytes))))

(defun dv-tbuf (st tr key bytes)
  "Buffer KEY on the second device (made the first time, BYTES)."
  (or (getf (dk-bufs st) key)
      (prog2 (cuda-current (dt-dev tr))
          (setf (getf (dk-bufs st) key) (cuda-alloc (dt-dev tr) bytes))
        (cuda-current (dk-dev st)))))

(defun dk-ffn-small (m st l li n tokv sb sa)
  "The FFN past the router for a short pass's N tokens (TOKV): routed on the
   CPU, each chosen expert once on its tokens' rows, on the device that holds
   it (or read through into a slot of the first's), the sums in slot order
   with the shared expert, HC post: SB into SA.  The slots it holds."
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (ff (ds4f-ff-exp m)) (nu (ds4f-n-used m)) (ne (ds4f-n-expert m))
         (clamp (svref (ds4f-clamps m) li)) (gb (dk-gate-bytes st)) (sbytes (dk-slot-bytes st))
         (base (dbuf-ptr (dk-pool st))) (tr (dk-tier st))
         (grb (second (dl-gate-exps l))) (drb (second (dl-down-exps l)))
         (norm2 (pb st :p-norm2 (* n d))) (logits (pb st :p-logits (* n ne)))
         (h (dv-host st :v-host (* 4 +dv-nt+ (+ ne d))))
         (slot-of (dk-slot-of st)) (pin (dk-pin st)) (used (dk-used st)) (tick (incf (dk-tick st)))
         (groups (make-hash-table)) (ws (make-array n)) (held '()))
    (declare (type (simple-array fixnum (*)) slot-of pin used))
    ;; the router's logits (and the FFN input, for the second device) back
    (cuda-copy-out h logits 0 (* 4 n ne))
    (when tr (cuda-copy-out (sb-sys:sap+ h (* 4 +dv-nt+ ne)) norm2 0 (* 4 n d)))
    (cuda-sync dev)
    (dk-arena-reset st)
    (dotimes (tk n)
      (let ((lg (f32vec ne)))
        (dotimes (i ne) (setf (aref lg i) (sb-sys:sap-ref-single h (* 4 (+ (* tk ne) i)))))
        (multiple-value-bind (sel w) (d4-route m l li nil (svref tokv tk) lg)
          (setf (svref ws tk) w)
          (dotimes (j nu) (push (cons tk j) (gethash (aref sel j) groups))))))
    ;; where each expert is computed: read through first, then held, then the second device's
    (let ((rt '()) (hit '()) (b '()))
      (dolist (e (sort (loop for e being the hash-keys of groups collect e) #'<))
        (let* ((key (+ (* li ne) e)) (s (aref slot-of key)) (users (reverse (gethash e groups))))
          (cond ((>= s 0) (setf (aref used s) tick) (incf (dk-hits st)) (incf (aref pin s)) (push s held)
                 (push (list s 0 users) hit))
                ((and tr (>= (aref (dt-slot-of tr) key) 0)) (incf (dt-hits tr))
                 (push (list (aref (dt-slot-of tr) key) users) b))
                (t (let ((s (d4-claim m st li e nil)))
                     (incf (dk-misses st)) (incf (aref pin s)) (push s held)
                     (push (list s (sb-sys:sap-int (d4-expert-sap m st li e)) users) rt))))))
      (let* ((a (append (nreverse rt) (nreverse hit))) (b (nreverse b))
             (rows-a (loop for x in a sum (length (third x)))) (rows-b (loop for x in b sum (length (second x))))
             (smap (make-array (* n nu))) (down (pb st :v-down (* (+ rows-a rows-b) d))) (r 0))
        (when a
          (let ((rmap '()) (wrow '()) (counts '()) (bases '())
                (x (pb st :v-x (* rows-a d))) (g (pb st :v-g (* rows-a ff))) (u (pb st :v-u (* rows-a ff)))
                (mid (pb st :v-mid (* rows-a ff))))
            (loop for (s src users) in a
                  do (push r bases) (push (length users) counts)
                     (loop for (tk . j) in users
                           do (push tk rmap) (push (aref (svref ws tk) j) wrow) (setf (aref smap (+ (* tk nu) j)) r) (incf r)))
            (let ((ptrs (dk-put st (loop for (s) in a collect (+ base (* s sbytes))) :type :u64))
                  (srcs (and (find-if #'plusp a :key #'second) (dk-put st (mapcar #'second a) :type :u64)))
                  (cnt (dk-put st (nreverse counts) :type :u32)) (bas (dk-put st (nreverse bases) :type :u32))
                  (na (length a)))
              (launch dev "d4_gather_rows" (ceiling (* rows-a d) 256) 256 (list d rows-a norm2 (dk-put st (nreverse rmap)) x))
              (launch dev "d4_mv_iq2_xxs" (list (ceiling ff 8) 1 na) 64 (list d ff grb (list :i64 0) d ff ptrs srcs x g cnt bas))
              (launch dev "d4_mv_iq2_xxs" (list (ceiling ff 8) 1 na) 64 (list d ff grb (list :i64 gb) d ff ptrs srcs x u cnt bas))
              (launch dev "d4_swiglu_w" (ceiling (* rows-a ff) 256) 256
                      (list rows-a ff clamp g u (dk-put st (nreverse wrow) :type :f32) mid nil nil))
              (launch dev "d4_mv_q2_k" (list (ceiling d 8) 1 na) 64
                      (list ff d drb (list :i64 (* 2 gb)) ff d ptrs srcs mid down cnt bas)))))
        (when b
          ;; the second device's: its inputs in one copy, its down rows back to pinned memory
          (let* ((tdev (dt-dev tr)) (tb (dt-pool tr)) (th (dv-host st :v-thost (* 1024 1024)))
                 (tout (dv-host st :v-tout (* 4 +dv-nt+ nu d)))
                 (o-rmap (* 4 n d)) (o-w (+ o-rmap (* 4 rows-b))) (o-cnt (+ o-w (* 4 rows-b)))
                 (o-bas (+ o-cnt (* 4 (length b)))) (o-ptr (* 8 (ceiling (+ o-bas (* 4 (length b))) 8)))
                 (bytes (+ o-ptr (* 8 (length b))))
                 (tin (dv-tbuf st tr :v-tin (* 1024 1024)))
                 (tx (dv-tbuf st tr :v-tx (* 4 +dv-nt+ nu d))) (tg (dv-tbuf st tr :v-tg (* 4 +dv-nt+ nu ff)))
                 (tu (dv-tbuf st tr :v-tu (* 4 +dv-nt+ nu ff))) (tmid (dv-tbuf st tr :v-tmid (* 4 +dv-nt+ nu ff)))
                 (tdown (dv-tbuf st tr :v-tdown (* 4 +dv-nt+ nu d))) (q 0) (k 0))
            (d4-memcpy th (sb-sys:sap+ h (* 4 +dv-nt+ ne)) (* 4 n d))
            (loop for (s users) in b
                  do (setf (sb-sys:sap-ref-64 th (+ o-ptr (* 8 k))) (+ (dbuf-ptr tb) (* s sbytes))
                           (sb-sys:sap-ref-32 th (+ o-cnt (* 4 k))) (length users)
                           (sb-sys:sap-ref-32 th (+ o-bas (* 4 k))) q)
                     (incf k)
                     (loop for (tk . j) in users
                           do (setf (sb-sys:sap-ref-32 th (+ o-rmap (* 4 q))) tk
                                    (sb-sys:sap-ref-single th (+ o-w (* 4 q))) (aref (svref ws tk) j)
                                    (aref smap (+ (* tk nu) j)) (+ rows-a q))
                              (incf q)))
            (cuda-current tdev)
            (cuda-copy-in tin 0 th bytes (cudev-stream tdev))
            (let ((cnt (cons tin o-cnt)) (bas (cons tin o-bas)) (ptrs (cons tin o-ptr)) (nb (length b)))
              (launch tdev "d4_gather_rows" (ceiling (* rows-b d) 256) 256 (list d rows-b tin (cons tin o-rmap) tx))
              (launch tdev "d4_mv_iq2_xxs" (list (ceiling ff 8) 1 nb) 64 (list d ff grb (list :i64 0) d ff ptrs nil tx tg cnt bas))
              (launch tdev "d4_mv_iq2_xxs" (list (ceiling ff 8) 1 nb) 64 (list d ff grb (list :i64 gb) d ff ptrs nil tx tu cnt bas))
              (launch tdev "d4_swiglu_w" (ceiling (* rows-b ff) 256) 256 (list rows-b ff clamp tg tu (cons tin o-w) tmid nil nil))
              (launch tdev "d4_mv_q2_k" (list (ceiling d 8) 1 nb) 64
                      (list ff d drb (list :i64 (* 2 gb)) ff d ptrs nil tmid tdown cnt bas)))
            (cuda-copy-out tout tdown 0 (* 4 rows-b d))
            (cuda-event-record (dt-event tr) (cudev-stream tdev))
            (cuda-current dev)
            (cuda-stream-wait (cudev-stream dev) (dt-event tr))
            (cuda-copy-in down (* 4 rows-a d) tout (* 4 rows-b d))))
        (let ((moe (pb st :p-moe (* n d))))
          (launch dev "d4_moe_sum_rows" (ceiling (* n d) 256) 256
                  (list d nu n down (dk-put st (coerce smap 'list)) (pb st :p-sout (* n d)) moe))
          (dk-hc-post-chunk m st moe sb (pb st :p-pcf (* n 32)) sa n))))
    held))

(defun dk-head-n (m st streams n)
  "Every one of N tokens' logits from STREAMS: decode's head, the output
   weights read once: a vector, a row of N-VOCAB a token."
  (let* ((dev (dk-dev st)) (d (ds4f-d m)) (nh (ds4f-n-hc m)) (hcd (* nh d)) (eps (ds4f-eps m)) (nv (ds4f-n-vocab m))
         (flat (pb st :v-hflat (* n hcd))) (pre (pb st :v-hpre (* n 16)))
         (embd (pb st :v-hembd (* n d))) (norm (pb st :v-hnorm (* n d))) (out (pb st :v-out (* n nv)))
         (rows (f32vec (* n nv))))
    (dk-rms dev streams nil flat hcd n eps)
    (dk-f16mm st "output_hc_fn.weight" flat pre n :y-stride 16)
    (launch dev "d4_head_mix" (list (ceiling d 256) n) 256
            (list d nh (ds4f-hc-eps m) pre (dvec st (ds4f-out-hc-scale m)) (dvec st (ds4f-out-hc-base m)) streams embd))
    (dk-rms dev embd (dvec st (ds4f-output-norm m)) norm d n eps)
    (dk-q8mm st "output.weight" norm out n)
    (sb-sys:with-pinned-objects (rows)
      (cuda-download out (sb-sys:vector-sap rows) (* 4 n nv)))
    rows))

(defun ds4f-cuda-pass (ctx tokens)
  "A short pass: TOKENS (at most +DV-NT+) at CTX's next positions, every
   layer at once, decode's products: each token's logits (DK-HEAD-N's)."
  (let* ((m (d4ctx-model ctx)) (st (ds4f-cuda-state m)) (dev (dk-dev st))
         (d (ds4f-d m)) (n-hc (ds4f-n-hc m)) (hcd (* n-hc d)) (pos0 (d4ctx-pos ctx))
         (n (length tokens)) (tokv (coerce tokens 'simple-vector)) (wf (ds4f-file m))
         (positions (loop for i below n collect (+ pos0 i)))
         (*dk-small* t)
         (*dk-mapped* (cons (or (getf (dk-bufs st) :p-mapped-sap)
                                (setf (getf (dk-bufs st) :p-mapped-sap) (cuda-host-alloc +dk-mapped-bytes+ :mapped t)))
                            0)))
    (declare (type u8vec wf))
    (assert (<= n +dv-nt+))
    (dk-ready m st ctx)
    (let ((sa (pb st :p-sa (* n hcd))) (sb (pb st :p-sb (* n hcd))) (held '()))
      (let ((e (f32vec (* n hcd))))
        (dotimes (tk n)
          (let ((o (+ (ds4f-embd-off m) (* (svref tokv tk) d 2))))
            (dotimes (j d)
              (let ((v (f16-at wf (+ o (* 2 j)))))
                (dotimes (h n-hc) (setf (aref e (+ (* tk hcd) (* h d) j)) v))))))
        (cuda-upload-vector sa e))
      (let ((ones (f32vec n))) (fill ones 1f0) (cuda-upload-vector (pb st :p-ones n) ones))
      (let* ((li-plain (position 0 (ds4f-ratios m))) (li-comp (position-if #'plusp (ds4f-ratios m)))
             (tables (loop for li in (list li-plain li-comp) for k from 0
                           collect (and li (cons (dk-cs-rows m st (if (zerop k) :p-cs0 :p-cs1) positions li)
                                                 (dk-cs-rows m st (if (zerop k) :p-ics0 :p-ics1) positions li t))))))
        (dotimes (li (ds4f-n-layer m))
          (let* ((l (svref (ds4f-layers m) li)) (c (svref (d4ctx-caches ctx) li))
                 (tab (if (plusp (dl-ratio l)) (second tables) (first tables))))
            (dk-attention-chunk m st l li c pos0 n sa sb (car tab) (cdr tab))
            (dk-ffn-head-chunk m st l li n sb)
            ;; (it syncs: the layer before is done with its slots, and the mapped arena)
            (let ((now (dk-ffn-small m st l li n tokv sb sa)))
              (dolist (s held) (decf (aref (dk-pin st) s)))
              (setf held now (cdr *dk-mapped*) 0)))))
      (incf (d4ctx-pos ctx) n)
      (prog1 (dk-head-n m st sa n)
        (cuda-sync dev)
        (dk-arena-reset st)
        (dolist (s held) (decf (aref (dk-pin st) s)))))))

;;; --- undoing a pass past the tokens kept --------------------------------------------

(defun ds4f-cuda-record (ctx n)
  "What a pass of N tokens over CTX writes that the kept tokens may not, on
   the device: per layer its counts, the ring's rows the pass will write,
   its compressors' windows."
  (let* ((m (d4ctx-model ctx)) (st (ds4f-cuda-state m)) (dev (dk-dev st)) (hd (ds4f-head-dim m)) (swa (ds4f-swa m)))
    (loop for c across (d4ctx-caches ctx) for li from 0
          collect (let ((ring (pb st (intern (format nil "V-RING-~D" li) :keyword) (* n hd))))
                    (launch dev "d4_ring_gather" (ceiling (* n hd) 256) 256
                            (list hd swa (mod (dc-raw-total c) swa) 0 n (dvec st (dc-raw c)) ring))
                    (list (dc-n-raw c) (dc-raw-total c) (dc-n-comp c) (dc-n-ix c) ring
                          (loop for v in (list (dc-st-kv c) (dc-st-score c) (dc-ix-kv c) (dc-ix-score c))
                                for k from 0
                                collect (and v (let ((b (pb st (intern (format nil "V-WIN-~D-~D" li k) :keyword) (length v))))
                                                 (cuda-copy-dd b 0 (dvec st v) 0 (* 4 (length v)))
                                                 b))))))))

(defun ds4f-cuda-undo (ctx rec pos0 keep n)
  "CTX after a pass of N tokens from POS0 (REC its record), as if only its
   first KEEP had gone through."
  (let* ((m (d4ctx-model ctx)) (st (ds4f-cuda-state m)) (dev (dk-dev st)) (hd (ds4f-head-dim m)) (swa (ds4f-swa m)))
    (loop for c across (d4ctx-caches ctx) for r in rec for li from 0
          do (destructuring-bind (n-raw total n-comp n-ix ring wins) r
               (let* ((l (svref (ds4f-layers m) li)) (ratio (dl-ratio l))
                      (bounds (if (plusp ratio) (- (floor (+ pos0 keep) ratio) (floor pos0 ratio)) 0)))
                 (setf (dc-n-raw c) (min swa (+ n-raw keep)) (dc-raw-total c) (+ total keep)
                       (dc-n-comp c) (+ n-comp bounds) (dc-n-ix c) (+ n-ix (if (= ratio 4) bounds 0)))
                 ;; the ring's rows past the kept tokens, as they were
                 (launch dev "d4_ring_store" (ceiling (* (- n keep) hd) 256) 256
                         (list hd swa (mod (+ total keep) swa) keep (- n keep) (dvec st (dc-raw c)) ring))
                 ;; the windows as they were, carried over the kept tokens
                 (loop for v in (list (dc-st-kv c) (dc-st-score c) (dc-ix-kv c) (dc-ix-score c)) for b in wins
                       when v do (cuda-copy-dd (dvec st v) 0 b 0 (* 4 (length v))))
                 (when (plusp ratio)
                   (let ((*dk-keep-proj* t))
                     (dk-comp-carry m st l c pos0 keep 0 (pb st (dk-proj-key 0 0 li) 1) (pb st (dk-proj-key 0 1 li) 1))
                     (when (= ratio 4)
                       (dk-comp-carry m st l c pos0 keep 1 (pb st (dk-proj-key 1 0 li) 1) (pb st (dk-proj-key 1 1 li) 1))))))))
    (setf (d4ctx-pos ctx) (+ pos0 keep))
    ctx))

(defvar *dv-k* 4
  "Guesses the next step takes at most: doubled when a step's are all
   confirmed, cut to what was confirmed (and one) when not -- a pass costs
   about a decode step and a third of one more a token (38 ms, then 10 ms a
   token, against 31 ms), so guesses that miss lose.")
(defvar *dv-cool* 0 "Steps to go a token at a time, after guesses missed at one guess a step.")
(defvar *dv-stats* (list 0 0 0) "Speculative steps, guesses, guesses confirmed.")

(defun ds4f-cuda-speculate (ctx tok draft)
  "TOK and its guesses DRAFT through one pass at CTX's next position; how
   many of the guesses its greedy choices confirm, CTX then past them (and
   the token after them its logits)."
  (when (plusp *dv-cool*)
    (decf *dv-cool*)
    (ds4f-forward ctx tok)
    (return-from ds4f-cuda-speculate 0))
  (let* ((m (d4ctx-model ctx)) (st (ds4f-cuda-state m)) (nv (ds4f-n-vocab m))
         (draft (subseq draft 0 (min (length draft) *dv-k* (1- +dv-nt+))))
         (pos0 (d4ctx-pos ctx)) (batch (cons tok draft)) (n (length batch)))
    (with-cuda
      (cuda-current (dk-dev st))
      (let* ((rec (ds4f-cuda-record ctx n))
             (rows (let ((*dk-keep-proj* t)) (ds4f-cuda-pass ctx batch)))
             (a (loop for g in draft for i from 0 while (= (row-argmax rows i nv) g) count t))
             (keep (1+ a)))
        (when (< keep n) (ds4f-cuda-undo ctx rec pos0 keep n) (cuda-sync (dk-dev st)))
        (replace (d4ctx-logits ctx) rows :start2 (* (1- keep) nv) :end2 (* keep nv))
        (incf (first *dv-stats*)) (incf (second *dv-stats*) (length draft)) (incf (third *dv-stats*) a)
        (cond ((= a (length draft)) (setf *dv-k* (min (1- +dv-nt+) (* 2 *dv-k*))))
              ((and (zerop a) (= *dv-k* 1)) (setf *dv-cool* 4))
              (t (setf *dv-k* (max 1 (1+ a)))))
        a))))
