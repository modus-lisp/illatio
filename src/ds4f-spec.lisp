;;;; ds4f-spec.lisp — DS4F decodes greedily a few tokens at a time, guessing
;;;; them from the context.
;;;
;;; Prompt lookup: the last few tokens have often been seen before (code
;;; being edited, text quoted, a tool's output echoed), and what followed
;;; then is a guess at what follows now.  The token and up to *DS4F-SPEC-K*
;;; guesses go through the GPU's prefill as one small chunk -- each layer's
;;; dense weights read once for all of them, its experts for their union --
;;; with the logits of each; the guesses are kept while the model's own
;;; greedy choice agrees, plus its choice where it first doesn't, so the
;;; tokens are the ones decoding one at a time would give.  What the chunk
;;; wrote past the kept tokens is undone: the raw ring's rows saved before
;;; are put back (the window still needs them), the compressed rows' counts
;;; wound back (rows past a count are never read), the compressors' windows
;;; restored and carried again over the kept tokens alone.

(in-package #:illatio)

(defparameter *ds4f-spec-k* 0
  "Guesses a step at most (0: none, the default).  Exact, but on this model
   streamed from the SSD it doesn't pay yet: a token's experts must be read
   whether it is guessed or not (about 70 ms of a 100 ms token), so a step
   saves at most the dense weights' second reading (about 30 ms a token),
   and a verifying pass costs more than that as things are.")
(defparameter *ds4f-spec-ngram* '(4 3 2) "The suffix lengths a guess is looked up by, longest first.")
(defvar *ds4f-spec-always* nil "For tests: a step with no guess goes through the prefill too.")

(defun ngram-draft (hist k)
  "Up to K tokens that followed the latest earlier occurrence of HIST's
   (a vector) longest suffix found, of the lengths *DS4F-SPEC-NGRAM*."
  (let ((len (length hist)))
    (dolist (n *ds4f-spec-ngram* '())
      (when (> len n)
        (loop for j from (- len 2) downto (1- n)  ; the occurrence ends at J, before the suffix's own end
              when (loop for i below n always (= (aref hist (- j i)) (aref hist (- len 1 i))))
                do (return-from ngram-draft
                     (loop for p from (1+ j) below (min len (+ j 1 k)) collect (aref hist p))))))))

(defun ds4f-undo-record (c n)
  "What a pass of N tokens over d4ctx C writes that the kept tokens may not:
   per layer, its counts, the raw ring rows the pass will write, the windows."
  (let* ((m (d4ctx-model c)) (hd (ds4f-head-dim m)) (swa (ds4f-swa m)))
    (map 'simple-vector
         (lambda (d)
           (list (dc-n-raw d) (dc-raw-total d) (dc-n-comp d) (dc-n-ix d)
                 (let ((rows (f32vec (* n hd))))
                   (dotimes (i n rows)
                     (replace rows (dc-raw d) :start1 (* i hd)
                                              :start2 (* (mod (+ (dc-raw-total d) i) swa) hd)
                                              :end2 (* (1+ (mod (+ (dc-raw-total d) i) swa)) hd))))
                 (and (dc-st-kv d) (copy-seq (dc-st-kv d))) (and (dc-st-score d) (copy-seq (dc-st-score d)))
                 (and (dc-ix-kv d) (copy-seq (dc-ix-kv d))) (and (dc-ix-score d) (copy-seq (dc-ix-score d)))))
         (d4ctx-caches c))))

(defun ds4f-undo-to (c undo pos0 keep n)
  "d4ctx C, a pass of N tokens from POS0 over it, as if only its first KEEP
   had gone through."
  (let* ((m (d4ctx-model c)) (hd (ds4f-head-dim m)) (swa (ds4f-swa m))
         (st (ds4f-gpu-state m)) (gpu (d4gpu-gpu st)) (*prefill-keep-proj* t))
    (with-gpu-work (e gpu)
      (loop for d across (d4ctx-caches c) for u across undo for li from 0
            do (destructuring-bind (n-raw total n-comp n-ix rows st-kv st-sc ix-kv ix-sc) u
                 (let* ((l (svref (ds4f-layers m) li)) (ratio (dl-ratio l))
                        (bounds (if (plusp ratio) (- (floor (+ pos0 keep) ratio) (floor pos0 ratio)) 0)))
                   (setf (dc-n-raw d) (min swa (+ n-raw keep)) (dc-raw-total d) (+ total keep)
                         (dc-n-comp d) (+ n-comp bounds) (dc-n-ix d) (+ n-ix (if (= ratio 4) bounds 0)))
                   ;; the ring rows past the kept tokens, as they were
                   (loop for i from keep below n
                         do (replace (dc-raw d) rows :start1 (* (mod (+ total i) swa) hd)
                                                     :start2 (* i hd) :end2 (* (1+ i) hd)))
                   ;; the windows as they were, carried over the kept tokens
                   (when (plusp ratio)
                     (replace (dc-st-kv d) st-kv) (replace (dc-st-score d) st-sc)
                     (multiple-value-bind (pkv psc)
                         (comp-proj-bufs st li 0 n (* (if (= ratio 4) 2 1) hd))
                       (enc-comp-carry m gpu e l d pos0 keep 0 pkv psc))
                     (when (= ratio 4)
                       (replace (dc-ix-kv d) ix-kv) (replace (dc-ix-score d) ix-sc)
                       (multiple-value-bind (pkv psc) (comp-proj-bufs st li 1 n (* 2 (ds4f-ix-dim m)))
                         (enc-comp-carry m gpu e l d pos0 keep 1 pkv psc))))))))
    (setf (d4ctx-pos c) (+ pos0 keep))
    c))

(defun row-argmax (rows row nv)
  (declare (type f32vec rows) (type fixnum row nv))
  (let ((o (* row nv)) (b 0))
    (declare (type fixnum o b))
    (loop for i from 1 below nv when (> (aref rows (+ o i)) (aref rows (+ o b))) do (setf b i))
    b))

(defun ds4f-speculate (ctx tok more)
  "For GENERATE: TOKEN at CTX's next position and the guesses its greedy
   choices confirm; those (:NO if there is nothing to guess with)."
  (let ((model (context-model ctx)))
    (unless (and (typep (model-impl model) 'ds4f) (ds4f-gpu-p :prefill) (plusp *ds4f-spec-k*))
      (return-from ds4f-speculate :no))
    (let* ((hist (coerce (append (context-tokens ctx) (list tok)) 'simple-vector))
           (draft (ngram-draft hist (min *ds4f-spec-k* more))))
      (unless (or draft *ds4f-spec-always*) (return-from ds4f-speculate :no))
      (let* ((c (ds4f-context ctx)) (m (d4ctx-model c)) (nv (ds4f-n-vocab m))
             (pos0 (context-pos ctx)) (batch (cons tok draft)) (n (length batch))
             (undo (progn (setf (d4ctx-pos c) pos0) (ds4f-undo-record c n)))
             (rows (let ((*prefill-keep-proj* t) (*q8-f16-scale* *ds4f-q8-f16-scale*))
                     (sb-int:with-float-traps-masked (:overflow :inexact :underflow)
                       (ds4f-gpu-prefill-1 c batch t :all-logits t))))
             (a (loop for d in draft for i from 0 while (= (row-argmax rows i nv) d) count t))
             (keep (1+ a)))
        (when (< keep n) (ds4f-undo-to c undo pos0 keep n))
        (replace (d4ctx-logits c) rows :start2 (* (1- keep) nv) :end2 (* keep nv))
        (setf (context-pos ctx) (+ pos0 keep))
        (subseq draft 0 a)))))

(pushnew 'ds4f-speculate *speculate-hooks*)
