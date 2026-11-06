;;;; judgment.lisp — the judgment-first experiment's batches and measurements
;;;; (docs/judgment-first.md).
;;;
;;; A verdict example is a story cut after some sentence, then "\nCoherent?"
;;; and " yes" or " no", then end-of-text: ordinary next-token prediction,
;;; weighted only where the answer is predicted.  The stories come from a
;;; pool of minimal pairs over the training stories (corrupt.lisp), each
;;; with its distance, so the curriculum can admit the near breaks first.
;;; A pair is kept as (story, kind, seed, distance) and remade when drawn:
;;; making one costs less than keeping its text.
;;;
;;; Rows are TT + 1 tokens (inputs the first TT, targets the last TT).  A
;;; verdict row holds whole examples, padded with end-of-text at weight 0:
;;; an example split across rows would lose its story.  A next-token row is
;;; the training stream's next TT + 1 tokens, every position at weight 1.

(in-package #:illatio)

(defstruct (jdata (:constructor %make-jdata))
  vocab sv               ; the teacher's tokenizer, the dense vocabulary
  texts                  ; the training stories (a vector of strings)
  names donors           ; for the breaks
  pool                   ; (story kind seed distance) vectors, nearest first
  stream (pos 0)         ; the training tokens (dense), and where the next row starts
  suffix yes no)         ; the verdict's tokens (dense)

(defun dense-tokens (jd text)
  "TEXT in the dense vocabulary, or NIL if it has a token outside it."
  (let ((ids (tokenize-with (jdata-vocab jd) text :add-bos nil)) (d (story-vocab-dense (jdata-sv jd))))
    (when (every (lambda (id) (gethash id d)) ids)
      (map '(simple-array (signed-byte 32) (*)) (lambda (id) (gethash id d)) ids))))

(defun make-jdata (&key (dir (merge-pathnames "workspace/data/tinystories/" (user-homedir-pathname)))
                        (teacher (merge-pathnames "models/qwen2.5-0.5b-instruct-q8_0.gguf" (user-homedir-pathname)))
                        (subdir "qwen/") (kinds '(:swap :referent :splice)) (seed 1) (out *standard-output*))
  "The training stories prepare-stories wrote, and a pool of minimal pairs of KINDS over them."
  (let* ((qdir (merge-pathnames subdir dir))
         (sv (read-story-vocab (merge-pathnames "vocab.bin" qdir)))
         (stream (let ((tk (read-token-shard (merge-pathnames "stories_train_000.bin" qdir))))
                   (map '(simple-array (signed-byte 32) (*)) #'identity tk)))
         (n-stories (count (story-vocab-eot sv) stream))
         (texts (let ((v '()))
                  (map-stories (merge-pathnames "TinyStoriesV2-GPT4-train.txt" dir) (lambda (s) (push s v)) :max n-stories)
                  (coerce (nreverse v) 'simple-vector)))
         (names (corpus-names (coerce (subseq texts 0 (min 20000 (length texts))) 'list)))
         (donors (map 'list #'split-sentences (subseq texts 0 (min 4000 (length texts)))))
         (jd (%make-jdata :vocab (make-vocab-from-gguf (read-gguf teacher)) :sv sv :texts texts
                          :names names :donors donors :stream stream)))
    (setf (jdata-suffix jd) (dense-tokens jd (first *verdict-strings*))
          (jdata-yes jd) (dense-tokens jd (second *verdict-strings*))
          (jdata-no jd) (dense-tokens jd (third *verdict-strings*)))
    (assert (and (= 1 (length (jdata-yes jd))) (= 1 (length (jdata-no jd)))))
    (let ((rng (make-rng seed)) (pool '()))
      (dotimes (i (length texts))
        (dolist (kind kinds)
          (let* ((s (1+ (funcall rng 1000000000)))
                 (p (make-minimal-pair (svref texts i) kind names donors (make-rng s))))
            (when p (push (vector i kind s (pair-distance p)) pool)))))
      (setf (jdata-pool jd) (sort (coerce pool 'simple-vector) #'< :key (lambda (e) (svref e 3)))))
    (format out "~:D stories, ~:D minimal pairs (~{~(~A~) ~:D~^, ~}); median distance ~:D chars~%"
            (length texts) (length (jdata-pool jd))
            (loop for k in kinds collect k collect (count k (jdata-pool jd) :key (lambda (e) (svref e 1))))
            (svref (svref (jdata-pool jd) (floor (length (jdata-pool jd)) 2)) 3))
    jd))

(defun pool-pair (jd e)
  "Pool entry E's pair, remade."
  (make-minimal-pair (svref (jdata-texts jd) (svref e 0)) (svref e 1) (jdata-names jd) (jdata-donors jd)
                     (make-rng (svref e 2))))

;;; --- rows ---------------------------------------------------------------------------

(defun verdict-example (jd max-distance tt rng &key shuffle label)
  "A verdict example's tokens and the index of its answer: a pair from the
   pool no farther than MAX-DISTANCE characters, intact or corrupt by a coin,
   cut after the same sentence at or after its break.  SHUFFLE: the answer
   by another coin, not by the story (arm E).  LABEL, if given, decides the
   answer instead: a function of the story's tokens and whether it is the
   corrupt one, true for yes (for sanity checks).  NIL if it does not fit TT."
  (let* ((pool (jdata-pool jd))
         (n (let ((lo 0) (hi (length pool)))  ; the pairs no farther than MAX-DISTANCE: the pool is sorted
              (loop while (< lo hi)
                    do (let ((mid (floor (+ lo hi) 2)))
                         (if (> (svref (svref pool mid) 3) max-distance) (setf hi mid) (setf lo (1+ mid)))))
              lo))
         (e (svref pool (funcall rng (max 1 n))))
         (p (pool-pair jd e))
         (corrupt (zerop (funcall rng 2)))
         (texts (verdict-texts p (make-rng (+ 7 (svref e 2) (funcall rng 1000)))))
         (story (dense-tokens jd (if corrupt (cdr texts) (car texts))))
         (answer (cond ((and label story) (if (funcall label story corrupt) (jdata-yes jd) (jdata-no jd)))
                       ((if shuffle (zerop (funcall rng 2)) corrupt) (jdata-no jd))
                       (t (jdata-yes jd)))))
    (when (and story (<= (+ (length story) (length (jdata-suffix jd)) 2) (1+ tt)))
      (let ((tokens (concatenate '(simple-array (signed-byte 32) (*))
                                 story (jdata-suffix jd) answer
                                 (vector (story-vocab-eot (jdata-sv jd))))))
        (values tokens (- (length tokens) 2))))))

(defun verdict-row (jd tt max-distance weight rng &key shuffle label)
  "TT + 1 tokens of whole verdict examples and their weights (WEIGHT where an
   answer is the target)."
  (let ((row (make-array (1+ tt) :element-type '(signed-byte 32) :initial-element (story-vocab-eot (jdata-sv jd))))
        (w (f32vec tt)) (at 0))
    (loop repeat 50
          do (multiple-value-bind (tokens answer) (verdict-example jd max-distance tt rng :shuffle shuffle :label label)
               (when tokens
                 (if (> (+ at (length tokens)) (1+ tt))
                     (return)
                     (progn (replace row tokens :start1 at)
                            ;; the answer at ANSWER is the target of the position before it
                            (setf (aref w (+ at answer -1)) weight)
                            (incf at (length tokens)))))))
    (values row w)))

(defun stream-row (jd tt)
  "The training stream's next TT + 1 tokens (the next row starts at its last)."
  (let* ((s (jdata-stream jd)) (n (length s)))
    (when (> (+ (jdata-pos jd) tt 1) n) (setf (jdata-pos jd) 0))
    (prog1 (subseq s (jdata-pos jd) (+ (jdata-pos jd) tt 1))
      (incf (jdata-pos jd) tt))))

(defun rows-batch (rows tt)
  "ROWS, (tokens . weights-or-nil), as inputs, targets and weights (NIL if none has any)."
  (let* ((b (length rows)) (x (make-array (* b tt) :element-type '(signed-byte 32)))
         (y (make-array (* b tt) :element-type '(signed-byte 32)))
         (weighted (some #'cdr rows)) (w (and weighted (f32vec (* b tt)))))
    (loop for (row . rw) in rows for r from 0
          do (replace x row :start1 (* r tt) :end2 tt)
             (replace y row :start1 (* r tt) :start2 1)
             (when weighted
               (if rw (replace w rw :start1 (* r tt)) (fill w 1f0 :start (* r tt) :end (* (1+ r) tt)))))
    (values x y w)))

(defun judgment-batches (jd b tt &key (verdict-share (constantly 0)) (verdict-weight 1f0)
                                      (distance (constantly most-positive-fixnum)) shuffle label (seed 3))
  "A BATCHES function for nanogpt-train: each of B rows is a verdict row with
   probability (VERDICT-SHARE progress), else a next-token row; a verdict row's
   breaks are no farther than (DISTANCE progress) characters.  Progress is
   the step over the steps.  SHUFFLE: arm E's labels."
  (let ((rng (make-rng seed)))
    (lambda (step steps)
      (let* ((progress (/ step (max 1 steps))) (share (funcall verdict-share progress))
             (rows (loop repeat b
                         collect (if (< (funcall rng 1000000) (* share 1000000))
                                     (multiple-value-bind (row w)
                                         (verdict-row jd tt (funcall distance progress) verdict-weight rng :shuffle shuffle :label label)
                                       (cons row w))
                                     (cons (stream-row jd tt) nil)))))
        (rows-batch rows tt)))))

;;; --- measurements --------------------------------------------------------------------

(defun story-logliks (m jd stories)
  "Each of STORIES' (dense token vectors) total log-likelihood under M, one
   story a row after end-of-text (truncated to the row)."
  (let* ((b (ngpt-b m)) (tt (ngpt-tt m)) (eot (story-vocab-eot (jdata-sv jd)))
         (out (make-array (length stories) :element-type 'double-float))
         (losses (f32vec (* b tt))))
    (ngpt-set-weights m nil)
    (loop for start from 0 below (length stories) by b
          do (let ((x (make-array (* b tt) :element-type '(signed-byte 32) :initial-element eot))
                   (y (make-array (* b tt) :element-type '(signed-byte 32) :initial-element eot))
                   (lens '()))
               (loop for i from start below (min (length stories) (+ start b)) for r from 0
                     do (let* ((s (elt stories i)) (len (min (length s) tt)))
                          ;; inputs: eot, s0 .. s(len-2); targets s0 .. s(len-1)
                          (replace x s :start1 (1+ (* r tt)) :end2 (1- len))
                          (replace y s :start1 (* r tt) :end2 len)
                          (push len lens)))
               (ngpt-loss m x y)
               (copy-from-gbuf (ngpt-abuf m) losses (lof (ngpt-acts m) :losses))
               (loop for len in (nreverse lens) for r from 0 for i from start
                     do (setf (aref out i)
                              (- (loop for k below len sum (coerce (aref losses (+ (* r tt) k)) 'double-float)))))))
    out))

(defun make-eval-pairs (jd texts kinds per-kind &key (seed 11))
  "PER-KIND minimal pairs of each of KINDS from TEXTS (held out): a list of
   (kind intact-tokens corrupt-tokens), whole stories."
  (let ((rng (make-rng seed)) (out '()))
    (dolist (kind kinds (nreverse out))
      (let ((made 0))
        (loop for tx in texts
              while (< made per-kind)
              do (let ((p (make-minimal-pair tx kind (jdata-names jd) (jdata-donors jd) rng)))
                   (when p
                     (let ((a (dense-tokens jd (join-sentences (pair-intact p))))
                           (c (dense-tokens jd (join-sentences (pair-corrupt p)))))
                       (when (and a c)
                         (push (list kind a c) out)
                         (incf made))))))))))

(defun minimal-pair-accuracy (m jd pairs)
  "The share of PAIRS whose intact story M finds likelier (a tie half), for each kind: an alist."
  (let* ((stories (loop for (nil a c) in pairs collect a collect c))
         (ll (story-logliks m jd stories))
         (kinds (remove-duplicates (mapcar #'first pairs))))
    (loop for kind in kinds
          collect (let ((n 0) (right 0))
                    (loop for (k) in pairs for i from 0
                          when (eq k kind)
                            do (incf n)
                               ;; a tie (an untrained model's, often) is half right
                               (incf right (let ((d (- (aref ll (* 2 i)) (aref ll (1+ (* 2 i))))))
                                             (cond ((plusp d) 1) ((zerop d) 1/2) (t 0)))))
                    (cons kind (coerce (/ right (max 1 n)) 'single-float))))))

(defun verdict-accuracy (m jd pairs)
  "The share of PAIRS' stories (both of each, whole) for which M, asked
   \"Coherent?\", finds the right answer likelier than the wrong one: for
   each kind, an alist.  Two passes over the same inputs, the answer
   position's target yes in one and no in the other."
  (let* ((b (ngpt-b m)) (tt (ngpt-tt m)) (eot (story-vocab-eot (jdata-sv jd)))
         (yes (aref (jdata-yes jd) 0)) (no (aref (jdata-no jd) 0))
         (items (loop for (kind a c) in pairs
                      collect (list kind a yes) collect (list kind c no)))
         (losses (f32vec (* b tt))) (right (make-hash-table)) (n (make-hash-table)))
    (ngpt-set-weights m nil)
    (loop for start from 0 below (length items) by b
          do (let ((x (make-array (* b tt) :element-type '(signed-byte 32) :initial-element eot))
                   (y (make-array (* b tt) :element-type '(signed-byte 32) :initial-element eot))
                   (at '()))
               (loop for (kind s) in (subseq items start (min (length items) (+ start b))) for r from 0
                     do (let* ((tokens (concatenate '(simple-array (signed-byte 32) (*)) s (jdata-suffix jd)))
                               (len (length tokens)) (keep (min len (- tt 1))) (from (- len keep)))
                          ;; the story's end and the question: input eot then the last KEEP tokens
                          (replace x tokens :start1 (1+ (* r tt)) :start2 from)
                          (replace y tokens :start1 (* r tt) :start2 from)
                          (push (+ (* r tt) keep) at)))
               (setf at (nreverse at))
               (flet ((answer-losses (target)
                        (dolist (i at) (setf (aref y i) target))
                        (ngpt-loss m x y)
                        (copy-from-gbuf (ngpt-abuf m) losses (lof (ngpt-acts m) :losses))
                        (mapcar (lambda (i) (aref losses i)) at)))
                 (loop for ly in (answer-losses yes) for ln in (answer-losses no)
                       for (kind nil truth) in (subseq items start)
                       do (incf (gethash kind n 0))
                          (when (if (= truth yes) (< ly ln) (< ln ly)) (incf (gethash kind right 0)))))))
    (loop for kind in (remove-duplicates (mapcar #'first pairs))
          collect (cons kind (coerce (/ (gethash kind right 0) (max 1 (gethash kind n 0))) 'single-float)))))
