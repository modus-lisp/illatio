;;;; stories.lisp — TinyStories for the judgment-first experiment
;;;; (docs/judgment-first.md): tokenized with a teacher's tokenizer, its ids
;;;; pruned to those the stories use, written as llm.c token shards.
;;;
;;; A 20M-parameter student cannot afford Qwen's 151K-row embedding, but it
;;; must share the teachers' tokens, so their distributions can be read over
;;; its vocabulary.  So the stories are tokenized by the teacher, and every id
;;; they use (with those of the verdict's words) is renumbered densely, in
;;; the teacher's order.  The map is kept beside the shards.
;;;
;;; A story file is TinyStories': stories separated by <|endoftext|> lines.
;;; In a shard, each story is followed by the end-of-text token.

(in-package #:illatio)

(defparameter *verdict-strings* '("\nCoherent?" " yes" " no")
  "What a verdict example appends to a story; its ids are always kept.")

(defun map-stories (path fn &key (max nil))
  "FN called on each story in the TinyStories file PATH (trimmed), the first MAX."
  (with-open-file (s path :external-format :utf-8)
    (let ((lines '()) (n 0))
      (flet ((flush ()
               (let ((text (string-trim '(#\Space #\Newline #\Return #\Tab)
                                        (format nil "~{~A~^~%~}" (nreverse lines)))))
                 (setf lines '())
                 (when (plusp (length text))
                   (funcall fn text)
                   (incf n)
                   (when (and max (>= n max)) (return-from map-stories n))))))
        (loop for line = (read-line s nil)
              while line
              do (if (string= (string-trim '(#\Space #\Return) line) "<|endoftext|>")
                     (flush)
                     (push line lines))
              finally (flush))
        n))))

(defun tokenize-stories (vocab path &key max-tokens max-stories)
  "The stories of PATH tokenized by VOCAB (no BOS), as a list of vectors, until
   MAX-TOKENS tokens or MAX-STORIES stories."
  (let ((out '()) (total 0))
    (block read
      (map-stories path
                   (lambda (text)
                     (let ((ids (coerce (tokenize-with vocab text :add-bos nil) '(simple-array (signed-byte 32) (*)))))
                       (push ids out)
                       (incf total (1+ (length ids)))
                       (when (and max-tokens (>= total max-tokens)) (return-from read))))
                   :max max-stories))
    (nreverse out)))

(defun write-token-shard (path tokens)
  "TOKENS (a vector of ids under 65536) as an llm.c token shard."
  (let ((h (make-array 256 :element-type '(signed-byte 32) :initial-element 0))
        (v (make-array (length tokens) :element-type '(unsigned-byte 16))))
    (replace h (list 20240520 1 (length tokens)))
    (replace v tokens)
    (ensure-directories-exist path)
    (let ((fd (sb-unix:unix-open (namestring (merge-pathnames path))
                                 (logior sb-unix:o_wronly sb-unix:o_creat sb-unix:o_trunc) #o644)))
      (unless fd (error "cannot write ~A" path))
      (unwind-protect (progn (raw-write fd h 1024) (raw-write fd v (* 2 (length v))))
        (sb-unix:unix-close fd)))
    path))

(defstruct (story-vocab (:constructor %make-story-vocab))
  kept      ; dense id -> the teacher's id
  dense     ; the teacher's id -> dense id (a hash table)
  eot)      ; the end-of-text token's dense id

(defun make-story-vocab (vocab story-lists)
  "The dense vocabulary of every id in STORY-LISTS (lists of id vectors), the
   verdict's words and end-of-text."
  (let ((seen (make-hash-table)) (eot (gethash "<|endoftext|>" (vocab-ids vocab))))
    (assert eot)
    (setf (gethash eot seen) t)
    (dolist (s *verdict-strings*)
      (dolist (id (coerce (tokenize-with vocab s :add-bos nil) 'list)) (setf (gethash id seen) t)))
    (dolist (stories story-lists)
      (dolist (ids stories) (loop for id across ids do (setf (gethash id seen) t))))
    (let* ((kept (sort (coerce (loop for k being the hash-keys of seen collect k) 'vector) #'<))
           (dense (make-hash-table)))
      (loop for id across kept for i from 0 do (setf (gethash id dense) i))
      (%make-story-vocab :kept kept :dense dense :eot (gethash eot dense)))))

(defun story-vocab-size (sv) (length (story-vocab-kept sv)))

(defun dense-ids (sv ids)
  (map '(simple-array (signed-byte 32) (*)) (lambda (id) (gethash id (story-vocab-dense sv))) ids))

(defun pack-stories (sv stories)
  "STORIES (teacher ids) in dense ids, each followed by end-of-text, as one vector."
  (let ((out (make-array (loop for s in stories sum (1+ (length s))) :element-type '(signed-byte 32))) (i 0))
    (dolist (s stories out)
      (loop for id across s do (setf (aref out i) (gethash id (story-vocab-dense sv))) (incf i))
      (setf (aref out i) (story-vocab-eot sv)) (incf i))))

(defun write-story-vocab (sv path)
  (let ((h (make-array 256 :element-type '(signed-byte 32) :initial-element 0))
        (v (coerce (story-vocab-kept sv) '(simple-array (signed-byte 32) (*)))))
    (replace h (list 20261107 1 (length v) (story-vocab-eot sv)))
    (let ((fd (sb-unix:unix-open (namestring (merge-pathnames path))
                                 (logior sb-unix:o_wronly sb-unix:o_creat sb-unix:o_trunc) #o644)))
      (unwind-protect (progn (raw-write fd h 1024) (raw-write fd v (* 4 (length v))))
        (sb-unix:unix-close fd)))))

(defun read-story-vocab (path)
  (with-raw-file (fd path)
    (let* ((h (read-header fd 20261107))
           (kept (make-array (aref h 2) :element-type '(signed-byte 32)))
           (dense (make-hash-table)))
      (raw-read fd kept (* 4 (length kept)))
      (loop for id across kept for i from 0 do (setf (gethash id dense) i))
      (%make-story-vocab :kept kept :dense dense :eot (aref h 3)))))

(defun prepare-stories (&key (dir (merge-pathnames "workspace/data/tinystories/" (user-homedir-pathname)))
                             (teacher (merge-pathnames "models/qwen2.5-0.5b-instruct-q8_0.gguf" (user-homedir-pathname)))
                             (train-tokens 25000000) (out *standard-output*))
  "Tokenize TinyStories V2 (GPT-4) with TEACHER's tokenizer: the first
   TRAIN-TOKENS tokens of the training stories and all the validation ones.
   Writes DIR/qwen/stories_{train,val}_000.bin and vocab.bin.  The vocabulary."
  (let* ((vocab (make-vocab-from-gguf (read-gguf teacher)))
         (train (tokenize-stories vocab (merge-pathnames "TinyStoriesV2-GPT4-train.txt" dir) :max-tokens train-tokens))
         (val (tokenize-stories vocab (merge-pathnames "TinyStoriesV2-GPT4-valid.txt" dir)))
         (sv (make-story-vocab vocab (list train val)))
         (qdir (merge-pathnames "qwen/" dir)))
    (ensure-directories-exist qdir)
    (let ((tr (pack-stories sv train)) (va (pack-stories sv val)))
      (write-token-shard (merge-pathnames "stories_train_000.bin" qdir) tr)
      (write-token-shard (merge-pathnames "stories_val_000.bin" qdir) va)
      (write-story-vocab sv (merge-pathnames "vocab.bin" qdir))
      (format out "~:D training stories, ~:D tokens; ~:D validation stories, ~:D tokens; vocabulary ~:D~%"
              (length train) (length tr) (length val) (length va) (story-vocab-size sv)))
    sv))
