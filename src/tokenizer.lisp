;;;; tokenizer.lisp — GPT-2 style byte-level BPE (tokenizer.ggml.model = gpt2).
;;;
;;; Three stages, as in llama.cpp:
;;;   1. special tokens (<|im_start|> ...) are cut out of the text first,
;;;      longest first, and become their own ids;
;;;   2. the rest is split into words by the model's pre-tokenizer pattern;
;;;   3. each word's UTF-8 bytes, spelt in GPT-2's printable byte alphabet,
;;;      are merged pairwise by merge rank until no ranked pair is left.

(in-package #:illatio)

(defstruct (vocab (:constructor %make-vocab))
  (pieces #() :type simple-vector)          ; id -> piece (byte alphabet)
  (types #() :type simple-vector)           ; id -> gguf token type
  (ids (make-hash-table :test 'equal))      ; piece -> id
  (ranks (make-hash-table :test 'equal))    ; "left right" -> rank
  (specials '())                            ; (piece . id), longest first
  (pre "default")
  (bos nil) (eos nil) (eot nil)
  (add-bos nil)
  (cache (make-hash-table :test 'equal)))   ; word -> ids

;;; gguf token types
(defconstant +tok-normal+ 1)
(defconstant +tok-control+ 3)
(defconstant +tok-user-defined+ 4)

;;; --- GPT-2's byte alphabet --------------------------------------------------
;;; Printable bytes stand for themselves; the other 68 are moved to U+0100 and
;;; up, so every byte is a visible character and a piece is a plain string.

(defparameter *byte-char*
  (let ((v (make-array 256)) (n 0))
    (dotimes (b 256 v)
      (setf (aref v b)
            (if (or (<= 33 b 126) (<= 161 b 172) (<= 174 b 255))
                (code-char b)
                (prog1 (code-char (+ 256 n)) (incf n)))))))

(defparameter *char-byte*
  (let ((h (make-hash-table)))
    (dotimes (b 256 h) (setf (gethash (aref *byte-char* b) h) b))))

(defun bytes-spelling (string)
  "STRING's UTF-8 bytes in the byte alphabet."
  (map 'string (lambda (b) (aref *byte-char* b))
       (sb-ext:string-to-octets string :external-format :utf-8)))

(defun spelling-bytes (piece)
  (map '(vector (unsigned-byte 8)) (lambda (c) (or (gethash c *char-byte*) 63)) piece))

;;; --- building ---------------------------------------------------------------

(defun make-vocab-from-gguf (g)
  (let* ((pieces (coerce (meta g "tokenizer.ggml.tokens") 'simple-vector))
         (types (coerce (or (meta g "tokenizer.ggml.token_type")
                            (make-array (length pieces) :initial-element 1))
                        'simple-vector))
         (v (%make-vocab :pieces pieces :types types
                         :pre (meta g "tokenizer.ggml.pre" "default")
                         :bos (meta g "tokenizer.ggml.bos_token_id")
                         :eos (meta g "tokenizer.ggml.eos_token_id")
                         :eot (meta g "tokenizer.ggml.eot_token_id")
                         :add-bos (meta g "tokenizer.ggml.add_bos_token"))))
    (unless (equal (meta g "tokenizer.ggml.model") "gpt2")
      (error "tokenizer ~S is not supported (only gpt2 byte-level BPE)"
             (meta g "tokenizer.ggml.model")))
    (loop for p across pieces for id from 0
          do (setf (gethash p (vocab-ids v)) id))
    (loop for m across (meta g "tokenizer.ggml.merges") for rank from 0
          do (setf (gethash m (vocab-ranks v)) rank))
    (setf (vocab-specials v)
          (sort (loop for p across pieces for ty across types for id from 0
                      when (member ty (list +tok-control+ +tok-user-defined+))
                        collect (cons p id))
                #'> :key (lambda (e) (length (car e)))))
    v))

;;; --- stage 1: specials ------------------------------------------------------
;;; Fragments are strings (still to be split) or ids (done).  Each special in
;;; turn, longest first, is cut out of every string fragment: that is
;;; llama.cpp's tokenizer_st_partition, and longest-first is what makes
;;; <|im_start|> win over any shorter special that is a prefix of it.

(defun cut (string piece id)
  "STRING with every occurrence of PIECE replaced by the fragment ID."
  (loop with start = 0
        for hit = (search piece string :start2 start)
        while hit
        when (> hit start) collect (subseq string start hit) into out
        collect id into out
        do (setf start (+ hit (length piece)))
        finally (return (if (< start (length string))
                            (append out (list (subseq string start)))
                            out))))

(defun split-specials (vocab text)
  (let ((frags (list text)))
    (loop for (piece . id) in (vocab-specials vocab)
          do (setf frags (loop for f in frags
                               if (and (stringp f) (search piece f)) append (cut f piece id)
                               else collect f)))
    frags))

;;; --- stage 2: words ---------------------------------------------------------
;;; The qwen2 pre-tokenizer pattern is
;;;
;;;   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
;;;   |[^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n]*
;;;   |\s*[\r\n]+ | \s+(?!\S) | \s+
;;;
;;; matched leftmost-first at each position.  llama3's pattern is the same
;;; but with \p{N}{1,3}; DIGITS is that bound.  Written out by hand, the way
;;; llama.cpp's own unicode_regex_split_custom_llama3 does it.

(declaim (inline letterp numberp* spacep newlinep))
(defun letterp (c) (and c (member (sb-unicode:general-category c) '(:lu :ll :lt :lm :lo))))
(defun numberp* (c) (and c (member (sb-unicode:general-category c) '(:nd :nl :no))))
(defun spacep (c) (and c (sb-unicode:whitespace-p c)))
(defun newlinep (c) (and c (or (char= c #\Return) (char= c #\Newline))))

(defun split-words (text &key (digits 1))
  (let ((n (length text)) (pos 0) (words '()))
    (flet ((at (i) (and (< i n) (char text i))))
      (loop while (< pos n)
            do (let* ((c (at pos))
                      (end
                        (or
                         ;; contractions
                         (and (char= c #\')
                              (let ((a (and (at (1+ pos)) (char-downcase (at (1+ pos)))))
                                    (b (and (at (+ pos 2)) (char-downcase (at (+ pos 2))))))
                                (cond ((member a '(#\s #\t #\m #\d)) (+ pos 2))
                                      ((or (and (eql a #\r) (eql b #\e))
                                           (and (eql a #\v) (eql b #\e))
                                           (and (eql a #\l) (eql b #\l)))
                                       (+ pos 3)))))
                         ;; [^\r\n\p{L}\p{N}]?\p{L}+
                         (let ((s (cond ((letterp c) pos)
                                        ((and (not (newlinep c)) (not (numberp* c))
                                              (letterp (at (1+ pos))))
                                         (1+ pos)))))
                           (and s (loop for i from s while (letterp (at i))
                                        finally (return i))))
                         ;; \p{N}{1,digits}
                         (and (numberp* c)
                              (loop for i from pos below (min n (+ pos digits))
                                    while (numberp* (at i))
                                    finally (return i)))
                         ;; ' ?[^\s\p{L}\p{N}]+[\r\n]*'
                         (let ((s (if (and (char= c #\Space) (at (1+ pos))) (1+ pos) pos)))
                           (flet ((other (x) (and x (not (spacep x)) (not (letterp x)) (not (numberp* x)))))
                             (when (other (at s))
                               (let ((i (loop for i from s while (other (at i)) finally (return i))))
                                 (loop while (newlinep (at i)) do (incf i))
                                 i))))
                         ;; the three whitespace alternatives
                         (when (spacep c)
                           (let ((run (loop for i from pos while (spacep (at i)) finally (return i))))
                             (or
                              ;; \s*[\r\n]+ : up to the last newline in the run
                              (loop with last = nil
                                    for i from pos below run
                                    when (newlinep (at i)) do (setf last i)
                                    finally (return (and last (1+ last))))
                              ;; \s+(?!\S) : all of it at the end of text, else all but one
                              (cond ((= run n) run)
                                    ((> (- run pos) 1) (1- run)))
                              ;; \s+
                              run)))
                         ;; nothing matched: one character on its own
                         (1+ pos))))
                 (push (subseq text pos end) words)
                 (setf pos end))))
    (nreverse words)))

(defun pre-tokenizer-digits (pre)
  (cond ((member pre '("qwen2" "deepseek-r1-qwen") :test #'equal) 1)
        ((member pre '("llama3" "llama-bpe" "smaug-bpe") :test #'equal) 3)
        (t (error "pre-tokenizer ~S is not supported yet" pre))))

;;; --- stage 3: merges --------------------------------------------------------

(defun bpe-word (vocab word)
  "Token ids for one pre-tokenized WORD."
  (or (gethash word (vocab-cache vocab))
      (setf (gethash word (vocab-cache vocab))
            (let ((syms (map 'list #'string (bytes-spelling word)))
                  (ranks (vocab-ranks vocab)))
              ;; Merge the lowest-ranked adjacent pair, leftmost on ties,
              ;; until no adjacent pair has a rank.
              (loop
                (let ((best nil) (best-rank nil))
                  (loop for (a b) on syms while b
                        for cell on syms
                        do (let ((r (gethash (concatenate 'string a " " b) ranks)))
                             (when (and r (or (null best-rank) (< r best-rank)))
                               (setf best cell best-rank r))))
                  (unless best (return))
                  (setf (car best) (concatenate 'string (first best) (second best))
                        (cdr best) (cddr best))))
              (loop for s in syms
                    for id = (gethash s (vocab-ids vocab))
                    if id collect id
                    ;; not in the vocabulary: fall back to its single bytes
                    else append (loop for ch across s
                                      collect (or (gethash (string ch) (vocab-ids vocab))
                                                  (error "no token for byte ~S" ch))))))))

(defun tokenize-with (vocab text &key (add-bos (vocab-add-bos vocab)))
  (let ((digits (pre-tokenizer-digits (vocab-pre vocab))))
    (append (and add-bos (vocab-bos vocab) (list (vocab-bos vocab)))
            (loop for f in (split-specials vocab text)
                  if (integerp f) collect f
                  else append (loop for w in (split-words f :digits digits)
                                    append (bpe-word vocab w))))))

;;; --- decoding ---------------------------------------------------------------

(defun token-bytes (vocab id)
  "The bytes token ID stands for.  A control token stands for its own name."
  (let ((piece (svref (vocab-pieces vocab) id)))
    (if (= (svref (vocab-types vocab) id) +tok-normal+)
        (spelling-bytes piece)
        (sb-ext:string-to-octets piece :external-format :utf-8))))

(defun detokenize-with (vocab ids)
  (utf8-decode
   (let ((out (make-array 0 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0)))
     (dolist (id ids (coerce out '(simple-array (unsigned-byte 8) (*))))
       (loop for b across (token-bytes vocab id) do (vector-push-extend b out))))))
