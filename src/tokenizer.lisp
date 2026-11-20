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
  (cache (make-hash-table :test 'equal))    ; word -> ids
  (kind :bpe)                               ; :bpe (gpt2) or :spm (llama's SentencePiece)
  (scores nil)                              ; :spm: id -> score, the higher merged first
  (add-space-prefix t))                     ; :spm: a text fragment after a special starts with a space

;;; gguf token types
(defconstant +tok-normal+ 1)
(defconstant +tok-control+ 3)
(defconstant +tok-user-defined+ 4)
(defconstant +tok-byte+ 6)

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
         (types (copy-seq (coerce (or (meta g "tokenizer.ggml.token_type")
                                      (make-array (length pieces) :initial-element 1))
                                  'simple-vector)))
         (v (%make-vocab :pieces pieces :types types
                         :pre (meta g "tokenizer.ggml.pre" "default")
                         :bos (meta g "tokenizer.ggml.bos_token_id")
                         :eos (meta g "tokenizer.ggml.eos_token_id")
                         :eot (meta g "tokenizer.ggml.eot_token_id")
                         :add-bos (meta g "tokenizer.ggml.add_bos_token"))))
    (cond ((equal (meta g "tokenizer.ggml.model") "gpt2"))
          ((equal (meta g "tokenizer.ggml.model") "llama")
           (return-from make-vocab-from-gguf (finish-spm-vocab v g)))
          (t (error "tokenizer ~S is not supported (gpt2 byte-level BPE and llama SentencePiece are)"
                    (meta g "tokenizer.ggml.model"))))
    ;; DeepSeek's markers (<｜User｜>, <think>, ...) are typed normal in
    ;; some conversions (V4.1's): they are specials all the same
    ;; and so is a "normal" piece with a character no byte spells (V4.1's
    ;; ｜DSML｜): byte-level BPE can't have made it, so it is matched whole in
    ;; text and stands for itself, as ds4 has it (not "?DSML?")
    (loop for p across pieces for id from 0
          when (and (= (svref types id) +tok-normal+)
                    (or (and (> (length p) 4) (string= "<｜" p :end2 2) (string= "｜>" p :start2 (- (length p) 2)))
                        (member p '("<think>" "</think>") :test #'string=)
                        (find-if-not (lambda (c) (gethash c *char-byte*)) p)))
            do (setf (svref types id) +tok-control+))
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

;;; --- llama's SentencePiece BPE (tokenizer.ggml.model = llama) ------------------
;;; As llama.cpp's llm_tokenizer_spm: no pre-tokenizer; a fragment's spaces
;;; become U+2581 (and, after a special or at the start, one is put in front);
;;; its characters are merged pairwise, the pair whose merged piece scores
;;; highest first (leftmost on ties), as long as the merged piece is a token;
;;; a character that is no token is spelt as its UTF-8 bytes' <0xXX> tokens.

(defun finish-spm-vocab (v g)
  (let ((pieces (vocab-pieces v)) (types (vocab-types v)))
    (setf (vocab-kind v) :spm
          (vocab-scores v) (coerce (meta g "tokenizer.ggml.scores") 'simple-vector)
          (vocab-add-space-prefix v) (let ((a (multiple-value-list (meta g "tokenizer.ggml.add_space_prefix" t))))
                                       (first a)))
    (loop for p across pieces for id from 0
          do (setf (gethash p (vocab-ids v)) id))
    (setf (vocab-specials v)
          (sort (loop for p across pieces for ty across types for id from 0
                      when (member ty (list +tok-control+ +tok-user-defined+))
                        collect (cons p id))
                #'> :key (lambda (e) (length (car e)))))
    v))

(defun spm-escape (text) (substitute (code-char #x2581) #\Space text))

(defun spm-byte-ids (vocab string)
  (loop for b across (sb-ext:string-to-octets string :external-format :utf-8)
        collect (or (gethash (format nil "<0x~2,'0X>" b) (vocab-ids vocab))
                    (error "no byte token for ~2,'0X" b))))

(defun spm-fragment (vocab text)
  "Token ids for one escaped fragment TEXT."
  (let* ((n (length text))
         (ids (vocab-ids vocab)) (scores (vocab-scores vocab))
         (len (make-array n :element-type 'fixnum :initial-element 1))
         (prev (make-array n :element-type 'fixnum))
         (next (make-array n :element-type 'fixnum))
         ;; a max-heap of #(score left right size): higher score, then smaller left, first
         (heap (make-array 64 :adjustable t :fill-pointer 0)))
    (when (zerop n) (return-from spm-fragment nil))
    (dotimes (i n) (setf (aref prev i) (1- i) (aref next i) (if (= i (1- n)) -1 (1+ i))))
    (labels ((better (a b) (or (> (svref a 0) (svref b 0))
                               (and (= (svref a 0) (svref b 0)) (< (svref a 1) (svref b 1)))))
             (push-heap (x)
               (vector-push-extend x heap)
               (loop with i = (1- (fill-pointer heap))
                     while (plusp i)
                     do (let ((p (floor (1- i) 2)))
                          (if (better (aref heap i) (aref heap p))
                              (progn (rotatef (aref heap i) (aref heap p)) (setf i p))
                              (return)))))
             (pop-heap ()
               (let ((top (aref heap 0)) (last (vector-pop heap)))
                 (when (plusp (fill-pointer heap))
                   (setf (aref heap 0) last)
                   (loop with i = 0 with m = (fill-pointer heap)
                         do (let* ((l (+ 1 (* 2 i))) (r (+ 2 (* 2 i))) (best i))
                              (when (and (< l m) (better (aref heap l) (aref heap best))) (setf best l))
                              (when (and (< r m) (better (aref heap r) (aref heap best))) (setf best r))
                              (if (= best i) (return) (progn (rotatef (aref heap i) (aref heap best)) (setf i best))))))
                 top))
             (try (l r)
               (when (and (>= l 0) (>= r 0))
                 (let ((id (gethash (subseq text l (+ r (aref len r))) ids)))
                   (when id (push-heap (vector (svref scores id) l r (+ (aref len l) (aref len r)))))))))
      (loop for i from 1 below n do (try (1- i) i))
      (loop while (plusp (fill-pointer heap))
            do (let* ((b (pop-heap)) (l (svref b 1)) (r (svref b 2)))
                 (unless (or (zerop (aref len l)) (zerop (aref len r))
                             (/= (+ (aref len l) (aref len r)) (svref b 3)))
                   (incf (aref len l) (aref len r))
                   (setf (aref len r) 0
                         (aref next l) (aref next r))
                   (when (>= (aref next r) 0) (setf (aref prev (aref next r)) l))
                   (try (aref prev l) l)
                   (try l (aref next l)))))
      (loop for i = 0 then (aref next i)
            while (>= i 0)
            for piece = (subseq text i (+ i (aref len i)))
            for id = (gethash piece ids)
            if id collect id
            else append (spm-byte-ids vocab piece)))))

(defun spm-tokenize (vocab text add-bos)
  (let ((out '()) (prev-special t))
    (when (and add-bos (vocab-bos vocab)
               ;; a chat template that writes the BOS itself (Nanbeige's BOS is
               ;; <|im_start|>) does not get a second one
               (not (eql 0 (search (svref (vocab-pieces vocab) (vocab-bos vocab)) text))))
      (push (vocab-bos vocab) out))
    (dolist (f (split-specials vocab text) (nreverse out))
      (if (integerp f)
          (progn (push f out) (setf prev-special t))
          (progn
            (dolist (id (spm-fragment vocab (spm-escape (if (and prev-special (vocab-add-space-prefix vocab))
                                                            (concatenate 'string " " f)
                                                            f))))
              (push id out))
            (setf prev-special nil))))))

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

;;; \p{L}, \p{N} and \s come from illatio's own tables (unicode-data.lisp,
;;; generated from SBCL's), not from the implementation: an implementation's
;;; character predicates are allowed to know only ASCII, and the tokens must
;;; not depend on which Lisp is running.

(defun in-ranges-p (cp ranges)
  "CP inside one of RANGES' inclusive [lo, hi] pairs (ascending): a binary search."
  (let ((lo 0) (hi (1- (floor (length ranges) 2))))
    (loop
      (when (> lo hi) (return nil))
      (let* ((mid (floor (+ lo hi) 2))
             (a (aref ranges (* 2 mid)))
             (b (aref ranges (1+ (* 2 mid)))))
        (cond ((< cp a) (setq hi (1- mid)))
              ((> cp b) (setq lo (1+ mid)))
              (t (return t)))))))

(defun letterp (c)
  (and c (let ((cp (char-code c)))
           (if (< cp 128) (alpha-char-p c) (in-ranges-p cp +letter-ranges+)))))
(defun numberp* (c)
  (and c (let ((cp (char-code c)))
           (if (< cp 128) (char<= #\0 c #\9) (in-ranges-p cp +number-ranges+)))))
(defun spacep (c) (and c (in-ranges-p (char-code c) +space-ranges+)))
(declaim (inline newlinep))
(defun newlinep (c) (and c (or (char= c #\Return) (char= c #\Newline))))

(defun markp (c)
  (and c (>= (char-code c) 768) (in-ranges-p (char-code c) +mark-ranges+)))

(defun split-words (text &key (digits 1) marks)
  "TEXT split as the pre-tokenizer's regex splits it.  MARKS (qwen35): a
   word is letters and combining marks, [\\p{L}\\p{M}]+, and a run of other
   characters excludes marks too."
  (let ((n (length text)) (pos 0) (words '()))
    (flet ((at (i) (and (< i n) (char text i)))
           (wordp (c) (or (letterp c) (and marks (markp c)))))
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
                         (let ((s (cond ((wordp c) pos)
                                        ((and (not (newlinep c)) (not (numberp* c))
                                              (wordp (at (1+ pos))))
                                         (1+ pos)))))
                           (and s (loop for i from s while (wordp (at i))
                                        finally (return i))))
                         ;; \p{N}{1,digits}
                         ;; (not FINALLY's I: after a BELOW end test its value is
                         ;; the spec's open question -- SBCL leaves the bound,
                         ;; modus, like ansi-test, the last one inside, and a
                         ;; one-digit run never advanced)
                         (and (numberp* c)
                              (let ((i pos) (lim (min n (+ pos digits))))
                                (loop while (and (< i lim) (numberp* (at i))) do (incf i))
                                i))
                         ;; ' ?[^\s\p{L}\p{N}]+[\r\n]*'
                         (let ((s (if (and (char= c #\Space) (at (1+ pos))) (1+ pos) pos)))
                           (flet ((other (x) (and x (not (spacep x)) (not (wordp x)) (not (numberp* x)))))
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

;;; DeepSeek V4 Flash declares tokenizer.ggml.pre = "joyai-llm".  This is
;;; ds4.c's split for it (bpe_tokenize_text), rule for rule: the JoyAI
;;; pattern
;;;   \p{N}{1,3} | [CJK/kana]+ | [P/S][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
;;;   |  ?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
;;; with ds4's choices: ASCII punctuation is the P/S class, any non-ASCII
;;; character is letter-like (CJK and kana runs are taken first), a
;;; punctuation run keeps its trailing newlines, and in a run of spaces
;;; before a word the last space goes with the word.

(defun joyai-punctp (c)
  (let ((k (char-code c)))
    (or (<= 33 k 47) (<= 58 k 64) (<= 91 k 96) (<= 123 k 126))))
(defun joyai-alphap (c) (let ((k (char-code c))) (or (<= 65 k 90) (<= 97 k 122))))
(defun joyai-digitp (c) (char<= #\0 c #\9))
(defun joyai-letterp (c) (or (joyai-alphap c) (>= (char-code c) 128)))
(defun joyai-cjkp (c)
  (let ((k (char-code c))) (or (<= #x4e00 k #x9fa5) (<= #x3040 k #x309f) (<= #x30a0 k #x30ff))))
(defun joyai-spacep (c) (member (char-code c) '(32 9 10 13 11 12)))
(defun joyai-newlinep (c) (member c '(#\Newline #\Return)))

(defun split-words-joyai (text)
  (let ((len (length text)) (pos 0) (words '()))
    (flet ((at (p) (and (< p len) (char text p))))
      (loop while (< pos len)
            do (let ((start pos) (c (char text pos)))
                 (flet ((letters () (loop while (and (< pos len) (joyai-letterp (char text pos))) do (incf pos)))
                        (punct-run () (loop while (and (< pos len) (joyai-punctp (char text pos))) do (incf pos))
                                      (loop while (and (< pos len) (joyai-newlinep (char text pos))) do (incf pos))))
                   (cond ((joyai-digitp c)
                          (loop repeat 3 while (and (< pos len) (joyai-digitp (char text pos))) do (incf pos)))
                         ((joyai-cjkp c)
                          (loop do (incf pos) while (and (< pos len) (joyai-cjkp (char text pos)))))
                         ((and (joyai-punctp c) (at (1+ pos)) (joyai-alphap (at (1+ pos))))
                          (incf pos)
                          (loop while (and (< pos len) (joyai-alphap (char text pos))) do (incf pos)))
                         ((joyai-letterp c) (letters))
                         ((and (not (joyai-newlinep c)) (not (joyai-punctp c))
                               (at (1+ pos)) (joyai-letterp (at (1+ pos))))
                          (incf pos) (letters))
                         ((and (char= c #\Space) (at (1+ pos)) (joyai-punctp (at (1+ pos))))
                          (incf pos) (punct-run))
                         ((joyai-punctp c) (punct-run))
                         ((joyai-spacep c)
                          (let ((p pos) (last-nl nil))
                            (loop while (and (< p len) (joyai-spacep (char text p)))
                                  do (when (joyai-newlinep (char text p)) (setf last-nl (1+ p)))
                                     (incf p))
                            (setf pos (cond (last-nl last-nl)
                                            ((and (< p len) (> p (1+ pos))
                                                  (or (joyai-letterp (char text p)) (joyai-punctp (char text p))))
                                             (1- p))
                                            (t p)))))
                         (t (incf pos))))
                 (when (= pos start) (incf pos))
                 (push (subseq text start pos) words))))
    (nreverse words)))

(defun pre-tokenizer-digits (pre)
  (cond ((member pre '("qwen2" "qwen35" "deepseek-r1-qwen") :test #'equal) 1)
        ((member pre '("llama3" "llama-bpe" "smaug-bpe" "glm4") :test #'equal) 3)
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
  (when (eq (vocab-kind vocab) :spm)
    (return-from tokenize-with (spm-tokenize vocab text add-bos)))
  (let* ((joyai (equal (vocab-pre vocab) "joyai-llm"))
         (digits (if joyai 3 (pre-tokenizer-digits (vocab-pre vocab))))
         (marks (equal (vocab-pre vocab) "qwen35")))
    (append (and add-bos (vocab-bos vocab) (list (vocab-bos vocab)))
            (loop for f in (split-specials vocab text)
                  if (integerp f) collect f
                  else append (loop for w in (if joyai (split-words-joyai f) (split-words f :digits digits :marks marks))
                                    append (bpe-word vocab w))))))

;;; --- decoding ---------------------------------------------------------------

(defun token-bytes (vocab id)
  "The bytes token ID stands for.  A control token stands for its own name."
  (let ((piece (svref (vocab-pieces vocab) id)))
    (when (eq (vocab-kind vocab) :spm)
      (return-from token-bytes
        (cond ((= (svref (vocab-types vocab) id) +tok-byte+)
               (make-array 1 :element-type '(unsigned-byte 8)
                             :initial-element (parse-integer piece :start 3 :end 5 :radix 16)))
              ((= (svref (vocab-types vocab) id) +tok-normal+)
               (sb-ext:string-to-octets (substitute #\Space (code-char #x2581) piece) :external-format :utf-8))
              (t (sb-ext:string-to-octets piece :external-format :utf-8)))))
    (if (= (svref (vocab-types vocab) id) +tok-normal+)
        (spelling-bytes piece)
        (sb-ext:string-to-octets piece :external-format :utf-8))))

(defun detokenize-with (vocab ids)
  (utf8-decode
   (let ((out (make-array 0 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0)))
     (dolist (id ids (coerce out '(simple-array (unsigned-byte 8) (*))))
       (loop for b across (token-bytes vocab id) do (vector-push-extend b out))))))
