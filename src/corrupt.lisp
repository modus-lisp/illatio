;;;; corrupt.lisp — minimal pairs for the judgment-first experiment
;;;; (docs/judgment-first.md): a story intact, and the same story with
;;;; exactly one break, made by program.
;;;
;;; Three kinds of break:
;;;   :swap    two sentences trade places (their text only; the spaces and
;;;            paragraph breaks between sentences stay where they were, or
;;;            the layout would give the swap away);
;;;   :rename  a character is called by another name from some sentence on
;;;            (the names drawn from the corpus's own, by frequency, so a
;;;            rare name is no tell);
;;;   :referent from some sentence on, one character is called by another's
;;;            name (both already in the story: a rename would bring in a
;;;            new name, and a bag of words can count names);
;;;   :splice  the ending, from some sentence on, is another story's of a
;;;            similar length.
;;; Each pair records the sentence where the break is (BREAK) and how far, in
;;; characters, the break is from the fact it violates (DISTANCE): for a swap
;;; the span between the two sentences, for a rename the span from the
;;; name's last use before the break, for a splice the length of what came
;;; before.  A verdict example cuts both stories after the same sentence, at
;;; or after the break.

(in-package #:illatio)

;;; --- sentences ------------------------------------------------------------------

(defun split-sentences (text)
  "TEXT as a vector of (body . separator).  A sentence ends at . ! or ?
   outside quotation marks and followed by whitespace, or at a line break: a
   quotation stays one piece with its attribution (\"Yes!\" Lily said.), so a
   swap cannot split one.  The whitespace after a sentence is its separator.
   Quotes are counted afresh on each line.  Bodies and separators,
   concatenated, are TEXT."
  (let ((out '()) (start 0) (n (length text)) (i 0) (quoted nil))
    (flet ((end-at (j)
             ;; the sentence is START..J; its separator the whitespace after
             (let ((k j))
               (loop while (and (< k n) (member (char text k) '(#\Space #\Newline #\Tab #\Return))) do (incf k))
               (push (cons (subseq text start j) (subseq text j k)) out)
               (setf start k i k quoted nil))))
      (loop while (< i n)
            do (let ((c (char text i)))
                 (cond ((char= c #\Newline) (if (> i start) (end-at i) (incf i)))
                       ((find c "\"“”")
                        (setf quoted (case c (#\“ t) (#\” nil) (t (not quoted))))
                        (incf i))
                       ((and (find c ".!?") (not quoted)
                             (or (= (1+ i) n) (member (char text (1+ i)) '(#\Space #\Newline #\Tab #\Return))))
                        (end-at (1+ i)))
                       (t (incf i))))))
    (when (< start n) (push (cons (subseq text start) "") out))
    (coerce (nreverse out) 'simple-vector)))

(defun join-sentences (sentences &optional (end (length sentences)))
  "The text of SENTENCES' first END, without the last one's separator."
  (with-output-to-string (o)
    (loop for i below end
          for (body . sep) = (svref sentences i)
          do (write-string body o)
             (when (< i (1- end)) (write-string sep o)))))

(defun sentence-start (sentences k)
  "The character offset of sentence K."
  (loop for i below k sum (+ (length (car (svref sentences i))) (length (cdr (svref sentences i))))))

;;; --- names ------------------------------------------------------------------------

(defun words-of (text)
  "TEXT's words (runs of letters), with each one's offset and whether it starts a sentence."
  (let ((out '()) (n (length text)) (i 0) (sentence-start t))
    (loop while (< i n)
          do (let ((c (char text i)))
               (cond ((alpha-char-p c)
                      (let ((j i))
                        (loop while (and (< j n) (alpha-char-p (char text j))) do (incf j))
                        (push (list (subseq text i j) i sentence-start) out)
                        (setf sentence-start nil i j)))
                     (t (when (find c ".!?\"\n") (setf sentence-start t))
                        (incf i)))))
    (nreverse out)))

(defparameter *not-names*
  '("Mr" "Mrs" "Ms" "Dr" "Miss" "Mom" "Mommy" "Mum" "Mama" "Dad" "Daddy" "Papa"
    "Grandma" "Grandpa" "Granny" "Nana" "God" "Santa")
  "Capitalized words that are titles or roles, not names to swap.")

(defun pronoun-gender (text)
  "+1 for each he/him/his in TEXT, -1 for each she/her/hers: their sum."
  (loop for (w) in (words-of text)
        sum (let ((w (string-downcase w)))
              (cond ((member w '("he" "him" "his" "himself") :test #'string=) 1)
                    ((member w '("she" "her" "hers" "herself") :test #'string=) -1)
                    (t 0)))))

(defun corpus-names (stories &key (min-count 50) (ratio 0.97) (sure 0.8))
  "The words of STORIES (texts) that are names: capitalized mid-sentence
   (only the first letter) at least MIN-COUNT times and almost never lower
   case, less *NOT-NAMES*.  A list of (name weight gender), heaviest first:
   the weight is the number of stories naming it in two sentences or more
   (how often it is a rename's source); the gender, :he or :she, from the
   pronouns in the sentences where it is the only name and in the next
   ones, when SURE of them; names of no sure gender are left out."
  (let ((cap (make-hash-table :test 'equal)) (low (make-hash-table :test 'equal)))
    (dolist (s stories)
      (loop for (w nil start) in (words-of s)
            do (cond ((lower-case-p (char w 0)) (incf (gethash w low 0)))
                     ((and (not start) (string= w (string-capitalize w)))
                      (incf (gethash (string-downcase w) cap 0))))))
    (let ((names (make-hash-table :test 'equal)) (weights (make-hash-table :test 'equal))
          (he (make-hash-table :test 'equal)) (she (make-hash-table :test 'equal)))
      (maphash (lambda (k n)
                 (when (and (>= n min-count) (> (length k) 1)
                            (not (member k *not-names* :test #'string-equal))
                            (>= (/ n (+ n (gethash k low 0))) ratio))
                   (setf (gethash (string-capitalize k) names) t)))
               cap)
      (dolist (s stories)
        (let ((seen (make-hash-table :test 'equal)) (sentences (split-sentences s)))
          (dotimes (k (length sentences))
            (let ((here (remove-duplicates (loop for (w) in (words-of (car (svref sentences k)))
                                                 when (gethash w names) collect w)
                                           :test #'string=)))
              (dolist (w here) (incf (gethash w seen 0)))
              (when (= 1 (length here))
                (let ((g (+ (pronoun-gender (car (svref sentences k)))
                            (if (< (1+ k) (length sentences)) (pronoun-gender (car (svref sentences (1+ k)))) 0))))
                  (cond ((plusp g) (incf (gethash (first here) he 0)))
                        ((minusp g) (incf (gethash (first here) she 0))))))))
          (maphash (lambda (w k) (when (>= k 2) (incf (gethash w weights 0)))) seen)))
      (sort (loop for w being the hash-keys of names
                  for m = (gethash w he 0) for f = (gethash w she 0)
                  for gender = (cond ((>= m (* sure (+ m f 1))) :he) ((>= f (* sure (+ m f 1))) :she))
                  when (and gender (plusp (gethash w weights 0)))
                    collect (list w (gethash w weights) gender))
            #'> :key #'second))))

;;; --- breaks ---------------------------------------------------------------------

(defstruct pair kind intact corrupt sentences break distance)

(defun pick (list rng) (nth (funcall rng (length list)) list))

(defun pick-weighted (list rng &key (weight #'second))
  "An element of LIST with probability its WEIGHT's share."
  (let ((r (funcall rng (reduce #'+ list :key weight))))
    (loop for e in list
          do (decf r (funcall weight e))
          when (< r 0) return e)))

(defun break-swap (sentences rng)
  "Two sentences, not the first, trade bodies."
  (let ((n (length sentences)))
    (when (>= n 4)
      (let* ((i (1+ (funcall rng (- n 2)))) (j (loop for j = (1+ (funcall rng (1- n))) unless (= j i) return j))
             (a (min i j)) (b (max i j))
             (s (map 'simple-vector #'copy-tree sentences)))
        ;; identical bodies would make no break
        (unless (string= (car (svref s a)) (car (svref s b)))
          (rotatef (car (svref s a)) (car (svref s b)))
          (make-pair :kind :swap :corrupt s :break a
                     :distance (- (sentence-start sentences b) (sentence-start sentences a))))))))

(defun replace-word (text old new)
  "TEXT with each whole word OLD replaced by NEW."
  (with-output-to-string (o)
    (let ((n (length text)) (m (length old)) (i 0))
      (loop while (< i n)
            do (if (and (<= (+ i m) n) (string= old text :start2 i :end2 (+ i m))
                        (or (zerop i) (not (alpha-char-p (char text (1- i)))))
                        (or (= (+ i m) n) (not (alpha-char-p (char text (+ i m))))))
                   (progn (write-string new o) (incf i m))
                   (progn (write-char (char text i) o) (incf i)))))))

(defun break-rename (sentences names rng)
  "A character named in two sentences or more is N1 before some sentence K
   after their first, and N2 from K on; the intact story calls them N2
   throughout.  N1 and N2 are drawn alike (by weight, of the character's
   gender, not named in the story), so neither side of a pair is likelier to
   hold any name -- the break is the change of name alone."
  (let ((n (length sentences)) (known (make-hash-table :test 'equal)))
    (dolist (nm names) (setf (gethash (first nm) known) nm))
    (let ((used '()))  ; (name sentences...)
      (dotimes (k n)
        (dolist (w (remove-duplicates (loop for (w) in (words-of (car (svref sentences k)))
                                            when (gethash w known) collect w)
                                      :test #'string=))
          (let ((e (assoc w used :test #'string=)))
            (if e (nconc e (list k)) (push (list w k) used)))))
      (let ((candidates (remove-if (lambda (e) (< (length e) 3)) used)))
        (when candidates
          (destructuring-bind (name first &rest later) (pick candidates rng)
            (let* ((gender (third (gethash name known)))
                   (pool (remove-if (lambda (nm) (or (not (eq (third nm) gender))
                                                     (assoc (first nm) used :test #'string=)))
                                    names)))
              (when (>= (length pool) 2)
                (let* ((a (first (pick-weighted pool rng)))
                       (b (loop repeat 50 for o = (first (pick-weighted pool rng))
                                unless (string= o a) return o))
                       ;; B is drawn given it is not A: which is first is a coin's, or B leans light
                       (swap (zerop (funcall rng 2)))
                       (n1 (if swap b a)) (n2 (if swap a b))
                       (k (pick later rng))
                       (prev (loop for j from (1- k) downto 0
                                   when (find name (words-of (car (svref sentences j))) :key #'first :test #'string=)
                                     return j)))
                  (when n2
                    (let ((intact (map 'simple-vector #'copy-tree sentences))
                          (corrupt (map 'simple-vector #'copy-tree sentences)))
                      (dotimes (j n)
                        (setf (car (svref intact j)) (replace-word (car (svref intact j)) name n2)
                              (car (svref corrupt j)) (replace-word (car (svref corrupt j)) name (if (< j k) n1 n2))))
                      (make-pair :kind :rename :intact intact :corrupt corrupt :break k
                                 :distance (- (sentence-start sentences k) (sentence-start sentences prev))))))))))))))

(defun break-referent (sentences names rng)
  "Two characters A and B both named before some sentence K, after which A
   is called B: a wrong referent.  Unlike a rename it brings in no name, so
   the corrupted story has the intact one's names (a bag of words can count
   names: a renamed story has one more)."
  (let ((n (length sentences)) (known (make-hash-table :test 'equal)))
    (dolist (nm names) (setf (gethash (first nm) known) t))
    (let ((uses '()))  ; (name sentence...)
      (dotimes (k n)
        (dolist (w (remove-duplicates (loop for (w) in (words-of (car (svref sentences k)))
                                            when (gethash w known) collect w)
                                      :test #'string=))
          (let ((e (assoc w uses :test #'string=)))
            (if e (nconc e (list k)) (push (list w k) uses)))))
      (when (>= (length uses) 2)
        (let* ((a (pick uses rng))
               (b (pick (remove a uses) rng))
               ;; K: a sentence naming A, after both have been named
               (ks (remove-if (lambda (k) (<= k (max (second a) (second b)))) (rest a))))
          (when ks
            (let* ((k (pick ks rng))
                   (prev (loop for j in (reverse (rest a)) when (< j k) return j))
                   (corrupt (map 'simple-vector #'copy-tree sentences)))
              (loop for j from k below n
                    do (setf (car (svref corrupt j)) (replace-word (car (svref corrupt j)) (first a) (first b))))
              (make-pair :kind :referent :corrupt corrupt :break k
                         :distance (- (sentence-start sentences k) (sentence-start sentences prev))))))))))

(defun break-splice (sentences donors rng)
  "From a sentence K past the middle, another story's ending, from its own
   second half (or its opening would show), of within 10% of the length."
  (let ((n (length sentences)))
    (when (>= n 4)
      (let* ((k (+ (ceiling n 2) (funcall rng (- n (ceiling n 2)))))
             (tail (- (sentence-start sentences n) (sentence-start sentences k)))
             (len (sentence-start sentences n)))
        (loop repeat 30
              do (let* ((d (pick donors rng)) (m (length d)))
                   (when (and (>= m 4) (>= (- m (- n k)) (floor m 2)))
                     (let* ((dk (- m (- n k)))
                            (s (concatenate 'simple-vector (subseq sentences 0 k) (subseq d dk)))
                            (newlen (sentence-start s (length s))))
                       (when (and (< (abs (- newlen len)) (* 0.1 len))
                                  (not (equalp (subseq d dk) (subseq sentences k))))
                         (return (make-pair :kind :splice :corrupt s :break k
                                            :distance (- len tail))))))))))))

(defun make-minimal-pair (text kind names donors rng)
  "TEXT and a break of KIND in it, or NIL when the story has none to make."
  (let* ((sentences (split-sentences text))
         (p (ecase kind
              (:swap (break-swap sentences rng))
              (:rename (break-rename sentences names rng))
              (:referent (break-referent sentences names rng))
              (:splice (break-splice sentences donors rng)))))
    (when p
      (unless (pair-intact p) (setf (pair-intact p) sentences))
      (setf (pair-sentences p) (length sentences))
      p)))

(defun verdict-texts (p rng)
  "The pair cut after the same sentence, at or after its break: (intact . corrupt)."
  (let* ((n (min (length (pair-intact p)) (length (pair-corrupt p))))
         (cut (+ 1 (pair-break p) (funcall rng (- n (pair-break p))))))
    (cons (join-sentences (pair-intact p) cut) (join-sentences (pair-corrupt p) cut))))

(defun make-rng (seed)
  "A function of N giving integers below N: xorshift64*."
  (let ((s (logior 1 (ldb (byte 64 0) seed))))
    (lambda (n)
      (multiple-value-bind (ns out) (xorshift-u32 s)
        (setf s ns)
        (mod out n)))))

;;; --- the shortcut audit ------------------------------------------------------------
;;;
;;; Before any training: can a bag of word bigrams tell the corrupted stories
;;; from the intact ones?  It sees no order beyond adjacent words, so above
;;; chance it is reading surface traces of the edit, not coherence.  Logistic
;;; regression on hashed bigram counts, both stories of a pair on the same
;;; side of the split.

(defun bigram-features (text &optional (bits 18))
  "TEXT's word bigrams (and words), hashed to BITS bits: an alist (index . count)."
  (let ((h (make-hash-table)) (prev "<s>") (mask (1- (ash 1 bits))))
    (flet ((add (s) (incf (gethash (logand (sxhash s) mask) h 0))))
      (loop for (w) in (words-of text)
            do (let ((w (string-downcase w)))
                 (add w) (add (concatenate 'string prev " " w)) (setf prev w))))
    (loop for k being the hash-keys of h using (hash-value v)
          collect (cons k (log (1+ v))))))

(defun logistic-accuracy (train test &key (bits 18) (epochs 8) (lr 0.1d0) (l2 1d-6))
  "TRAIN and TEST: lists of (features . label), label 0 or 1.  Fit by SGD on
   TRAIN; the accuracy on TEST."
  (let ((w (make-array (ash 1 bits) :element-type 'double-float :initial-element 0d0)) (b 0d0)
        (train (coerce train 'simple-vector)))
    (flet ((score (f) (+ b (loop for (k . v) in f sum (* v (aref w k))))))
      (dotimes (e epochs)
        (let ((rate (/ lr (1+ e))))
          (loop for (f . y) across train
                do (let* ((z (score f)) (p (/ 1 (1+ (exp (- (max -30d0 (min 30d0 z)))))))
                          (g (- p y)))
                     (decf b (* rate g))
                     (loop for (k . v) in f
                           do (setf (aref w k) (- (* (aref w k) (- 1 (* rate l2))) (* rate g v))))))))
      (/ (count-if (lambda (x) (eql (if (> (score (car x)) 0) 1 0) (cdr x))) test)
         (max 1 (length test))))))

(defun shortcut-audit (texts kinds &key (per-kind 4000) (seed 7) (out *standard-output*))
  "For each of KINDS, PER-KIND minimal pairs from TEXTS cut as verdict
   examples, and a bigram model's held-out accuracy at telling the corrupted
   from the intact (chance 0.5).  An alist (kind . accuracy)."
  (let* ((rng (make-rng seed)) (names (corpus-names texts))
         (donors (mapcar #'split-sentences (subseq texts 0 (min 4000 (length texts))))))
    (loop for kind in kinds
          collect (let ((pairs '()))
                    (loop for tx in texts
                          while (< (length pairs) per-kind)
                          do (let ((p (make-minimal-pair tx kind names donors rng)))
                               (when p (push (verdict-texts p rng) pairs))))
                    (let* ((split (floor (* 4 (length pairs)) 5))
                           (rows (lambda (ps) (loop for (a . c) in ps
                                                    collect (cons (bigram-features a) 0)
                                                    collect (cons (bigram-features c) 1))))
                           (acc (logistic-accuracy (funcall rows (subseq pairs 0 split))
                                                   (funcall rows (subseq pairs split)))))
                      (format out "~8A ~:D pairs: bigram accuracy ~,3F~%" kind (length pairs) acc)
                      (finish-output out)
                      (cons kind acc))))))
