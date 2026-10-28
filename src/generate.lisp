;;;; generate.lisp — sampling, and turning a prompt into a stream of text.
;;;
;;; A context remembers the tokens its KV cache holds.  GENERATE is given the
;;; whole prompt every time and evaluates only what differs from that
;;; history: an agent resends its entire conversation each turn, and
;;; re-running all of it would make every turn as slow as the first.  The
;;; cache is indexed by position, so rewinding to the common prefix is just
;;; moving POS back -- positions past it get overwritten.

(in-package #:illatio)

;;; --- sampling ---------------------------------------------------------------

(defstruct (sampler (:constructor %make-sampler))
  (temperature 0.8 :type real)
  (top-k 40 :type fixnum)
  (top-p 0.95 :type real)
  (min-p 0.05 :type real)
  (random-state (make-random-state t)))

(defun make-sampler (&key (temperature 0.8) (top-k 40) (top-p 0.95) (min-p 0.05) seed)
  "Temperature 0 is greedy.  Otherwise llama.cpp's default chain, in its
   order: top-k, top-p, min-p, then temperature, then a draw."
  (%make-sampler :temperature temperature :top-k (if (plusp top-k) top-k most-positive-fixnum)
                 :top-p top-p :min-p min-p
                 :random-state (if seed
                                   (sb-ext:seed-random-state seed)
                                   (make-random-state t))))

(defun sample (sampler logits)
  (declare (type f32vec logits))
  (let ((temp (sampler-temperature sampler)))
    (if (<= temp 0)
        (argmax logits)
        (let* ((k (min (sampler-top-k sampler) (length logits)))
               (cands (mapcar #'car (top-logprobs logits k)))
               (mx (aref logits (first cands)))
               ;; probabilities at temperature 1 decide what survives top-p
               ;; and min-p; temperature only reshapes the survivors
               (ps (mapcar (lambda (i) (exp (coerce (- (aref logits i) mx) 'double-float))) cands))
               (z (reduce #'+ ps))
               (ps (mapcar (lambda (p) (/ p z)) ps)))
          ;; top-p: the shortest head whose mass reaches top-p
          (let ((n (loop for p in ps for i from 1 sum p into acc
                         when (>= acc (sampler-top-p sampler)) return i
                         finally (return (length ps)))))
            (setf cands (subseq cands 0 n) ps (subseq ps 0 n)))
          ;; min-p: drop what is less than min-p times the most likely
          (let ((floor (* (sampler-min-p sampler) (first ps))))
            (loop for p in ps for c in cands
                  when (>= p floor) collect c into kc and collect p into kp
                  finally (setf cands kc ps kp)))
          (let* ((ws (mapcar (lambda (i) (exp (coerce (/ (- (aref logits i) mx) temp) 'double-float)))
                             cands))
                 (r (random (reduce #'+ ws) (sampler-random-state sampler))))
            (loop for w in ws for c in cands
                  do (decf r w)
                  when (<= r 0) return c
                  finally (return (car (last cands)))))))))

;;; --- prompt history ---------------------------------------------------------

(defun context-tokens (ctx) (getf (context-plist ctx) :tokens))
(defun (setf context-tokens) (v ctx) (setf (getf (context-plist ctx) :tokens) v))

(defun sync-prompt (ctx ids)
  "Make the cache hold exactly IDS, re-evaluating only past the prefix it
   already has, and leave the logits of the last one.  Returns how many
   tokens were reused."
  (let* ((have (context-tokens ctx))
         (common (or (mismatch have ids) (length ids))))
    ;; The last prompt token is always re-run even when cached: its logits
    ;; are what the next token is sampled from, and we don't keep logits
    ;; per position.
    (setf common (min common (1- (length ids))))
    (setf (context-pos ctx) common
          (context-tokens ctx) (subseq ids 0 common))
    (eval-tokens ctx (nthcdr common ids))
    (setf (context-tokens ctx) (copy-list ids))
    common))

;;; --- text that is safe to show ----------------------------------------------
;;; A token can end in the middle of a UTF-8 character (byte-level BPE splits
;;; wherever the merges say).  Streamed text holds back an incomplete tail
;;; until the bytes that finish it arrive.

(defun utf8-complete-length (bytes)
  "How many leading BYTES form whole characters."
  (let ((n (length bytes)))
    (loop for back from 1 to (min 4 n)
          for b = (aref bytes (- n back))
          do (cond ((< b #x80) (return n))                     ; ascii: complete
                   ((>= b #xc0)                                 ; a lead byte
                    (let ((need (cond ((>= b #xf0) 4) ((>= b #xe0) 3) (t 2))))
                      (return (if (>= back need) n (- n back)))))))
    n))

;;; --- generation -------------------------------------------------------------

(defun stop-tokens (model)
  "Tokens that end a turn: EOS, EOT, and the chat end-of-turn marker."
  (let ((v (model-vocab model)))
    (remove-duplicates
     (remove nil (list* (vocab-eos v) (vocab-eot v)
                        (loop for name in '("<|im_end|>" "<|endoftext|>" "<|eot_id|>" "<|end_of_text|>")
                              collect (gethash name (vocab-ids v))))))))

(defun generate (ctx prompt &key (max-tokens 256) (sampler (make-sampler :temperature 0))
                                 (stop '()) on-text (stop-tokens (stop-tokens (context-model ctx))))
  "Continue PROMPT (a string or token ids) in CTX.  ON-TEXT, if given, is
   called with each new piece of text as it becomes printable.  Stops at a
   stop token, after MAX-TOKENS, or when the text contains one of the STOP
   strings (which is not included).  Returns (values text token-ids
   finish-reason reused-prompt-tokens)."
  (let* ((m (context-model ctx))
         (ids (if (stringp prompt) (tokenize m prompt) prompt))
         (room (- (context-n-ctx ctx) (length ids))))
    (when (<= room 0)
      (error "the prompt is ~D tokens; the context holds ~D" (length ids) (context-n-ctx ctx)))
    (let ((reused (sync-prompt ctx ids))
          (bytes (make-array 0 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0))
          (shown 0)                     ; bytes already passed to ON-TEXT
          (out '())
          (reason :length)
          (max-stop (reduce #'max stop :key #'length :initial-value 0)))
      (flet ((text-so-far () (utf8-decode (coerce bytes '(simple-array (unsigned-byte 8) (*))))))
        (loop repeat (min max-tokens room)
              do (let ((tok (sample sampler (context-logits ctx))))
                   (when (member tok stop-tokens) (setf reason :stop) (return))
                   (push tok out)
                   (loop for b across (token-bytes (model-vocab m) tok) do (vector-push-extend b bytes))
                   (forward ctx tok)
                   (setf (context-tokens ctx) (append (context-tokens ctx) (list tok)))
                   (let ((text (and stop (text-so-far))))
                     (when (and stop (some (lambda (s) (search s text)) stop))
                       (setf reason :stop) (return)))
                   (when on-text
                     ;; hold back an unfinished character, and enough of the
                     ;; tail that a stop string can't be half-shown
                     (let ((ready (- (utf8-complete-length bytes) (if stop (* 4 max-stop) 0))))
                       (when (> ready shown)
                         (let ((ready (utf8-complete-length (subseq bytes 0 ready))))
                           (when (> ready shown)
                             (funcall on-text (utf8-decode (coerce (subseq bytes shown ready)
                                                                   '(simple-array (unsigned-byte 8) (*)))))
                             (setf shown ready))))))))
        (let* ((text (text-so-far))
               (cut (and stop (loop for s in stop
                                    for at = (search s text)
                                    when at minimize at))))
          (when cut (setf text (subseq text 0 cut)))
          (when on-text
            (let ((already (length (utf8-decode (coerce (subseq bytes 0 shown)
                                                        '(simple-array (unsigned-byte 8) (*)))))))
              (when (> (length text) already)
                (funcall on-text (subseq text already)))))
          (values text (nreverse out) reason reused))))))

(defun complete (model prompt &rest keys &key (n-ctx 4096) &allow-other-keys)
  "One-shot: a fresh context, PROMPT continued.  See GENERATE for KEYS."
  (remf keys :n-ctx)
  (let ((ctx (make-context model :n-ctx n-ctx)))
    (unwind-protect (apply #'generate ctx prompt keys)
      (free-context ctx))))
