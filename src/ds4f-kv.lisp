;;;; ds4f-kv.lisp — DS4F's context state on disk: a long prompt prefilled once.
;;;
;;; As ds4's disk KV cache (ds4-server --kv-disk-dir, ds4-agent's
;;; sysprompt.kv): a prefill of a long prompt saves the context's state at
;;; the message boundaries in it -- before a <｜User｜> or <｜Assistant｜>
;;; token, so the end of a system prompt is one -- and at its end: the raw
;;; ring, the compressed and indexer rows, the compressors' windows, with the
;;; tokens they hold.  A context starting over takes the longest saved
;;; prefix of its new prompt instead of prefilling it: a new conversation
;;; with the same system prompt, or an old one carried on, starts where the
;;; saved one left off.  DeepSeek V4's state is small
;;; (about 50 MB at 6K tokens): the rows are kept through F16 already, so
;;; they are stored so, losslessly; the windows (f32, -1e30 where empty) as
;;; they are.  A saved prefix is always shorter than the prompt that takes
;;; it, so at least one token is evaluated and its logits are fresh.

(in-package #:illatio)

(defvar *kv-dir* (merge-pathnames ".cache/illatio/kv/" (user-homedir-pathname))
  "Where context states are saved; NIL saves none.")

(defparameter *kv-disk-gb* 8 "The saved states' budget; the oldest go first.")
(defparameter *kv-min-tokens* 512 "Prefills shorter than this save nothing.")
(defparameter *kv-min-gap* 512 "A saved state at least this many tokens past the last one.")

(defconstant +kv-magic+ #x31564b46344453)  ; "DS4FKV1"

(defun ds4f-model-key (m)
  "What a saved state must match: the model file, its size and its date."
  (let ((p (gguf-path (ds4f-gguf m))))
    (format nil "~A|~D|~D" (namestring (truename p))
            (with-open-file (s p :element-type '(unsigned-byte 8)) (file-length s))
            (file-write-date p))))

;;; --- raw bytes ------------------------------------------------------------------------

(defun write-u64 (s n) (dotimes (i 8) (write-byte (ldb (byte 8 (* 8 i)) n) s)))
(defun read-u64 (s) (let ((n 0)) (dotimes (i 8 n) (setf n (logior n (ash (read-byte s) (* 8 i)))))))

(defun write-f32s (s v n)
  "V's first N floats, their bits."
  (declare (type f32vec v) (type fixnum n))
  (let ((b (make-array (* 4 n) :element-type '(unsigned-byte 8))))
    (dotimes (i n) (let ((x (ldb (byte 32 0) (sb-kernel:single-float-bits (aref v i)))))
                     (setf (aref b (* 4 i)) (ldb (byte 8 0) x) (aref b (+ (* 4 i) 1)) (ldb (byte 8 8) x)
                           (aref b (+ (* 4 i) 2)) (ldb (byte 8 16) x) (aref b (+ (* 4 i) 3)) (ldb (byte 8 24) x))))
    (write-sequence b s)))

(defun read-f32s (s v n)
  (declare (type f32vec v) (type fixnum n))
  (let ((b (make-array (* 4 n) :element-type '(unsigned-byte 8))))
    (read-sequence b s)
    (dotimes (i n v)
      (let ((x (logior (aref b (* 4 i)) (ash (aref b (+ (* 4 i) 1)) 8)
                       (ash (aref b (+ (* 4 i) 2)) 16) (ash (aref b (+ (* 4 i) 3)) 24))))
        (setf (aref v i) (sb-kernel:make-single-float (if (>= x #x80000000) (- x #x100000000) x)))))))

(defun write-f16s (s v n)
  "V's first N floats through F16 (they are F16 values already)."
  (declare (type f32vec v) (type fixnum n))
  (let ((b (make-array (* 2 n) :element-type '(unsigned-byte 8))))
    (dotimes (i n) (let ((h (single-to-half (aref v i))))
                     (setf (aref b (* 2 i)) (ldb (byte 8 0) h) (aref b (1+ (* 2 i))) (ldb (byte 8 8) h))))
    (write-sequence b s)))

(defun read-f16s (s v n)
  (declare (type f32vec v) (type fixnum n))
  (let ((b (make-array (* 2 n) :element-type '(unsigned-byte 8))))
    (read-sequence b s)
    (dotimes (i n v) (setf (aref v i) (half-to-single (logior (aref b (* 2 i)) (ash (aref b (1+ (* 2 i))) 8)))))))

;;; --- a state ----------------------------------------------------------------------------

(defun tokens-hash (tokens)
  "FNV-1a over TOKENS' ids (SXHASH of a vector looks at little of it)."
  (let ((h #xcbf29ce484222325))
    (dolist (tk tokens h)
      (dotimes (i 4)
        (setf h (ldb (byte 64 0) (* (logxor h (ldb (byte 8 (* 8 i)) tk)) #x100000001b3)))))))

(defun kv-file (tokens)
  (merge-pathnames (format nil "~16,'0X-~D.kv" (tokens-hash tokens) (length tokens)) *kv-dir*))

(defun write-kv-header (s key tokens)
  (let ((kb (map '(vector (unsigned-byte 8)) #'char-code key)))
    (write-u64 s +kv-magic+) (write-u64 s (length kb)) (write-sequence kb s)
    (write-u64 s (length tokens)) (dolist (tk tokens) (write-u64 s tk))))

(defun read-kv-header (s)
  "(values model-key tokens), or NIL if S isn't a saved state."
  (when (= (read-u64 s) +kv-magic+)
    (let* ((kn (read-u64 s)) (kb (make-array kn :element-type '(unsigned-byte 8))))
      (read-sequence kb s)
      (let ((n (read-u64 s)))
        (values (map 'string #'code-char kb) (loop repeat n collect (read-u64 s)))))))

(defun ds4f-save-state (c tokens)
  "Save d4ctx C's state, which holds TOKENS, under *KV-DIR*."
  (let* ((m (d4ctx-model c)) (hd (ds4f-head-dim m)) (ixd (ds4f-ix-dim m)) (swa (ds4f-swa m))
         (path (kv-file tokens)) (tmp (make-pathname :type "tmp" :defaults path)))
    (assert (= (d4ctx-pos c) (length tokens)))
    (ensure-directories-exist path)
    (with-open-file (s tmp :direction :output :element-type '(unsigned-byte 8) :if-exists :supersede)
      (write-kv-header s (ds4f-model-key m) tokens)
      (loop for d across (d4ctx-caches c)
            do (write-u64 s (dc-n-raw d)) (write-u64 s (dc-raw-total d))
               (write-u64 s (dc-n-comp d)) (write-u64 s (dc-n-ix d))
               (write-f16s s (dc-raw d) (* swa hd))
               (when (dc-comp d)
                 (write-f16s s (dc-comp d) (* (dc-n-comp d) hd))
                 (write-f32s s (dc-st-kv d) (length (dc-st-kv d)))
                 (write-f32s s (dc-st-score d) (length (dc-st-score d))))
               (when (dc-ix-comp d)
                 (write-f16s s (dc-ix-comp d) (* (dc-n-ix d) ixd))
                 (write-f32s s (dc-ix-kv d) (length (dc-ix-kv d)))
                 (write-f32s s (dc-ix-score d) (length (dc-ix-score d))))))
    (rename-file tmp path)
    (kv-trim)
    path))

(defun ds4f-load-state (c path)
  "Load the state at PATH into d4ctx C (fresh): the tokens it holds, or NIL
   when it doesn't fit C."
  (let* ((m (d4ctx-model c)) (hd (ds4f-head-dim m)) (ixd (ds4f-ix-dim m)) (swa (ds4f-swa m)))
    (with-open-file (s path :element-type '(unsigned-byte 8))
      (multiple-value-bind (key tokens) (read-kv-header s)
        (when (and key (string= key (ds4f-model-key m)))
          (loop for d across (d4ctx-caches c)
                do (setf (dc-n-raw d) (read-u64 s) (dc-raw-total d) (read-u64 s)
                         (dc-n-comp d) (read-u64 s) (dc-n-ix d) (read-u64 s))
                   (read-f16s s (dc-raw d) (* swa hd))
                   (when (dc-comp d)
                     (when (> (* (dc-n-comp d) hd) (length (dc-comp d))) (return-from ds4f-load-state nil))
                     (read-f16s s (dc-comp d) (* (dc-n-comp d) hd))
                     (read-f32s s (dc-st-kv d) (length (dc-st-kv d)))
                     (read-f32s s (dc-st-score d) (length (dc-st-score d))))
                   (when (dc-ix-comp d)
                     (read-f16s s (dc-ix-comp d) (* (dc-n-ix d) ixd))
                     (read-f32s s (dc-ix-kv d) (length (dc-ix-kv d)))
                     (read-f32s s (dc-ix-score d) (length (dc-ix-score d)))))
          (setf (d4ctx-pos c) (length tokens))
          tokens)))))

(defun kv-files ()
  (and *kv-dir* (probe-file *kv-dir*) (directory (merge-pathnames "*.kv" *kv-dir*))))

(defun kv-trim ()
  "The oldest saved states deleted until the rest fit *KV-DISK-GB*."
  (let* ((files (sort (mapcar (lambda (p) (cons p (file-write-date p))) (kv-files)) #'> :key #'cdr))
         (budget (* *kv-disk-gb* 1000000000)) (total 0))
    (dolist (f files)
      (incf total (with-open-file (s (car f) :element-type '(unsigned-byte 8)) (file-length s)))
      (when (> total budget) (delete-file (car f))))))

(defun ds4f-longest-saved (m ids)
  "The saved state whose tokens are the longest prefix of IDS shorter than
   IDS: (values path tokens), or NIL."
  (let ((key (ds4f-model-key m)) (best nil) (best-tokens nil) (n (length ids)))
    (dolist (p (kv-files) (values best best-tokens))
      (ignore-errors
       (with-open-file (s p :element-type '(unsigned-byte 8))
         (multiple-value-bind (k tokens) (read-kv-header s)
           (let ((len (length tokens)))
             (when (and k (string= k key) (< len n) (> len (length best-tokens))
                        (= (or (mismatch tokens ids) len) len))
               (setf best p best-tokens tokens)))))))))

;;; --- the engine's side ------------------------------------------------------------------

(defun ds4f-restore-prefix (ctx ids)
  "For SYNC-PROMPT: a fresh ds4f context takes the longest saved prefix of
   IDS; how many tokens it holds then (NIL: not a ds4f context, or none)."
  (when (and *kv-dir* (typep (model-impl (context-model ctx)) 'ds4f) (zerop (context-pos ctx)))
    (multiple-value-bind (path tokens) (ds4f-longest-saved (model-impl (context-model ctx)) ids)
      (when path
        (let ((got (ds4f-load-state (ds4f-context ctx) path)))
          (when got
            (setf (context-pos ctx) (length got))
            (length got)))))))

(pushnew 'ds4f-restore-prefix *restore-prefix-hooks*)

(defun ds4f-role-ids (m)
  "The ids of the chat format's turn tokens: a message ends before one."
  (let ((ids (vocab-ids (ds4f-vocab m))))
    (remove nil (mapcar (lambda (name) (gethash name ids)) '("<｜User｜>" "<｜Assistant｜>")))))

(defun ds4f-eval-saving (c prefix tokens)
  "DS4F-PREFILL of TOKENS after PREFIX (the tokens C holds), its state saved
   at the message boundaries in them (*KV-MIN-GAP* apart at least) and at
   their end, when TOKENS are *KV-MIN-TOKENS* or more."
  (if (or (null *kv-dir*) (< (length tokens) *kv-min-tokens*))
      (ds4f-prefill c tokens)
      (let* ((roles (ds4f-role-ids (d4ctx-model c))) (pos0 (d4ctx-pos c)) (end (+ pos0 (length tokens)))
             (cuts (let ((last pos0) (out '()))
                     (loop for tk in tokens for p from pos0
                           when (and (member tk roles) (> p pos0) (>= (- p last) *kv-min-gap*) (< p end))
                             do (push p out) (setf last p))
                     (nreverse (cons end out))))
             (done (reverse prefix)) (rest tokens))
        (dolist (cut cuts)
          (let* ((n (- cut (d4ctx-pos c))) (piece (subseq rest 0 n)))
            (setf rest (nthcdr n rest))
            (ds4f-prefill c piece :logits (null rest))
            (setf done (revappend piece done))
            (handler-case (ds4f-save-state c (reverse done))
              (error (e) (warn "illatio: could not save the context's state: ~A" e))))))))
