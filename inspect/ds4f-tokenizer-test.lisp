;;;; ds4f-tokenizer-test.lisp — DeepSeek V4 Flash's tokenizer against ds4's ids.
;;;
;;;   sbcl --non-interactive --load inspect/ds4f-tokenizer-test.lisp
;;;
;;; vectors/ds4f-tokens.sexp (collect-ds4f-tokens.py) holds texts chosen to
;;; exercise the joyai-llm split -- digits, accents, CJK and kana, emoji,
;;; indentation and tabs, punctuation before newlines, CRLF, special tokens --
;;; with ds4's token ids for each.  Every one must match, and decode back to
;;; its text.  Needs the GGUF ($ILLATIO_DS4F) for its vocabulary.

(load (merge-pathnames "load.lisp" (or *load-truename* *default-pathname-defaults*)))
(in-package #:illatio)

(let* ((path (or (uiop:getenv "ILLATIO_DS4F")
                 (namestring (merge-pathnames "workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
                                              (user-homedir-pathname)))))
       (v (make-vocab-from-gguf (read-gguf path)))
       (fails 0))
  (dolist (c (with-open-file (s (merge-pathnames "vectors/ds4f-tokens.sexp" (or *load-truename* *default-pathname-defaults*)))
               (loop for c = (read s nil) while c collect c)))
    (destructuring-bind (text ids) c
      (let ((ours (tokenize-with v text :add-bos nil)))
        (cond ((not (equal ours ids))
               (incf fails)
               (let ((k (mismatch ours ids)))
                 (format t "~&  FAIL ~S~%    at ~D: ours ~S~%           ds4  ~S~%" text k
                         (mapcar (lambda (i) (svref (vocab-pieces v) i)) (subseq ours k (min (length ours) (+ k 5))))
                         (mapcar (lambda (i) (svref (vocab-pieces v) i)) (subseq ids k (min (length ids) (+ k 5)))))))
              ((not (string= (detokenize-with v ids) text))
               (incf fails)
               (format t "~&  FAIL ~S decodes to ~S~%" text (detokenize-with v ids)))
              (t (format t "~&  ok   ~D tokens  ~S~%" (length ids) (subseq text 0 (min 40 (length text)))))))))
  (format t "~&~:[ok~;~:*~D failures~]~%" (and (plusp fails) fails))
  (uiop:quit (if (zerop fails) 0 1)))
