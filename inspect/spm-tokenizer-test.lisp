;;;; spm-tokenizer-test.lisp — llama's SentencePiece BPE (tokenizer.ggml.model
;;;; = llama) tokenizes as llama.cpp does.
;;;
;;;   sbcl --non-interactive --load inspect/spm-tokenizer-test.lisp
;;;
;;; vectors/spm-nanbeige.sexp is llama-tokenize's output (--ids --no-bos) for
;;; each text, on Nanbeige4.1-3B: specials, the space put in front of a
;;; fragment, whitespace runs, and characters that fall back to <0xXX> bytes.
;;; Needs the model's vocabulary: ~/models/nanbeige4.1-3b-q4_k_m.gguf or
;;; $ILLATIO_TEST_SPM.

(load (merge-pathnames "load.lisp" (or *load-truename* *default-pathname-defaults*)))
(in-package #:illatio)

(let ((path (or (uiop:getenv "ILLATIO_TEST_SPM")
                (namestring (merge-pathnames "models/nanbeige4.1-3b-q4_k_m.gguf" (user-homedir-pathname)))))
      (failures 0))
  (unless (probe-file path)
    (format t "~&skip: no model at ~A~%" path)
    (uiop:quit 2))
  (let ((v (make-vocab-from-gguf (read-gguf path))))
    (dolist (case (with-open-file (s (merge-pathnames "vectors/spm-nanbeige.sexp" *load-truename*)
                                     :external-format :utf-8)
                    (read s)))
      (destructuring-bind (text want) case
        (let ((got (tokenize-with v text :add-bos nil)))
          (format t "~&  ~:[FAIL~;ok  ~] ~D tokens: ~S~%" (equal got want) (length want)
                  (subseq text 0 (min 40 (length text))))
          (unless (equal got want)
            (incf failures)
            (let ((k (mismatch got want)))
              (format t "       first difference at ~D: ~S, llama.cpp ~S~%" k
                      (subseq got k (min (length got) (+ k 5))) (subseq want k (min (length want) (+ k 5))))))
          ;; and back: the space SentencePiece put in front of each fragment aside
          (let ((back (detokenize-with v got)))
            (unless (search (string-left-trim " " (remove-if (lambda (c) (char= c #\Space)) text))
                            (remove-if (lambda (c) (char= c #\Space)) back))
              (incf failures)
              (format t "       FAIL round trip: ~S~%" back)))))))
  (format t "~&~:[ok~;~:*~D failures~]~%" (and (plusp failures) failures))
  (uiop:quit (if (zerop failures) 0 1)))
