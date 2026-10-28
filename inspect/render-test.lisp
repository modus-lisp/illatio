;;;; render-test.lisp — our chat template renders what llama.cpp renders.
;;;
;;;   sbcl --dynamic-space-size 4096 --non-interactive --load inspect/render-test.lisp
;;;
;;; vectors/render/NAME.request.json is a chat-completions request;
;;; NAME.expected is llama-server's /apply-template output for it.  One
;;; space is one token, so the bar is byte equality.  Also checks that tool
;;; calls come back out of a reply, including the doubled-brace form small
;;; models copy from Qwen's own template text.

(load (merge-pathnames "load.lisp" (or *load-truename* *default-pathname-defaults*)))
(in-package #:illatio)

(defvar *here* (make-pathname :name nil :type nil :defaults *load-truename*))
(defvar *failures* 0)
(defun check (ok what)
  (format t "~&  ~:[FAIL~;ok  ~] ~A~%" ok what)
  (unless ok (incf *failures*)))

(defun slurp (path)
  (with-open-file (s path :external-format :utf-8)
    (let ((str (make-string (file-length s))))
      (subseq str 0 (read-sequence str s)))))

;; Only the template text is needed, not the weights.
(defvar *m* (%make-model :gguf (read-gguf (or (uiop:getenv "ILLATIO_TEST_MODEL")
                                              (merge-pathnames "models/qwen2.5-0.5b-instruct-q8_0.gguf"
                                                               (user-homedir-pathname))))
                         :name "test"))

(dolist (name '("plain" "tools"))
  (let* ((req (json-simple:parse (slurp (merge-pathnames (format nil "vectors/render/~A.request.json" name) *here*))))
         (want (slurp (merge-pathnames (format nil "vectors/render/~A.expected" name) *here*)))
         (got (render-chat *m* (jget req "messages") (jget req "tools"))))
    (check (string= got want)
           (format nil "~A: rendering = llama.cpp's~@[ (first difference at ~D)~]" name (mismatch got want)))))

(flet ((calls (text) (multiple-value-list (parse-tool-calls text))))
  (destructuring-bind (content calls)
      (calls (format nil "Sure.~%<tool_call>~%{\"name\": \"f\", \"arguments\": {\"a\": 1}}~%</tool_call>"))
    (check (and (equal content "Sure.") (= 1 (length calls))
                (equal (jget (first calls) "function" "name") "f")
                (equal (jget (first calls) "function" "arguments") "{\"a\":1}"))
           "a tool call is parsed out of the reply"))
  (destructuring-bind (content calls)
      (calls (format nil "<tool_call>~%{{\"name\": \"f\", \"arguments\": {}}}~%</tool_call>"))
    (check (and (equal content "") (= 1 (length calls))) "the doubled-brace form parses"))
  (destructuring-bind (content calls) (calls "<tool_call>not json</tool_call> after")
    (check (and (null calls) (search "not json" content))
           "a block that isn't JSON stays in the content")))

(format t "~&~:[ok~;~:*~D failures~]~%" (and (plusp *failures*) *failures*))
(uiop:quit (if (zerop *failures*) 0 1))
