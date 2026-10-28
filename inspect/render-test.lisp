;;;; render-test.lisp — our chat template renders what llama.cpp renders.
;;;
;;;   sbcl --dynamic-space-size 4096 --non-interactive --load inspect/render-test.lisp
;;;
;;; vectors/render/NAME.request.json is a chat-completions request;
;;; NAME.expected is llama-server's /apply-template output for it (Qwen2.5's
;;; template; vectors/render-qwen3/ the same for Qwen3's).  One
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

(defun check-renderings (m dir names)
  (dolist (name names)
    (let* ((req (json-simple:parse (slurp (merge-pathnames (format nil "vectors/~A/~A.request.json" dir name) *here*))))
           (want (slurp (merge-pathnames (format nil "vectors/~A/~A.expected" dir name) *here*)))
           (got (render-chat m (jget req "messages") (jget req "tools"))))
      (check (string= got want)
             (format nil "~A/~A: rendering = llama.cpp's~@[ (first difference at ~D)~]" dir name (mismatch got want))))))

(check-renderings *m* "render" '("plain" "tools"))

;; Qwen3's template, when its model is here (~/models, or $ILLATIO_TEST_QWEN3)
(let ((q3 (or (uiop:getenv "ILLATIO_TEST_QWEN3")
              (namestring (merge-pathnames "models/qwen3-4b-instruct-2507-q8_0.gguf" (user-homedir-pathname))))))
  (if (probe-file q3)
      (check-renderings (%make-model :gguf (read-gguf q3) :name "qwen3") "render-qwen3"
                        '("plain" "system" "tools-nosys" "tools-sys" "multistep"
                          "think" "think-mid" "after-tool-final"))
      (format t "~&  skip Qwen3 renderings: no model at ~A~%" q3)))

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
