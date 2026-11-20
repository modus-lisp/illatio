;;;; render-test.lisp — our chat template renders what llama.cpp renders.
;;;
;;;   sbcl --dynamic-space-size 4096 --non-interactive --load inspect/render-test.lisp
;;;
;;; vectors/render/NAME.request.json is a chat-completions request;
;;; NAME.expected is llama-server's /apply-template output for it (Qwen2.5's
;;; template; vectors/render-qwen3/ the same for Qwen3's, render-qwen35/ for
;;; Qwen3.5's, whose tool calls are XML).  One
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
           (got (render-chat m (jget req "messages") (jget req "tools") :thinking (model-thinking m req))))
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

;; Qwen3.5's template, when its model is here (~/models, or $ILLATIO_TEST_QWEN35)
(let ((q35 (or (uiop:getenv "ILLATIO_TEST_QWEN35")
               (namestring (merge-pathnames "models/qwen3.5-4b-q4_k_m.gguf" (user-homedir-pathname))))))
  (if (probe-file q35)
      (check-renderings (%make-model :gguf (read-gguf q35) :name "qwen35" :arch "qwen35") "render-qwen35"
                        '("plain" "tools" "nothink"))
      (format t "~&  skip Qwen3.5 renderings: no model at ~A~%" q35)))

;; Qwen3.5's XML calls, typed by the request's schemas
(let ((tools (json-simple:parse "[{\"type\":\"function\",\"function\":{\"name\":\"Read\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\"}}}}},{\"type\":\"function\",\"function\":{\"name\":\"Edit\",\"parameters\":{\"type\":\"object\",\"properties\":{\"old\":{\"type\":\"string\"},\"all\":{\"type\":\"boolean\"}}}}}]")))
  (flet ((reply (text) (multiple-value-list (qwen35-parse-reply text nil tools))))
    (destructuring-bind (content calls reasoning)
        (reply (format nil "Reading it.~%~%<tool_call>~%<function=Read>~%<parameter=path>~%a.txt~%</parameter>~%<parameter=limit>~%20~%</parameter>~%</function>~%</tool_call>"))
      (check (and (equal content "Reading it.") (null reasoning) (= 1 (length calls))
                  (equal (jget (first calls) "function" "arguments") "{\"path\":\"a.txt\",\"limit\":20}"))
             "an XML call is parsed, its integer typed by the schema"))
    (destructuring-bind (content calls reasoning)
        (reply (format nil "<tool_call>~%<function=Edit>~%<parameter=old>~%(defun f ()~%  1)~%</parameter>~%<parameter=all>~%true~%</parameter>~%</function>~%</tool_call>~%<tool_call>~%<function=Read>~%<parameter=path>~%b~%</function>"))
      (declare (ignore reasoning))
      (check (and (equal content "") (= 2 (length calls))
                  (equal (jget (first calls) "function" "arguments") "{\"old\":\"(defun f ()\\n  1)\",\"all\":true}")
                  (equal (jget (second calls) "function" "arguments") "{\"path\":\"b\"}"))
             "two calls: a multi-line string kept, a boolean typed, an unclosed call still read"))
    (destructuring-bind (content calls reasoning)
        (reply (format nil "<tool_call>~%{\"name\": \"Read\", \"arguments\": {\"path\": \"z\"}}~%</tool_call>"))
      (declare (ignore content reasoning))
      (check (and (= 1 (length calls)) (equal (jget (first calls) "function" "arguments") "{\"path\":\"z\"}"))
             "a Hermes JSON call inside Qwen3.5's tags is read too"))
    (destructuring-bind (content calls reasoning)
        (multiple-value-list (qwen35-parse-reply (format nil "weighing it~%</think>~%~%Just text.") t tools))
      (check (and (equal content "Just text.") (null calls) (equal reasoning "weighing it"))
             "reasoning before </think> is split out"))))

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
