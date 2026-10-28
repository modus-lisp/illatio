;;;; bin/illatio.lisp — the command line.
;;;
;;;   illatio [FLAGS] "text"     continue TEXT (no chat template), streamed
;;;   illatio [FLAGS] --chat     talk to it; one line is one user turn
;;;   illatio [FLAGS] --serve [[HOST]:PORT]
;;;                              OpenAI-compatible API (default 127.0.0.1:9931)
;;;
;;; Modes are flags; a bare word is always text.  bin/illatio is the
;;; launcher (it keeps a core image so startup is instant).

(defpackage #:illatio.cli
  (:use #:cl #:illatio)
  (:import-from #:illatio #:make-engine #:serve #:chat-completion #:hobj #:jget
                #:default-thread-count #:make-sampler)
  (:export #:main))

(in-package #:illatio.cli)

(defparameter +usage+
  "illatio — language-model inference in pure Common Lisp

usage:
  illatio [flags] \"text\"          continue the text, streamed (raw, no template)
  illatio [flags] --chat [\"text\"] chat; each line you type is a user turn
  illatio [flags] --serve [ADDR]  OpenAI-compatible server; ADDR is [HOST]:PORT
                                  (default 127.0.0.1:9931)

flags:
  -m, --model PATH    GGUF file (default $ILLATIO_MODEL, then
                      ~/models/qwen2.5-0.5b-instruct-q8_0.gguf)
  -n, --predict N     at most N tokens per reply (default 512)
  --temp T            temperature; 0 is greedy (default 0.7; raw text: 0)
  --seed S            sampling seed
  -c, --ctx N         context size in tokens (default 8192)
  -t, --threads N     worker threads (default: the performance cores)
  -s, --system TEXT   system prompt for --chat
  -h, --help          this
")

(defun die (fmt &rest args)
  (format *error-output* "illatio: ~?~%" fmt args)
  (uiop:quit 2))

(defun parse-args (args)
  (let ((opts (list :predict 512 :ctx 8192)) (text '()))
    (loop while args
          do (let ((a (pop args)))
               (flet ((val () (or (pop args) (die "~A needs a value" a)))
                      (int (s) (or (parse-integer s :junk-allowed t) (die "~A is not a number" s))))
                 (cond
                   ((member a '("-h" "--help") :test #'string=) (setf (getf opts :help) t))
                   ((member a '("-m" "--model") :test #'string=) (setf (getf opts :model) (val)))
                   ((member a '("-n" "--predict") :test #'string=) (setf (getf opts :predict) (int (val))))
                   ((string= a "--temp") (setf (getf opts :temp) (let ((*read-default-float-format* 'single-float))
                                                                    (coerce (read-from-string (val)) 'single-float))))
                   ((string= a "--seed") (setf (getf opts :seed) (int (val))))
                   ((member a '("-c" "--ctx") :test #'string=) (setf (getf opts :ctx) (int (val))))
                   ((member a '("-t" "--threads") :test #'string=) (setf (getf opts :threads) (int (val))))
                   ((member a '("-s" "--system") :test #'string=) (setf (getf opts :system) (val)))
                   ((string= a "--chat") (setf (getf opts :mode) :chat))
                   ((string= a "--serve")
                    (setf (getf opts :mode) :serve)
                    (when (and args (find #\: (first args)) (not (eql 0 (search "-" (first args)))))
                      (setf (getf opts :addr) (pop args))))
                   ((and (> (length a) 1) (char= (char a 0) #\-) (not (string= a "--")))
                    (die "unknown flag ~A (see --help)" a))
                   ((string= a "--") (setf text (append text args) args '()))
                   (t (setf text (append text (list a))))))))
    (values opts (format nil "~{~A~^ ~}" text))))

(defun model-path (opts)
  (let ((p (or (getf opts :model) (uiop:getenv "ILLATIO_MODEL")
               (namestring (merge-pathnames "models/qwen2.5-0.5b-instruct-q8_0.gguf" (user-homedir-pathname))))))
    (unless (probe-file p)
      (die "no model at ~A~%  pass -m PATH, or set ILLATIO_MODEL; the README says where to get one" p))
    p))

(defun parse-addr (addr)
  "[HOST]:PORT -> (values host port)."
  (if (null addr)
      (values "127.0.0.1" 9931)
      (let ((c (position #\: addr :from-end t)))
        (values (if (and c (plusp c)) (subseq addr 0 c) "127.0.0.1")
                (or (parse-integer addr :start (if c (1+ c) 0) :junk-allowed t)
                    (die "bad address ~A: want [HOST]:PORT" addr))))))

(defun load-it (opts)
  (let ((path (model-path opts))
        (t0 (get-internal-real-time)))
    (format *error-output* "~&illatio: loading ~A~%" (file-namestring path))
    (let ((m (load-model path :threads (or (getf opts :threads) (default-thread-count)))))
      (format *error-output* "illatio: ~A, ~D layers, ~D threads, ~,1Fs~%"
              (model-name m) (illatio::model-n-layer m) (illatio::pool-size (illatio::model-pool m))
              (/ (- (get-internal-real-time) t0) internal-time-units-per-second))
      m)))

(defun out (text) (write-string text) (force-output))

(defun run-complete (opts text)
  (let* ((m (load-it opts))
         (ctx (make-context m :n-ctx (getf opts :ctx))))
    (out text)
    (generate ctx text :max-tokens (getf opts :predict) :on-text #'out
                       :sampler (make-sampler :temperature (or (getf opts :temp) 0) :seed (getf opts :seed)))
    (terpri)))

(defun run-chat (opts first)
  (let* ((m (load-it opts))
         (engine (make-engine m :n-ctx (getf opts :ctx)))
         (messages (if (getf opts :system)
                       (list (hobj "role" "system" "content" (getf opts :system)))
                       '())))
    (loop
      (let ((line (or (shiftf first nil)
                      (progn (format t "~&> ") (force-output) (read-line *standard-input* nil)))))
        (unless line (terpri) (return))
        (when (plusp (length (string-trim " " line)))
          (setf messages (append messages (list (hobj "role" "user" "content" line))))
          (let* ((resp (chat-completion engine (hobj "messages" (coerce messages 'vector)
                                                     "max_tokens" (getf opts :predict)
                                                     "temperature" (or (getf opts :temp) 0.7)
                                                     "seed" (or (getf opts :seed) 'null))
                                        :on-text #'out))
                 (msg (jget (aref (jget resp "choices") 0) "message"))
                 (tm (jget resp "timings")))
            (format *error-output* "~&  [~D tok, ~,1F tok/s; prompt ~D new + ~D cached]~%"
                    (jget tm "predicted_n")
                    (/ (jget tm "predicted_n") (max 1d-3 (/ (jget tm "predicted_ms") 1000)))
                    (jget tm "prompt_n") (jget tm "cache_n"))
            (setf messages (append messages (list msg)))))))))

(defun run-serve (opts)
  (multiple-value-bind (host port) (parse-addr (getf opts :addr))
    (let* ((m (load-it opts))
           (engine (make-engine m :n-ctx (getf opts :ctx))))
      (serve engine :host host :port port)
      (format *error-output* "illatio: serving http://~A:~D/v1 (context ~D)~%" host port (getf opts :ctx))
      (loop (sleep 3600)))))

(defun main (args)
  (multiple-value-bind (opts text) (parse-args args)
    (cond ((getf opts :help) (write-string +usage+))
          ((eq (getf opts :mode) :serve) (run-serve opts))
          ((eq (getf opts :mode) :chat) (run-chat opts (and (plusp (length text)) text)))
          ((plusp (length text)) (run-complete opts text))
          (t (write-string +usage+) (uiop:quit 2)))))
