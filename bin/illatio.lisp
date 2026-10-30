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
  illatio [flags] --bench         llama-bench's tests, in its table: -p prompt
                                  tokens, -n generated, -d context depth (each a
                                  comma list), -r repetitions; e.g.
                                    illatio --bench --gpu -m M -p 512 -n 128 -d 0,4096
                                    llama-bench -ngl 99 -m M -p 512 -n 128 -d 0,4096

flags:
  -m, --model PATH    GGUF file (default $ILLATIO_MODEL, then
                      ~/models/qwen2.5-0.5b-instruct-q8_0.gguf)
  -n, --predict N     at most N tokens per reply (default 512)
  --temp T            temperature; 0 is greedy (default 0.7; raw text: 0)
  --seed S            sampling seed
  -c, --ctx N         context size in tokens (default 8192)
  -t, --threads N     worker threads (default: the performance cores)
  --gpu               run the model on the GPU: a qwen35 model on Metal (DeepSeek
                      V4 Flash always uses it on a Mac); qwen2, qwen3 and llama
                      on CUDA
  --device N          the CUDA device to run on (implies --gpu; default 0)
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
                      (int (s) (or (parse-integer s :junk-allowed t) (die "~A is not a number" s)))
                      (int-list (s)
                        (mapcar (lambda (x) (or (parse-integer x :junk-allowed t) (die "~A is not a number" x)))
                                (uiop:split-string s :separator ","))))
                 (cond
                   ((member a '("-h" "--help") :test #'string=) (setf (getf opts :help) t))
                   ((member a '("-m" "--model") :test #'string=) (setf (getf opts :model) (val)))
                   ((member a '("-n" "--predict") :test #'string=)
                    (let ((v (val)))           ; --bench takes a list, as llama-bench does
                      (setf (getf opts :predict) (int v) (getf opts :bench-n) (int-list v))))
                   ((string= a "--temp") (setf (getf opts :temp) (let ((*read-default-float-format* 'single-float))
                                                                    (coerce (read-from-string (val)) 'single-float))))
                   ((string= a "--seed") (setf (getf opts :seed) (int (val))))
                   ((member a '("-c" "--ctx") :test #'string=) (setf (getf opts :ctx) (int (val))))
                   ((member a '("-t" "--threads") :test #'string=) (setf (getf opts :threads) (int (val))))
                   ((member a '("-s" "--system") :test #'string=) (setf (getf opts :system) (val)))
                   ((string= a "--chat") (setf (getf opts :mode) :chat))
                   ((string= a "--gpu") (setf (getf opts :gpu) t))
                   ((string= a "--device") (setf (getf opts :gpu) t (getf opts :device) (int (val))))
                   ((string= a "--bench") (setf (getf opts :mode) :bench))
                   ((string= a "-p") (setf (getf opts :bench-p) (int-list (val))))
                   ((string= a "-d") (setf (getf opts :bench-d) (int-list (val))))
                   ((string= a "-r") (setf (getf opts :bench-r) (int (val))))
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
    (let ((m (if (getf opts :gpu)
                 #+(and sbcl darwin) (illatio::load-model-gpu path)
                 #+(and sbcl linux) (illatio::load-model-cuda path :device (or (getf opts :device) 0))
                 #-(or (and sbcl darwin) (and sbcl linux)) (die "--gpu needs Metal (macOS) or CUDA (Linux)")
                 (load-model path :threads (or (getf opts :threads) (default-thread-count))))))
      (format *error-output* "illatio: ~A, ~D layers, ~:[~D threads~*~;~*~A~], ~,1Fs~%"
              (model-name m) (illatio::model-n-layer m)
              (getf opts :gpu) (illatio::pool-size (illatio::model-pool m))
              #+(and sbcl darwin) (and (getf opts :gpu)
                                       (illatio::gpu-name (illatio::gmodel-gpu (illatio::model-gpu m))))
              #+(and sbcl linux) (and (getf opts :gpu)
                                      (illatio::cudev-name (illatio::cmodel-dev (illatio::model-cuda m))))
              #-(or (and sbcl darwin) (and sbcl linux)) nil
              (/ (- (get-internal-real-time) t0) internal-time-units-per-second))
      m)))

(defun out (text) (write-string text) (force-output))

(defun run-complete (opts text)
  (let* ((m (load-it opts))
         (ctx (make-context m :n-ctx (getf opts :ctx))))
    (out text)
    (let ((t0 (get-internal-real-time)))
      (multiple-value-bind (s ids)
          (generate ctx text :max-tokens (getf opts :predict) :on-text #'out
                             :sampler (make-sampler :temperature (or (getf opts :temp) 0) :seed (getf opts :seed)))
        (declare (ignore s))
        (terpri)
        ;; llama-simple's closing line, timed as it times it (prompt included)
        (let ((secs (/ (- (get-internal-real-time) t0) internal-time-units-per-second)))
          (format *error-output* "illatio: decoded ~D tokens in ~,2F s, speed: ~,2F t/s~%"
                  (length ids) secs (/ (length ids) (max secs 1d-6))))))))

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

(defun run-bench (opts)
  "llama-bench's pp and tg tests, timed the way it times them: random
   tokens; pp a prompt of P in one go (prefill), tg N tokens decoded one at a
   time; each after a context already D deep (not timed); the mean and
   standard deviation of R runs, after a warm-up."
  (let* ((m (load-it opts))
         (path (model-path opts))
         (ps (getf opts :bench-p '(512))) (ns (getf opts :bench-n '(128)))
         (ds (getf opts :bench-d '(0))) (reps (getf opts :bench-r 3))
         (state (sb-ext:seed-random-state 42))
         (vocab (illatio::model-n-vocab m)))
    (labels ((toks (k) (loop repeat k collect (random (min vocab 150000) state)))
             (secs (t0) (/ (- (get-internal-real-time) t0) internal-time-units-per-second))
             (one (kind k d)
               (let ((ctx (make-context m :n-ctx (+ d k 64))))
                 (unwind-protect
                      (progn
                        (when (plusp d) (eval-tokens ctx (toks d)))
                        (ecase kind
                          (:pp (let ((ids (toks k)) (t0 (get-internal-real-time)))
                                 (eval-tokens ctx ids)
                                 (/ k (secs t0))))
                          (:tg (when (zerop d) (eval-tokens ctx (toks 1)))
                               (let ((ids (toks k)) (t0 (get-internal-real-time)))
                                 (dolist (tok ids) (forward ctx tok))
                                 (/ k (secs t0))))))
                   (free-context ctx)))))
      (one :pp 32 0) (one :tg 4 0)                    ; warm-up: kernels, pages
      (format t "~&| model | size | backend | threads | test | t/s |~%|---|---|---|---|---|---|~%")
      (dolist (d ds)
        (dolist (test (append (loop for p in ps when (plusp p) collect (list :pp p))
                              (loop for n in ns when (plusp n) collect (list :tg n))))
          (destructuring-bind (kind k) test
            (let* ((xs (loop repeat reps collect (one kind k d)))
                   (mean (/ (reduce #'+ xs) reps))
                   (sd (sqrt (/ (reduce #'+ (mapcar (lambda (x) (expt (- x mean) 2)) xs)) (max 1 (1- reps))))))
              (format t "| ~A | ~,2F GiB | ~A | ~A | ~(~A~)~D~@[ @ d~D~] | ~,2F ± ~,2F |~%"
                      (illatio::model-name m)
                      (/ (with-open-file (s path :element-type '(unsigned-byte 8)) (file-length s)) (expt 2d0 30))
                      (if (getf opts :gpu) #+darwin "Metal" #-darwin "CUDA" "CPU")
                      (if (getf opts :gpu) "-" (illatio::pool-size (illatio::model-pool m)))
                      kind k (and (plusp d) d) mean sd)
              (force-output))))))))

(defun main (args)
  (multiple-value-bind (opts text) (parse-args args)
    (cond ((getf opts :help) (write-string +usage+))
          ((eq (getf opts :mode) :serve) (run-serve opts))
          ((eq (getf opts :mode) :bench) (run-bench opts))
          ((eq (getf opts :mode) :chat) (run-chat opts (and (plusp (length text)) text)))
          ((plusp (length text)) (run-complete opts text))
          (t (write-string +usage+) (uiop:quit 2)))))
