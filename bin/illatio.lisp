;;;; bin/illatio.lisp — the command line.
;;;
;;;   illatio [FLAGS] "text"     continue TEXT (no chat template), streamed
;;;   illatio [FLAGS] --chat     talk to it; one line is one user turn
;;;   illatio [FLAGS] --serve [[HOST]:PORT]
;;;                              OpenAI-compatible API (default 127.0.0.1:9931)
;;;
;;;   illatio-server [FLAGS] [--host HOST] [--port PORT]
;;;                              the server as a command of its own (llama-server's
;;;                              way): bin/illatio-server, the same core
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
                                  (default 127.0.0.1:9931; or --host, --port); as a
                                  command of its own: bin/illatio-server
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
  --steer FILE        directional steering (ds4's): one direction per layer, f32
                      rows of the model's width (a ds4 .f32 file loads as it is),
                      taken out of each layer's FFN output; --steer-build makes one
  --steer-ffn F       its scale there: 1 removes the direction, 0 leaves it, a
                      negative scale amplifies it (default 1)
  --steer-attn F      its scale on each attention output (default 0)
  --steer-build OUT --targets FILE --contrasts FILE
                      make a steering file: the mean last-token FFN output (or
                      attention's, with --steer-at attn) of the target prompts
                      less the contrast prompts', per layer (one prompt a line,
                      each a user turn in the model's chat template; --raw: as is)
  -h, --help          this
")

(defparameter +server-usage+
  "illatio-server — illatio's OpenAI-compatible HTTP server

usage:
  illatio-server [flags]

  POST /v1/chat/completions   (\"stream\": true for server-sent events)
  POST /v1/images/generations (with --images)
  GET  /v1/models
  GET  /health

flags:
  -m, --model PATH    GGUF file (default $ILLATIO_MODEL)
  --host HOST         address to listen on (default 127.0.0.1: there is no
                      authentication; listen wider only behind something that has it)
  --port PORT         port (default 9931)
  --path DIR          serve DIR's files too, at / (a web UI: e.g. llama.cpp's
                      tools/server/public_simplechat, or public for its full one)
  --reasoning-format F  deepseek: a reply's thinking in reasoning_content
                      (default); none: in its content, as <think>...</think>
                      (for a client that shows only content, as SimpleChat)
  -c, --ctx N         context size in tokens (default 8192)
  -n, --predict N     max_tokens for a request that gives none (default 2048)
  -t, --threads N     worker threads (default: the performance cores)
  --gpu               run the model on the GPU (DeepSeek V4/V4.1 Flash use CUDA
                      or Metal by themselves)
  --device N          the CUDA device to run on (implies --gpu; default 0)
  --mmproj PATH       a vision projector (llama.cpp's mmproj GGUF): pictures in
                      requests' image_url parts (DeepSeek V4.1 Flash, with --gpu)
  --mmproj-device N   the CUDA device it runs on (default the last)
  --images DIR        also make pictures: ex-nihilo (a sibling checkout) with the
                      Qwen-Image weights in DIR, at POST /v1/images/generations
                      and GET /images.  They wait in host memory; a picture
                      borrows the GPUs between chat requests, the chat model
                      moved to host memory meanwhile (DeepSeek's expert pools
                      given up, and made again after)
  --image-encoder FILE  the prompt encoder instead of DIR's te-q8_0.gguf
  --image-device N    the CUDA device the pictures are made on (default 0)
  --image-encoder-device N|cpu  where the prompt is encoded (default 1)
  --steer FILE        directional steering (ds4's): one direction per layer, f32
                      rows of the model's width (a ds4 .f32 file loads as it is),
                      taken out of each layer's FFN output; --steer-build makes one
  --steer-ffn F       its scale there: 1 removes the direction, 0 leaves it, a
                      negative scale amplifies it (default 1)
  --steer-attn F      its scale on each attention output (default 0)
                      (a request's \"steering\": {\"ffn\": F, \"attn\": F} sets them for it)
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
                      (num (s) (let ((*read-default-float-format* 'single-float))
                                 (let ((x (ignore-errors (read-from-string s))))
                                   (if (realp x) (coerce x 'single-float) (die "~A is not a number" s)))))
                      (int-list (s)
                        (mapcar (lambda (x) (or (parse-integer x :junk-allowed t) (die "~A is not a number" x)))
                                (uiop:split-string s :separator ","))))
                 (cond
                   ((member a '("-h" "--help") :test #'string=) (setf (getf opts :help) t))
                   ((member a '("-m" "--model") :test #'string=) (setf (getf opts :model) (val)))
                   ((member a '("-n" "--predict") :test #'string=)
                    (let ((v (val)))           ; --bench takes a list, as llama-bench does
                      (setf (getf opts :predict) (int v) (getf opts :predict-given) t
                            (getf opts :bench-n) (int-list v))))
                   ((string= a "--temp") (setf (getf opts :temp) (let ((*read-default-float-format* 'single-float))
                                                                    (coerce (read-from-string (val)) 'single-float))))
                   ((string= a "--seed") (setf (getf opts :seed) (int (val))))
                   ((member a '("-c" "--ctx") :test #'string=) (setf (getf opts :ctx) (int (val))))
                   ((member a '("-t" "--threads") :test #'string=) (setf (getf opts :threads) (int (val))))
                   ((member a '("-s" "--system") :test #'string=) (setf (getf opts :system) (val)))
                   ((string= a "--chat") (setf (getf opts :mode) :chat))
                   ((string= a "--gpu") (setf (getf opts :gpu) t))
                   ((string= a "--mmproj") (setf (getf opts :mmproj) (val)))
                   ((string= a "--mmproj-device") (setf (getf opts :mmproj-device) (int (val))))
                   ((string= a "--device") (setf (getf opts :gpu) t (getf opts :device) (int (val))))
                   ((string= a "--bench") (setf (getf opts :mode) :bench))
                   ((string= a "-p") (setf (getf opts :bench-p) (int-list (val))))
                   ((string= a "-d") (setf (getf opts :bench-d) (int-list (val))))
                   ((string= a "-r") (setf (getf opts :bench-r) (int (val))))
                   ((string= a "--host") (setf (getf opts :host) (val)))
                   ((string= a "--port") (setf (getf opts :port) (int (val))))
                   ((string= a "--path") (setf (getf opts :static) (val)))
                   ((string= a "--steer") (setf (getf opts :steer) (val)))
                   ((string= a "--steer-ffn") (setf (getf opts :steer-ffn) (num (val))))
                   ((string= a "--steer-attn") (setf (getf opts :steer-attn) (num (val))))
                   ((string= a "--steer-build") (setf (getf opts :mode) :steer-build (getf opts :steer-out) (val)))
                   ((string= a "--targets") (setf (getf opts :targets) (val)))
                   ((string= a "--contrasts") (setf (getf opts :contrasts) (val)))
                   ((string= a "--steer-at")
                    (let ((v (val)))
                      (unless (member v '("ffn" "attn") :test #'string=) (die "--steer-at is ffn or attn"))
                      (setf (getf opts :steer-at) (if (string= v "ffn") :ffn :attn))))
                   ((string= a "--raw") (setf (getf opts :raw) t))
                   ((string= a "--images") (setf (getf opts :images) (val)))
                   ((string= a "--image-encoder") (setf (getf opts :image-encoder) (val)))
                   ((string= a "--image-device") (setf (getf opts :image-device) (int (val))))
                   ((string= a "--image-encoder-device")
                    (let ((v (val))) (setf (getf opts :image-encoder-device) (if (string-equal v "cpu") :cpu (int v)))))
                   ((string= a "--reasoning-format")
                    (let ((v (val)))
                      (unless (member v '("none" "deepseek") :test #'string=) (die "--reasoning-format is none or deepseek"))
                      (setf (getf opts :reasoning-inline) (string= v "none"))))
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

(defun parse-addr (addr &optional host port)
  "[HOST]:PORT -> (values host port); HOST and PORT (--host, --port) over it."
  (if (null addr)
      (values (or host "127.0.0.1") (or port 9931))
      (let ((c (position #\: addr :from-end t)))
        (values (or host (if (and c (plusp c)) (subseq addr 0 c) "127.0.0.1"))
                (or port
                    (parse-integer addr :start (if c (1+ c) 0) :junk-allowed t)
                    (die "bad address ~A: want [HOST]:PORT" addr))))))

(defun load-it (opts)
  (let ((path (model-path opts))
        (t0 (get-internal-real-time)))
    (format *error-output* "~&illatio: loading ~A~%" (file-namestring path))
    ;; the vision encoder first: its device memory is taken before the model's pool sizes itself
    (when (getf opts :mmproj)
      #+(and sbcl linux)
      (if (getf opts :gpu)
          (illatio::load-vision-cuda (getf opts :mmproj)
                                     :device (or (getf opts :mmproj-device) (1- (illatio::cuda-device-count))))
          (die "--mmproj needs --gpu (the vision encoder runs on CUDA)"))
      #-(and sbcl linux) (die "--mmproj needs CUDA (Linux)"))
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
                                      (if (illatio::model-cuda m)
                                          (illatio::cudev-name (illatio::cmodel-dev (illatio::model-cuda m)))
                                          "CUDA, its experts in host memory"))
              #-(or (and sbcl darwin) (and sbcl linux)) nil
              (/ (- (get-internal-real-time) t0) internal-time-units-per-second))
      (when (getf opts :steer)
        (unless (probe-file (getf opts :steer)) (die "--steer: no file ~A" (getf opts :steer)))
        (setf illatio::*steering*
              (handler-case (illatio::load-steering (getf opts :steer) m :ffn (getf opts :steer-ffn 1f0)
                                                                         :attn (getf opts :steer-attn 0f0))
                (error (e) (die "--steer: ~A" e))))
        (format *error-output* "illatio: steering by ~A (~D layers; ffn ~,2F, attention ~,2F)~%"
                (file-namestring (getf opts :steer)) (illatio::steering-n-layer illatio::*steering*)
                (illatio::steering-ffn illatio::*steering*) (illatio::steering-attn illatio::*steering*)))
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

(defun load-images (opts)
  "ex-nihilo's image model, when --images asks for it: loaded (ASDF finds the
   sibling checkout; illatio does not depend on it) and moved to host
   memory before the chat model sizes its device memory.  Its tenant."
  (let ((dir (getf opts :images)))
    (when dir
      (unless (uiop:directory-exists-p dir) (die "--images: no directory ~A" dir))
      (when (and (getf opts :image-encoder) (not (probe-file (getf opts :image-encoder))))
        (die "--image-encoder: no file ~A" (getf opts :image-encoder)))
      (unless (asdf:find-system "ex-nihilo/server" nil)
        (die "--images needs ex-nihilo, a checkout beside illatio's"))
      (handler-bind ((warning #'muffle-warning))
        (asdf:load-system "ex-nihilo/server"))
      (let ((ed (getf opts :image-encoder-device 1)))
        (uiop:symbol-call '#:ex-nihilo '#:make-tenant
                          :weights dir :encoder (getf opts :image-encoder)
                          :device (getf opts :image-device 0) :encoder-device (if (eq ed :cpu) nil ed))))))

(defun run-serve (opts)
  (multiple-value-bind (host port) (parse-addr (getf opts :addr) (getf opts :host) (getf opts :port))
    (let* ((images (load-images opts))
           ;; with pictures the chat model's device memory is recorded, to be lent to them
           (m (if images (illatio::load-lendable (lambda () (load-it opts))) (load-it opts)))
           (engine (apply #'make-engine m :n-ctx (getf opts :ctx)
                          (and (getf opts :predict-given) (list :max-tokens (getf opts :predict))))))
      (when (and (getf opts :static) (not (uiop:directory-exists-p (getf opts :static))))
        (die "--path: no directory ~A" (getf opts :static)))
      (setf illatio::*reasoning-inline* (getf opts :reasoning-inline))
      (when images (uiop:symbol-call '#:ex-nihilo '#:attach-tenant images engine))
      (serve engine :host host :port port :static (getf opts :static))
      (format *error-output* "illatio: serving http://~A:~D/v1 (context ~D)~%" host port (getf opts :ctx))
      (when (getf opts :static)
        (format *error-output* "illatio: and ~A at http://~A:~D/~%" (getf opts :static) host port))
      (when images
        (format *error-output* "illatio: and pictures at http://~A:~D/v1/images/generations (try http://~A:~D/images)~%"
                host port host port))
      (loop (sleep 3600)))))

(defun prompt-lines (path what)
  (unless (and path (probe-file path)) (die "--steer-build needs ~A FILE" what))
  (or (remove-if (lambda (l) (zerop (length (string-trim " " l)))) (uiop:read-file-lines path))
      (die "~A: no prompts" path)))

(defun run-steer-build (opts)
  (let* ((targets (prompt-lines (getf opts :targets) "--targets"))
         (contrasts (prompt-lines (getf opts :contrasts) "--contrasts"))
         (m (load-it (list* :steer nil opts)))
         (wrap (lambda (p)
                 (if (getf opts :raw) p
                     (illatio::render-chat m (vector (hobj "role" "user" "content" p)) nil :thinking nil)))))
    (illatio::build-direction m (mapcar wrap targets) (mapcar wrap contrasts)
                              :kind (getf opts :steer-at :ffn) :out (getf opts :steer-out))
    (format *error-output* "illatio: wrote ~A (and its .json)~%" (getf opts :steer-out))))

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
  ;; bin/illatio-server: the server, its flags, its usage
  (when (equal (first args) "--server-command")
    (multiple-value-bind (opts text) (parse-args (rest args))
      (cond ((getf opts :help) (write-string +server-usage+))
            ((plusp (length text)) (die "illatio-server takes no text (~A); see --help" text))
            (t (run-serve opts))))
    (return-from main))
  (multiple-value-bind (opts text) (parse-args args)
    (cond ((getf opts :help) (write-string +usage+))
          ((eq (getf opts :mode) :serve) (run-serve opts))
          ((eq (getf opts :mode) :bench) (run-bench opts))
          ((eq (getf opts :mode) :steer-build) (run-steer-build opts))
          ((eq (getf opts :mode) :chat) (run-chat opts (and (plusp (length text)) text)))
          ((plusp (length text)) (run-complete opts text))
          (t (write-string +usage+) (uiop:quit 2)))))
