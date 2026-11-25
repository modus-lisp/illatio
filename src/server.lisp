;;;; server.lisp — the OpenAI-compatible HTTP face of CHAT-COMPLETION.
;;;
;;; Enough HTTP/1.1 for API clients: one request per connection, a
;;; Content-Length body, JSON or server-sent events back (the HTTP is
;;; cl-transport/http-server's).  Everything model-shaped happens in
;;; CHAT-COMPLETION; this file only routes and moves bytes.
;;;
;;;   GET  /health                -> {"status":"ok"}
;;;   GET  /v1/models             -> the one model, with its context_length
;;;   POST /v1/chat/completions   -> blocking, or "stream": true for SSE
;;;        (/chat/completions too, as llama-server takes it)
;;;   GET  anything else          -> a file under STATIC (SERVE's), if given:
;;;        a web UI's, e.g. llama.cpp's (tools/server/public_simplechat)
;;;
;;; Another program may add routes (*ROUTES*) and models (*MORE-MODELS*),
;;; and borrow the GPUs between requests (LEND-DEVICES): ex-nihilo's
;;; pictures, loaded by bin/illatio-server --images.

(in-package #:illatio)

;;; The HTTP itself -- request parsing, responses, SSE framing, static files, the listener
;;; -- is cl-transport/http-server, which grew out of this file.  What stays here is
;;; illatio's: its routes and the stream-level helpers (RESPOND, WRITE-UTF8, SEND-EVENT)
;;; that another program's route functions (*ROUTES*) are written against.

(defun write-utf8 (s string)
  (write-sequence (sb-ext:string-to-octets string :external-format :utf-8) s))

(defun respond (s status object &key (type "application/json"))
  "A whole response on stream S: OBJECT a string, or JSON (json-simple) for anything else."
  (cl-transport.http-server:write-response
   s status (if (stringp object) object (json-simple:stringify object)) :content-type type))

(defun static-file (root path)
  "(values file gzipped-p content-type) for request PATH under directory ROOT, or NIL:
   / is index.html; FILE.gz stands in for a missing FILE (sent gzip-encoded, as
   llama.cpp's public/index.html.gz is meant to be); nothing outside the directory."
  (cl-transport.http-server:static-file root path))

;;; --- other programs' routes, and the GPUs lent to them -----------------------------

(defvar *routes* '()
  "(METHOD PATHS FUNCTION) entries: a request for one of PATHS by METHOD is
   FUNCTION's, called with the stream and the body (a string); it writes the whole
   response to the stream (RESPOND, or WRITE-UTF8 and SEND-EVENT).")

(defvar *more-models* '() "More entries for GET /v1/models (objects).")

(defun add-route (method paths function)
  (setf *routes* (cons (list method paths function)
                       (remove-if (lambda (r) (and (string= (first r) method) (intersection (second r) paths :test #'string=)))
                                  *routes*))))

(defun model-lend (m)
  "Move model M's device memory aside where it is a cache: (values thunk
   bytes), THUNK putting it back.  DeepSeek's and GLM's experts' pools on
   CUDA, DeepSeek V4's on Metal; other models keep theirs (0 bytes)."
  #+(and sbcl linux)
  (let ((impl (model-impl m)))
    (when (and impl (member (model-arch m) '("deepseek4" "deepseek41" "glm5-next") :test #'equal))
      (let ((st (getf (ds4f-gpu-plist impl) :cuda)))
        (when st (return-from model-lend (d4-lend impl st))))))
  #+(and sbcl darwin)
  (let ((impl (model-impl m)))
    (when (and impl (equal (model-arch m) "deepseek4"))
      (return-from model-lend (ds4f-gpu-lend impl))))
  (values (lambda ()) 0))

(defun park-memory (parking)
  "PARKING's device memory given up (CUDA's moved to host memory; Metal's
   released from its working set): the bytes."
  #+(and sbcl linux) (cuda-park parking)
  #+(and sbcl darwin) (gpu-park parking))

(defun unpark-memory (parking)
  "PARKING's device memory back, as it was."
  #+(and sbcl linux) (cuda-unpark parking)
  #+(and sbcl darwin) (gpu-unpark parking))

(defvar *chat-parking* nil
  "The chat model's device memory, recorded (LOAD-LENDABLE) so that it can
   be moved to host memory while the GPUs are lent; or NIL.")

(defun load-lendable (thunk)
  "THUNK's result (a model loaded), its device memory recorded in
   *CHAT-PARKING* (as weights), and all device memory allocated outside
   another record from now on (contexts' caches, scratch) as well."
  (setf *chat-parking* (make-parking "chat"))
  (prog1 (with-parking-record (*chat-parking* :const t) (funcall thunk))
    #+(and sbcl linux) (setf *cuda-alloc-hook* (parking-recorder *chat-parking* nil))
    #+(and sbcl darwin) (setf *gpu-alloc-hook* (parking-recorder *chat-parking* nil))))

(defun lend-devices (engine function)
  "FUNCTION called with ENGINE's GPUs as free as can be: between its
   requests (its lock held), its model's caches given up (MODEL-LEND) and,
   when it was loaded by LOAD-LENDABLE, the rest of its device memory moved
   to host memory; all of it back after."
  (sb-thread:with-mutex ((engine-lock engine))
    (let ((t0 (get-internal-real-time)))
      (multiple-value-bind (restore lent) (model-lend (engine-model engine))
        (let ((t1 nil))
          (unwind-protect
               (let ((parked (if *chat-parking* (park-memory *chat-parking*) 0)))
                 (format *error-output* "illatio: the GPU lent: ~,1F GB of caches given up, ~,1F GB parked, ~,1Fs~%"
                         (/ lent 1d9) (/ parked 1d9) (/ (- (get-internal-real-time) t0) internal-time-units-per-second))
                 (unwind-protect (funcall function)
                   (setf t1 (get-internal-real-time))
                   (when *chat-parking* (unpark-memory *chat-parking*))))
            ;; the caches made again whatever happened (a failed park included)
            (funcall restore)
            (when t1
              (format *error-output* "illatio: the GPU back, ~,1Fs~%"
                      (/ (- (get-internal-real-time) t1) internal-time-units-per-second)))))))))

(defun error-object (message &optional (type "invalid_request_error"))
  (hobj "error" (hobj "message" message "type" type)))

(defun engine-sees-p (engine)
  "Whether ENGINE's model takes pictures: a vision encoder loaded, for a model it serves."
  (and *vision* (equal (model-arch (engine-model engine)) "deepseek41") t))

(defun models-object (engine)
  (let ((m (engine-model engine)))
    (hobj "object" "list"
          "data" (apply #'vector (hobj "id" (model-name m)
                               "object" "model"
                               "owned_by" "illatio"
                               ;; what a client may send in one conversation:
                               ;; this context's size, not the model's training length
                               "context_length" (context-n-ctx (engine-context engine))
                               "status" (hobj "value" "loaded")
                               ;; what it reads and writes, as OpenRouter's model list says it
                               "architecture" (hobj "input_modalities" (if (engine-sees-p engine) #("text" "image") #("text"))
                                                    "output_modalities" #("text"))
                               "meta" (hobj "n_ctx_train" (model-context-length m)
                                            "n_vocab" (model-n-vocab m)))
                         *more-models*))))

;; GET /props: what llama.cpp's web UI asks of its server (its model, context size, and whether
;; images or audio may be attached -- neither, here)
(defun props-object (engine)
  (hobj "model_path" (model-name (engine-model engine))
        "build_info" "illatio"
        "n_ctx" (context-n-ctx (engine-context engine))
        "modalities" (hobj "vision" (engine-sees-p engine) "audio" nil)))

;;; --- streaming --------------------------------------------------------------
;;; The chunks come from CHAT-COMPLETION-CHUNKS; here they only become SSE.

(defun send-event (s object)
  (cl-transport.http-server:send-event s (json-simple:stringify object)))

(defun stream-completion (engine req s)
  (cl-transport.http-server:write-head s 200 :content-type "text/event-stream"
                                             :headers '(("Cache-Control" . "no-cache")))
  (force-output s)
  ;; past the header a failure can't be a 500: it is an error event, as OpenAI's and llama.cpp's
  ;; streams carry one, so a client sees why its reply stopped rather than an empty one
  (handler-case (chat-completion-chunks engine req (lambda (chunk) (send-event s chunk)))
    (stream-error (e) (error e))
    (error (e)
      (format *error-output* "~&illatio: a streamed request failed: ~A~%" e)
      (send-event s (error-object (princ-to-string e) "server_error"))))
  (cl-transport.http-server:send-event s "[DONE]"))

;;; --- the listener -----------------------------------------------------------

(defun handle-request (engine request &optional static)
  "Answer REQUEST (a cl-transport.http-server request) for ENGINE."
  (let* ((s (cl-transport.http-server:request-stream request))
         (method (cl-transport.http-server:request-method request))
         (path (cl-transport.http-server:request-path request)))
    ;; everything below writes the response to S itself
    (setf (cl-transport.http-server:request-responded-p request) t)
    (handler-case
        (cond
          ((and (string= method "GET") (string= path "/health"))
           (respond s "200 OK" (hobj "status" "ok")))
          ((and (string= method "GET") (string= path "/v1/models"))
           (respond s "200 OK" (models-object engine)))
          ((and (string= method "GET") (string= path "/props"))
           (respond s "200 OK" (props-object engine)))
          ((and (string= method "POST") (member path '("/v1/chat/completions" "/chat/completions") :test #'string=))
           (let ((req (handler-case (json-simple:parse (cl-transport.http-server:request-body-string request))
                        (error () nil))))
             (cond ((not (hash-table-p req))
                    (respond s "400 Bad Request" (error-object "the body is not a JSON object")))
                   ((not (vectorp (jget req "messages")))
                    (respond s "400 Bad Request" (error-object "messages must be an array")))
                   ((jget req "stream") (stream-completion engine req s))
                   (t (respond s "200 OK" (chat-completion engine req))))))
          ((let ((r (find-if (lambda (r) (and (string= method (first r)) (member path (second r) :test #'string=)))
                             *routes*)))
             (when r (funcall (third r) s (cl-transport.http-server:request-body-string request)) t)))
          ((and static (member method '("GET" "HEAD") :test #'string=) (static-file static path))
           (setf (cl-transport.http-server:request-responded-p request) nil)
           (multiple-value-bind (file gzipped type) (static-file static path)
             (cl-transport.http-server:respond-file request file :gzipped gzipped :content-type type)))
          (t (respond s "404 Not Found" (error-object (format nil "no route for ~A ~A" method path)
                                                       "not_found"))))
      (stream-error () nil)               ; the client went away
      (error (e)
        (format *error-output* "~&illatio: a request failed: ~A~%" e)
        (ignore-errors (respond s "500 Internal Server Error"
                                (error-object (princ-to-string e) "server_error")))))))

(defun serve (engine &key (host "127.0.0.1") (port 9931) static)
  "Answer OpenAI chat completions for ENGINE on HOST:PORT, and GET requests
   for the files under directory STATIC when given (a web UI).  Returns a
   function that stops the server, and the server (cl-transport/http-server's;
   PORT 0 binds an ephemeral port, which SERVER-PORT reads back).  Loopback by
   default: this is a model with no authentication in front of it."
  (when static (assert (uiop:directory-exists-p static) () "no directory ~A to serve" static))
  (let ((server (cl-transport.http-server:start-server
                 :host host :port port :name "illatio"
                 :handler (lambda (request) (handle-request engine request static) t))))
    (values (lambda () (cl-transport.http-server:stop-server server))
            server)))
