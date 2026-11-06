;;;; server.lisp — the OpenAI-compatible HTTP face of CHAT-COMPLETION.
;;;
;;; Enough HTTP/1.1 for API clients: one request per connection, a
;;; Content-Length body, JSON or server-sent events back.  Everything
;;; model-shaped happens in CHAT-COMPLETION; this file only moves bytes.
;;;
;;;   GET  /health                -> {"status":"ok"}
;;;   GET  /v1/models             -> the one model, with its context_length
;;;   POST /v1/chat/completions   -> blocking, or "stream": true for SSE
;;;        (/chat/completions too, as llama-server takes it)
;;;   GET  anything else          -> a file under STATIC (SERVE's), if given:
;;;        a web UI's, e.g. llama.cpp's (tools/server/public_simplechat)

(in-package #:illatio)

(defun read-line-crlf (s)
  "One header line from byte stream S, without its CRLF; NIL at EOF."
  (let ((out (make-array 0 :element-type 'character :adjustable t :fill-pointer 0)))
    (loop for b = (read-byte s nil)
          do (cond ((null b) (return (and (plusp (length out)) (coerce out 'simple-string))))
                   ((= b 10) (return (string-right-trim '(#\Return) (coerce out 'simple-string))))
                   (t (vector-push-extend (code-char b) out))))))

(defun read-request (s)
  "(values method path headers body) for the request on S, or NIL."
  (let ((line (read-line-crlf s)))
    (when line
      (destructuring-bind (&optional method path &rest ignore)
          (uiop:split-string line :separator " ")
        (declare (ignore ignore))
        (let ((headers (loop for h = (read-line-crlf s)
                             while (and h (plusp (length h)))
                             collect (let ((c (position #\: h)))
                                       (cons (string-downcase (subseq h 0 c))
                                             (string-trim " " (subseq h (1+ c))))))))
          (let* ((n (parse-integer (or (cdr (assoc "content-length" headers :test #'string=)) "0")))
                 (body (make-array n :element-type '(unsigned-byte 8))))
            (read-sequence body s)
            (values method path headers (utf8-decode body))))))))

(defun write-utf8 (s string)
  (write-sequence (sb-ext:string-to-octets string :external-format :utf-8) s))

(defun respond (s status object &key (type "application/json"))
  (let ((body (sb-ext:string-to-octets (if (stringp object) object (json-simple:stringify object))
                                       :external-format :utf-8)))
    (write-utf8 s (format nil "HTTP/1.1 ~A~C~CContent-Type: ~A~C~CContent-Length: ~D~C~CConnection: close~C~C~C~C"
                          status #\Return #\Newline type #\Return #\Newline (length body)
                          #\Return #\Newline #\Return #\Newline #\Return #\Newline))
    (write-sequence body s)
    (force-output s)))

;;; --- static files ----------------------------------------------------------
;;; A directory's files, read-only: / is index.html; FILE.gz stands in for a
;;; missing FILE (sent gzip-encoded, as llama.cpp's public/index.html.gz is
;;; meant to be); nothing outside the directory (no "..", no dot files).

(defparameter +content-types+
  '(("html" . "text/html; charset=utf-8") ("js" . "text/javascript; charset=utf-8")
    ("mjs" . "text/javascript; charset=utf-8") ("css" . "text/css; charset=utf-8")
    ("json" . "application/json") ("svg" . "image/svg+xml") ("png" . "image/png")
    ("jpg" . "image/jpeg") ("webp" . "image/webp") ("ico" . "image/x-icon")
    ("txt" . "text/plain; charset=utf-8") ("md" . "text/plain; charset=utf-8")
    ("woff2" . "font/woff2") ("wasm" . "application/wasm")))

(defun static-file (root path)
  "(values file gzipped-p content-type) for request PATH under directory
   ROOT, or NIL."
  (let* ((rel (string-left-trim "/" (if (string= path "/") "/index.html" path)))
         (parts (uiop:split-string rel :separator "/")))
    (unless (or (some (lambda (p) (or (zerop (length p)) (char= (char p 0) #\.))) parts)
                (find #\\ rel))
      (let* ((name (car (last parts))) (dot (position #\. name :from-end t))
             (type (or (cdr (assoc (and dot (string-downcase (subseq name (1+ dot)))) +content-types+ :test #'equal))
                       "application/octet-stream"))
             (file (merge-pathnames (uiop:parse-unix-namestring rel) (uiop:ensure-directory-pathname root))))
        (cond ((uiop:file-exists-p file) (values file nil type))
              ((uiop:file-exists-p (concatenate 'string (namestring file) ".gz"))
               (values (pathname (concatenate 'string (namestring file) ".gz")) t type)))))))

(defun respond-file (s file gzipped type)
  (let ((body (with-open-file (in file :element-type '(unsigned-byte 8))
                (let ((v (make-array (file-length in) :element-type '(unsigned-byte 8))))
                  (read-sequence v in) v))))
    (flet ((line (fmt &rest args) (format nil "~?~C~C" fmt args #\Return #\Newline)))
      (write-utf8 s (concatenate 'string (line "HTTP/1.1 200 OK") (line "Content-Type: ~A" type)
                                 (if gzipped (line "Content-Encoding: gzip") "")
                                 (line "Content-Length: ~D" (length body)) (line "Cache-Control: no-cache")
                                 (line "Connection: close") (line ""))))
    (write-sequence body s)
    (force-output s)))

(defun error-object (message &optional (type "invalid_request_error"))
  (hobj "error" (hobj "message" message "type" type)))

(defun models-object (engine)
  (let ((m (engine-model engine)))
    (hobj "object" "list"
          "data" (vector (hobj "id" (model-name m)
                               "object" "model"
                               "owned_by" "illatio"
                               ;; what a client may send in one conversation:
                               ;; this context's size, not the model's training length
                               "context_length" (context-n-ctx (engine-context engine))
                               "status" (hobj "value" "loaded")
                               "meta" (hobj "n_ctx_train" (model-context-length m)
                                            "n_vocab" (model-n-vocab m)))))))

;; GET /props: what llama.cpp's web UI asks of its server (its model, context size, and whether
;; images or audio may be attached -- neither, here)
(defun props-object (engine)
  (hobj "model_path" (model-name (engine-model engine))
        "build_info" "illatio"
        "n_ctx" (context-n-ctx (engine-context engine))
        "modalities" (hobj "vision" nil "audio" nil)))

;;; --- streaming --------------------------------------------------------------
;;; The chunks come from CHAT-COMPLETION-CHUNKS; here they only become SSE.

(defun send-event (s object)
  (write-utf8 s (format nil "data: ~A~%~%" (json-simple:stringify object)))
  (force-output s))

(defun stream-completion (engine req s)
  (write-utf8 s (format nil "HTTP/1.1 200 OK~C~CContent-Type: text/event-stream~C~CCache-Control: no-cache~C~CConnection: close~C~C~C~C"
                        #\Return #\Newline #\Return #\Newline #\Return #\Newline
                        #\Return #\Newline #\Return #\Newline))
  (chat-completion-chunks engine req (lambda (chunk) (send-event s chunk)))
  (write-utf8 s (format nil "data: [DONE]~%~%"))
  (force-output s))

;;; --- the listener -----------------------------------------------------------

(defun handle-connection (engine s &optional static)
  (handler-case
      (multiple-value-bind (method path headers body) (read-request s)
        (declare (ignore headers))
        (let ((path (subseq path 0 (or (position #\? path) (length path)))))
          (cond
            ((null method))
            ((and (string= method "GET") (string= path "/health"))
             (respond s "200 OK" (hobj "status" "ok")))
            ((and (string= method "GET") (string= path "/v1/models"))
             (respond s "200 OK" (models-object engine)))
            ((and (string= method "GET") (string= path "/props"))
             (respond s "200 OK" (props-object engine)))
            ((and (string= method "POST") (member path '("/v1/chat/completions" "/chat/completions") :test #'string=))
             (let ((req (handler-case (json-simple:parse body)
                          (error () nil))))
               (cond ((not (hash-table-p req))
                      (respond s "400 Bad Request" (error-object "the body is not a JSON object")))
                     ((not (vectorp (jget req "messages")))
                      (respond s "400 Bad Request" (error-object "messages must be an array")))
                     ((jget req "stream") (stream-completion engine req s))
                     (t (respond s "200 OK" (chat-completion engine req))))))
            ((and static (string= method "GET") (static-file static path))
             (multiple-value-call #'respond-file s (static-file static path)))
            (t (respond s "404 Not Found" (error-object (format nil "no route for ~A ~A" method path)
                                                         "not_found"))))))
    (stream-error () nil)               ; the client went away
    (error (e)
      (ignore-errors (respond s "500 Internal Server Error"
                              (error-object (princ-to-string e) "server_error"))))))

(defun serve (engine &key (host "127.0.0.1") (port 9931) static)
  "Answer OpenAI chat completions for ENGINE on HOST:PORT, and GET requests
   for the files under directory STATIC when given (a web UI).  Returns a
   function that stops the server.  Loopback by default: this is a model
   with no authentication in front of it."
  (when static (assert (uiop:directory-exists-p static) () "no directory ~A to serve" static))
  (cl-transport:expose (lambda (stream peer)
                         (declare (ignore peer))
                         (handle-connection engine stream static))
                       :backend :tcp :host host :port port))
