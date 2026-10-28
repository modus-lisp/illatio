;;;; server.lisp — the OpenAI-compatible HTTP face of CHAT-COMPLETION.
;;;
;;; Enough HTTP/1.1 for API clients: one request per connection, a
;;; Content-Length body, JSON or server-sent events back.  Everything
;;; model-shaped happens in CHAT-COMPLETION; this file only moves bytes.
;;;
;;;   GET  /health                -> {"status":"ok"}
;;;   GET  /v1/models             -> the one model, with its context_length
;;;   POST /v1/chat/completions   -> blocking, or "stream": true for SSE

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

(defun handle-connection (engine s)
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
            ((and (string= method "POST") (string= path "/v1/chat/completions"))
             (let ((req (handler-case (json-simple:parse body)
                          (error () nil))))
               (cond ((not (hash-table-p req))
                      (respond s "400 Bad Request" (error-object "the body is not a JSON object")))
                     ((not (vectorp (jget req "messages")))
                      (respond s "400 Bad Request" (error-object "messages must be an array")))
                     ((jget req "stream") (stream-completion engine req s))
                     (t (respond s "200 OK" (chat-completion engine req))))))
            (t (respond s "404 Not Found" (error-object (format nil "no route for ~A ~A" method path)
                                                         "not_found"))))))
    (stream-error () nil)               ; the client went away
    (error (e)
      (ignore-errors (respond s "500 Internal Server Error"
                              (error-object (princ-to-string e) "server_error"))))))

(defun serve (engine &key (host "127.0.0.1") (port 9931))
  "Answer OpenAI chat completions for ENGINE on HOST:PORT.  Returns a
   function that stops the server.  Loopback by default: this is a model
   with no authentication in front of it."
  (cl-transport:expose (lambda (stream peer)
                         (declare (ignore peer))
                         (handle-connection engine stream))
                       :backend :tcp :host host :port port))
