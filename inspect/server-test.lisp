;;;; server-test.lisp — the HTTP face, without a model.
;;;
;;;   sbcl --non-interactive --load inspect/server-test.lisp
;;;
;;; Everything SERVE answers that does not need weights: /health, a request
;;; refused before it reaches the model (400s), another program's route
;;; (*ROUTES*, written against RESPOND / WRITE-UTF8 / SEND-EVENT as ex-nihilo's
;;; are), static files with index.html and FILE.gz, the 404, and a 500 from a
;;; route that fails.  The client is Python's http.client -- not ours -- so this
;;; checks the bytes against an HTTP implementation illatio had no part in.
;;; 127.0.0.1, an ephemeral port, closed at the end.

(defvar cl-user::*system* "illatio/server")
(load (merge-pathnames "load.lisp" (or *load-truename* *default-pathname-defaults*)))
(in-package #:illatio)

(defvar *failures* 0)
(defvar *checks* 0)
(defun check (ok what)
  (incf *checks*)
  (format t "~&  ~:[FAIL~;ok  ~] ~A~%" ok what)
  (unless ok (incf *failures*)))

(defvar *static*
  (let ((d (merge-pathnames (format nil "illatio-server-test-~D/" (random 1000000 (make-random-state t)))
                            (uiop:ensure-directory-pathname (or (uiop:getenv "TMPDIR") "/tmp")))))
    (ensure-directories-exist d)
    (with-open-file (o (merge-pathnames "index.html" d) :direction :output :if-exists :supersede)
      (write-string "<title>ui</title>" o))
    (with-open-file (o (merge-pathnames "app.js.gz" d) :direction :output :element-type '(unsigned-byte 8)
                                                        :if-exists :supersede)
      (write-sequence (coerce #(31 139 8 0) '(vector (unsigned-byte 8))) o))
    d))

;; another program's routes, the way ex-nihilo writes them: (stream body) -> the whole response
(add-route "POST" '("/v1/echo") (lambda (s body) (respond s "200 OK" (hobj "got" body))))
(add-route "GET" '("/v1/ticks")
           (lambda (s body)
             (declare (ignore body))
             (cl-transport.http-server:write-head s 200 :content-type "text/event-stream")
             (dotimes (i 3) (send-event s (hobj "n" i)))
             (write-utf8 s (format nil "data: [DONE]~%~%"))
             (force-output s)))
(add-route "GET" '("/v1/broken") (lambda (s body) (declare (ignore s body)) (error "a route that fails")))

(multiple-value-bind (stop server) (serve nil :host "127.0.0.1" :port 0 :static (namestring *static*))
  (let* ((port (cl-transport.http-server:server-port server))
         (script (format nil "
import http.client, json
port = ~D
def req(method, path, body=None, headers={}):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    c.request(method, path, body=body, headers=headers)
    r = c.getresponse()
    return [r.status, r.getheader('Content-Type'), r.getheader('Content-Encoding'), r.read().decode('latin-1')]
out = {}
out['health'] = req('GET', '/health')
out['notjson'] = req('POST', '/v1/chat/completions', 'not json', {'Content-Type': 'application/json'})
out['nomsgs'] = req('POST', '/chat/completions', json.dumps({'model': 'x'}), {'Content-Type': 'application/json'})
out['echo'] = req('POST', '/v1/echo', 'h\\u00e9llo'.encode('utf-8'))
out['ticks'] = req('GET', '/v1/ticks')
out['broken'] = req('GET', '/v1/broken')
out['index'] = req('GET', '/')
out['gz'] = req('GET', '/app.js')
out['missing'] = req('GET', '/nope.css')
out['dots'] = req('GET', '/../illatio.asd')
print(json.dumps(out))
" port))
         (got (with-output-to-string (o)
                (uiop:run-program (list "python3" "-I" "-c" script) :output o :error-output t)))
         (r (json-simple:parse got)))
    (flet ((at (k) (jget r k)))
      (check (and (= 200 (aref (at "health") 0)) (equal (aref (at "health") 1) "application/json")
                  (string= (aref (at "health") 3) "{\"status\":\"ok\"}"))
             "GET /health")
      (check (and (= 400 (aref (at "notjson") 0)) (search "not a JSON object" (aref (at "notjson") 3)))
             "a body that is not JSON: 400, before the model")
      (check (and (= 400 (aref (at "nomsgs") 0)) (search "messages must be an array" (aref (at "nomsgs") 3)))
             "/chat/completions without messages: 400")
      (check (and (= 200 (aref (at "echo") 0))
                  (string= (aref (at "echo") 3)
                           (map 'string #'code-char (sb-ext:string-to-octets "{\"got\":\"héllo\"}" :external-format :utf-8))))
             "another program's route gets the body as UTF-8 text")
      (check (and (= 200 (aref (at "ticks") 0)) (equal (aref (at "ticks") 1) "text/event-stream")
                  (string= (aref (at "ticks") 3)
                           (format nil "data: {\"n\":0}~%~%data: {\"n\":1}~%~%data: {\"n\":2}~%~%data: [DONE]~%~%")))
             "a route's server-sent events")
      (check (and (= 500 (aref (at "broken") 0)) (search "a route that fails" (aref (at "broken") 3)))
             "a failing route: 500 with the error object")
      (check (and (= 200 (aref (at "index") 0)) (equal (aref (at "index") 1) "text/html; charset=utf-8")
                  (string= (aref (at "index") 3) "<title>ui</title>"))
             "static / is index.html")
      (check (and (= 200 (aref (at "gz") 0)) (equal (aref (at "gz") 2) "gzip")
                  (equal (aref (at "gz") 1) "text/javascript; charset=utf-8"))
             "FILE.gz stands in for FILE, gzip-encoded")
      (check (and (= 404 (aref (at "missing") 0)) (search "not_found" (aref (at "missing") 3))) "a missing file: 404")
      (check (= 404 (aref (at "dots") 0)) "nothing outside the directory")))
  (funcall stop)
  (check (not (cl-transport.http-server:server-running-p server)) "the stop function stops it")
  (check (handler-case (progn (cl-transport.sockets:tcp-close
                               (cl-transport.sockets:tcp-connect "127.0.0.1" (cl-transport.http-server:server-port server)))
                              nil)
           (error () t))
         "and nothing is listening after"))

(format t "~&~D checks, ~D failed~%" *checks* *failures*)
(uiop:quit (if (zerop *failures*) 0 1))
