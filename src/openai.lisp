;;;; openai.lisp — OpenAI chat completions, computed in this image.
;;;
;;; CHAT-COMPLETION takes a parsed /v1/chat/completions request (json-simple's
;;; hash tables) and returns the response object.  The HTTP server is a thin
;;; wrapper around it; so is operandi's --illatio, which calls it directly and
;;; never opens a socket.
;;;
;;; The prompt is rendered by Lisp written for the model's template, not by a
;;; Jinja interpreter: there are few templates we care about, and a renderer
;;; we can read is one we can test byte for byte against llama.cpp's.

(in-package #:illatio)

;;; --- JSON the way the template writes it ------------------------------------
;;; Jinja's tojson is Python's json.dumps: ", " and ": " separators, keys in
;;; the order given, non-ASCII left as is.  json-simple writes compact JSON,
;;; and one space is one token, so we write our own.

(defun write-pyjson (x s)
  (cond ((hash-table-p x)
         (write-char #\{ s)
         (let ((first t))
           (maphash (lambda (k v)
                      (unless first (write-string ", " s))
                      (setf first nil)
                      (json-simple:write-json-string k s)
                      (write-string ": " s)
                      (write-pyjson v s))
                    x))
         (write-char #\} s))
        ((and (vectorp x) (not (stringp x)))
         (write-char #\[ s)
         (loop for v across x for i from 0
               do (when (plusp i) (write-string ", " s))
                  (write-pyjson v s))
         (write-char #\] s))
        ((stringp x) (json-simple:write-json-string x s))
        ((eq x t) (write-string "true" s))
        ((null x) (write-string "false" s))
        ((eq x 'null) (write-string "null" s))
        (t (write-string (json-simple:stringify x) s))))

(defun pyjson (x) (with-output-to-string (s) (write-pyjson x s)))

(defun jget (obj &rest keys)
  (loop for k in keys
        while (hash-table-p obj)
        do (setf obj (gethash k obj))
        finally (return (if (eq obj 'null) nil obj))))

(defun content-string (content)
  "A message's content as text: a string, or the text parts of a parts array."
  (cond ((stringp content) content)
        ((vectorp content)
         (with-output-to-string (s)
           (loop for part across content
                 when (equal (jget part "type") "text")
                   do (write-string (or (jget part "text") "") s))))
        (t "")))

;;; --- ids ----------------------------------------------------------------------
;;; Seeded on first use, not at load: a state made at load time is frozen into
;;; a saved core, and every process started from it would mint the same ids.

(defvar *id-random* nil)

(defun fresh-id (prefix &optional (digits 12))
  (format nil "~A~36R" prefix
          (random (expt 36 digits) (or *id-random* (setf *id-random* (make-random-state t))))))

;;; --- qwen2.5's template -----------------------------------------------------

(defparameter +qwen-default-system+
  "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")

(defun tool-call-arguments (call)
  "A call's arguments as an object.  OpenAI sends them as a JSON string;
   the template writes them as an object, so decode, and keep a string that
   isn't JSON as it is."
  (let ((a (jget call "function" "arguments")))
    (if (stringp a)
        (or (ignore-errors (json-simple:parse a)) a)
        a)))

(defun render-qwen-chat (messages tools &key (add-generation-prompt t))
  "MESSAGES and TOOLS rendered as Qwen2.5's chat template renders them."
  (let ((msgs (coerce messages 'list)))
    (with-output-to-string (s)
      (flet ((emit (&rest strings) (dolist (x strings) (write-string x s))))
        (let ((sys (and msgs (equal (jget (first msgs) "role") "system")
                        (content-string (jget (first msgs) "content")))))
          (if (and tools (plusp (length tools)))
              (progn
                (emit "<|im_start|>system" (string #\Newline) (or sys +qwen-default-system+)
                      (format nil "~%~%# Tools~%~%You may call one or more functions to assist with the user query.~%~%You are provided with function signatures within <tools></tools> XML tags:~%<tools>"))
                (loop for tool across tools do (emit (string #\Newline) (pyjson tool)))
                (emit (format nil "~%</tools>~%~%For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:~%<tool_call>~%{{\"name\": <function-name>, \"arguments\": <args-json-object>}}~%</tool_call><|im_end|>~%")))
              (emit "<|im_start|>system" (string #\Newline) (or sys +qwen-default-system+)
                    "<|im_end|>" (string #\Newline))))
        (loop for (m . rest) on msgs
              for prev = nil then role
              for role = (jget m "role")
              for i from 0
              for content = (content-string (jget m "content"))
              for calls = (let ((c (jget m "tool_calls"))) (and (vectorp c) (plusp (length c)) c))
              do (cond
                   ((or (equal role "user")
                        (and (equal role "system") (plusp i))
                        (and (equal role "assistant") (not calls)))
                    (emit "<|im_start|>" role (string #\Newline) content "<|im_end|>" (string #\Newline)))
                   ((equal role "assistant")
                    (emit "<|im_start|>assistant")
                    (when (plusp (length content)) (emit (string #\Newline) content))
                    (loop for call across calls
                          do (emit (format nil "~%<tool_call>~%{\"name\": \"")
                                   (or (jget call "function" "name") (jget call "name") "")
                                   "\", \"arguments\": "
                                   (pyjson (tool-call-arguments call))
                                   (format nil "}~%</tool_call>")))
                    (emit "<|im_end|>" (string #\Newline)))
                   ((equal role "tool")
                    (unless (equal prev "tool") (emit "<|im_start|>user"))
                    (emit (format nil "~%<tool_response>~%") content (format nil "~%</tool_response>"))
                    (unless (and rest (equal (jget (first rest) "role") "tool"))
                      (emit "<|im_end|>" (string #\Newline))))))
        (when add-generation-prompt
          (emit "<|im_start|>assistant" (string #\Newline)))))))

(defun render-chat (model messages &optional tools)
  (let ((template (meta (model-gguf model) "tokenizer.chat_template" "")))
    (cond ((and (search "<|im_start|>" template) (search "<tool_call>" template))
           (render-qwen-chat messages tools))
          (t (error "no renderer for ~A's chat template yet" (model-name model))))))

;;; --- the model's answer, as a message ---------------------------------------

(defun parse-tool-calls (text)
  "Split TEXT into (values content tool-calls): every <tool_call>{...}</tool_call>
   becomes a call, the rest is content.  A block whose JSON doesn't parse is
   left in the content, so nothing the model said is silently dropped."
  (let ((calls '()) (content (make-string-output-stream)) (pos 0) (n 0))
    (loop
      (let ((open (search "<tool_call>" text :start2 pos)))
        (unless open (write-string text content :start pos) (return))
        (let* ((close (search "</tool_call>" text :start2 open))
               (body (and close (string-trim '(#\Space #\Newline #\Tab #\Return)
                                             (subseq text (+ open 11) close))))
               (obj (and body (or (ignore-errors (json-simple:parse body))
                                  ;; Qwen's template shows the format as {{...}} (a
                                  ;; Jinja escape that isn't one, inside a string
                                  ;; literal), and small models copy it faithfully.
                                  (and (> (length body) 4)
                                       (string= "{{" body :end2 2)
                                       (ignore-errors
                                        (json-simple:parse (subseq body 1 (1- (length body))))))))))
          (if (and (hash-table-p obj) (stringp (jget obj "name")))
              (progn
                (write-string text content :start pos :end open)
                (let ((args (gethash "arguments" obj)))
                  (push (let ((h (make-hash-table :test 'equal)) (f (make-hash-table :test 'equal)))
                          (setf (gethash "id" h) (progn (incf n) (fresh-id "call_" 8))
                                (gethash "type" h) "function"
                                (gethash "name" f) (jget obj "name")
                                (gethash "arguments" f) (if (stringp args) args
                                                            (json-simple:stringify (or args (make-hash-table))))
                                (gethash "function" h) f)
                          h)
                        calls))
                (setf pos (+ close 12)))
              (progn
                (write-string text content :start pos :end (or close (length text)))
                (setf pos (if close (+ close 12) (length text)))
                (when close (write-string "</tool_call>" content)))))))
    (values (string-trim '(#\Space #\Newline) (get-output-stream-string content))
            (nreverse calls))))

;;; --- the endpoint -----------------------------------------------------------

(defstruct (engine (:constructor %make-engine))
  model context
  (lock (sb-thread:make-mutex :name "illatio engine")))

(defun make-engine (model &key (n-ctx (min 32768 (model-context-length model))))
  "A model and the one live context requests share.  One request runs at a
   time; consecutive requests that share a prefix (an agent's growing
   conversation) reuse its KV cache."
  (%make-engine :model model :context (make-context model :n-ctx n-ctx)))

(defun hobj (&rest kv)
  (let ((h (make-hash-table :test 'equal)))
    (loop for (k v) on kv by #'cddr do (setf (gethash k h) v))
    h))

(defun request-sampler (req)
  (flet ((num (k default) (let ((v (jget req k))) (if (realp v) v default))))
    (make-sampler :temperature (num "temperature" 0.8)
                  :top-k (num "top_k" 40) :top-p (num "top_p" 0.95) :min-p (num "min_p" 0.05)
                  :seed (let ((s (jget req "seed"))) (and (integerp s) s)))))

(defun request-stop (req)
  (let ((s (jget req "stop")))
    (cond ((stringp s) (list s))
          ((vectorp s) (coerce (remove-if-not #'stringp s) 'list)))))

(defun chat-completion (engine req &key on-text)
  "Answer the chat-completions request REQ (a parsed JSON object).  ON-TEXT,
   if given, gets the reply's text as it is generated (for streaming).
   Returns the response object."
  (sb-thread:with-mutex ((engine-lock engine))
    (let* ((m (engine-model engine))
           (ctx (engine-context engine))
           (prompt (render-chat m (jget req "messages") (jget req "tools")))
           (ids (tokenize m prompt))
           (max (or (jget req "max_completion_tokens") (jget req "max_tokens") 2048))
           (t0 (get-internal-real-time))
           (t1 nil))
      (multiple-value-bind (text out reason reused)
          (generate ctx ids :max-tokens max :sampler (request-sampler req)
                            :stop (request-stop req)
                            :on-text (lambda (piece)
                                       (unless t1 (setf t1 (get-internal-real-time)))
                                       (when on-text (funcall on-text piece))))
        (let ((t2 (get-internal-real-time)))
          (unless t1 (setf t1 t2))
          (multiple-value-bind (content calls) (parse-tool-calls text)
            (flet ((ms (a b) (/ (* 1000d0 (- b a)) internal-time-units-per-second)))
              (hobj "id" (fresh-id "chatcmpl-")
                    "object" "chat.completion"
                    "created" (- (get-universal-time) 2208988800)
                    "model" (model-name m)
                    "choices" (vector
                               (hobj "index" 0
                                     "message" (let ((msg (hobj "role" "assistant" "content" content)))
                                                 (when calls (setf (gethash "tool_calls" msg) (coerce calls 'vector)))
                                                 msg)
                                     "finish_reason" (cond (calls "tool_calls")
                                                           ((eq reason :length) "length")
                                                           (t "stop"))))
                    "usage" (hobj "prompt_tokens" (length ids)
                                  "completion_tokens" (length out)
                                  "total_tokens" (+ (length ids) (length out)))
                    ;; llama.cpp's own timings block, which operandi reads
                    "timings" (hobj "cache_n" reused
                                    "prompt_n" (- (length ids) reused)
                                    "prompt_ms" (ms t0 t1)
                                    "predicted_n" (length out)
                                    "predicted_ms" (ms t1 t2))))))))))

;;; --- streamed -----------------------------------------------------------------
;;; The same answer as a sequence of chat.completion.chunk objects -- what an
;;; SSE stream carries, before it is text.  The server writes them as events;
;;; a Lisp caller in this image (operandi --illatio) folds them directly.

(defun chunk-object (id model delta &optional finish)
  (hobj "id" id "object" "chat.completion.chunk"
        "created" (- (get-universal-time) 2208988800)
        "model" model
        "choices" (vector (hobj "index" 0 "delta" delta "finish_reason" (or finish 'null)))))

(defun make-tool-call-filter (emit)
  "A text sink that passes text to EMIT until a <tool_call> starts, holding
   back a tail that might be the start of that tag.  The calls themselves
   are sent whole, parsed, once the reply is done."
  (let ((pending "") (stopped nil) (tag "<tool_call>"))
    (lambda (piece)
      (unless stopped
        (let* ((text (concatenate 'string pending piece))
               (at (search tag text)))
          (if at
              (progn (when (plusp at) (funcall emit (subseq text 0 at)))
                     (setf stopped t pending ""))
              ;; keep the longest suffix that is a proper prefix of the tag
              (let ((keep (loop for k from (min (1- (length tag)) (length text)) downto 1
                                when (string= text tag :start1 (- (length text) k) :end2 k)
                                  return k
                                finally (return 0))))
                (when (> (length text) keep)
                  (funcall emit (subseq text 0 (- (length text) keep))))
                (setf pending (subseq text (- (length text) keep))))))))))

(defun chat-completion-chunks (engine req on-chunk)
  "Answer REQ as CHAT-COMPLETION does, calling ON-CHUNK with each
   chat.completion.chunk object as the reply is generated: content deltas,
   then any tool calls, then the finish reason, then usage and timings.
   Returns the complete response object as well."
  (let* ((id (fresh-id "chatcmpl-"))
         (name (model-name (engine-model engine)))
         (sent-role nil)
         (emit (lambda (text)
                 (funcall on-chunk
                          (chunk-object id name (if sent-role
                                                    (hobj "content" text)
                                                    (progn (setf sent-role t)
                                                           (hobj "role" "assistant" "content" text)))))))
         (resp (chat-completion engine req
                                :on-text (if (jget req "tools") (make-tool-call-filter emit) emit)))
         (choice (aref (jget resp "choices") 0))
         (calls (jget choice "message" "tool_calls")))
    (when calls
      (funcall on-chunk
               (chunk-object id name
                             (hobj "tool_calls"
                                   (map 'vector (lambda (c i)
                                                  (hobj "index" i "id" (jget c "id") "type" "function"
                                                        "function" (jget c "function")))
                                        calls (loop for i below (length calls) collect i))))))
    (funcall on-chunk (chunk-object id name (hobj) (jget choice "finish_reason")))
    (funcall on-chunk (hobj "id" id "object" "chat.completion.chunk" "model" name
                            "choices" (vector) "usage" (jget resp "usage")
                            "timings" (jget resp "timings")))
    resp))
