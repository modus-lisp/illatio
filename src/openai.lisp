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

;;; --- qwen3's template ---------------------------------------------------------
;;; Qwen2.5's, less the default system prompt, plus <think> handling: an
;;; assistant turn after the last real user query is written with a <think>
;;; block when it is the final message or has reasoning_content.  A tool
;;; call's arguments that arrive as a JSON string are written as they are.

(defun strip-newlines (string &key (left t) (right t))
  "STRING without leading (LEFT) and trailing (RIGHT) newlines: Python's
   lstrip / rstrip / strip with '\\n'."
  (let ((start 0) (end (length string)))
    (when left (loop while (and (< start end) (char= (char string start) #\Newline)) do (incf start)))
    (when right (loop while (and (> end start) (char= (char string (1- end)) #\Newline)) do (decf end)))
    (subseq string start end)))

(defun split-reasoning (content)
  "(VALUES reasoning content) as the template splits a turn at </think>."
  (let ((close (search "</think>" content)))
    (if (null close)
        (values "" content)
        (let* ((before (strip-newlines (subseq content 0 close) :left nil))
               (open (search "<think>" before :from-end t))
               (reasoning (strip-newlines (if open (subseq before (+ open 7)) before) :right nil))
               (last-close (search "</think>" content :from-end t)))
          (values reasoning (strip-newlines (subseq content (+ last-close 8)) :right nil))))))

(defun render-qwen3-chat (messages tools
                          &key (add-generation-prompt
                                (let ((l (and (plusp (length messages))
                                              (elt messages (1- (length messages))))))
                                  (not (and l (equal (jget l "role") "assistant"))))))
  "MESSAGES and TOOLS rendered as Qwen3's chat template renders them."
  (let* ((msgs (coerce messages 'list))
         (n (length msgs))
         ;; the last user message that is not a wrapped tool response
         (last-query
           (or (loop for i from (1- n) downto 0
                     for m = (nth i msgs)
                     for c = (jget m "content")
                     when (and (equal (jget m "role") "user") (stringp c)
                               (not (and (eql 0 (search "<tool_response>" c))
                                         (let ((e (search "</tool_response>" c :from-end t)))
                                           (and e (= (+ e 16) (length c)))))))
                       return i)
               (1- n))))
    (with-output-to-string (s)
      (flet ((emit (&rest strings) (dolist (x strings) (write-string x s))))
        (let ((sys (and msgs (equal (jget (first msgs) "role") "system")
                        (content-string (jget (first msgs) "content")))))
          (if (and tools (plusp (length tools)))
              (progn
                (emit "<|im_start|>system" (string #\Newline))
                (when sys (emit sys (format nil "~%~%")))
                (emit (format nil "# Tools~%~%You may call one or more functions to assist with the user query.~%~%You are provided with function signatures within <tools></tools> XML tags:~%<tools>"))
                (loop for tool across tools do (emit (string #\Newline) (pyjson tool)))
                (emit (format nil "~%</tools>~%~%For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:~%<tool_call>~%{\"name\": <function-name>, \"arguments\": <args-json-object>}~%</tool_call><|im_end|>~%")))
              (when sys
                (emit "<|im_start|>system" (string #\Newline) sys "<|im_end|>" (string #\Newline)))))
        (loop for (m . rest) on msgs
              for prev = nil then role
              for role = (jget m "role")
              for i from 0
              for raw = (jget m "content")
              for content = (if (stringp raw) raw "")
              for calls = (let ((c (jget m "tool_calls"))) (and (vectorp c) (plusp (length c)) c))
              do (cond
                   ((or (equal role "user") (and (equal role "system") (plusp i)))
                    (emit "<|im_start|>" role (string #\Newline) content "<|im_end|>" (string #\Newline)))
                   ((equal role "assistant")
                    ;; llama-server splits reasoning out of earlier turns, not
                    ;; out of a final (prefill) one, which it leaves as sent
                    (multiple-value-bind (reasoning body)
                        (let ((rc (jget m "reasoning_content")))
                          (cond ((stringp rc) (values rc content))
                                ((null rest) (values "" content))
                                (t (split-reasoning content))))
                      (if (and (> i last-query) (or (null rest) (plusp (length reasoning))))
                          (emit "<|im_start|>assistant" (format nil "~%<think>~%")
                                (strip-newlines reasoning) (format nil "~%</think>~%~%")
                                (strip-newlines body :right nil))
                          (emit "<|im_start|>assistant" (string #\Newline) body))
                      (when calls
                        (loop for call across calls for first = t then nil
                              do (when (or (not first) (plusp (length body))) (emit (string #\Newline)))
                                 (progn
                                   (emit (format nil "<tool_call>~%{\"name\": \"")
                                         (or (jget call "function" "name") (jget call "name") "")
                                         "\", \"arguments\": "
                                         ;; a JSON string is written re-serialized,
                                         ;; as llama-server passes it to the template
                                         (pyjson (tool-call-arguments call))
                                         (format nil "}~%</tool_call>")))))
                      ;; a final assistant message is a prefill: left open
                      (when rest (emit "<|im_end|>" (string #\Newline)))))
                   ((equal role "tool")
                    (unless (equal prev "tool") (emit "<|im_start|>user"))
                    (emit (format nil "~%<tool_response>~%") content (format nil "~%</tool_response>"))
                    (unless (and rest (equal (jget (first rest) "role") "tool"))
                      (emit "<|im_end|>" (string #\Newline))))))
        (when add-generation-prompt
          (emit "<|im_start|>assistant" (string #\Newline)))))))

(defun render-chat (model messages &optional tools &key (thinking t))
  (let ((template (meta (model-gguf model) "tokenizer.chat_template" "")))
    (cond ((member (model-arch model) '("deepseek4" "deepseek41") :test #'equal)
           (render-ds4f-chat messages tools :thinking thinking :v41 (equal (model-arch model) "deepseek41")))
          ((and (search "<|im_start|>" template) (search "<tool_call>" template)
                (search "</think>" template))
           (render-qwen3-chat messages tools))
          ((and (search "<|im_start|>" template) (search "<tool_call>" template))
           (render-qwen-chat messages tools))
          ;; No template in the file (some finetunes ship without one): a
          ;; Qwen model gets Qwen3's ChatML, which is what llama-server falls
          ;; back to (ChatML) with Qwen3's <think> handling of the history
          ((and (string= template "")
                (member (model-arch model) '("qwen2" "qwen3" "qwen35") :test #'equal))
           (unless (getf (model-notes model) :template-warned)
             (setf (getf (model-notes model) :template-warned) t)
             (format *error-output* "~&illatio: ~A has no chat template; using Qwen3's ChatML~%"
                     (model-name model)))
           (render-qwen3-chat messages tools))
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
           (thinking (request-thinking req))
           (ds4f (member (model-arch m) '("deepseek4" "deepseek41") :test #'equal))
           (prompt (render-chat m (jget req "messages") (jget req "tools") :thinking thinking))
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
          (multiple-value-bind (content calls reasoning)
              (if ds4f (ds4f-parse-reply text thinking) (parse-tool-calls text))
            (flet ((ms (a b) (/ (* 1000d0 (- b a)) internal-time-units-per-second)))
              (hobj "id" (fresh-id "chatcmpl-")
                    "object" "chat.completion"
                    "created" (- (get-universal-time) 2208988800)
                    "model" (model-name m)
                    "choices" (vector
                               (hobj "index" 0
                                     "message" (let ((msg (hobj "role" "assistant" "content" content)))
                                                 (when calls (setf (gethash "tool_calls" msg) (coerce calls 'vector)))
                                                 (when (and reasoning (plusp (length reasoning)))
                                                   (setf (gethash "reasoning_content" msg) reasoning))
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
         (emit-reasoning (lambda (text)
                           (funcall on-chunk
                                    (chunk-object id name (if sent-role
                                                              (hobj "reasoning_content" text)
                                                              (progn (setf sent-role t)
                                                                     (hobj "role" "assistant" "reasoning_content" text)))))))
         (resp (chat-completion engine req
                                :on-text (cond ((member (model-arch (engine-model engine)) '("deepseek4" "deepseek41") :test #'equal)
                                                (make-ds4f-stream-filter emit emit-reasoning (request-thinking req)))
                                               ((jget req "tools") (make-tool-call-filter emit))
                                               (t emit))))
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

;;; --- DeepSeek V4 Flash: its chat format, as ds4's server renders it ----------------
;;; (ds4_server.c, render_deepseek_chat_prompt_text): the tool schemas, one
;;; JSON object a line under ds4's instructions, come before the client's
;;; system text; turns are <｜User｜> / <｜Assistant｜>; an assistant turn
;;; opens with its reasoning in <think>..</think> where thinking keeps it, else
;;; </think>; tool calls are DSML blocks; tool output is wrapped in
;;; <tool_result>, only its own closing tag escaped.  Thinking is on unless the
;;; request turns it off, as in ds4's server.

(defparameter +ds4f-tools-header+
  (concatenate 'string
   "## Tools" (string #\Newline) (string #\Newline)
   "You have access to a set of tools to help answer the user question. "
   "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:" (string #\Newline) (string #\Newline)
   "<｜DSML｜tool_calls>" (string #\Newline)
   "<｜DSML｜invoke name=\"$TOOL_NAME\">" (string #\Newline)
   "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>" (string #\Newline)
   "..." (string #\Newline)
   "</｜DSML｜invoke>" (string #\Newline)
   "<｜DSML｜invoke name=\"$TOOL_NAME2\">" (string #\Newline)
   "..." (string #\Newline)
   "</｜DSML｜invoke>" (string #\Newline)
   "</｜DSML｜tool_calls>" (string #\Newline) (string #\Newline)
   "String parameters should be specified as raw text and set `string=\"true\"`. "
   "Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes. "
   "Only if a string value itself contains the exact closing parameter tag `</｜DSML｜parameter>`, write that tag as `&lt;/｜DSML｜parameter>` inside the value. "
   "For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`." (string #\Newline) (string #\Newline)
   "When thinking mode is enabled, finish reasoning with </think> before any tool calls or final response." (string #\Newline) (string #\Newline)
   "Otherwise, output directly after </think> with tool calls or final response." (string #\Newline) (string #\Newline)
   "### Available Tool Schemas" (string #\Newline) (string #\Newline)))

(defparameter +ds4f-tools-footer+
  (concatenate 'string (string #\Newline) (string #\Newline)
               "You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. "
               "Use the exact parameter names from the schemas."))

(defun request-thinking (req)
  "Whether REQ wants the model to think: on, as in ds4's server, unless
   \"thinking\" or chat_template_kwargs.enable_thinking is given false (or
   {\"type\": \"disabled\"}), or reasoning_effort is \"none\".  (A JSON false
   and a missing key both read as NIL, so presence is asked of the table.)"
  (flet ((given-false (h key)
           (and (hash-table-p h)
                (multiple-value-bind (v present) (gethash key h) (and present (null v))))))
    (let ((th (and (hash-table-p req) (gethash "thinking" req))))
      (not (or (given-false req "thinking")
               (and (hash-table-p th) (equal (jget th "type") "disabled"))
               (given-false (and (hash-table-p req) (gethash "chat_template_kwargs" req)) "enable_thinking")
               (equal (jget req "reasoning_effort") "none"))))))

(defun ds4f-escape-close (text end)
  "TEXT with END (a closing tag) written as &lt;..., and any spelling of that
   escape with one &amp; more, so the text can't close its wrapper early
   (ds4_tool_text_needs_escape)."
  (with-output-to-string (o)
    (let ((i 0) (n (length text)) (tail (subseq end 1)))
      (loop while (< i n)
            do (cond ((and (<= (+ i (length end)) n) (string= end text :start2 i :end2 (+ i (length end))))
                      (write-string "&lt;" o) (incf i))
                     ((and (char= (char text i) #\&)
                           (let ((j (1+ i)))
                             (loop while (and (<= (+ j 4) n) (string= "amp;" text :start2 j :end2 (+ j 4))) do (incf j 4))
                             (and (<= (+ j 3 (length tail)) n) (string= "lt;" text :start2 j :end2 (+ j 3))
                                  (string= tail text :start2 (+ j 3) :end2 (+ j 3 (length tail))))))
                      (write-string "&amp;" o) (incf i))
                     (t (write-char (char text i) o) (incf i)))))))

(defun ds4f-unescape-close (text end)
  "ds4_tool_text_unescape: one level of DS4F-ESCAPE-CLOSE undone."
  (with-output-to-string (o)
    (let ((i 0) (n (length text)) (tail (subseq end 1)))
      (loop while (< i n)
            do (if (and (char= (char text i) #\&)
                        (let ((j (1+ i)))
                          (loop while (and (<= (+ j 4) n) (string= "amp;" text :start2 j :end2 (+ j 4))) do (incf j 4))
                          (and (<= (+ j 3 (length tail)) n) (string= "lt;" text :start2 j :end2 (+ j 3))
                               (string= tail text :start2 (+ j 3) :end2 (+ j 3 (length tail))))))
                   (if (and (<= (+ i 5) n) (string= "&amp;" text :start2 i :end2 (+ i 5)))
                       (progn (write-char #\& o) (incf i 5))
                       (progn (write-char #\< o) (incf i 4)))
                   (progn (write-char (char text i) o) (incf i)))))))

(defun ds4f-attr-escape (s)
  (with-output-to-string (o)
    (loop for c across s
          do (case c (#\& (write-string "&amp;" o)) (#\< (write-string "&lt;" o)) (#\> (write-string "&gt;" o))
                   (#\" (write-string "&quot;" o)) (t (write-char c o))))))

(defun ds4f-dsml-calls (calls)
  "append_dsml_tool_calls_text: CALLS (OpenAI tool_calls) as a DSML block."
  (with-output-to-string (o)
    (format o "~%~%<｜DSML｜tool_calls>~%")
    (loop for c across calls
          do (let* ((f (jget c "function")) (args (jget f "arguments"))
                    (obj (if (stringp args) (ignore-errors (json-simple:parse args)) args)))
               (format o "<｜DSML｜invoke name=\"~A\">~%" (ds4f-attr-escape (or (jget f "name") "")))
               (if (hash-table-p obj)
                   (maphash (lambda (k v)
                              (format o "<｜DSML｜parameter name=\"~A\" string=\"~:[false~;true~]\">~A</｜DSML｜parameter>~%"
                                      (ds4f-attr-escape k) (stringp v)
                                      (if (stringp v)
                                          (ds4f-escape-close v "</｜DSML｜parameter>")
                                          (let ((j (json-simple:stringify v)))
                                            ;; append_dsml_json_literal: the closing tag's < as \u003c
                                            (with-output-to-string (o2)
                                              (loop with end = "</｜DSML｜parameter>"
                                                    for i from 0 below (length j)
                                                    do (if (and (<= (+ i (length end)) (length j))
                                                                (string= end j :start2 i :end2 (+ i (length end))))
                                                           (write-string "\\u003c" o2)
                                                           (write-char (char j i) o2))))))))
                            obj)
                   (format o "<｜DSML｜parameter name=\"arguments\" string=\"true\">~A</｜DSML｜parameter>~%"
                           (ds4f-escape-close (if (stringp args) args "") "</｜DSML｜parameter>")))
               (format o "</｜DSML｜invoke>~%")))
    (write-string "</｜DSML｜tool_calls>" o)))

(defun render-ds4f-chat (messages tools &key (thinking t) v41)
  "DeepSeek V4's prompt; V4.1's (V41) opens a system turn with <｜System｜>,
   with thinking on by ds4's default reasoning effort (high, 75)."
  (let* ((msgs (coerce messages 'list))
         (tools (and tools (plusp (length tools)) (coerce tools 'list)))
         (tool-context (or tools
                           (some (lambda (m) (or (and (equal (jget m "role") "assistant") (jget m "tool_calls")
                                                      (plusp (length (jget m "tool_calls"))))
                                                 (member (jget m "role") '("tool" "function") :test #'equal)))
                                 msgs)))
         (last-user (loop for m in msgs for i from 0
                          when (member (jget m "role") '("user" "tool" "function") :test #'equal) maximize i))
         (system (with-output-to-string (o)
                   (when tools
                     (write-string +ds4f-tools-header+ o)
                     (loop for tool in tools for first = t then nil
                           do (unless first (write-char #\Newline o))
                              (write-string (json-simple:stringify (or (jget tool "function") tool)) o))
                     (write-string +ds4f-tools-footer+ o))))
         (pending-assistant nil) (pending-tool nil))
    (dolist (m msgs)
      (when (equal (jget m "role") "system")
        (setf system (concatenate 'string system (if (plusp (length system)) (format nil "~%~%") "")
                                  (content-string (jget m "content"))))))
    (with-output-to-string (o)
      (write-string "<｜begin▁of▁sentence｜>" o)
      (when (and v41 thinking)
        (format o "<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)~%~%"))
      (when (and v41 (not thinking) (plusp (length system)))
        (write-string "<｜System｜>" o))
      (write-string system o)
      (loop for m in msgs for i from 0
            for role = (jget m "role")
            do (cond ((equal role "system"))
                     ((equal role "user")
                      (write-string "<｜User｜>" o) (write-string (content-string (jget m "content")) o)
                      (setf pending-assistant t pending-tool nil))
                     ((member role '("tool" "function") :test #'equal)
                      (unless pending-tool (write-string "<｜User｜>" o))
                      (write-string "<tool_result>" o)
                      (write-string (ds4f-escape-close (content-string (jget m "content")) "</tool_result>") o)
                      (write-string "</tool_result>" o)
                      (setf pending-assistant t pending-tool t))
                     ((equal role "assistant")
                      (when pending-assistant
                        (write-string "<｜Assistant｜>" o)
                        (if (and thinking (or tool-context (> i last-user)))
                            (format o "<think>~A</think>" (or (jget m "reasoning_content") ""))
                            (write-string "</think>" o)))
                      (write-string (content-string (jget m "content")) o)
                      (let ((calls (jget m "tool_calls")))
                        (when (and calls (plusp (length calls))) (write-string (ds4f-dsml-calls calls) o)))
                      (write-string "<｜end▁of▁sentence｜>" o)
                      (setf pending-assistant nil pending-tool nil))))
      (when pending-assistant
        (write-string "<｜Assistant｜>" o)
        (write-string (if thinking "<think>" "</think>") o)))))

(defun ds4f-parse-reply (text thinking)
  "(values content tool-calls reasoning) from the model's reply: with
   thinking, the reasoning is what comes before </think>; tool calls are
   the DSML block's invokes, string parameters unescaped as ds4 does,
   others parsed as JSON."
  (let* ((close (and thinking (search "</think>" text)))
         (reasoning (cond (close (subseq text 0 close)) (thinking text) (t nil)))
         (rest (cond (close (subseq text (+ close 8))) (thinking "") (t text)))
         (start (search "<｜DSML｜tool_calls>" rest))
         (content (string-trim '(#\Space #\Newline #\Tab #\Return) (subseq rest 0 (or start (length rest)))))
         (calls '()))
    (when start
      (let ((pos start) (pend "</｜DSML｜parameter>"))
        (loop
          (let* ((inv (search "<｜DSML｜invoke name=\"" rest :start2 pos))
                 (inv-end (and inv (search "</｜DSML｜invoke>" rest :start2 inv))))
            (unless (and inv inv-end) (return))
            (let* ((name-start (+ inv (length "<｜DSML｜invoke name=\"")))
                   (name-end (search "\"" rest :start2 name-start))
                   (args (make-hash-table :test 'equal))
                   (p (search ">" rest :start2 name-end)))
              (loop
                (let ((ps (search "<｜DSML｜parameter name=\"" rest :start2 p :end2 inv-end)))
                  (unless ps (return))
                  (let* ((ks (+ ps (length "<｜DSML｜parameter name=\"")))
                         (ke (search "\"" rest :start2 ks))
                         (strp (search "string=\"true\"" rest :start2 ke :end2 (search ">" rest :start2 ke)))
                         (vs (1+ (search ">" rest :start2 ke)))
                         (ve (or (search pend rest :start2 vs) inv-end))
                         (raw (subseq rest vs ve)))
                    (setf (gethash (ds4f-attr-unescape (subseq rest ks ke)) args)
                          (if strp
                              (ds4f-unescape-close raw pend)
                              (let ((v (ignore-errors (json-simple:parse raw)))) (if v v raw))))
                    (setf p (+ ve (length pend))))))
              (push (hobj "id" (fresh-id "call_" 8) "type" "function"
                          "function" (hobj "name" (ds4f-attr-unescape (subseq rest name-start name-end))
                                           "arguments" (json-simple:stringify args)))
                    calls)
              (setf pos (+ inv-end (length "</｜DSML｜invoke>"))))))))
    (values content (nreverse calls) reasoning)))

(defun ds4f-attr-unescape (s)
  (let ((out s))
    (loop for (from to) in '(("&quot;" "\"") ("&gt;" ">") ("&lt;" "<") ("&amp;" "&"))
          do (setf out (with-output-to-string (o)
                         (loop with i = 0 while (< i (length out))
                               do (if (and (<= (+ i (length from)) (length out))
                                           (string= from out :start2 i :end2 (+ i (length from))))
                                      (progn (write-string to o) (incf i (length from)))
                                      (progn (write-char (char out i) o) (incf i)))))))
    out))

(defun make-ds4f-stream-filter (emit-content emit-reasoning thinking)
  "A text sink for a DeepSeek reply: reasoning (when THINKING, up to
   </think>) to EMIT-REASONING, then content to EMIT-CONTENT until a DSML
   tool-call block starts (the calls are sent whole once the reply is done).
   A tail that might begin the next tag is held back."
  (let ((pending "") (state (if thinking :reasoning :content)))
    (labels ((tag () (if (eq state :reasoning) "</think>" "<｜DSML｜tool_calls>"))
             (feed (text)
               (unless (eq state :done)
                 (let* ((tag (tag)) (at (search tag text)))
                   (if at
                       (progn (when (plusp at)
                                (funcall (if (eq state :reasoning) emit-reasoning emit-content) (subseq text 0 at)))
                              (setf pending "")
                              (if (eq state :reasoning)
                                  (progn (setf state :content) (feed (subseq text (+ at (length tag)))))
                                  (setf state :done)))
                       (let ((keep (loop for k from (min (1- (length tag)) (length text)) downto 1
                                         when (string= text tag :start1 (- (length text) k) :end2 k)
                                           return k
                                         finally (return 0))))
                         (when (> (length text) keep)
                           (funcall (if (eq state :reasoning) emit-reasoning emit-content)
                                    (subseq text 0 (- (length text) keep))))
                         (setf pending (subseq text (- (length text) keep)))))))))
      (lambda (piece) (feed (concatenate 'string pending piece))))))
