;;;; pool.lisp — a fixed set of worker threads that run one job at a time.
;;;
;;; A token's forward pass is a few hundred short parallel steps (each
;;; matrix-vector product, each attention), so threads are made once and
;;; woken per step; making a thread per step would cost more than the step.
;;; A job is (FN INDEX COUNT): each of COUNT participants, the caller being
;;; one of them, does its INDEXth share.

(in-package #:illatio)

;;; Between two steps of a token the workers SPIN on the job generation (an
;;; atomic word) rather than sleep on the condition variable: a wake through
;;; the kernel and a join cost ~38 us a step, which at ~290 steps a token was
;;; 43% of a Qwen3-4B decode on ten threads.  A worker that has spun +SPIN+
;;; rounds with nothing new -- between tokens, while a user types -- parks on
;;; the condition variable as before.  The caller spins for the stragglers.
;;; (modus's thread shim has no atomic words; there the pool parks always.)

(defconstant +spin+ 200000
  "Spin rounds before a worker parks: a few hundred microseconds, longer
   than any gap between two steps of a token.")

(defstruct (pool (:constructor %make-pool))
  (threads '())
  (size 1 :type fixnum)                 ; participants, caller included
  (lock (sb-thread:make-mutex :name "illatio pool"))
  (wake (sb-thread:make-waitqueue))
  (idle (sb-thread:make-waitqueue))
  (job nil)
  (generation 0 :type #+sbcl sb-ext:word #-sbcl fixnum)
  (pending 0 :type #+sbcl sb-ext:word #-sbcl fixnum)
  (sleepers 0 :type #+sbcl sb-ext:word #-sbcl fixnum) ; workers parked on WAKE
  (failure nil)
  (stop nil))

(defun default-thread-count ()
  "The machine's core count, which is what a CPU-bound matvec wants."
  (or #+sbcl
      (ignore-errors
       (let ((s (with-output-to-string (o)
                  (sb-ext:run-program "/usr/sbin/sysctl" '("-n" "hw.perflevel0.physicalcpu")
                                      :output o :error nil))))
         (let ((n (parse-integer s :junk-allowed t)))
           (and n (plusp n) n))))
      (ignore-errors
       (with-open-file (s "/proc/cpuinfo")
         (loop for line = (read-line s nil) while line
               count (eql 0 (search "processor" line)))))
      4))

(defun make-pool (&optional (size (default-thread-count)))
  (let ((p (%make-pool :size (max 1 size))))
    (setf (pool-threads p)
          (loop for i from 1 below (pool-size p)
                ;; the generation a worker starts from is read here, not when
                ;; it first runs: a job posted before then would be taken for
                ;; one it had seen, and never done
                collect (let ((index i) (gen (pool-generation p)))
                          (sb-thread:make-thread (lambda () (pool-worker p index gen))
                                                 :name (format nil "illatio worker ~D" i)))))
    p))

(defun run-job (p job index)
  "JOB's share INDEX; an error is kept for the caller, not raised here."
  (let ((err (handler-case (progn (funcall job index (pool-size p)) nil)
               (error (e) e))))
    (when err
      (sb-thread:with-mutex ((pool-lock p))
        (unless (pool-failure p) (setf (pool-failure p) err))))))

#+sbcl
(defun pool-worker (p index seen)
  ;; C's float semantics, not SBCL's traps: an exp that overflows is
  ;; infinity, and a sigmoid or silu of it the 0 or 1 ggml computes.  The
  ;; mode is per thread, so each worker sets its own (FORWARD the caller's).
  (sb-int:with-float-traps-masked (:overflow :inexact)
    (pool-worker-1 p index seen)))

#+sbcl
(defun pool-worker-1 (p index seen)
  (progn
    (loop
      ;; spin for the next generation; park if it is long in coming
      (let ((n 0))
        (declare (type fixnum n))
        (loop until (or (/= (pool-generation p) seen) (pool-stop p) (> (incf n) +spin+))
              do (sb-ext:spin-loop-hint)))
      (when (and (= (pool-generation p) seen) (not (pool-stop p)))
        (sb-thread:with-mutex ((pool-lock p))
          ;; counted BEFORE the generation is read again: either the caller
          ;; sees a sleeper and broadcasts, or this read sees its new generation
          (sb-ext:atomic-incf (pool-sleepers p))
          (sb-thread:barrier (:memory))
          (loop while (and (= (pool-generation p) seen) (not (pool-stop p)))
                do (sb-thread:condition-wait (pool-wake p) (pool-lock p)))
          (sb-ext:atomic-decf (pool-sleepers p))))
      (when (pool-stop p) (return))
      (setf seen (pool-generation p))
      (sb-thread:barrier (:read))
      (run-job p (pool-job p) index)
      (sb-thread:barrier (:write))
      (sb-ext:atomic-decf (pool-pending p)))))

#-sbcl
(defun pool-worker (p index &optional (seen 0))
  (progn
    (loop
      (let (job)
        (sb-thread:with-mutex ((pool-lock p))
          (loop until (or (pool-stop p) (/= seen (pool-generation p)))
                do (sb-thread:condition-wait (pool-wake p) (pool-lock p)))
          (when (pool-stop p) (return))
          (setf seen (pool-generation p)
                job (pool-job p)))
        (run-job p job index)
        (sb-thread:with-mutex ((pool-lock p))
          (when (zerop (decf (pool-pending p)))
            (sb-thread:condition-broadcast (pool-idle p))))))))

(defun pool-run (p fn)
  "Run (FN INDEX COUNT) on every participant of P and return when all are done."
  (declare (type function fn))
  (when (= (pool-size p) 1)
    (return-from pool-run (funcall fn 0 1)))
  #+sbcl
  (progn
    (setf (pool-job p) fn
          (pool-failure p) nil
          (pool-pending p) (1- (pool-size p)))
    (sb-thread:barrier (:write))
    (sb-ext:atomic-incf (pool-generation p))
    (sb-thread:barrier (:memory))
    (when (plusp (pool-sleepers p))
      (sb-thread:with-mutex ((pool-lock p))
        (sb-thread:condition-broadcast (pool-wake p))))
    (funcall fn 0 (pool-size p))
    (loop until (zerop (pool-pending p)) do (sb-ext:spin-loop-hint))
    (sb-thread:barrier (:read)))
  #-sbcl
  (progn
    (sb-thread:with-mutex ((pool-lock p))
      (setf (pool-job p) fn
            (pool-failure p) nil
            (pool-pending p) (1- (pool-size p)))
      (incf (pool-generation p))
      (sb-thread:condition-broadcast (pool-wake p)))
    (funcall fn 0 (pool-size p))
    (sb-thread:with-mutex ((pool-lock p))
      (loop until (zerop (pool-pending p))
            do (sb-thread:condition-wait (pool-idle p) (pool-lock p)))))
  (let ((e (pool-failure p)))
    (when e (error e))))

(defun pool-stop! (p)
  (sb-thread:with-mutex ((pool-lock p))
    (setf (pool-stop p) t)
    (sb-thread:condition-broadcast (pool-wake p)))
  (mapc #'sb-thread:join-thread (pool-threads p))
  (setf (pool-threads p) '()))

(declaim (inline share))
(defun share (n index count)
  "The [start, end) slice of N items that participant INDEX of COUNT does."
  (declare (type fixnum n index count))
  (values (floor (* n index) count) (floor (* n (1+ index)) count)))

(defmacro parallel-for ((var n pool) &body body)
  "Run BODY for each VAR in [0, N), the range split across POOL."
  (let ((i (gensym)) (c (gensym)) (s (gensym)) (e (gensym)) (nn (gensym)))
    `(let ((,nn ,n))
       (declare (type fixnum ,nn))
       (pool-run ,pool
                 (lambda (,i ,c)
                   (declare (type fixnum ,i ,c))
                   (multiple-value-bind (,s ,e) (share ,nn ,i ,c)
                     (declare (type fixnum ,s ,e))
                     (loop for ,var of-type fixnum from ,s below ,e
                           do (progn ,@body))))))))
