;;;; pool.lisp — a fixed set of worker threads that run one job at a time.
;;;
;;; A token's forward pass is a few hundred short parallel steps (each
;;; matrix-vector product, each attention), so threads are made once and
;;; woken per step; making a thread per step would cost more than the step.
;;; A job is (FN INDEX COUNT): each of COUNT participants, the caller being
;;; one of them, does its INDEXth share.

(in-package #:illatio)

(defstruct (pool (:constructor %make-pool))
  (threads '())
  (size 1 :type fixnum)                 ; participants, caller included
  (lock (sb-thread:make-mutex :name "illatio pool"))
  (wake (sb-thread:make-waitqueue))
  (idle (sb-thread:make-waitqueue))
  (job nil)
  (generation 0 :type fixnum)
  (pending 0 :type fixnum)
  (failure nil)
  (stop nil))

(defun default-thread-count ()
  "The machine's core count, which is what a CPU-bound matvec wants."
  (or (ignore-errors
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
                collect (let ((index i))
                          (sb-thread:make-thread (lambda () (pool-worker p index))
                                                 :name (format nil "illatio worker ~D" i)))))
    p))

(defun pool-worker (p index)
  (let ((seen 0))
    (loop
      (let (job)
        (sb-thread:with-mutex ((pool-lock p))
          (loop until (or (pool-stop p) (/= seen (pool-generation p)))
                do (sb-thread:condition-wait (pool-wake p) (pool-lock p)))
          (when (pool-stop p) (return))
          (setf seen (pool-generation p)
                job (pool-job p)))
        (let ((err (handler-case (progn (funcall job index (pool-size p)) nil)
                     (error (e) e))))
          (sb-thread:with-mutex ((pool-lock p))
            (when (and err (not (pool-failure p))) (setf (pool-failure p) err))
            (when (zerop (decf (pool-pending p)))
              (sb-thread:condition-broadcast (pool-idle p)))))))))

(defun pool-run (p fn)
  "Run (FN INDEX COUNT) on every participant of P and return when all are done."
  (declare (type function fn))
  (if (= (pool-size p) 1)
      (funcall fn 0 1)
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
                do (sb-thread:condition-wait (pool-idle p) (pool-lock p)))
          (let ((e (pool-failure p)))
            (when e (error e)))))))

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
