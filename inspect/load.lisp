;;;; load.lisp — load illatio (and SYSTEM, default "illatio/openai") from this
;;;; checkout, finding sibling repos (json-simple, cl-transport) beside it.

(require :asdf)
(defvar *illatio-root* (merge-pathnames "../" (make-pathname :name nil :type nil :defaults *load-truename*)))
(asdf:initialize-source-registry
 `(:source-registry (:directory ,*illatio-root*) (:tree ,(merge-pathnames "../" *illatio-root*))
                    :inherit-configuration))
(handler-bind ((warning #'muffle-warning))
  (asdf:load-system (or (and (boundp 'cl-user::*system*) (symbol-value 'cl-user::*system*))
                        "illatio/openai")))
