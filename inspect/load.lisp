;;;; load.lisp — load illatio (and SYSTEM, default "illatio/openai") from this
;;;; checkout, finding sibling repos (json-simple, cl-transport) beside it.

(require :asdf)
(defvar *illatio-root* (merge-pathnames "../" (make-pathname :name nil :type nil :defaults *load-truename*)))
;; modus's source registry is its own tree, which an INITIALIZE-SOURCE-REGISTRY puts ahead
;; of everything (measured: a release tree's cl-transport then wins over the sibling
;; checkout), so there the checkout and its siblings go on the central registry instead
(let* ((here (make-pathname :name nil :type nil :version nil :defaults *load-truename*))
       (root (make-pathname :directory (butlast (pathname-directory here)) :defaults here))
       (above (butlast (pathname-directory root))))
  (dolist (sibling '("json-simple" "cl-transport"))
    (let ((d (make-pathname :directory (append above (list sibling)) :defaults root)))
      (push d asdf:*central-registry*)))   ; harmless when absent
  (push root asdf:*central-registry*))
#-modus
(progn
 (asdf:initialize-source-registry
 `(:source-registry (:directory ,*illatio-root*) (:tree ,(merge-pathnames "../" *illatio-root*))
                    :inherit-configuration)))
(handler-bind ((warning #'muffle-warning))
  (asdf:load-system (or (and (boundp 'cl-user::*system*) (symbol-value 'cl-user::*system*))
                        "illatio/openai")))
