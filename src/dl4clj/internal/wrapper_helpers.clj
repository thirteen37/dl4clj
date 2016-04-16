(ns dl4clj.internal.wrapper-helpers
  (:require [camel-snake-kebab.core :refer [->camelCase]]))

(defn- make-method [method-keyword]
  (symbol (str "." (name (->camelCase method-keyword)))))

(defn make-methods [method-args]
  (map (fn [[method arg]] (list (make-method method) arg)) method-args))

