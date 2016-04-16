(ns dl4clj.models.word2vec.word2vec
  (:require [camel-snake-kebab.core :refer [->camelCase]])
  (:import [org.deeplearning4j.models.word2vec Word2Vec$Builder]))

(defmacro build [& {:as args}]
  `(.build
    (doto (Word2Vec$Builder.)
      ~@(map (fn [[method arg]] (list (symbol (str "." (name (->camelCase method)))) arg))
             args))))
