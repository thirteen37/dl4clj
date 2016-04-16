(ns dl4clj.models.word2vec.word2vec
  (:require [dl4clj.internal.wrapper-helpers :refer [make-methods]])
  (:import [org.deeplearning4j.models.word2vec Word2Vec$Builder]))

(defmacro build [& {:as args}]
  `(-> (Word2Vec$Builder.)
       ~@(make-methods args)
       .build))

(defn fit [word2vec]
  (.fit word2vec))

(defn words-nearest [word2vec word n]
  (.wordsNearest word2vec word n))
