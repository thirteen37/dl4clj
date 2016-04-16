(ns dl4clj.models.embeddings.inmemory.in-memory-lookup-table
  (:require [dl4clj.internal.wrapper-helpers :refer [make-methods]])
  (:import [org.deeplearning4j.models.embeddings.inmemory InMemoryLookupTable$Builder]))

(defmacro build [& {:as args}]
  `(-> (InMemoryLookupTable$Builder.)
       ~@(make-methods args)
       .build))
