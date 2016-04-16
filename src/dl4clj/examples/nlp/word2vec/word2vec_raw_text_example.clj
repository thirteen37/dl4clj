(ns dl4clj.examples.nlp.word2vec.word2vec-raw-text-example
  (:require [clojure.java.io :refer [resource]]
            [taoensso.timbre :as timbre :refer (info)]
            [dl4clj.models.embeddings.loader.word-vector-serializer :refer [write-word-vectors]]
            [dl4clj.models.word2vec.word2vec :refer [fit words-nearest] :as word2vec])
  (:import [org.deeplearning4j.models.embeddings.loader WordVectorSerializer]
           [org.deeplearning4j.text.sentenceiterator BasicLineIterator SentenceIterator]
           [org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor]
           [org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory TokenizerFactory]
           [org.deeplearning4j.ui UiServer]))

(defn -main [& args]
  (let [file-path (-> "raw_sentences.txt" resource .getFile)]
    (info file-path)
    (info "Load & Vectorize Sentences....")
    (let [iter (BasicLineIterator. file-path)
          t (DefaultTokenizerFactory.)]
      (.setTokenPreProcessor t (CommonPreprocessor.))
      (info "Building model....")
      (let [vec (word2vec/build :min-word-frequency 5
                                :iterations 1
                                :layer-size 100
                                :seed 42
                                :window-size 5
                                :iterate iter
                                :tokenizer-factory t)]
        (info "Fitting Word2Vec model....")
        (fit vec)
        (info "Writing word vectors to text file....")
        (write-word-vectors vec "pathToWriteto.txt")
        (info "Closest Words:")
        (println (words-nearest vec "day" 10))
        (let [server (UiServer/getInstance)]
          (println "Started on port " (.getPort server)))))))
