(ns dl4clj.examples.nlp.word2vec.word2vec-raw-text-example
  (:require [taoensso.timbre :as timbre :refer (info)])
  (:import [org.canova.api.util ClassPathResource]
           [org.deeplearning4j.models.embeddings.loader WordVectorSerializer]
           [org.deeplearning4j.models.word2vec Word2Vec$Builder]
           [org.deeplearning4j.text.sentenceiterator BasicLineIterator SentenceIterator]
           [org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor]
           [org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory TokenizerFactory]
           [org.deeplearning4j.ui UiServer]))

(defn -main [& args]
  (let [file-path (-> "raw_sentences.txt" ClassPathResource. .getFile .getAbsolutePath)]
    (info "Load & Vectorize Sentences....")
    (let [iter (BasicLineIterator. file-path)
          t (DefaultTokenizerFactory.)] (.setTokenPreProcessor t (CommonPreprocessor.))
         (info "Building model....")
         (let [vec (-> (Word2Vec$Builder.)
                       (.minWordFrequency 5)
                       (.iterations 1)
                       (.layerSize 100)
                       (.seed 42)
                       (.windowSize 5)
                       (.iterate iter)
                       (.tokenizerFactory t)
                       .build)]
           (info "Fitting Word2Vec model....")
           (.fit vec)
           (info "Writing word vectors to text file....")
           (WordVectorSerializer/writeWordVectors vec "pathToWriteto.txt")
           (info "Closest Words:")
           (println (.wordsNearest vec "day" 10))
           (let [server (UiServer/getInstance)]
             (println "Started on port " (.getPort server)))))))
