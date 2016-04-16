(ns dl4clj.examples.nlp.word2vec.word2vec-uptraining-example
  (:require [clojure.java.io :refer [resource]]
            [taoensso.timbre :as timbre :refer (info)]
            [dl4clj.models.word2vec.word2vec :as word2vec]
            [dl4clj.models.embeddings.inmemory.in-memory-lookup-table :as imlt])
  (:import [org.canova.api.util ClassPathResource]
           [org.deeplearning4j.models.embeddings WeightLookupTable]
           [org.deeplearning4j.models.embeddings.loader WordVectorSerializer]
           [org.deeplearning4j.models.word2vec.wordstore.inmemory InMemoryLookupCache]
           [org.deeplearning4j.text.sentenceiterator BasicLineIterator SentenceIterator]
           [org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor]
           [org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory TokenizerFactory]))

(defn -main [& args]
  (let [file-path (-> "raw_sentences.txt" resource .getFile)]
    (info "Load & Vectorize Sentences....")
    (let [iter (BasicLineIterator. file-path)
          t (DefaultTokenizerFactory.)]
      (.setTokenPreProcessor t (CommonPreprocessor.))
      (let [cache (InMemoryLookupCache.)
            table (imlt/build :vector-length 100
                              :use-ada-grad false
                              :cache cache
                              :lr 0.025)]
        (info "Building model....")
        (let [vec (word2vec/build :min-word-frequency 5
                                  :iterations 1
                                  :epochs 1
                                  :layer-size 100
                                  :seed 42
                                  :window-size 5
                                  :iterate iter
                                  :tokenizer-factory t
                                  :lookup-table table
                                  :vocab-cache cache)]
          (info "Fitting Word2Vec model....")
          (.fit vec)
          (info "Closest words to 'day' on 1st run:" (.wordsNearest vec "day" 10))
          (WordVectorSerializer/writeFullModel vec "pathToSaveModel.txt")
          (let [word2vec (WordVectorSerializer/loadFullModel "pathToSaveModel.txt")
                iterator (BasicLineIterator. file-path)
                tokenizerFactory (DefaultTokenizerFactory.)]
            (.setTokenPreProcessor tokenizerFactory (CommonPreprocessor.))
            (.setTokenizerFactory word2vec tokenizerFactory)
            (.setSentenceIter word2vec iterator)
            (info "Word2vec uptraining...")
            (.fit word2vec)
            (info "Closest words to 'day' on 2nd run:" (.wordsNearest word2vec "day" 10))))))))
