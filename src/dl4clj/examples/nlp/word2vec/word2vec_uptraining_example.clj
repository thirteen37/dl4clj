(ns dl4clj.examples.nlp.word2vec.word2vec-uptraining-example
  (:require [clojure.java.io :refer [resource]]
            [taoensso.timbre :as timbre :refer (info)])
  (:import [org.canova.api.util ClassPathResource]
           [org.deeplearning4j.models.embeddings WeightLookupTable]
           [org.deeplearning4j.models.embeddings.inmemory InMemoryLookupTable$Builder]
           [org.deeplearning4j.models.embeddings.loader WordVectorSerializer]
           [org.deeplearning4j.models.word2vec VocabWord Word2Vec$Builder]
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
            table (-> (InMemoryLookupTable$Builder.)
                      (.vectorLength 100)
                      (.useAdaGrad false)
                      (.cache cache)
                      (.lr 0.025)
                      .build)]
        (info "Building model....")
        (let [vec (-> (Word2Vec$Builder.)
                      (.minWordFrequency 5)
                      (.iterations 1)
                      (.epochs 1)
                      (.layerSize 100)
                      (.seed 42)
                      (.windowSize 5)
                      (.iterate iter)
                      (.tokenizerFactory t)
                      (.lookupTable table)
                      (.vocabCache cache)
                      .build)]
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