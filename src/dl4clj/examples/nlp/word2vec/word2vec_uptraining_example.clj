(ns dl4clj.examples.nlp.word2vec.word2vec-uptraining-example
  (:require [clojure.java.io :refer [resource]]
            [taoensso.timbre :as timbre :refer (info)]
            [dl4clj.models.embeddings.inmemory.in-memory-lookup-table :as imlt]
            [dl4clj.models.embeddings.loader.word-vector-serializer :refer [load-full-model write-full-model]]
            [dl4clj.models.word2vec.word2vec :refer [fit words-nearest] :as word2vec])
  (:import [org.deeplearning4j.models.word2vec.wordstore.inmemory InMemoryLookupCache]
           [org.deeplearning4j.text.sentenceiterator BasicLineIterator]
           [org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor]
           [org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory]))

;;; This is simple example for model weights update after initial vocab building.
;;; If you have built your w2v model, and some time later you've decided that it can be additionally trained over new corpus, here's an example how to do it.
;;;
;;; PLEASE NOTE: At this moment, no new words will be added to vocabulary/model. Only weights update process will be issued. It's often called "frozen vocab training".
(defn -main [& args]
  ;; Initial model training phase
  (let [file-path (-> "raw_sentences.txt" resource .getFile)]
    (info "Load & Vectorize Sentences....")
    (let [;; Strip white space before and after for each line
          iter (BasicLineIterator. file-path)
          ;; Split on white spaces in the line to get words
          t (DefaultTokenizerFactory.)]
      (.setTokenPreProcessor t (CommonPreprocessor.))
      ;; manual creation of VocabCache and WeightLookupTable usually isn't necessary
      ;; but in this case we'll need them
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
          (fit vec)
          (info "Closest words to 'day' on 1st run:" (words-nearest vec "day" 10))
          ;; at this moment we're supposed to have model built, and it can be saved for future use.
          (write-full-model vec "pathToSaveModel.txt"))))
    ;; Let's assume that some time passed, and now we have new corpus to be used to weights update.
    ;; Instead of building new model over joint corpus, we can use weights update mode.
    (let [word2vec (load-full-model "pathToSaveModel.txt")
          ;; PLEASE NOTE: after model is restored, it's still required to set SentenceIterator and TokenizerFactory, if you're going to train this model
          iterator (BasicLineIterator. file-path)
          tokenizerFactory (DefaultTokenizerFactory.)]
      (.setTokenPreProcessor tokenizerFactory (CommonPreprocessor.))
      (.setTokenizerFactory word2vec tokenizerFactory)
      (.setSentenceIter word2vec iterator)
      (info "Word2vec uptraining...")
      (fit word2vec)
      (info "Closest words to 'day' on 2nd run:" (words-nearest word2vec "day" 10))
      ;; Model can be saved for future use now
      )))
