(ns dl4clj.models.embeddings.loader.word-vector-serializer
  (:import [org.deeplearning4j.models.embeddings.loader WordVectorSerializer]))

(defn write-word-vectors [word-vectors file-path]
  (WordVectorSerializer/writeWordVectors word-vectors file-path))

(defn write-full-model [word-vectors file-path]
  (WordVectorSerializer/writeFullModel word-vectors file-path))

(defn load-full-model [file-path]
  (WordVectorSerializer/loadFullModel file-path))
