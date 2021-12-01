# import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_eager_execution()


import sentencepiece as spm
import tensorflow_hub as hub
import numpy as np


class USE_lite:

    @tf.function
    def __init__(self):
        with tf.Session() as sess:
            module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
            spm_path = sess.run(module(signature="spm_path"))
            # spm_path now contains a path to the SentencePiece model stored inside the
            # TF-Hub module

        sp = spm.SentencePieceProcessor()
        sp.Load(spm_path)

        self.input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
        self.encodings = module(
            inputs=dict(
                values=self.input_placeholder.values,
                indices=self.input_placeholder.indices,
                dense_shape=self.input_placeholder.dense_shape))

        with tf.Session() as sess:
            spm_path = sess.run(module(signature="spm_path"))

        self.sp = spm.SentencePieceProcessor()
        with tf.io.gfile.GFile(spm_path, mode="rb") as f:
            self.sp.LoadFromSerializedProto(f.read())
        print("SentencePiece model loaded at {}.".format(spm_path))

    def process_to_IDs_in_sparse_format(self, sentences):
        # An utility method that processes sentences with the sentence piece processor
        # 'sp' and returns the results in tf.SparseTensor-similar format:
        # (values, indices, dense_shape)
        ids = [self.sp.EncodeAsIds(x) for x in sentences]
        max_len = max(len(x) for x in ids)
        dense_shape = (len(ids), max_len)
        values = [item for sublist in ids for item in sublist]
        indices = [[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]
        return values, indices, dense_shape

    @tf.function
    def __call__(self, *args, **kwargs):
        values, indices, dense_shape = self.process_to_IDs_in_sparse_format(args[0])

        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(
                self.encodings,
                feed_dict={self.input_placeholder.values: values,
                           self.input_placeholder.indices: indices,
                           self.input_placeholder.dense_shape: dense_shape})

            return np.array(message_embeddings).tolist()

            # for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
            #     print("Message: {}".format(messages[i]))
            #     print("Embedding size: {}".format(len(message_embedding)))
            #     message_embedding_snippet = ", ".join(
            #         (str(x) for x in message_embedding[:3]))
            #     print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
