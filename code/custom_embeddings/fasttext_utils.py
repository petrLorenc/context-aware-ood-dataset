import pickle
from abc import ABC, abstractmethod

import numpy as np
import re

from custom_embeddings.twitter_tokenizer import TweetTokenizer


def format_brackets(token):
    """
    Formats bracket tokens.
    :param token: string token
    :return: formatted token
    """

    if token == '-LRB-':
        token = '('
    elif token == '-RRB-':
        token = ')'
    elif token == '-RSB-':
        token = ']'
    elif token == '-LSB-':
        token = '['
    elif token == '-LCB-':
        token = '{'
    elif token == '-RCB-':
        token = '}'
    return token


def tokenize(tokenizer, sentence, to_lower=True):
    """
    Tokenize sentence.
    :param tokenizer: a tokenizer implementing the NLTK tokenizer interface
    :param sentence: a string to be tokenized
    :param to_lower: lowercase input
    :return: tokenized string
    """

    sentence = sentence.strip()
    sentence = ' '.join([format_brackets(x) for x in tokenizer.tokenize(sentence)])
    if to_lower:
        sentence = sentence.lower()
    sentence = re.sub(r"((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))", "<url>", sentence)
    sentence = re.sub(r"(@[^\s]+)", "<user>", sentence)
    sentence = re.sub(r'\{ ([^\{\}]+) \}', r'{\1}', sentence)  # entity tokens
    filter(lambda word: r" " not in word, sentence)
    return sentence


def generate_batches(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class EmbeddingBase:
    """
    Base class for embedding algorithms.
    This class needs to be derived.
    """

    def __init__(self, alg: str, dim: int):
        super(EmbeddingBase, self).__init__()
        self.tokenizer = TweetTokenizer()
        self.alg = alg
        self.dim = dim
        self._oov_vectors = {}

    def vector_batch(self, batch: list, normalize=True):
        return np.asarray([self.vector(x, normalize) for x in batch]).reshape(-1, self.dim)

    def vector(self, s, normalize=True):
        """
        Transform sentence into vector representation.
        :param s: sentence
        :param normalize: normalize to unit length
        :return: transformed sentence
        """

        if not s.strip():
            return np.zeros((self.dim,))

        s = tokenize(tokenizer=self.tokenizer, sentence=s, to_lower=True)
        vec = self.vector_preprocessed(s)

        if normalize:
            vec /= np.linalg.norm(vec)

        return vec

    def __call__(self, *args, **kwargs):
        return [self.vector(s) for s in args[0]]

    def process(self, data):
        def generator():
            for sentence in data:
                yield self.vector_preprocessed(' '.join(sentence).lower(), average=False)

        return generator

    @abstractmethod
    def vector_preprocessed(self, sentence, average=True):
        """
        Compute embedding vector for a preprocessed sentence.
        :param sentence: preprocessed sentence
        :param average: if True (default) returns average of the vectors for each words, otherwise returns sequence of vectors
        :return: encoded sentence
        """
        pass

    def oov_vector(self, word, norm=30.0):
        """
        Function for handling out-of-vocabulary words.
        Creates a random embedding and stores it for reuse with the same OOV word.
        :param word: OOV word
        :param norm: norm of the resulting vector (weight in sentence)
        :return: OOV word embedding
        """

        if word in self._oov_vectors:
            return self._oov_vectors[word]

        vec = np.random.rand(self.dim) - 0.5
        vec /= np.linalg.norm(vec) / norm

        self._oov_vectors[word] = vec
        return vec

    def serialize(self) -> bytes:
        return pickle.dumps(self._oov_vectors)

    def deserialize(self, data: bytes):
        self._oov_vectors = pickle.loads(data)


class CompressedEmbeddingBase(EmbeddingBase):
    """
    Base class for compressed embedding algorithms.
    :param path: where to look for model file
    :param model: model file (model.pickle)
    """

    def __init__(self, alg: str, model_data):
        super(CompressedEmbeddingBase, self).__init__(alg, model_data['dim'])

        self._cb = model_data['codebook']
        self._vocab = model_data['vectors']
        self._norms = model_data['norms']

    def vector_preprocessed(self, sentence, average=True):
        """
        Compute embedding vector for a preprocessed sentence.
        :param sentence: preprocessed sentence
        :param average: if True (default) returns average of the vectors for each words, otherwise returns sequence of vectors
        :return: encoded sentence
        """

        tokens = sentence.split()
        vectors = []

        for token in tokens:
            try:
                vec = self.decode_vec(self._vocab[token]) * self._norms[token]
                vectors.append(vec)
            except KeyError:
                vectors.append(self.oov_vector(token))

        if average:
            return np.average(vectors, axis=0)
        else:
            return vectors

    def decode_vec(self, vec):
        """
        Decode compressed vector.
        :param vec: numpy vector
        :return: decoded numpy vector
        """

        out = [self._cb[idx] for idx in vec]
        return np.concatenate(out)
