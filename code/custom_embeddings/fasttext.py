import pickle
import numpy as np

from custom_embeddings.fasttext_utils import CompressedEmbeddingBase


def get_subwords(word, nmin=3, nmax=6):
    """
    Get a list of n-grams in <word> with n between <nmin> and <nmax>.
    :param word: input word
    :param nmin: lower bound on n
    :param nmax: upper bound on n
    :return: list of n-grams in <word>
    """

    word = f'<{word}>'
    return [word[i:j] for i in range(len(word))
            for j in range(i + nmin, 1 + min(i + nmax, len(word)))]


class FastText(CompressedEmbeddingBase):
    """
    Compressed FastText embedding algorithm.
    :param path: where to look for model file
    :param model: model file (model.pickle)
    """

    def __init__(self, model_data, **kwargs):
        super(FastText, self).__init__('FastText', model_data)
        self.alg = 'FastText'


class FastTextSW(CompressedEmbeddingBase):
    """
    Compressed FastText embedding algorithm with sub-word information.
    :param path: where to look for model files
    :param model: model file (model.pickle)
    :param sw: sub-word file (model_sw.pickle)
    """

    def __init__(self, model_data_path, sw_data_path, **kwargs):
        def load_model_data(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            return data
        model_data = load_model_data(model_data_path)
        sw_data = load_model_data(sw_data_path)

        super(FastTextSW, self).__init__('FastTextSW', model_data)

        self.lang = "en"
        assert self.dim == sw_data['dim']
        self._sw_cb = sw_data['codebook']
        self._sw_vocab = sw_data['vectors']
        self._sw_norms = sw_data['norms']
        self.output_dim = self.dim

    def oov_vector(self, word, norm=None):
        """
        Function for handling out-of-vocabulary words.
        Creates OOV embedding from sub-word embeddings.
        :param word: oov word
        :param norm: (inherited, not used)
        :return: oov word embedding
        """

        if word in self._oov_vectors:
            return self._oov_vectors[word]

        sws = get_subwords(word, nmin=3, nmax=4)
        vectors = []
        sws_found = False
        for sw in sws:
            try:
                vec = self.decode_sw_vec(self._sw_vocab[sw]) * self._sw_norms[sw]
                vectors.append(vec)
                sws_found = True
            except KeyError:
                vectors.append(np.zeros((self.dim,)))

        if not sws_found:
            return super().oov_vector(word)

        vec = np.average(vectors, axis=0)
        # self._oov_vectors[word] = vec
        return vec

    def decode_sw_vec(self, vec):
        """
        Decode compressed sub-word vector.
        :param vec: numpy vector
        :return: decoded numpy vector
        """

        out = [self._sw_cb[idx] for idx in vec]
        return np.concatenate(out)
