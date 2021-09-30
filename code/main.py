from utils import print_results, cross_val_evaluate
from CosineSimilarity import CosineSimilarity
from NeuralNets import BaselineNNExtraLayer

import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

RANDOM_SELECTION = False  # am I testing using the random selection of IN intents?
repetitions = 5  # number of evaluations when using random selection
LIMIT_NUM_SENTS = None  # either None (i.e. no limit) or int with value > 0 (i.e. maximal number of sentences per class).
# LIMIT_NUM_SENTS is ignored when RANDOM_SELECTION is True

imports = []

# ------------------------------------------------------------
from ood_train import evaluate

imports.append((evaluate, [
    BaselineNNExtraLayer(),
    LogisticRegression(class_weight="balanced", max_iter=2000)
]))
# ------------------------------------------------------------
from ood_threshold import evaluate
#
imports.append((evaluate, [
    CosineSimilarity()
]))

dataset_name = 'HIDDEN'
categories = ['animals', 'books', 'education', 'fashion', 'food', 'habits']
              # 'movies', 'music', 'science',
              # 'smalltalk', 'sports', 'travel']  # all
# ------------------------------------------------------------



time_pretraining = None  # keeps track of pre-training time in seconds (if there's ADB pre-training)

embedding_functions = {}  # uncomment them one by one when measuring memory usage or pre-training time
# embedding_functions['use_dan'] = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embedding_functions['use_dan'] = hub.load("/media/petrlorenc/Data/universal-sentence-encoder_4")
# embedding_functions['use_tran'] = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
# embedding_functions['use_tran'] = hub.load("/media/petrlorenc/Data/universal-sentence-encoder_5")
# embedding_functions['sbert'] = SentenceTransformer('stsb-roberta-base').encode

for i in imports:
    evaluate = i[0]

    for emb_name, embed_f in embedding_functions.items():
        for model in i[1]:
            model_name = type(model).__name__

            results_dct, emb_name = cross_val_evaluate(categories, evaluate, model, model_name, emb_name,
                                                           embed_f, LIMIT_NUM_SENTS)

            print_results(dataset_name, model_name, emb_name, results_dct)