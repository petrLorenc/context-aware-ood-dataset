imports = []
LIMIT_NUM_SENTS = None

from evaluate.one_class_based import evaluate

from models.one_class_svm import OneClassSVMModel
import tensorflow_hub as hub
from custom_embeddings.fasttext import FastTextSW
from sklearn.svm import OneClassSVM

imports.append({
    "evaluation_fn": evaluate,
    "algorithms": [OneClassSVMModel(OneClassSVM(gamma='auto'))],
    "evaluation_fn_arg": {"limit_num_sents": LIMIT_NUM_SENTS},
    "embeddings": [
        {"embedding_name": "fasttext", "embedding_model": FastTextSW(model_data_path="../data/embeddings/wiki_en_50k.pickle", sw_data_path="../data/embeddings/wiki_en_sw_100k.pickle")}
        # {"embedding_name": "use4", "embedding_model": hub.load("/media/petrlorenc/Data/universal-sentence-encoder_4")}
        # {"embedding_name": "use5", "embedding_model": hub.load("/media/petrlorenc/Data/universal-sentence-encoder_5")}
        # {"embedding_name": "use4_finetuned", "embedding_model": hub.load("/media/petrlorenc/Data/universal-sentence-encoder_fine")}

    ],
    "test_keys": [""]
    # "test_keys": ["", "emoji", "spaces", "insert", "uppercase", "regex", "regex_2", "swap", "random_char"] # all chosen to study robustness
    # "test_keys": # enriching_keys = ["", "random_char", "emoji", "spaces", "deletion", "insert", "swap", "uppercase", "regex", "end_char", "end_word"] # all
})