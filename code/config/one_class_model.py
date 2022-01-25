imports = []
LIMIT_NUM_SENTS = None

import os
from utils.dataset.generate import DatasetType, DatasetReturnType
from evaluate.one_class_based import evaluate

from constants import ROOT_DIR, TENSORFLOW_ROOT, BERT_ROOT, HUGGING_FACE_ROOT
from custom_embeddings.huggingface import HuggingFaceModel
from custom_embeddings.BERT_style import TensorflowBERT
from models.one_class_svm import OneClassSklearnModel
from sklearn.mixture import GaussianMixture
import tensorflow_hub as hub
from custom_embeddings.fasttext import FastTextSW
from sklearn.svm import OneClassSVM
from constants import TENSORFLOW_ROOT, ROOT_DIR

APPROACH = "one_class_model"

imports.append({
    "debug": False,
    "evaluation_fn": evaluate,
    "algorithms": [OneClassSklearnModel(OneClassSVM(gamma='auto'))],
    # "algorithms": [OneClassSklearnModel(GaussianMixture(n_components=10, covariance_type='full', n_init=3))],
    "evaluation_fn_arg": {"limit_num_sents": LIMIT_NUM_SENTS},
    "embeddings": [
        # {"embedding_name": "fasttext", "embedding_model": FastTextSW(model_data_path="../data/embeddings/wiki_en_50k.pickle",
        #                                                              sw_data_path="../data/embeddings/wiki_en_sw_100k.pickle")},
        {"embedding_name": "use4", "embedding_model": hub.load(os.path.join(TENSORFLOW_ROOT, "universal-sentence-encoder-4"))},
        {"embedding_name": "use5", "embedding_model": hub.load(os.path.join(TENSORFLOW_ROOT, "universal-sentence-encoder-5"))},
        # {"embedding_name": "use4_finetuned", "embedding_model": hub.load(os.path.join(TENSORFLOW_ROOT,"universal-sentence-encoder-finetuned-4"))},
        # {"embedding_name": "use5-finetuned", "embedding_model": hub.load(os.path.join(TENSORFLOW_ROOT,"universal-sentence-encoder-finetuned-5"))},
        {"embedding_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "embedding_model": HuggingFaceModel(
            tokenizer_path=os.path.join(HUGGING_FACE_ROOT, "paraphrase-multilingual-MiniLM-L12-v2"),
            model_path=os.path.join(HUGGING_FACE_ROOT, "paraphrase-multilingual-MiniLM-L12-v2"))},
        {"embedding_name": "sentence-transformers/all-mpnet-base-v2", "embedding_model": HuggingFaceModel(
            tokenizer_path=os.path.join(HUGGING_FACE_ROOT, "all-mpnet-base-v2"),
            model_path=os.path.join(HUGGING_FACE_ROOT, "all-mpnet-base-v2"))},
        {"embedding_name": "sentence-transformers/paraphrase-MiniLM-L3-v2", "embedding_model": HuggingFaceModel(
            tokenizer_path=os.path.join(HUGGING_FACE_ROOT, "paraphrase-MiniLM-L3-v2"),
            model_path=os.path.join(HUGGING_FACE_ROOT, "paraphrase-MiniLM-L3-v2"))},
        {"embedding_name": "BERT-finetuned-normal",
         "embedding_model": TensorflowBERT(model_path=os.path.join(BERT_ROOT, "bert_en_uncased_L-12_H-768_A-12"))},
        {"embedding_name": "BERT-finetuned-small",
         "embedding_model": TensorflowBERT(model_path=os.path.join(BERT_ROOT, "bert_en_uncased_L-2_H-128_A-2"))}

    ],
    "test_keys": [""],
    # "test_keys": ["", "emoji", "spaces", "insert", "uppercase", "regex", "regex_2", "swap", "random_char"] # all chosen to study robustness
    # "test_keys": # enriching_keys = ["", "random_char", "emoji", "spaces", "deletion", "insert", "swap", "uppercase", "regex", "end_char", "end_word"] # all

    "dataset_args": {
        "name": "OUR_DATASET",
        "categories": {},
        "dataset_type": DatasetType.FLATTEN,
        "return_type": DatasetReturnType.YIELD_SEPARATELY,
        # "annotated_files_path": os.path.join(ROOT_DIR, 'data', 'dialogues'),
        "annotated_files_path": os.path.join(ROOT_DIR, 'data', 'flatten_dialogues', 'annotated', 'ok'),
        # "annotated_files_path": os.path.join(ROOT_DIR, 'data', 'flatten_dialogues', 'annotated', 'all_in_one'),
        # "annotated_files_path": os.path.join(ROOT_DIR, 'data', 'flatten_dialogues', 'annotated', 'ok_small'),
        "global_path": os.path.join(ROOT_DIR, 'data', "flatten_dialogues", "annotated", "ok", "global.json")
        # "global_path": None
    }

})
