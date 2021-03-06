imports = []

LIMIT_NUM_SENTS = None  # # either None (i.e. no limit) or int with value > 0 (i.e. maximal number of sentences per class).

import os

from constants import ROOT_DIR, TENSORFLOW_ROOT, BERT_ROOT, HUGGING_FACE_ROOT
from utils.dataset.generate import DatasetType, DatasetReturnType
from custom_embeddings.fasttext import FastTextSW
from custom_embeddings.huggingface import HuggingFaceModel
from custom_embeddings.BERT_style import TensorflowBERT
import tensorflow_hub as hub
from models.sklearn_models import SklearnLogisticRegression, SklearnMLPClassifier, SklearnRandomForestClassifier
from models.neural_nets import OwnLogisticRegression, BaselineNNExtraLayer, BaselineNNTwoExtraLayer
from models.cosine_similarity import CosineSimilarity
from evaluate.local_illusionist import evaluate
# from evaluate.one_layer_based import evaluate
from evaluate.local_illusionist import find_best_threshold
from tensorflow.keras.losses import SparseCategoricalCrossentropy

APPROACH = "local_illusionist"

imports.append({
    "debug": False,
    "evaluation_fn": evaluate,
    "algorithms": [
        CosineSimilarity(),
        SklearnLogisticRegression(),
        SklearnMLPClassifier(),
        SklearnRandomForestClassifier(),
        # OwnLogisticRegression(loss_function=SparseCategoricalCrossentropy(), epochs=10, batch_size=8, init_learning_rate=0.0005),
        BaselineNNExtraLayer(loss_function=SparseCategoricalCrossentropy(), epochs=25, batch_size=64, init_learning_rate=0.0005),
        BaselineNNTwoExtraLayer(loss_function=SparseCategoricalCrossentropy(), epochs=25, batch_size=64, init_learning_rate=0.0005)
    ],
    "evaluation_fn_arg": {"limit_num_sents": LIMIT_NUM_SENTS, "find_best_threshold_fn": find_best_threshold},
    # "evaluation_fn_arg": {"limit_num_sents": LIMIT_NUM_SENTS, "find_best_threshold_fn": lambda x: 0.0},
    "embeddings": [
        # {"embedding_name": "fasttext", "embedding_model": FastTextSW(model_data_path="../data/embeddings/wiki_en_50k.pickle",
        #                                                              sw_data_path="../data/embeddings/wiki_en_sw_100k.pickle")},
        {"embedding_name": "use4", "embedding_model": hub.load(os.path.join(TENSORFLOW_ROOT, "universal-sentence-encoder-4"))},
        {"embedding_name": "use5", "embedding_model": hub.load(os.path.join(TENSORFLOW_ROOT, "universal-sentence-encoder-5"))},

        {"embedding_name": "sentence-transformers/all-mpnet-base-v2", "embedding_model": HuggingFaceModel(
            tokenizer_path=os.path.join(HUGGING_FACE_ROOT, "all-mpnet-base-v2"),
            model_path=os.path.join(HUGGING_FACE_ROOT, "all-mpnet-base-v2"))},

        {"embedding_name": "sentence-transformers/all-distilroberta-v1", "embedding_model": HuggingFaceModel(
            tokenizer_path=os.path.join(HUGGING_FACE_ROOT, "all-distilroberta-v1"),
            model_path=os.path.join(HUGGING_FACE_ROOT, "all-distilroberta-v1"))},

        {"embedding_name": "sentence-transformers/all-MiniLM-L12-v2", "embedding_model": HuggingFaceModel(
            tokenizer_path=os.path.join(HUGGING_FACE_ROOT, "all-MiniLM-L12-v2"),
            model_path=os.path.join(HUGGING_FACE_ROOT, "all-MiniLM-L12-v2"))},

        {"embedding_name": "sentence-transformers/all-MiniLM-L6-v2", "embedding_model": HuggingFaceModel(
            tokenizer_path=os.path.join(HUGGING_FACE_ROOT, "all-MiniLM-L6-v2"),
            model_path=os.path.join(HUGGING_FACE_ROOT, "all-MiniLM-L6-v2"))},

        {"embedding_name": "sentence-transformers/multi-qa-distilbert-cos-v1", "embedding_model": HuggingFaceModel(
            tokenizer_path=os.path.join(HUGGING_FACE_ROOT, "multi-qa-distilbert-cos-v1"),
            model_path=os.path.join(HUGGING_FACE_ROOT, "multi-qa-distilbert-cos-v1"))},

        {"embedding_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "embedding_model": HuggingFaceModel(
            tokenizer_path=os.path.join(HUGGING_FACE_ROOT, "paraphrase-multilingual-mpnet-base-v2"),
            model_path=os.path.join(HUGGING_FACE_ROOT, "paraphrase-multilingual-mpnet-base-v2"))},

        {"embedding_name": "sentence-transformers/paraphrase-albert-small-v2", "embedding_model": HuggingFaceModel(
            tokenizer_path=os.path.join(HUGGING_FACE_ROOT, "paraphrase-albert-small-v2"),
            model_path=os.path.join(HUGGING_FACE_ROOT, "paraphrase-albert-small-v2"))},

        {"embedding_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "embedding_model": HuggingFaceModel(
            tokenizer_path=os.path.join(HUGGING_FACE_ROOT, "paraphrase-multilingual-MiniLM-L12-v2"),
            model_path=os.path.join(HUGGING_FACE_ROOT, "paraphrase-multilingual-MiniLM-L12-v2"))},

        {"embedding_name": "sentence-transformers/paraphrase-MiniLM-L3-v2", "embedding_model": HuggingFaceModel(
            tokenizer_path=os.path.join(HUGGING_FACE_ROOT, "paraphrase-MiniLM-L3-v2"),
            model_path=os.path.join(HUGGING_FACE_ROOT, "paraphrase-MiniLM-L3-v2"))},

        {"embedding_name": "sentence-transformers/bert-base-nli-mean-tokens", "embedding_model": HuggingFaceModel(
            tokenizer_path=os.path.join(HUGGING_FACE_ROOT, "bert-base-nli-mean-tokens"),
            model_path=os.path.join(HUGGING_FACE_ROOT, "bert-base-nli-mean-tokens"))},

        # {"embedding_name": "use4_finetuned", "embedding_model": hub.load(os.path.join(TENSORFLOW_ROOT,"universal-sentence-encoder-finetuned-4"))},
        # {"embedding_name": "use5-finetuned", "embedding_model": hub.load(os.path.join(TENSORFLOW_ROOT,"universal-sentence-encoder-finetuned-5"))},

        # {"embedding_name": "BERT-finetuned-normal",
        #  "embedding_model": TensorflowBERT(model_path=os.path.join(BERT_ROOT, "bert_en_uncased_L-12_H-768_A-12"))},
        # {"embedding_name": "BERT-finetuned-small",
        #  "embedding_model": TensorflowBERT(model_path=os.path.join(BERT_ROOT, "bert_en_uncased_L-2_H-128_A-2"))}
    ],
    "test_keys": [""],
    # "test_keys": ["", "emoji", "spaces", "insert", "uppercase", "regex", "regex_2", "swap", "random_char"] # all chosen to study robustness
    # "test_keys": # enriching_keys = ["", "random_char", "emoji", "spaces", "deletion", "insert", "swap", "uppercase", "regex", "end_char", "end_word"] # all

    "dataset_args": {
        "name": "OUR_DATASET",
        "categories": {},
        "dataset_type": DatasetType.FLATTEN,
        # "dataset_type": DatasetType.CLINC150,
        "return_type": DatasetReturnType.YIELD_SEPARATELY,
        # "annotated_files_path": os.path.join(ROOT_DIR, 'data', 'dialogues'),
        # "annotated_files_path": os.path.join(ROOT_DIR, 'data', 'clinc', 'data', 'data_full.json'),
        "annotated_files_path": os.path.join(ROOT_DIR, 'data', 'flatten_dialogues', 'annotated', 'ok'),
        # "annotated_files_path": os.path.join(ROOT_DIR, 'data', 'flatten_dialogues', 'annotated', 'all_in_one'),
        # "annotated_files_path": os.path.join(ROOT_DIR, 'data', 'flatten_dialogues', 'annotated', 'ok_small'),
        "global_path": os.path.join(ROOT_DIR, 'data', "flatten_dialogues", "annotated", "ok", "global.json")
        # "global_path": os.path.join(ROOT_DIR, 'data', "clinc", "data", "domains.json")
        # "global_path": None
    }
})
