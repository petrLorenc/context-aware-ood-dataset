imports = []

LIMIT_NUM_SENTS = None

from constants import ROOT_DIR, TENSORFLOW_ROOT, BERT_ROOT, HUGGING_FACE_ROOT
import os
from utils.dataset.generate import DatasetType, DatasetReturnType
from evaluate.threshold_based import evaluate
from models.sklearn_models import SklearnLogisticRegression
from models.neural_nets import OwnLogisticRegression, BaselineNNExtraLayer
from models.cosine_similarity import CosineSimilarity
from custom_embeddings.fasttext import FastTextSW
from evaluate.threshold_based import find_best_threshold
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow_hub as hub
from custom_embeddings.BERT_style import TensorflowBERT
from custom_embeddings.huggingface import HuggingFaceModel

APPROACH = "threshold_based"

imports.append({
    "debug": False,
    "evaluation_fn": evaluate,
    "algorithms": [
        CosineSimilarity(),
        SklearnLogisticRegression(),
        # OwnLogisticRegression(loss_function=SparseCategoricalCrossentropy(), epochs=25, batch_size=8, init_learning_rate=0.0005),
        BaselineNNExtraLayer(loss_function=SparseCategoricalCrossentropy(), epochs=50, batch_size=8, init_learning_rate=0.0005),
    ],
    # "evaluation_fn_arg": {"limit_num_sents": LIMIT_NUM_SENTS, "find_best_threshold_fn": lambda x: 0.55},
    "evaluation_fn_arg": {"limit_num_sents": LIMIT_NUM_SENTS, "find_best_threshold_fn": find_best_threshold},
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
    ],
    "test_keys": [""],
    # "test_keys": ["", "emoji", "spaces", "insert", "uppercase", "regex", "regex_2", "swap", "random_char"] # all chosen to study robustness
    # "test_keys": # enriching_keys = ["", "random_char", "emoji", "spaces", "deletion", "insert", "swap", "uppercase", "regex", "end_char", "end_word"] # all

    "dataset_args": {
        "name": "OUR_DATASET",
        "categories": {},
        "annotated_files_path": os.path.join(ROOT_DIR, 'data', 'flatten_dialogues', 'annotated', 'ok'),
        "global_path": os.path.join(ROOT_DIR, 'data', "flatten_dialogues", "annotated", "ok", "global.json"),
        "dataset_type": DatasetType.FLATTEN,
        "return_type": DatasetReturnType.YIELD_SEPARATELY}

})
