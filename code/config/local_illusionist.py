imports = []

LIMIT_NUM_SENTS = None  # # either None (i.e. no limit) or int with value > 0 (i.e. maximal number of sentences per class).
categories = [
    'animals',
    'books', 'education',
    'fashion', 'food', 'habits',
    'movies', 'music', 'science', 'smalltalk',
    'sports', 'travel'
]
import os

from constants import ROOT_DIR, TENSORFLOW_ROOT
from utils.dataset.generate import DatasetType, DatasetReturnType
from custom_embeddings.fasttext import FastTextSW
from custom_embeddings.huggingface import HuggingFaceModel
import tensorflow_hub as hub
from models.sklearn_models import SklearnLogisticRegression
from evaluate.local_illusionist import evaluate
from evaluate.local_illusionist import find_best_threshold

imports.append({
    "evaluation_fn": evaluate,
    "algorithms": [SklearnLogisticRegression()],
    # "algorithms": [BaselineNNExtraLayer()],
    # "evaluation_fn_arg": {"limit_num_sents": LIMIT_NUM_SENTS, "find_best_threshold_fn": find_best_threshold},
    "evaluation_fn_arg": {"limit_num_sents": LIMIT_NUM_SENTS, "find_best_threshold_fn": lambda x: 0.55},
    "embeddings": [
        # {"embedding_name": "fasttext", "embedding_model": FastTextSW(model_data_path="../data/embeddings/wiki_en_50k.pickle", sw_data_path="../data/embeddings/wiki_en_sw_100k.pickle")}
        # {"embedding_name": "use4", "embedding_model": hub.load(os.path.join(TENSORFLOW_ROOT,universal-sentence-encoder_4"))}
        # {"embedding_name": "use5", "embedding_model": hub.load(os.path.join(TENSORFLOW_ROOT,universal-sentence-encoder_5"))}
        # {"embedding_name": "use4_finetuned", "embedding_model": hub.load(os.path.join(TENSORFLOW_ROOT,universal-sentence-encoder-finetuned-4"))}
        {"embedding_name": "use5-finetuned", "embedding_model": hub.load(os.path.join(TENSORFLOW_ROOT,"universal-sentence-encoder-finetuned-5"))}
        # {"embedding_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "embedding_model": HuggingFaceModel(tokenizer_path="/media/petrlorenc/Data/hugging_face_models/paraphrase-multilingual-MiniLM-L12-v2", model_path="/media/petrlorenc/Data/hugging_face_models/paraphrase-multilingual-MiniLM-L12-v2")}
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
        # "annotated_files_path": os.path.join(ROOT_DIR, 'data', 'flatten_dialogues', 'annotated', 'ok_small'),
        "global_path": os.path.join(ROOT_DIR, 'data', "flatten_dialogues", "annotated", "ok", "global.json")
    }
})
