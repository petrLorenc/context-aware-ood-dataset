imports = []

LIMIT_NUM_SENTS = None  # # either None (i.e. no limit) or int with value > 0 (i.e. maximal number of sentences per class).
categories = [
    'animals',
    'books', 'education',
    'fashion', 'food', 'habits',
    'movies', 'music', 'science', 'smalltalk',
    'sports', 'travel'
]

from utils.dataset.generate import DatasetType, DatasetReturnType
from custom_embeddings.fasttext import FastTextSW
from custom_embeddings.huggingface import HuggingFaceModel
import tensorflow_hub as hub
from models.embedding_net import BERTEmbeddingModel
from evaluate.embedding_based import evaluate
import os
from constants import ROOT_DIR, TENSORFLOW_ROOT

imports.append({
    "evaluation_fn": evaluate,
    # "algorithms": [BERTEmbeddingModel(tfhub_handle_preprocess=os.path.join(TENSORFLOW_ROOT, "prepro"), tfhub_handle_encoder=os.path.join(TENSORFLOW_ROOT, "encoder"))],
    "algorithms": [BERTEmbeddingModel(tfhub_handle_encoder="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1",
                                      tfhub_handle_preprocess="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
                                      seq_length=128,
                                      use_masking_when_training=True,
                                      epoch=10,
                                      batch_size=32,
                                      init_learning_rate=3e-5)],
    "evaluation_fn_arg": {"limit_num_sents": LIMIT_NUM_SENTS},
    "embeddings": [
        {"embedding_name": "implicit", "embedding_model": None}
    ],
    "test_keys": [""],
    "dataset_args": {
        "name": "OUR_DATASET",
        "categories": {},
        "dataset_type": DatasetType.FLATTEN,
        "return_type": DatasetReturnType.RETURN_ALL,
        # "annotated_files_path": os.path.join(ROOT_DIR, 'data', 'dialogues'),
        # "annotated_files_path": os.path.join(ROOT_DIR, 'data', 'flatten_dialogues', 'annotated', 'ok'),
        "annotated_files_path": os.path.join(ROOT_DIR, 'data', 'flatten_dialogues', 'annotated', 'ok_small'),
        "global_path": os.path.join(ROOT_DIR, 'data', "flatten_dialogues", "annotated", "ok", "global.json")}
})
