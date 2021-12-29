imports = []

LIMIT_NUM_SENTS = None

from evaluate.threshold_based import evaluate
from models.sklearn_models import SklearnLogisticRegression
from models.neural_nets import OwnLogisticRegression, BaselineNNExtraLayer
from models.cosine_similarity import CosineSimilarity
from custom_embeddings.fasttext import FastTextSW

imports.append({
    "evaluation_fn": evaluate,
    "algorithms": [SklearnLogisticRegression(), OwnLogisticRegression(), BaselineNNExtraLayer(), CosineSimilarity()],
    "evaluation_fn_arg": {"limit_num_sents": LIMIT_NUM_SENTS, "find_best_threshold_fn": lambda x: 0.55},
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
