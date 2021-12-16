import wandb

from utils.utils import print_results, generate_dataset, aggregate_results
from models.sklearn_models import SklearnLogisticRegression
from models.neural_nets import OwnLogisticRegression, BaselineNNExtraLayer
from models.cosine_similarity import CosineSimilarity
from custom_embeddings.fasttext import FastTextSW
from utils.utils import find_best_threshold

import tensorflow_hub as hub

LIMIT_NUM_SENTS = None  # either None (i.e. no limit) or int with value > 0 (i.e. maximal number of sentences per class).


imports = []

from evaluate.remote_illusionist import evaluate

imports.append({
    "evaluation_fn": evaluate,
    "algorithms": ["remote"],
    "evaluation_fn_arg": {"limit_num_sents": LIMIT_NUM_SENTS},
    "embeddings": [
      # {"embedding_name": "ALBERT", "embedding_model": "HuggingFace"}
      {"embedding_name": "use4", "embedding_model": "UniversalSentenceEncoder"}
      # {"embedding_name": "fasttext", "embedding_model": "FastTextSW"}
    ],
    "test_keys": [""]
    # "test_keys": ["", "emoji", "spaces", "insert", "uppercase", "regex", "regex_2", "swap", "random_char"] # all chosen to study robustness
    # "test_keys": # enriching_keys = ["", "random_char", "emoji", "spaces", "deletion", "insert", "swap", "uppercase", "regex", "end_char", "end_word"] # all
})

# from evaluate.local_illusionist import evaluate
# from evaluate.local_illusionist import find_best_threshold
#
# imports.append({
#     "evaluation_fn": evaluate,
#     "algorithms": [SklearnLogisticRegression()],
#     "evaluation_fn_arg": {"limit_num_sents": LIMIT_NUM_SENTS, "find_best_threshold_fn": find_best_threshold},
#     "embeddings": [
#         {"embedding_name": "fasttext", "embedding_model": FastTextSW(model_data_path="../data/embeddings/wiki_en_50k.pickle", sw_data_path="../data/embeddings/wiki_en_sw_100k.pickle")}
#         # {"embedding_name": "use4", "embedding_model": hub.load("/media/petrlorenc/Data/universal-sentence-encoder_4")}
#         # {"embedding_name": "use5", "embedding_model": hub.load("/media/petrlorenc/Data/universal-sentence-encoder_5")}
#         # {"embedding_name": "use4_finetuned", "embedding_model": hub.load("/media/petrlorenc/Data/universal-sentence-encoder_fine")}
#
#     ],
#     "test_keys": [""]
#     # "test_keys": ["", "emoji", "spaces", "insert", "uppercase", "regex", "regex_2", "swap", "random_char"] # all chosen to study robustness
#     # "test_keys": # enriching_keys = ["", "random_char", "emoji", "spaces", "deletion", "insert", "swap", "uppercase", "regex", "end_char", "end_word"] # all
# })

# from evaluate.threshold_based import evaluate
#
# imports.append({
#     "evaluation_fn": evaluate,
#     "algorithms": [SklearnLogisticRegression(), OwnLogisticRegression(), BaselineNNExtraLayer(), CosineSimilarity()],
#     "evaluation_fn_arg": {"limit_num_sents": LIMIT_NUM_SENTS, "find_best_threshold_fn": find_best_threshold},
#     "embeddings": [
#         {"embedding_name": "fasttext", "embedding_model": FastTextSW(model_data_path="../data/embeddings/wiki_en_50k.pickle", sw_data_path="../data/embeddings/wiki_en_sw_100k.pickle")}
#         # {"embedding_name": "use4", "embedding_model": hub.load("/media/petrlorenc/Data/universal-sentence-encoder_4")}
#         # {"embedding_name": "use5", "embedding_model": hub.load("/media/petrlorenc/Data/universal-sentence-encoder_5")}
#         # {"embedding_name": "use4_finetuned", "embedding_model": hub.load("/media/petrlorenc/Data/universal-sentence-encoder_fine")}
#
#     ],
#     "test_keys": [""]
#     # "test_keys": ["", "emoji", "spaces", "insert", "uppercase", "regex", "regex_2", "swap", "random_char"] # all chosen to study robustness
#     # "test_keys": # enriching_keys = ["", "random_char", "emoji", "spaces", "deletion", "insert", "swap", "uppercase", "regex", "end_char", "end_word"] # all
# })


dataset_name = 'ALQUIST'
categories = [
    'animals',
    'books', 'education',
    'fashion', 'food', 'habits',
    'movies', 'music', 'science', 'smalltalk',
    'sports', 'travel'
]

# dataset_name = 'CLINC150'
# categories = [
#     '10',
#     "5",
#     "2"
# ]
# ------------------------------------------------------------

if __name__ == '__main__':
    for i in imports:
        evaluate_fn = i["evaluation_fn"]

        for embeddings in i["embeddings"]:
            emb_name = embeddings["embedding_name"]
            embedding_model = embeddings["embedding_model"]

            for k in i["test_keys"]:
                test_label = "test" + ("_" if k else "") + k

                for classification_model in i["algorithms"]:
                    if type(classification_model) == str:
                        model_name = classification_model
                    else:
                        model_name = type(classification_model).__name__

                    wandb.init(project='robust-intent-recognition', entity='alquist')
                    config = wandb.config
                    config.dataset_name = dataset_name
                    config.model_name = model_name
                    config.emb_name = emb_name
                    config.test_type = test_label

                    dct_results_lst = []

                    for dataset in generate_dataset(categories, test_label):
                        args = i["evaluation_fn_arg"]
                        args["dataset"] = dataset
                        args["embedding_model"] = embedding_model
                        args["classification_model"] = classification_model

                        results_dct = evaluate_fn(**args)
                        dct_results_lst.append(results_dct)
                        print(results_dct)
                        print()

                    results_dct = aggregate_results(dct_results_lst)

                    for k, v in results_dct.items():
                        wandb.log({k: v})
                    print_results(dataset_name, model_name, emb_name, results_dct)
                    wandb.finish()
