imports = []

LIMIT_NUM_SENTS = None  # either None (i.e. no limit) or int with value > 0 (i.e. maximal number of sentences per class).

from evaluate.remote_illusionist import evaluate

imports.append({
    "evaluation_fn": evaluate,
    "algorithms": ["remote"],
    "evaluation_fn_arg": {"limit_num_sents": LIMIT_NUM_SENTS},
    "embeddings": [
      # {"embedding_name": "ALBERT", "embedding_model": "HuggingFace"}
      # {"embedding_name": "use4", "embedding_model": "UniversalSentenceEncoder"}
      {"embedding_name": "fasttext", "embedding_model": "FastTextSW"}
    ],
    "test_keys": [""]
    # "test_keys": ["", "emoji", "spaces", "insert", "uppercase", "regex", "regex_2", "swap", "random_char"] # all chosen to study robustness
    # "test_keys": # enriching_keys = ["", "random_char", "emoji", "spaces", "deletion", "insert", "swap", "uppercase", "regex", "end_char", "end_word"] # all
})
