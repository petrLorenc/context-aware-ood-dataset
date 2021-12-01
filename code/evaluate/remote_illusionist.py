from utils.utils import Split
from utils.testing import Testing

import time, psutil
import numpy as np
import requests

INFERENCE_URL = "http://localhost:8090/query/multi_model?key=test"
DELETE_URL = "http://localhost:8090/intent/invalidate/<XY>?key=test"
TRAIN_URL = "http://localhost:8095/training/models/<XY>?key=test"


def evaluate(dataset, model, model_name, embed_f, limit_num_sents, find_best_threshold_fn):
    split = Split(embed_f)
    threshold = 0.0
    # TRAINING
    start_time_train = time.time()

    # Split dataset
    X_train, y_train = split.get_X_y(dataset['train'], limit_num_sents=limit_num_sents, raw=True)
    training_template_local = {
        '_id': 'local',
        'auth_key': 'test',
        'model': {
            'name': 'test',
            'algorithm': 'UniversalSentenceEncoder',
            'lang': 'en',
            'use_tfidf': False,
            'approach': "hybrid"
        },
        'qa': {k: {"questions": [sentence for sentence, v in zip(X_train, y_train) if v == k], "answer": k, "threshold": threshold} for k in list(set(y_train))}
    }
    response = requests.post(TRAIN_URL.replace("<XY>", "local"), json=training_template_local).json()

    X_global, y_global = split.get_X_y(dataset["global_train"], raw=True)
    training_template_global = {
        '_id': 'global',
        'auth_key': 'test',
        'model': {
            'name': 'test',
            'algorithm': 'UniversalSentenceEncoder',
            'lang': 'en',
            'use_tfidf': False,
            'approach': "hybrid"
        },
        'qa': {k: {"questions": [sentence for sentence, v in zip(X_global, y_global) if v == k], "answer": k, "threshold": threshold} for k in list(set(y_global))}
    }
    response = requests.post(TRAIN_URL.replace("<XY>", "global"), json=training_template_global).json()

    end_time_train = time.time()

    memory = psutil.Process().memory_full_info().uss / (1024 ** 2)  # in megabytes

    # TESTING
    results_dct = {"results": {}}
    start_time_inference = time.time()

    # Split dataset
    X_test, y_test = split.get_X_y(dataset['test'], limit_num_sents=None, raw=True)

    predictions = []

    for test_sentence in X_test:
        response = requests.post(INFERENCE_URL, json={
            "query": test_sentence,
            "_ids": ["local", "global"],
            "denied_answers": [],
            "allowed_answers": [],
            "use_threshold": True,
            "n": 0
        }).json()

        if response[0]["_id"] == "local" and response[0]["answer"] != "outofdomain":
            predictions.append(response[0]["answer"])
        else:
            predictions.append(split.intents_dct['ood'])

    results_dct["results"]["local_intents"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=split.intents_dct['ood'],
                                                                       focus="IND")

    # # # Split dataset
    X_test, y_test = split.get_X_y(dataset['global_test'], limit_num_sents=None)
    # y_test = [999] * len(X_test)
    predictions = []
    for test_sentence in X_test:
        response = requests.post(INFERENCE_URL, json={
            "query": test_sentence,
            "_ids": ["local", "global"],
            "denied_answers": [],
            "allowed_answers": [],
            "use_threshold": True,
            "n": 0
        }).json()

        if response[0]["_id"] == "global" and response[0]["answer"] != "outofdomain":
            predictions.append(response[0]["answer"])
        else:
            predictions.append(split.intents_dct['ood'])

    results_dct["results"]["global_intents"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=split.intents_dct['ood'],
                                                                        focus="IND")

    # # Split dataset
    X_test, _ = split.get_X_y(dataset['global_ood'] + dataset["local_ood"] + dataset["garbage"], limit_num_sents=None)
    y_test = [999] * len(X_test)
    predictions = []

    for test_sentence in X_test:
        response = requests.post(INFERENCE_URL, json={
            "query": test_sentence,
            "_ids": ["local", "global"],
            "denied_answers": [],
            "allowed_answers": [],
            "use_threshold": True,
            "n": 0
        }).json()

        if response[0]["answer"] == "outofdomain":
            predictions.append(999)
        else:
            predictions.append(-1)

    results_dct["results"]["ood"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=split.intents_dct['ood'], focus="OOD")

    end_time_inference = time.time()

    requests.delete(TRAIN_URL.replace("<XY>", "local"))
    requests.delete(TRAIN_URL.replace("<XY>", "global"))
    requests.post(DELETE_URL.replace("<XY>", "local"))
    requests.post(DELETE_URL.replace("<XY>", "global"))

    results_dct['time_train'] = round(end_time_train - start_time_train, 1)
    results_dct['time_inference'] = round(end_time_inference - start_time_inference, 1)
    results_dct['threshold'] = [0.50]  # store threshold value
    results_dct['memory'] = round(memory, 1)

    return results_dct
