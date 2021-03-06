from utils.utils import TransformToEmbeddings
from utils.testing import Testing

import time, psutil
import numpy as np
import requests

INFERENCE_URL = "http://localhost:8090/query/multi_model?key=test"
DELETE_URL = "http://localhost:8090/intent/invalidate/<XY>?key=test"
TRAIN_URL = "http://localhost:8095/training/models/<XY>?key=test"


def evaluate(dataset, classification_model, embedding_model, limit_num_sents):
    split = TransformToEmbeddings(None)
    threshold = 0.0
    local_id = "local2"
    global_id = "global2"
    # TRAINING
    start_time_train = time.time()

    # Split dataset
    X_train, y_train = split.get_X_y(dataset['train'], limit_num_sents=limit_num_sents, raw=True)
    training_template_local = {
        '_id': local_id,
        'auth_key': 'test',
        'model': {
            'name': 'test',
            'embedding': embedding_model,
            'algorithm': "hybrid",
            'lang': 'en',
            'use_tfidf': False,
            'approach': "hybrid"
        },
        'qa': {k: {"questions": [sentence for sentence, v in zip(X_train, y_train) if v == k], "answer": k, "threshold": threshold} for k in list(set(y_train))}
    }
    _ = requests.post(TRAIN_URL.replace("<XY>", local_id), json=training_template_local).json()

    X_global, y_global = split.get_X_y(dataset["global_train"], raw=True)
    training_template_global = {
        '_id': global_id,
        'auth_key': 'test',
        'model': {
            'name': 'test',
            'embedding': embedding_model,
            'algorithm': "hybrid",
            'lang': 'en',
            'use_tfidf': False,
            'approach': "hybrid"
        },
        'qa': {k: {"questions": [sentence for sentence, v in zip(X_global, y_global) if v == k], "answer": k, "threshold": threshold} for k in list(set(y_global))}
    }
    _ = requests.post(TRAIN_URL.replace("<XY>", global_id), json=training_template_global).json()

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
            "_ids": [local_id, global_id],
            "denied_answers": [],
            "allowed_answers": [],
            "use_threshold": True,
            "n": 0
        }).json()

        if response[0]["_id"] == local_id and response[0]["answer"] != "outofdomain":
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
            "_ids": [local_id, global_id],
            "denied_answers": [],
            "allowed_answers": [],
            "use_threshold": True,
            "n": 0
        }).json()

        if response[0]["_id"] == global_id and response[0]["answer"] != "outofdomain":
            predictions.append(response[0]["answer"])
        else:
            predictions.append(split.intents_dct['ood'])

    results_dct["results"]["global_intents"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=split.intents_dct['ood'],
                                                                        focus="IND")

    # # Split dataset
    X_test, _ = split.get_X_y(dataset['global_ood'] + dataset["local_ood"] + dataset["garbage"], limit_num_sents=None)
    # X_test = ["tell me more about your relatives", "what is the weather in Prague"]
    y_test = [split.intents_dct['ood']] * len(X_test)
    predictions = []

    for test_sentence in X_test:
        response = requests.post(INFERENCE_URL, json={
            "query": test_sentence,
            "_ids": [local_id, global_id],
            "denied_answers": [],
            "allowed_answers": [],
            "use_threshold": True,
            "n": 0
        }).json()

        if response[0]["answer"] == "outofdomain":
            predictions.append(split.intents_dct['ood'])
        else:
            predictions.append(999)

    results_dct["results"]["ood"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=split.intents_dct['ood'], focus="OOD")

    end_time_inference = time.time()

    requests.delete(TRAIN_URL.replace("<XY>", local_id))
    requests.delete(TRAIN_URL.replace("<XY>", global_id))
    requests.post(DELETE_URL.replace("<XY>", local_id))
    requests.post(DELETE_URL.replace("<XY>", global_id))

    results_dct['time_train'] = round(end_time_train - start_time_train, 1)
    results_dct['time_inference'] = round(end_time_inference - start_time_inference, 1)
    results_dct['threshold'] = [0.50]  # store threshold value
    results_dct['memory'] = round(memory, 1)

    return results_dct
