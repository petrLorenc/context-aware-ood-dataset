import psutil
import time

import numpy as np
import tensorflow as tf

from models import AbstractModel
from utils.testing import Testing
from utils.utils import TransformToEmbeddings
from custom_embeddings.BERT_style import ContextBased

def find_best_threshold(X_train):
    sim = np.inner(X_train, X_train)
    sim = np.exp(sim - 1)
    second_smallest = np.argsort(-sim)[:, 1]
    if second_smallest.shape[0] <= 1:
        second_smallest_sim = [0.5]
    else:
        second_smallest_sim = np.take(sim, second_smallest, axis=1)[:, 1]
    return np.average(second_smallest_sim)


def evaluate(dataset, classification_model, embedding_model, limit_num_sents, find_best_threshold_fn):
    local_split = TransformToEmbeddings(embedding_model, dataset["context"] if issubclass(type(embedding_model), ContextBased) else None)

    # TRAINING
    start_time_train = time.time()

    # Split dataset
    X_train, y_train = local_split.get_X_y(dataset['train'], limit_num_sents=limit_num_sents)
    X_val, y_val = local_split.get_X_y(dataset['val'], limit_num_sents=limit_num_sents)

    # Train
    local_model = classification_model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    ood_thresholds_local = {}

    for idx, cls in enumerate(np.unique(y_train)):
        ood_thresholds_local[idx] = find_best_threshold_fn([x for x, y in zip(X_train, y_train) if y == cls])

    end_time_train = time.time()

    memory = psutil.Process().memory_full_info().uss / (1024 ** 2)  # in megabytes

    # TESTING
    results_dct = {"results": {}}
    start_time_inference = time.time()

    # Split dataset
    X_test, y_test = local_split.get_X_y(dataset['test'], limit_num_sents=None)

    corr_local = np.inner(X_train, X_test)
    corr_local = np.exp(corr_local - 1)

    predictions = []
    for idx in range(len(X_test)):
        pred_probs = classification_model.predict_proba_with(local_model, X_test[idx].numpy().reshape(1, -1))
        match = False
        for max_idx in np.argsort(-pred_probs).squeeze():
            if corr_local[:, idx][y_train.numpy().squeeze() == max_idx].max() > ood_thresholds_local[max_idx]:
                predictions.append(int(max_idx))
                match = True
                break

        if match is False:
            predictions.append(local_split.intents_dct['ood'])

    results_dct["results"]["local_intents"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=local_split.intents_dct['ood'],
                                                                       focus="IND")

    # # Split dataset
    X_test, _ = local_split.get_X_y(dataset['global_ood'] + dataset["local_ood"] + dataset["garbage"], limit_num_sents=None)
    # X_test = ["tell me more about your relatives", "what is the weather in Prague"]
    # X_test = tf.convert_to_tensor(embedding_model(X_test), dtype='float32')
    y_test = [local_split.intents_dct['ood']] * len(X_test)

    corr_local = np.inner(X_train, X_test)
    corr_local = np.exp(corr_local - 1)

    predictions = []
    for idx in range(len(X_test)):
        match = False
        for max_idx in ood_thresholds_local.keys():
            if corr_local[:, idx][y_train.numpy().squeeze() == max_idx].max() > ood_thresholds_local[max_idx]:
                predictions.append(999)
                match = True
                break
        if match:
            continue

        predictions.append(local_split.intents_dct['ood'])

    results_dct["results"]["ood"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=local_split.intents_dct['ood'], focus="OOD")

    end_time_inference = time.time()

    results_dct['time_train'] = round(end_time_train - start_time_train, 1)
    results_dct['time_inference'] = round(end_time_inference - start_time_inference, 1)
    results_dct['threshold_local'] = ood_thresholds_local  # store threshold value
    results_dct['threshold_global'] = ood_thresholds_local  # store threshold value
    results_dct['memory'] = round(memory, 1)

    return results_dct
