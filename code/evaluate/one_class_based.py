import psutil
import time

import numpy as np

from utils.testing import Testing
from utils.utils import TransformToEmbeddings


def evaluate(dataset, classification_model, embedding_model, limit_num_sents):
    local_split = TransformToEmbeddings(embedding_model)
    global_split = TransformToEmbeddings(embedding_model)

    # TRAINING
    start_time_train = time.time()

    # Split dataset
    X_train, y_train = local_split.get_X_y(dataset['train'], limit_num_sents=limit_num_sents)
    X_val, y_val = local_split.get_X_y(dataset['val'], limit_num_sents=limit_num_sents)

    X_global, y_global = global_split.get_X_y(dataset["global_train"])
    X_global_val, y_global_val = global_split.get_X_y(dataset["global_val"])

    # Train
    local_model, local_one_class_models, local_density_threshold = classification_model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    global_model, global_one_class_models, global_density_threshold = classification_model.fit(X_train=X_global, y_train=y_global, X_val=X_global_val, y_val=y_global_val)

    end_time_train = time.time()

    memory = psutil.Process().memory_full_info().uss / (1024 ** 2)  # in megabytes

    # TESTING
    results_dct = {"results": {}}
    start_time_inference = time.time()

    # Split dataset
    X_test, y_test = local_split.get_X_y(dataset['test'], limit_num_sents=None)

    predictions = []
    for idx in range(len(X_test)):
        if classification_model.belong_to(local_one_class_models, local_density_threshold, X_test[idx].numpy().reshape(1, -1)):
            pred_probs = classification_model.predict_proba_with(local_model, X_test[idx].numpy().reshape(1, -1))
            predictions.append(int(np.argmax(-pred_probs)))
        else:
            predictions.append(local_split.intents_dct['ood'])

    results_dct["results"]["local_intents"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=global_split.intents_dct['ood'],
                                                                       focus="IND")

    # # # Split dataset
    X_test, y_test = global_split.get_X_y(dataset['global_test'], limit_num_sents=None)

    predictions = []
    for idx in range(len(X_test)):
        if classification_model.belong_to(global_one_class_models, global_density_threshold, X_test[idx].numpy().reshape(1, -1)):
            pred_probs = classification_model.predict_proba_with(global_model, X_test[idx].numpy().reshape(1, -1))
            predictions.append(int(np.argmax(-pred_probs)))
        else:
            predictions.append(local_split.intents_dct['ood'])

    results_dct["results"]["global_intents"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=global_split.intents_dct['ood'],
                                                                        focus="IND")

    # Split dataset
    X_test, _ = global_split.get_X_y(dataset['global_ood'] + dataset["local_ood"] + dataset["garbage"], limit_num_sents=None)
    y_test = [global_split.intents_dct['ood']] * len(X_test)

    predictions = []
    for idx in range(len(X_test)):
        if classification_model.belong_to(local_one_class_models, local_density_threshold, X_test[idx].numpy().reshape(1, -1)) or classification_model.belong_to(global_one_class_models, global_density_threshold, X_test[idx].numpy().reshape(1, -1)):
            predictions.append(999)
        else:
            predictions.append(local_split.intents_dct['ood'])

    results_dct["results"]["ood"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=local_split.intents_dct['ood'], focus="OOD")

    end_time_inference = time.time()

    results_dct['time_train'] = round(end_time_train - start_time_train, 1)
    results_dct['time_inference'] = round(end_time_inference - start_time_inference, 1)
    results_dct['memory'] = round(memory, 1)

    return results_dct
