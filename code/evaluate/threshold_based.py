from utils.utils import TransformToEmbeddings
from utils.testing import Testing

import time, psutil
import numpy as np
from sklearn.model_selection import KFold

from models.sklearn_models import SklearnLogisticRegression
from models.neural_nets import OwnLogisticRegression, BaselineNNExtraLayer
from models.cosine_similarity import CosineSimilarity


# def find_best_threshold(X, y, classification_model):
#     if type(classification_model) == CosineSimilarity:
#         sim = np.inner(X, X)
#         sim = np.exp(sim - 1)
#         second_smallest = np.argsort(-sim)[:, 1]
#         if second_smallest.shape[0] <= 1:
#             second_smallest_sim = [0.5]
#         else:
#             second_smallest_sim = np.take(sim, second_smallest, axis=1)[:, 1]
#         return np.average(second_smallest_sim)
#     elif type(classification_model) == SklearnLogisticRegression or type(classification_model) == OwnLogisticRegression or type(classification_model) == BaselineNNExtraLayer:
#         probs = classification_model.predict_proba(X)
#         return np.average(probs)


def find_best_threshold(X):
    return np.average([x[1] for x in X])


def evaluate(dataset, classification_model, embedding_model, limit_num_sents, find_best_threshold_fn):
    split = TransformToEmbeddings(embedding_model)

    # TRAINING
    start_time_train = time.time()

    # Split dataset
    X_train, y_train = split.get_X_y(dataset['train'] + dataset["global_train"], limit_num_sents=limit_num_sents)
    X_val, y_val = split.get_X_y(dataset['val'] + dataset["global_val"], limit_num_sents=limit_num_sents)

    # Train
    classification_model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    # Find threshold
    val_predictions_labels = []  # used to find threshold
    pred_probs = classification_model.predict_proba(X_val)  # function available in both scikit-learn and TF-Keras, returns numpy array

    pred_labels = np.argmax(pred_probs, axis=1)
    pred_similarities = np.take_along_axis(pred_probs, indices=np.expand_dims(pred_labels, axis=1), axis=1).squeeze()

    predictions = np.column_stack([pred_labels, pred_similarities])  # 2D list of [pred_label, similarity]

    for (pred_label, similarity), true_label in zip(predictions, y_val):
        val_predictions_labels.append((pred_label, similarity, true_label.numpy()))

    threshold = find_best_threshold_fn(val_predictions_labels)

    end_time_train = time.time()

    memory = psutil.Process().memory_full_info().uss / (1024 ** 2)  # in megabytes

    # TESTING
    results_dct = {"results": {}}
    start_time_inference = time.time()

    X_test, y_test = split.get_X_y(dataset['test'], limit_num_sents=None)

    pred_probs = classification_model.predict_proba(X_test)  # returns numpy array

    pred_labels = np.argmax(pred_probs, axis=1)
    pred_similarities = np.take_along_axis(pred_probs, indices=np.expand_dims(pred_labels, axis=1), axis=1).squeeze()

    predictions = []
    for pred_label, pred_similarity in zip(pred_labels, pred_similarities):
        if pred_similarity < threshold:
            predictions.append(split.intents_dct['ood'])
        else:
            predictions.append(pred_label)

    results_dct["results"]["local_intents"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=split.intents_dct['ood'], focus="IND")

    ###################################################################
    X_test, y_test = split.get_X_y(dataset['global_test'], limit_num_sents=None)

    pred_probs = classification_model.predict_proba(X_test)  # returns numpy array

    pred_labels = np.argmax(pred_probs, axis=1)
    pred_similarities = np.take_along_axis(pred_probs, indices=np.expand_dims(pred_labels, axis=1), axis=1).squeeze()

    predictions = []
    for pred_label, pred_similarity in zip(pred_labels, pred_similarities):
        if pred_similarity < threshold:
            predictions.append(split.intents_dct['ood'])
        else:
            predictions.append(pred_label)

    results_dct["results"]["global_intents"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=split.intents_dct['ood'], focus="IND")

    #######################################################################
    X_test, _ = split.get_X_y(dataset['global_ood'] + dataset["local_ood"] + dataset["garbage"], limit_num_sents=None)
    y_test = [split.intents_dct['ood']] * len(X_test)

    pred_probs = classification_model.predict_proba(X_test)  # returns numpy array

    pred_labels = np.argmax(pred_probs, axis=1)
    pred_similarities = np.take_along_axis(pred_probs, indices=np.expand_dims(pred_labels, axis=1), axis=1).squeeze()

    predictions = []
    for pred_similarity in pred_similarities:
        if pred_similarity < threshold:
            predictions.append(split.intents_dct['ood'])
        else:
            predictions.append(-999)
    # Test
    results_dct["results"]["ood"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=split.intents_dct['ood'], focus="OOD")

    end_time_inference = time.time()

    results_dct['time_train'] = round(end_time_train - start_time_train, 1)
    results_dct['time_inference'] = round(end_time_inference - start_time_inference, 1)
    results_dct['threshold'] = threshold  # store threshold value
    results_dct['memory'] = round(memory, 1)

    return results_dct
