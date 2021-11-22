from utils import Split, NEEDS_VAL
from testing import Testing

import time, psutil
import numpy as np


# def find_best_threshold(val_predictions_labels, oos_label):
#     """
#     Function used to find the best threshold in oos-threshold.
#     :param:            val_predictions_labels - prediction on the validation set, list
#                         oos_label - encodes oos label, int
#     :returns:           threshold - best threshold
#     """
#
#     # Initialize search for best threshold
#     thresholds = np.linspace(0, 1, 101)
#     previous_val_accuracy = 0
#     threshold = 0
#
#     # Find best threshold
#     for idx, tr in enumerate(thresholds):
#         val_accuracy_correct = 0
#         val_accuracy_out_of = 0
#
#         for pred, true_label in val_predictions_labels:
#             pred_label = pred[0]
#             similarity = pred[1]
#
#             if similarity < tr:
#                 pred_label = oos_label
#
#             if pred_label == true_label:
#                 val_accuracy_correct += 1
#
#             val_accuracy_out_of += 1
#
#         val_accuracy = val_accuracy_correct / val_accuracy_out_of
#
#         if val_accuracy < previous_val_accuracy:
#             threshold = thresholds[idx - 1]  # best threshold is the previous one
#             break
#
#         previous_val_accuracy = val_accuracy
#         threshold = tr
#
#     return threshold


def evaluate(dataset, model, model_name, embed_f, limit_num_sents, find_best_threshold_fn):
    split = Split(embed_f)

    # TRAINING
    start_time_train = time.time()

    # Split dataset
    X_train, y_train = split.get_X_y(dataset['train'], limit_num_sents=limit_num_sents)
    X_val, y_val = split.get_X_y(dataset['val'], limit_num_sents=limit_num_sents)

    X_global, _ = split.get_X_y(dataset["global_train"])

    # Train
    model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    # Find threshold
    val_predictions_labels = []  # used to find threshold
    # pred_probs = model.predict_proba(X_val)  # function available in both scikit-learn and TF-Keras, returns numpy array

    # pred_labels = np.argmax(pred_probs, axis=1)
    # pred_similarities = np.take_along_axis(pred_probs, indices=np.expand_dims(pred_labels, axis=1), axis=1).squeeze()

    # predictions = np.column_stack([pred_labels, pred_similarities])  # 2D list of [pred_label, similarity]

    # for pred, true_label in zip(predictions, y_val):
    #     val_predictions_labels.append((pred, true_label))

    # threshold = find_best_threshold_fn(val_predictions_labels, split.intents_dct['ood'])
    # if threshold is None:
    #     threshold = find_best_threshold(val_predictions_labels, split.intents_dct['ood'])

    threshold = 0.0
    ood_thresholds = {}

    # sim = np.inner(X_train, X_train)
    # for idx, cls in enumerate(np.unique(y_train)):
    #     second_smallest = np.argsort(-sim[y_train.numpy().squeeze() == idx])[:, 1]
    #     if second_smallest.shape[0] <= 1:
    #         second_smallest_sim = [threshold]
    #     else:
    #         second_smallest_sim = np.take(sim[y_train.numpy().squeeze() == idx], second_smallest, axis=1)[:, 1]
    #         # second_smallest_sim = list(map(lambda x: x if x > 0.0 else 0.0, second_smallest_sim))
    #     ood_thresholds[idx] = np.average(second_smallest_sim)

    for idx, cls in enumerate(np.unique(y_train)):
        ood_thresholds[idx] = threshold

    end_time_train = time.time()

    memory = psutil.Process().memory_full_info().uss / (1024 ** 2)  # in megabytes

    # TESTING
    results_dct = {"results": {}}
    start_time_inference = time.time()

    # Split dataset
    X_test, y_test = split.get_X_y(dataset['test'], limit_num_sents=None)
    corr_local = np.inner(X_test, X_train)
    corr_global = np.inner(X_test, X_global)
    predictions = []
    for idx in range(len(X_test)):
        if max(corr_local[idx]) > max(corr_global[idx]):
            pred_probs = model.predict_proba(X_test[idx].numpy().reshape(1, -1))
            match = False
            for max_idx in np.argsort(-pred_probs).squeeze():
                if corr_local.transpose()[y_train.numpy().squeeze() == max_idx].max() > ood_thresholds[max_idx]:
                    predictions.append(int(max_idx))
                    match = True
                    break

            if match is False:
                predictions.append(split.intents_dct['ood'])
        else:
            predictions.append(split.intents_dct['ood'])

    results_dct["results"]["local_intents"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=split.intents_dct['ood'], focus="IND")

    # # # Split dataset
    X_test, _ = split.get_X_y(dataset['global_test'], limit_num_sents=None)
    y_test = [999] * len(X_test)

    corr_local = np.inner(X_test, X_train)
    corr_global = np.inner(X_test, X_global)
    predictions = []
    for idx in range(len(X_test)):
        if corr_local[idx].max() > corr_global[idx].max():
            pred_probs = model.predict_proba(X_test[idx].numpy().reshape(1, -1))
            match = False
            for max_idx in np.argsort(-pred_probs).squeeze():
                if corr_local.transpose()[y_train.numpy().squeeze() == max_idx].max() > ood_thresholds[max_idx]:
                    predictions.append(998)
                    match = True
                    break

            if match is False:
                predictions.append(split.intents_dct['ood'])
        else:
            if corr_global[idx].max() > threshold:
                predictions.append(999)
            else:
                predictions.append(split.intents_dct['ood'])

    results_dct["results"]["global_intents"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=split.intents_dct['ood'], focus="IND")

    #
    #
    # # Split dataset
    X_test, _ = split.get_X_y(dataset['global_ood'] + dataset["local_ood"] + dataset["garbage"], limit_num_sents=None)
    y_test = [split.intents_dct['ood']] * len(X_test)

    corr_local = np.inner(X_test, X_train)
    corr_global = np.inner(X_test, X_global)
    predictions = []
    for idx in range(len(X_test)):
        match = False
        for max_idx in ood_thresholds.keys():
            if corr_global.max() > ood_thresholds[max_idx] or corr_local.max() > ood_thresholds[max_idx]:
                predictions.append(999)
                match = True
                break
        if match is False:
            predictions.append(split.intents_dct['ood'])

    results_dct["results"]["ood"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=split.intents_dct['ood'], focus="OOD")
    #
    # #
    # # # Split dataset
    # X_test, y_test = split.get_X_y(dataset['local_ood'], limit_num_sents=None)
    # # X_test, y_test = split.get_X_y(dataset['global_train'] + dataset['global_val'] + dataset['global_test'], limit_num_sents=None)
    # corr_local = np.inner(X_test, X_train)
    # corr_global = np.inner(X_test, X_global)
    # predictions = []
    # for idx in range(len(X_test)):
    #     if max(corr_local[idx]) > max(corr_global[idx]) and max(corr_local[idx]) > threshold:
    #         pred_probs = model.predict_proba(X_test[idx].numpy().reshape(1,-1))
    #         predictions.append(np.argmax(pred_probs))
    #     else:
    #         predictions.append(split.intents_dct['ood'])
    #
    # results_dct["results"]["local_ood"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=split.intents_dct['ood'], focus="OOD")
    #
    # #
    # # # Split dataset
    # X_test, y_test = split.get_X_y(dataset['garbage'], limit_num_sents=None)
    # # X_test, y_test = split.get_X_y(dataset['global_train'] + dataset['global_val'] + dataset['global_test'], limit_num_sents=None)
    # corr_local = np.inner(X_test, X_train)
    # corr_global = np.inner(X_test, X_global)
    # predictions = []
    # for idx in range(len(X_test)):
    #     if max(corr_local[idx]) > max(corr_global[idx]) and max(corr_local[idx]) > threshold:
    #         pred_probs = model.predict_proba(X_test[idx].numpy().reshape(1,-1))
    #         predictions.append(np.argmax(pred_probs))
    #     else:
    #         predictions.append(split.intents_dct['ood'])
    #
    # results_dct["results"]["garbage"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=split.intents_dct['ood'], focus="GARBAGE")

    end_time_inference = time.time()

    results_dct['time_train'] = round(end_time_train - start_time_train, 1)
    results_dct['time_inference'] = round(end_time_inference - start_time_inference, 1)
    results_dct['threshold'] = ood_thresholds  # store threshold value
    results_dct['memory'] = round(memory, 1)

    return results_dct
