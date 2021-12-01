from utils import Split
from utils.testing import Testing

import time, psutil
import numpy as np


def find_best_threshold(val_predictions_labels, oos_label):
    """
    Function used to find the best threshold in oos-threshold.
    :param:            val_predictions_labels - prediction on the validation set, list
                        oos_label - encodes oos label, int
    :returns:           threshold - best threshold
    """

    # Initialize search for best threshold
    thresholds = np.linspace(0, 1, 101)
    previous_val_accuracy = 0
    threshold = 0

    # Find best threshold
    for idx, tr in enumerate(thresholds):
        val_accuracy_correct = 0
        val_accuracy_out_of = 0

        for pred, true_label in val_predictions_labels:
            pred_label = pred[0]
            similarity = pred[1]

            if similarity < tr:
                pred_label = oos_label

            if pred_label == true_label:
                val_accuracy_correct += 1

            val_accuracy_out_of += 1

        val_accuracy = val_accuracy_correct / val_accuracy_out_of

        if val_accuracy < previous_val_accuracy:
            threshold = thresholds[idx - 1]  # best threshold is the previous one
            break

        previous_val_accuracy = val_accuracy
        threshold = tr

    return threshold


def evaluate(dataset, model, model_name, embed_f, limit_num_sents, find_best_threshold_fn):
    split = Split(embed_f)

    # TRAINING
    start_time_train = time.time()

    # Split dataset
    X_train, y_train = split.get_X_y(dataset['train'] + dataset["global_train"], limit_num_sents=limit_num_sents)
    X_val, y_val = split.get_X_y(dataset['val'] + dataset["global_val"], limit_num_sents=limit_num_sents)

    # Train
    model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    # Find threshold
    val_predictions_labels = []  # used to find threshold
    pred_probs = model.predict_proba(X_val)  # function available in both scikit-learn and TF-Keras, returns numpy array

    pred_labels = np.argmax(pred_probs, axis=1)
    pred_similarities = np.take_along_axis(pred_probs, indices=np.expand_dims(pred_labels, axis=1), axis=1).squeeze()

    predictions = np.column_stack([pred_labels, pred_similarities])  # 2D list of [pred_label, similarity]

    for pred, true_label in zip(predictions, y_val):
        val_predictions_labels.append((pred, true_label))

    threshold = find_best_threshold_fn(val_predictions_labels, split.intents_dct['ood'])
    if threshold is None:
        threshold = find_best_threshold(val_predictions_labels, split.intents_dct['ood'])

    end_time_train = time.time()

    memory = psutil.Process().memory_full_info().uss / (1024 ** 2)  # in megabytes

    # TESTING
    results_dct = {"results": {}}
    start_time_inference = time.time()

    # Split dataset
    X_test, y_test = split.get_X_y(dataset['test'], limit_num_sents=None)
    # Tess
    results_dct["results"]["local_intents"] = Testing.test_threshold(model, X_test, y_test, split.intents_dct['ood'], threshold, focus="IND")

    # Split dataset
    X_test, y_test = split.get_X_y(dataset['global_test'], limit_num_sents=None)
    # y_test = [split.intents_dct['ood']] * len(X_test)
    # Test
    results_dct["results"]["global_intents"] = Testing.test_threshold(model, X_test, y_test, split.intents_dct['ood'], threshold, focus="IND")

    # # Split dataset
    # X_test, y_test = split.get_X_y(dataset['global_ood'], limit_num_sents=None)
    # # Test
    # results_dct["results"]["global_ood"] = Testing.test_threshold(model, X_test, y_test, split.intents_dct['ood'], threshold,  focus="OOD")
    #
    # # Split dataset
    # X_test, y_test = split.get_X_y(dataset['local_ood'], limit_num_sents=None)
    # # Test
    # results_dct["results"]["local_ood"] = Testing.test_threshold(model, X_test, y_test, split.intents_dct['ood'], threshold,  focus="OOD")
    #
    # # Split dataset
    # X_test, y_test = split.get_X_y(dataset['garbage'], limit_num_sents=None)
    # # Test
    # results_dct["results"]["garbage"] = Testing.test_threshold(model, X_test, y_test, split.intents_dct['ood'], threshold,  focus="GARBAGE")

    end_time_inference = time.time()

    results_dct['time_train'] = round(end_time_train - start_time_train, 1)
    results_dct['time_inference'] = round(end_time_inference - start_time_inference, 1)
    results_dct['threshold'] = threshold  # store threshold value
    results_dct['memory'] = round(memory, 1)

    return results_dct
