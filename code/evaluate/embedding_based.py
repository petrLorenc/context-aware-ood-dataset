import psutil
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from models.abstract_neural_net import AbstractModel
from utils.testing import Testing
from utils.utils import TransformToEmbeddings


def evaluate(dataset, embedding_model, classification_model, limit_num_sents):
    local_split = TransformToEmbeddings(embedding_model)
    global_split = TransformToEmbeddings(embedding_model)

    # Split dataset
    X_train_with_context, y_train_with_context = [], []
    X_val_with_context, y_val_with_context = [], []

    X_test_with_context, y_test_with_context, begin_end_masks = [], [], []
    global_X_test_with_context, global_y_test_with_context = [], []


    for d in dataset:
        max_idx = max(y_train_with_context) if len(y_train_with_context) else 0
        mask_idx_begin = max_idx
        for s, y in d['train']:
            for c in d["context"]:
                X_train_with_context.append(f"{c}<SEP>{s}")
                y_train_with_context.append(y + max_idx)
        mask_idx_end = max(y_train_with_context)
        for s, y in d['val']:
            for c in d["context"]:
                X_val_with_context.append(f"{c}<SEP>{s}")
                y_val_with_context.append(y + max_idx)
        for s, y in d['test']:
            for c in d["context"]:
                X_test_with_context.append(f"{c}<SEP>{s}")
                y_test_with_context.append(y + max_idx)
        begin_end_masks.append((mask_idx_begin, mask_idx_end))


    max_idx = max(y_train_with_context) if len(y_train_with_context) else 0
    global_mask_idx_begin = max_idx
    for d in dataset:
        for s, y in d['global_train']:
            for c in d["context"]:
                X_train_with_context.append(f"{c}<SEP>{s}")
                y_train_with_context.append(y + max_idx)
        for s, y in d['global_val']:
            for c in d["context"]:
                X_val_with_context.append(f"{c}<SEP>{s}")
                y_val_with_context.append(y + max_idx)
        for s, y in d['global_test']:
            for c in d["context"]:
                global_X_test_with_context.append(f"{c}<SEP>{s}")
                global_y_test_with_context.append(y + max_idx)
    global_mask_idx_end = max(y_train_with_context)

    start_time_train = time.time()
    # Train
    num_classes = max(y_train_with_context) + 1
    classification_model.create_model(num_classes)
    y_train_with_context = to_categorical(y_train_with_context, num_classes=num_classes)
    y_val_with_context = to_categorical(y_val_with_context, num_classes=num_classes)

    mask_test_with_context = np.zeros_like(y_train_with_context)
    for idx, (b, e) in enumerate(begin_end_masks):
        mask_test_with_context[idx, b:e] = 1
        mask_test_with_context[idx, global_mask_idx_begin:global_mask_idx_end] = 1

    history = classification_model.fit(X_train=X_train_with_context, y_train=y_train_with_context, X_val=X_val_with_context, y_val=y_val_with_context)

    end_time_train = time.time()

    memory = psutil.Process().memory_full_info().uss / (1024 ** 2)  # in megabytes

    # TESTING
    results_dct = {"results": {}}
    start_time_inference = time.time()

    predictions = []
    for idx, s in enumerate(X_test_with_context):
        pred = classification_model.predict_proba([s], mask_test_with_context[idx])
        predictions.append(np.argmax(pred))

    results_dct["results"]["local_intents"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test_with_context, oos_label=global_split.intents_dct['ood'],
                                                                       focus="IND")

    # # # Split dataset
    predictions = []
    for idx, s in enumerate(global_X_test_with_context):
        pred = classification_model.predict_proba([s], mask_test_with_context[idx])
        predictions.append(np.argmax(pred))

    results_dct["results"]["global_intents"] = Testing.test_illusionist(y_pred=predictions, y_test=global_y_test_with_context, oos_label=global_split.intents_dct['ood'],
                                                                        focus="IND")

    #
    # X_test, _ = global_split.get_X_y(dataset['global_ood'] + dataset["local_ood"] + dataset["garbage"], limit_num_sents=None)
    # # X_test = ["tell me more about your relatives", "what is the weather in Prague"]
    # # X_test = tf.convert_to_tensor(embedding_model(X_test), dtype='float32')
    # y_test = [global_split.intents_dct['ood']] * len(X_test)
    #
    # corr_local = np.inner(X_train, X_test)
    # corr_local = np.exp(corr_local - 1)
    #
    # corr_global = np.inner(X_global, X_test)
    # corr_global = np.exp(corr_global - 1)
    #
    # predictions = []
    # for idx in range(len(X_test)):
    #     match = False
    #     for max_idx in ood_thresholds_local.keys():
    #         if corr_local[:, idx][y_train.numpy().squeeze() == max_idx].max() > ood_thresholds_local[max_idx]:
    #             predictions.append(999)
    #             match = True
    #             break
    #     if match:
    #         continue
    #     if not match:
    #         for max_idx in ood_thresholds_global.keys():
    #             if corr_global[:, idx][y_global.numpy().squeeze() == max_idx].max() > ood_thresholds_global[max_idx]:
    #                 predictions.append(999)
    #                 match = True
    #                 break
    #     if match:
    #         continue
    #
    #     predictions.append(global_split.intents_dct['ood'])
    #
    # results_dct["results"]["ood"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=local_split.intents_dct['ood'], focus="OOD")

    end_time_inference = time.time()

    results_dct['time_train'] = round(end_time_train - start_time_train, 1)
    results_dct['time_inference'] = round(end_time_inference - start_time_inference, 1)
    results_dct['threshold_local'] = []  # store threshold value
    results_dct['threshold_global'] = []  # store threshold value
    results_dct['memory'] = round(memory, 1)

    return results_dct
