import psutil
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from models import AbstractModel
from utils.testing import Testing
from utils.utils import TransformToEmbeddings


def evaluate(dataset, embedding_model, classification_model, get_threshold):
    local_split = TransformToEmbeddings(embedding_model)
    global_split = TransformToEmbeddings(embedding_model)

    # Split dataset
    X_train_context, X_train_utterance, y_train_with_context, begin_end_mask_train = [], [], [], []
    X_val_context, X_val_utterance, y_val_with_context, begin_end_mask_val = [], [], [], []

    local_X_test_context, local_X_test_utterance, local_y_test, begin_end_masks_test_local, begin_end_masks_test_global = [], [], [], [], []
    global_X_test_context, global_X_test_utterances, global_y_test = [], [], []

    local_X_ood_context, local_X_ood_utterance, begin_end_masks_ood_local = [], [], []

    for d in dataset:
        max_idx = max(y_train_with_context) + 1 if len(y_train_with_context) else 0
        mask_idx_begin = max_idx
        num_examples_begin = len(X_train_utterance)
        for s, y in d['train']:
            for c in d["context"]:
                X_train_context.append(f"{c}")
                X_train_utterance.append(f"{s}")
                y_train_with_context.append(y + max_idx)
        mask_idx_end = max(y_train_with_context) + 1
        num_examples_end = len(X_train_utterance)
        begin_end_mask_train.append((num_examples_begin, num_examples_end, mask_idx_begin, mask_idx_end))

        num_examples_begin = len(X_val_utterance)
        for s, y in d['val']:
            for c in d["context"]:
                X_val_context.append(f"{c}")
                X_val_utterance.append(f"{s}")
                y_val_with_context.append(y + max_idx)
        num_examples_end = len(X_val_utterance)
        begin_end_mask_val.append((num_examples_begin, num_examples_end, mask_idx_begin, mask_idx_end))

        num_examples_begin = len(local_X_test_utterance)
        for s, y in d['test']:
            for c in d["context"]:
                local_X_test_context.append(f"{c}")
                local_X_test_utterance.append(f"{s}")
                local_y_test.append(y + max_idx)
        num_examples_end = len(local_X_test_utterance)
        begin_end_masks_test_local.append((num_examples_begin, num_examples_end, mask_idx_begin, mask_idx_end))

        num_examples_begin = len(local_X_ood_context)
        for s, y in d['local_ood']:
            for c in d["context"]:
                local_X_ood_context.append(f"{c}")
                local_X_ood_utterance.append(f"{s}")
        for s, y in d['garbage']:
            for c in d["context"]:
                local_X_ood_context.append(f"{c}")
                local_X_ood_utterance.append(f"{s}")
        for s, y in d['global_ood']:
            for c in d["context"]:
                local_X_ood_context.append(f"{c}")
                local_X_ood_utterance.append(f"{s}")
        num_examples_end = len(local_X_ood_context)
        begin_end_masks_ood_local.append((num_examples_begin, num_examples_end, mask_idx_begin, mask_idx_end))

    max_idx = max(y_train_with_context) + 1 if len(y_train_with_context) else 0
    global_mask_idx_begin = max_idx
    for idx, d in enumerate(dataset):
        _, _, b, e = begin_end_mask_train[idx]
        begin_mask = len(X_train_utterance)
        for s, y in d['global_train']:
            for c in d["context"]:
                X_train_context.append(f"{c}")
                X_train_utterance.append(f"{s}")
                y_train_with_context.append(y + max_idx)
        begin_end_mask_train.append((begin_mask, len(X_train_utterance), b, e))

        _, _, b, e = begin_end_mask_val[idx]
        begin_mask = len(X_val_utterance)
        for s, y in d['global_val']:
            for c in d["context"]:
                X_val_context.append(f"{c}")
                X_val_utterance.append(f"{s}")
                y_val_with_context.append(y + max_idx)
        begin_end_mask_val.append((begin_mask, len(X_val_utterance), b, e))

        _, _, b, e = begin_end_masks_test_local[idx]
        begin_mask = len(global_X_test_utterances)
        for s, y in d['global_test']:
            for c in d["context"]:
                global_X_test_context.append(f"{c}")
                global_X_test_utterances.append(f"{s}")
                global_y_test.append(y + max_idx)
        begin_end_masks_test_global.append((begin_mask, len(global_X_test_utterances), b, e))

    global_mask_idx_end = max(y_train_with_context) + 1

    start_time_train = time.time()
    # Train
    num_classes = max(y_train_with_context) + 1
    classification_model.create_model(num_classes)
    y_train_with_context = to_categorical(y_train_with_context, num_classes=num_classes)
    y_val_with_context = to_categorical(y_val_with_context, num_classes=num_classes)

    mask_train = np.zeros_like(y_train_with_context)
    for b_idx, e_idx, b, e in begin_end_mask_train:
        mask_train[b_idx:e_idx, b:e] = 1
    mask_train[:, global_mask_idx_begin:global_mask_idx_end] = 1

    mask_val = np.zeros_like(y_val_with_context)
    for b_idx, e_idx, b, e in begin_end_mask_val:
        mask_val[b_idx:e_idx, b:e] = 1
    mask_val[:, global_mask_idx_begin:global_mask_idx_end] = 1

    mask_test_local = np.zeros(shape=(len(local_y_test), num_classes))
    for b_idx, e_idx, b, e in begin_end_masks_test_local:
        mask_test_local[b_idx:e_idx, b:e] = 1
    mask_test_local[:, global_mask_idx_begin:global_mask_idx_end] = 1

    mask_test_global = np.zeros(shape=(len(global_y_test), num_classes))
    for b_idx, e_idx, b, e in begin_end_masks_test_global:
        mask_test_global[b_idx:e_idx, b:e] = 1
    mask_test_global[:, global_mask_idx_begin:global_mask_idx_end] = 1

    mask_ood = np.zeros(shape=(len(local_X_ood_context), num_classes))
    for b_idx, e_idx, b, e in begin_end_masks_ood_local:
        mask_ood[b_idx:e_idx, b:e] = 1
    mask_ood[:, global_mask_idx_begin:global_mask_idx_end] = 1

    history = classification_model.fit(X_train=(X_train_context, X_train_utterance, mask_train), y_train=y_train_with_context,
                                       X_val=(X_val_context, X_val_utterance, mask_val), y_val=y_val_with_context)

    end_time_train = time.time()

    memory = psutil.Process().memory_full_info().uss / (1024 ** 2)  # in megabytes

    # TESTING
    results_dct = {"results": {}}
    start_time_inference = time.time()

    _predictions = classification_model.predict_proba((local_X_test_context, local_X_test_utterance, mask_test_local))
    predictions = list(map(lambda x: np.argmax(x) if np.max(x) > get_threshold(None) else global_split.intents_dct['ood'], _predictions))
    threshold_local = np.max(_predictions, axis=1)

    results_dct["results"]["local_intents"] = Testing.test_illusionist(y_pred=predictions, y_test=local_y_test,
                                                                       oos_label=global_split.intents_dct['ood'],
                                                                       focus="IND")

    # # # Split dataset
    _predictions = classification_model.predict_proba((global_X_test_context, global_X_test_utterances, mask_test_global))
    predictions = list(map(lambda x: np.argmax(x) if np.max(x) > get_threshold(None) else global_split.intents_dct['ood'], _predictions))
    threshold_global = np.max(_predictions, axis=1)

    results_dct["results"]["global_intents"] = Testing.test_illusionist(y_pred=predictions, y_test=global_y_test,
                                                                        oos_label=global_split.intents_dct['ood'],
                                                                        focus="IND")

    y_test = [global_split.intents_dct['ood']] * len(local_X_ood_context)
    _predictions = classification_model.predict_proba((local_X_ood_context, local_X_ood_utterance, mask_ood))
    predictions = list(map(lambda x: np.argmax(x) if np.max(x) > get_threshold(None) else global_split.intents_dct['ood'], _predictions))

    results_dct["results"]["ood"] = Testing.test_illusionist(y_pred=predictions, y_test=y_test, oos_label=local_split.intents_dct['ood'], focus="OOD")

    end_time_inference = time.time()

    results_dct['time_train'] = round(end_time_train - start_time_train, 1)
    results_dct['time_inference'] = round(end_time_inference - start_time_inference, 1)
    results_dct['threshold_local'] = np.average(threshold_local)  # store threshold value
    results_dct['threshold_global'] = np.average(threshold_global)  # store threshold value
    results_dct['memory'] = round(memory, 1)

    return results_dct
