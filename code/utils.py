import os, random, json
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split

EXTRA_LAYER_ACT_F = tf.keras.activations.relu  # specifies the activation function of the extra layer in NNs
NEEDS_VAL = ['BaselineNN', 'BaselineNNExtraLayer']
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results')


class Split:
    """
    Class used when splitting the training, validation and test set.

    :attributes:            intents_dct - keys: intent labels, values: unique ids, dict
                            new_key_value - keeps track of the newest unique id for intents_dct, int
                            embed_f - function that encodes sentences as embeddings
    """

    def __init__(self, embed_f):
        self.intents_dct = {
            "ood": -1,
            "global": -2,
            "garbage": -3
        }
        self.new_key_value = 0
        self.embed_f = embed_f

    def get_X_y(self, lst, limit_num_sents):
        """
        Splits a part (contained in lst) of dataset into sentences and intents.

        :param:             lst - contains the dataset, list
                            limit_num_sents - specifies (if not None) the limited number of sentences per intent, int
        :returns:           X - sentences encoded as embeddings, tf.Tensor OR sentences, list
                            y - intents, tf.Tensor
        """

        X = []
        y = []

        if limit_num_sents:  # these aren't needed normally
            random.shuffle(lst)
            label_occur_count = {}

        for sent, label in lst:
            if label not in self.intents_dct.keys():
                self.intents_dct[label] = self.new_key_value
                self.new_key_value += 1

            if limit_num_sents and label != 'ood':  # don't limit number of OOD sentences
                if label not in label_occur_count.keys():
                    label_occur_count[label] = 0

                if label_occur_count[label] == limit_num_sents:  # skip sentence and label if reached limit
                    continue

                label_occur_count[label] += 1

            X.append(sent)
            y.append(self.intents_dct[label])

        if self.embed_f is not None:
            X = self.embed_f(X)
            X = tf.convert_to_tensor(X, dtype='float32')

        y = tf.convert_to_tensor(y, dtype='int32')

        return X, y


def print_results(dataset_name: str, model_name: str, emb_name: str, results_dct: dict):
    """Helper print function."""

    print(f'dataset_name: {dataset_name}, model_name: {model_name}, embedding_name: {emb_name} -- {results_dct}\n')


def get_unsplit_Xy_ID_OOD(dialogue_path, key):
    category = dialogue_path.split(sep=os.sep)[-2]
    ood_path = os.path.join("..", "data", 'cross_ood', f'{category}_ood.json')
    # ood_path = os.path.join("..", "data", 'clinc_ood', f'{category}_ood.json')

    with open(dialogue_path) as f:
        # print(dialogue_path)
        dialogue = json.load(f)

    decision_nodes = dialogue['decisionNodes']
    intents_sel = []

    intents_decision_node = {}

    for node in decision_nodes:
        intents_decision_node[node] = []

        Xy_ID = []

        for intent in dialogue['links'][str(node)]:
            if str(intent) not in dialogue['intents'].keys():
                continue

            for sent in dialogue['intents'][str(intent)][key]:
                Xy_ID.append([sent, str(intent)])

        for intent in dialogue['globalIntents']:
            for sent in dialogue['globalIntents'][intent][key]:
                Xy_ID.append([sent, intent])

        Xy_OOD = []

        with open(ood_path) as f:
            ood_cross = json.load(f)

        if key == "train":
            for sent in ood_cross['ood_train']:
                Xy_OOD.append([sent, 'oos'])
        else:
            for sent in ood_cross['ood_test']:
                Xy_OOD.append([sent, 'oos'])

        intents_decision_node[node] = (Xy_ID, Xy_OOD)

    return intents_decision_node


def get_ood(dialogue_path, level):
    category = dialogue_path.split(sep=os.sep)[-2]
    # ood_path = os.path.join("..", "data", 'cross_ood', f'{category}_ood.json')
    ood_path = os.path.join("..", "data", 'clinc_ood', f'{category}_ood.json')

    with open(dialogue_path) as f:
        # print(dialogue_path)
        dialogue = json.load(f)

    decision_nodes = dialogue['decisionNodes']

    intents_decision_node = {}

    for node in decision_nodes:
        intents_decision_node[node] = []

        if level == "local":
            for sent in dialogue["outOfDomain"][str(node)]:
                intents_decision_node[node].append([sent, "ood"])

        elif level == "global":
            with open(ood_path) as f:
                ood_cross = json.load(f)

            for sent in ood_cross['ood_train']:
                intents_decision_node[node].append([sent, 'ood'])
            for sent in ood_cross['ood_test']:
                intents_decision_node[node].append([sent, 'ood'])
        else:
            raise Exception(f"Unknown level {level}")

    return intents_decision_node


def get_intent(dialogue_path, key, level):
    with open(dialogue_path) as f:
        dialogue = json.load(f)

    decision_nodes = dialogue['decisionNodes']

    intents_decision_node = {}

    for node in decision_nodes:
        intents_decision_node[node] = []

        if level == "local":
            for intent in dialogue['links'][str(node)]:
                if str(intent) not in dialogue['intents'].keys():
                    continue

                for sent in dialogue['intents'][str(intent)][key]:
                    intents_decision_node[node].append([sent, str(intent)])
        elif level == "global":
            for intent in dialogue['globalIntents']:
                for sent in dialogue['globalIntents'][intent][key]:
                    intents_decision_node[node].append([sent, "global"])
        else:
            raise Exception(f"Unknown level {level}")

    return intents_decision_node


def get_garbage():
    garbages = []
    with open(os.path.join("..", 'data', "garbage", "garbage.txt"), "r") as f:
        for sent in f.readlines():
            garbages.append([sent.strip(), "garbage"])
    return garbages


def iterative_evalutation(categories, evaluate, model, model_name, emb_name, embed_f, limit_num_sents, find_best_threshold_fn):
    original_emb_name = emb_name
    dct_results_lst = []
    total_time_pretraining = 0

    for cat in categories:
        cat_path = os.path.join("..", 'data', "dialogues", cat)
        dataset_paths = [os.path.join(cat_path, ds) for ds in os.listdir(cat_path)]

        for dialogue_path in dataset_paths:
            # Xy_ID_tr, Xy_OOD_tr = get_unsplit_Xy_ID_OOD(dialogue_path, key="train")
            # Xy_ID_te, Xy_OOD_te = get_unsplit_Xy_ID_OOD(dialogue_path, key="test")
            print(dialogue_path)

            local_data_train = get_intent(dialogue_path, key="train", level="local")
            local_data_val = get_intent(dialogue_path, key="val", level="local")
            local_data_test = get_intent(dialogue_path, key="test", level="local")

            global_data_train = get_intent(dialogue_path, key="test", level="global")
            global_data_valid = get_intent(dialogue_path, key="test", level="global")
            global_data_test = get_intent(dialogue_path, key="test", level="global")

            local_ood = get_ood(dialogue_path, level="local")
            global_ood = get_ood(dialogue_path, level="global")
            garbage = get_garbage()

            dataset = {}

            for decision_node in local_data_train.keys():
                print(decision_node)
                dataset['train'] = local_data_train[decision_node]
                dataset['val'] = local_data_val[decision_node]
                dataset['test'] = local_data_test[decision_node]

                dataset['global_train'] = global_data_train[decision_node]
                dataset['global_val'] = global_data_valid[decision_node]
                dataset['global_test'] = global_data_test[decision_node]

                dataset["local_ood"] = local_ood[decision_node]
                dataset["global_ood"] = global_ood[decision_node]
                dataset["garbage"] = garbage

                # !!
                results_dct = evaluate(dataset, model, model_name, embed_f, limit_num_sents, find_best_threshold_fn)
                dct_results_lst.append(results_dct)
                print(results_dct)
            print()

    results_dct = {}
    num_results = len(dct_results_lst)

    for dct in dct_results_lst:
        for level in dct["results"].keys():
            if level not in results_dct:
                results_dct[level] = {}
            for key in dct["results"][level]:
                if key not in results_dct[level]:
                    results_dct[level][key] = 0

                results_dct[level][key] += dct["results"][level][key]

            for key in dct:
                if key == "results":
                    continue
                if "additional" not in results_dct:
                    results_dct["additional"] = {}
                if key not in results_dct["additional"]:
                    if key == "threshold":
                        results_dct["additional"][key] = []
                    else:
                        results_dct["additional"][key] = 0

                if key == "threshold":
                    results_dct["additional"][key].append(dct[key])
                else:
                    results_dct["additional"][key] += dct[key]

    for level in results_dct:
        # if level not in ['additional']:  # keep track of total train and inference time
        for key in results_dct[level]:
            if key != "threshold":
                results_dct[level][key] /= num_results
                results_dct[level][key] = round(results_dct[level][key], 1)

    if total_time_pretraining != 0:
        results_dct["additional"]['time_pretraining'] = round(total_time_pretraining, 1)

    return results_dct, emb_name
