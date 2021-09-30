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
        self.intents_dct = {}
        self.new_key_value = 0
        self.embed_f = embed_f

    def get_X_y(self, lst, limit_num_sents, set_type: str):
        """
        Splits a part (contained in lst) of dataset into sentences and intents.

        :param:             lst - contains the dataset, list
                            limit_num_sents - specifies (if not None) the limited number of sentences per intent, int
                            set_type - deprecated; specifies the type of the received dataset (train, val or test), str
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

            if limit_num_sents and label != 'oos':  # don't limit number of OOD sentences
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


def cross_val_evaluate(categories, evaluate, model, model_name, emb_name, embed_f, limit_num_sents):
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
            data_train = get_unsplit_Xy_ID_OOD(dialogue_path, key="train")
            data_val = get_unsplit_Xy_ID_OOD(dialogue_path, key="val")
            data_test = get_unsplit_Xy_ID_OOD(dialogue_path, key="test")

            # y_ID = [x[1] for x in Xy_ID]
            # y_OOD = [x[1] for x in Xy_OOD]

            dataset = {}

            for decision_node in data_train.keys():
                print("X", end="")
                Xy_ID_tr, Xy_OOD_tr = data_train[decision_node]
                Xy_ID_val, Xy_OOD_val = data_val[decision_node]
                Xy_ID_te, Xy_OOD_te = data_test[decision_node]

                dataset['train'] = Xy_ID_tr
                dataset['val'] = Xy_ID_val
                dataset['test'] = Xy_ID_te

                # train_idx_OOD, val_idx_OOD = train_test_split(train_idx_OOD, test_size=0.2)
                # TODO Xy_OOD_te
                dataset['oos_train'] = Xy_OOD_tr
                dataset['oos_val'] = Xy_OOD_val
                dataset['oos_test'] = Xy_OOD_te

                results_dct = evaluate(dataset, model, model_name, embed_f, limit_num_sents)
                dct_results_lst.append(results_dct)
            print()

    results_dct = {}
    num_results = len(dct_results_lst)

    for dct in dct_results_lst:
        for key in dct:
            if key not in results_dct:
                results_dct[key] = 0

            results_dct[key] += dct[key]

    for key in results_dct:
        if key not in ['time_train', 'time_inference']:  # keep track of total train and inference time
            results_dct[key] /= num_results

        results_dct[key] = round(results_dct[key], 1)

    if total_time_pretraining != 0:
        results_dct['time_pretraining'] = round(total_time_pretraining, 1)

    return results_dct, emb_name
