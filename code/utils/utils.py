import os, random, json
import tensorflow as tf

from constants import ROOT_DIR

EXTRA_LAYER_ACT_F = tf.keras.activations.relu  # specifies the activation function of the extra layer in NNs


class TransformToEmbeddings:
    """
    Class used when splitting the training, validation and test set.

    :attributes:            intents_dct - keys: intent labels, values: unique ids, dict
                            new_key_value - keeps track of the newest unique id for intents_dct, int
                            embed_f - function that encodes sentences as embeddings
    """

    def __init__(self, embed_f, contexts=None):
        self.intents_dct = {
            "ood": -1,
            "global": -2,
            "garbage": -3
        }
        self.new_key_value = 0
        self.embed_f = embed_f
        self.contexts = contexts

    def get_X_y(self, lst, limit_num_sents=None, raw=False):
        """
        Splits a part (contained in lst) of dataset into sentences and intents.

        :param:             lst - contains the dataset, list
                            limit_num_sents - specifies (if not None) the limited number of sentences per intent, int
        :returns:           X - sentences encoded as embeddings, tf.Tensor OR sentences, list
                            y - intents, tf.Tensor
        """
        contexts = []
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

            if self.contexts is not None:
                contexts.append(random.choice(self.contexts))
            X.append(sent)
            y.append(self.intents_dct[label])

        if not raw:
            if self.embed_f is not None:
                if len(X) != 0:
                    if self.contexts is not None:
                        X = self.embed_f((contexts, X))
                    else:
                        X = self.embed_f(X)
                X = tf.convert_to_tensor(X, dtype='float32')

            y = tf.convert_to_tensor(y, dtype='int32')

        return X, y


def print_results(dataset_name: str, model_name: str, emb_name: str, results_dct: dict):
    """Helper print function."""

    print(f'dataset_name: {dataset_name}, model_name: {model_name}, embedding_name: {emb_name} -- {results_dct}\n')


def aggregate_results(dct_results_lst):
    # Aggregate results
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
                    if key == "threshold_local" or key == "threshold_global" or key == "threshold":
                        results_dct["additional"][key] = []
                    else:
                        results_dct["additional"][key] = 0

                if key == "threshold_local" or key == "threshold_global" or key == "threshold":
                    results_dct["additional"][key].append(dct[key])
                else:
                    results_dct["additional"][key] += dct[key]

    for level in results_dct:
        # if level not in ['additional']:  # keep track of total train and inference time
        for key in results_dct[level]:
            if key != "threshold_local" and key != "threshold_global" and key != "threshold":
                results_dct[level][key] /= num_results
                results_dct[level][key] = round(results_dct[level][key], 1)

    return results_dct
