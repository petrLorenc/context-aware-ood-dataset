import json
import os

from constants import ROOT_DIR
from enum import Enum, auto
from abc import ABC, abstractmethod, ABCMeta


class DatasetType(Enum):
    ORIGINAL = auto()
    FLATTEN = auto()
    CLINC150 = auto()
    ROASTD = auto()
    GENERAL = auto()


class DatasetReturnType(Enum):
    YIELD_SEPARATELY = auto()
    RETURN_ALL = auto()


class DatasetGenerator(ABC):
    def __init__(self, dialogue_path):
        self.dialogue_path = dialogue_path
        with open(self.dialogue_path) as f:
            self.dialogue = json.load(f)

    @abstractmethod
    def get_ood(self, level):
        raise NotImplemented()

    @abstractmethod
    def get_intent(self, key, level):
        raise NotImplemented()

    @abstractmethod
    def get_garbage(self):
        raise NotImplemented()


class DatasetOriginal(DatasetGenerator):
    def get_ood(self, level):
        category = self.dialogue_path.split(sep=os.sep)[-2]
        # ood_path = os.path.join("..", "data", 'cross_ood', f'{category}_ood.json')
        ood_path = os.path.join(ROOT_DIR, "data", 'clinc_ood', f'{category}_ood.json')
        decision_nodes = self.dialogue['decisionNodes']

        intents_decision_node = {}

        for node in decision_nodes:
            intents_decision_node[node] = []

            if level == "local":
                for sent in self.dialogue["outOfDomain"][str(node)]:
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

    def get_intent(self, key, level):
        decision_nodes = self.dialogue['decisionNodes']

        intents_decision_node = {}

        for node in decision_nodes:
            intents_decision_node[node] = []

            if level == "local":
                for intent in self.dialogue['links'][str(node)]:
                    if str(intent) not in self.dialogue['intents'].keys() or key not in self.dialogue['intents'][str(intent)]:
                        continue

                    for sent in self.dialogue['intents'][str(intent)][key]:
                        intents_decision_node[node].append([sent, str(intent)])
            elif level == "global":
                for intent in self.dialogue['globalIntents']:
                    if key not in self.dialogue['globalIntents'][intent]:
                        continue

                    for sent in self.dialogue['globalIntents'][intent][key]:
                        intents_decision_node[node].append([sent, str(intent)])
            else:
                raise Exception(f"Unknown level {level}")

        return intents_decision_node

    def get_garbage(self):
        garbages = []
        with open(os.path.join(ROOT_DIR, 'data', "garbage", "garbage.txt"), "r") as f:
            for sent in f.readlines():
                garbages.append([sent.strip(), "garbage"])
        return garbages


class DatasetFlatten(DatasetGenerator):

    def __init__(self, dialogue_path, global_dialogue_path):
        with open(global_dialogue_path) as f:
            self.global_dialogue = json.load(f)
        super().__init__(dialogue_path)

    def get_ood(self, level):
        node = self.dialogue["reference"]["decision_node"]
        intents_decision_node = {node: []}

        if level == "local":
            for sent in self.dialogue["out_of_domain"]:
                intents_decision_node[node].append([sent, "ood"])

        elif level == "global":
            for sent in self.global_dialogue['ood']:
                intents_decision_node[node].append([sent, 'ood'])
        else:
            raise Exception(f"Unknown level {level}")

        return intents_decision_node

    def get_intent(self, key, level):
        node = self.dialogue["reference"]["decision_node"]
        intents_decision_node = {node: []}

        if level == "local":
            for idx, intent in enumerate(self.dialogue['user_response']):
                for sent in self.dialogue['user_response'][idx][key]:
                    intents_decision_node[node].append([sent, idx])
        elif level == "global":
            for intent in self.global_dialogue.keys():
                if intent == "ood":
                    continue
                for sent in self.global_dialogue[intent][key]:
                    intents_decision_node[node].append([sent, str(intent)])
        else:
            raise Exception(f"Unknown level {level}")

        return intents_decision_node

    def get_garbage(self):
        garbages = []
        with open(os.path.join(ROOT_DIR, 'data', "garbage", "garbage.txt"), "r") as f:
            for sent in f.readlines():
                garbages.append([sent.strip(), "garbage"])
        return garbages


def generate_dataset(categories,
                     test_label,
                     dataset_type: DatasetType = DatasetType.ORIGINAL,
                     return_type: DatasetReturnType = DatasetReturnType.YIELD_SEPARATELY):
    dataset_grouped = []
    if dataset_type is DatasetType.ORIGINAL:

        # Prepare files
        for cat in categories:
            annotated_files_path = os.path.join(ROOT_DIR, 'data', "dialogues", cat)
            files_path = [os.path.join(annotated_files_path, ds) for ds in os.listdir(annotated_files_path)]

            for file_path in files_path:
                print(file_path)
                datasetGenerator = DatasetOriginal(file_path)

                local_data_train = datasetGenerator.get_intent(key="train", level="local")
                local_data_val = datasetGenerator.get_intent(key="val", level="local")
                local_data_test = datasetGenerator.get_intent(key=test_label, level="local")

                global_data_train = datasetGenerator.get_intent(key="train", level="global")
                global_data_valid = datasetGenerator.get_intent(key="val", level="global")
                global_data_test = datasetGenerator.get_intent(key=test_label, level="global")

                local_ood = datasetGenerator.get_ood(level="local")
                global_ood = datasetGenerator.get_ood(level="global")
                garbage = datasetGenerator.get_garbage()

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

                    if return_type is DatasetReturnType.YIELD_SEPARATELY:
                        yield dataset
                    elif return_type is DatasetReturnType.RETURN_ALL:
                        dataset_grouped.append(dataset)

        if return_type is DatasetReturnType.RETURN_ALL:
            return [dataset_grouped]

    elif dataset_type is DatasetType.FLATTEN:
        # Prepare files
        annotated_files_path = os.path.join(ROOT_DIR, 'data', "flatten_dialogues", "annotated", "ok")
        global_path = os.path.join(ROOT_DIR, 'data', "flatten_dialogues", "annotated", "ok", "global.json")
        files_path = [os.path.join(annotated_files_path, ds) for ds in os.listdir(annotated_files_path)]

        for file_path in files_path:
            print(file_path)
            if file_path.endswith("global.json"):
                continue
            datasetGenerator = DatasetFlatten(file_path, global_path)

            local_data_train = datasetGenerator.get_intent(key="train", level="local")
            local_data_val = datasetGenerator.get_intent(key="val", level="local")
            local_data_test = datasetGenerator.get_intent(key=test_label, level="local")

            global_data_train = datasetGenerator.get_intent(key="train", level="global")
            global_data_valid = datasetGenerator.get_intent(key="val", level="global")
            global_data_test = datasetGenerator.get_intent(key=test_label, level="global")

            local_ood = datasetGenerator.get_ood(level="local")
            global_ood = datasetGenerator.get_ood(level="global")
            garbage = datasetGenerator.get_garbage()

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

                if return_type is DatasetReturnType.YIELD_SEPARATELY:
                    yield dataset
                elif return_type is DatasetReturnType.RETURN_ALL:
                    dataset_grouped.append(dataset)

            if return_type is DatasetReturnType.RETURN_ALL:
                return [dataset_grouped]
