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
        if global_dialogue_path is not None:
            with open(global_dialogue_path) as f:
                self.global_dialogue = json.load(f)
        else:
            self.global_dialogue = {
                "user_response": {},
                "out_of_domain": []
            }
        super().__init__(dialogue_path)

    def get_ood(self, level):
        node = self.dialogue["reference"]["decision_node"] if "reference" in self.dialogue else "NO_REFERENCE"
        intents_decision_node = {node: []}

        if level == "local":
            for sent in self.dialogue["out_of_domain"]:
                intents_decision_node[node].append([sent, "ood"])

        elif level == "global":
            for sent in self.global_dialogue['out_of_domain']:
                intents_decision_node[node].append([sent, 'ood'])
        else:
            raise Exception(f"Unknown level {level}")

        return intents_decision_node

    def get_intent(self, key, level):
        node = self.dialogue["reference"]["decision_node"] if "reference" in self.dialogue else "NO_REFERENCE"
        intents_decision_node = {node: []}

        if level == "local":
            for idx, _ in enumerate(self.dialogue['user_response']):
                for sent in self.dialogue['user_response'][idx][key]:
                    intents_decision_node[node].append([sent, idx])
        elif level == "global":
            for idx, _ in enumerate(self.global_dialogue['user_response']):
                for sent in self.global_dialogue['user_response'][idx][key]:
                    intents_decision_node[node].append([sent, idx])
        else:
            raise Exception(f"Unknown level {level}")

        return intents_decision_node

    def get_garbage(self):
        garbages = []
        with open(os.path.join(ROOT_DIR, 'data', "garbage", "garbage.txt"), "r") as f:
            for sent in f.readlines():
                garbages.append([sent.strip(), "garbage"])
        return garbages

    def get_context(self):
        return self.dialogue["bot_response"] if "bot_response" in self.dialogue else []


class DatasetCLINC(DatasetGenerator):

    def __init__(self, dialogue_path, domains_path):
        super().__init__(dialogue_path)
        if domains_path:
            with open(domains_path) as f:
                self.domains = json.load(f)

    def get_ood(self, level):
        node = self.dialogue["reference"]["decision_node"] if "reference" in self.dialogue else "NO_REFERENCE"
        intents_decision_node = {node: []}
        if level == "local":
            other_intents = list(set([x[1] for x in self.dialogue["train"] if x[1] in self.domains["travel"]]))
            local_intents = [[x[0], other_intents.index(x[1])] for x in self.dialogue["train"] if x[1] in self.domains["travel"]]
            intents_decision_node[node] = local_intents
        if level == "global":
            intents_decision_node = {node: [[x[0], "ood"] for x in self.dialogue["oos_test"]]}

        return intents_decision_node

    def get_intent(self, key, level):
        node = self.dialogue["reference"]["decision_node"] if "reference" in self.dialogue else "NO_REFERENCE"
        intents_decision_node = {node: []}

        if level == "local":
            other_intents = list(set([x[1] for x in self.dialogue[key] if x[1] in self.domains["work"]]))
            local_intents = [[x[0], other_intents.index(x[1])] for x in self.dialogue[key] if x[1] in self.domains["work"]]
            intents_decision_node[node] = local_intents

        elif level == "global":
            global_intents = [[x[0], self.domains["meta"].index(x[1])] for x in self.dialogue[key] if x[1] in self.domains["meta"]]
            intents_decision_node[node] = global_intents
        else:
            raise Exception(f"Unknown level {level}")

        return intents_decision_node

    def get_garbage(self):
        garbages = []
        with open(os.path.join(ROOT_DIR, 'data', "garbage", "garbage.txt"), "r") as f:
            for sent in f.readlines():
                garbages.append([sent.strip(), "garbage"])
        return garbages

    def get_context(self):
        return self.dialogue["bot_response"] if "bot_response" in self.dialogue else []

def generate_dataset(name,
                     categories,
                     test_label,
                     dataset_type: DatasetType = DatasetType.ORIGINAL,
                     return_type: DatasetReturnType = DatasetReturnType.YIELD_SEPARATELY,
                     annotated_files_path=None,
                     global_path=None
                     ):
    dataset_grouped = []
    if dataset_type is DatasetType.ORIGINAL:

        # Prepare files
        for cat in categories:
            annotated_files_path_cat = os.path.join(annotated_files_path, cat)
            files_path = [os.path.join(annotated_files_path_cat, ds) for ds in os.listdir(annotated_files_path_cat)]

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
            yield dataset_grouped

    ##############################################################################################################
    elif dataset_type is DatasetType.FLATTEN:
        # Prepare files
        files_path = [os.path.join(annotated_files_path, ds) for ds in os.listdir(annotated_files_path)]

        for file_path in files_path:
            if file_path.endswith("global.json"):
                continue
            if not file_path.endswith(".json"):
                continue
            print(file_path)
            datasetGenerator = DatasetFlatten(file_path, global_path)

            context = datasetGenerator.get_context()
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
                dataset["context"] = context

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
            yield dataset_grouped

    ##########################################################################################3
    elif dataset_type is DatasetType.CLINC150:
        datasetGenerator = DatasetCLINC(dialogue_path=annotated_files_path, domains_path=global_path)

        context = datasetGenerator.get_context()
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
            dataset["context"] = context

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
            yield dataset_grouped
