import wandb

import tensorflow as tf
import tensorflow_text
import time

from utils.utils import print_results, aggregate_results
from utils.dataset.generate import DatasetType
from utils.dataset.generate import generate_dataset

# from config.one_class_model import imports, APPROACH
# from config.embedding_based import imports, APPROACH
from config.local_illusionist import imports, APPROACH
# from config.threshold_based import imports, APPROACH

# tf.debugging.set_log_device_placement(True)

print("GPUS:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

gpus = tf.config.list_logical_devices('GPU')
print(gpus)

if gpus:
    strategy = tf.distribute.MirroredStrategy(gpus)
else:
    strategy = tf.distribute.OneDeviceStrategy(device="CPU")


def main():
    for i in imports:
        evaluate_fn = i["evaluation_fn"]
        dataset_args = i["dataset_args"]
        for embeddings in i["embeddings"]:
            emb_name = embeddings["embedding_name"]
            if i["debug"]:
                print(emb_name)
            embedding_model = embeddings["embedding_model"]

            for k in i["test_keys"]:
                test_label = "test" + ("_" if k else "") + k

                for classification_model in i["algorithms"]:
                    if type(classification_model) == str:
                        model_name = classification_model
                    else:
                        model_name = type(classification_model).__name__

                    wandb.init(project='robust-intent-recognition', entity='alquist')
                    config = wandb.config
                    config.dataset_name = dataset_args["name"]
                    config.approach = APPROACH
                    config.model_name = model_name
                    config.emb_name = emb_name
                    config.test_type = test_label
                    config.dataset_type = str(dataset_args["dataset_type"])
                    config.classification_model = classification_model.to_dict()

                    dct_results_lst = []

                    for dataset in generate_dataset(test_label=test_label,
                                                    **dataset_args):
                        args = i["evaluation_fn_arg"]
                        args["dataset"] = dataset
                        args["embedding_model"] = embedding_model
                        args["classification_model"] = classification_model

                        results_dct = evaluate_fn(**args)
                        dct_results_lst.append(results_dct)
                        if i["debug"]:
                            print(results_dct)
                            print()

                    results_dct = aggregate_results(dct_results_lst)

                    for k, v in results_dct.items():
                        wandb.log({k: v})
                    print_results(dataset_args["name"], model_name, emb_name, results_dct)
                    wandb.finish()


if __name__ == '__main__':
    # with strategy.scope():
    main()
