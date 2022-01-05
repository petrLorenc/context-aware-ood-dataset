import wandb

from utils.utils import print_results, aggregate_results
from utils.dataset.generate import DatasetType
from utils.dataset.generate import generate_dataset

# from config.one_class_model import imports
from config.embedding_based import imports
# from config.local_illusionist import imports

categories = [
    'animals',
    'books', 'education',
    'fashion', 'food', 'habits',
    'movies', 'music', 'science', 'smalltalk',
    'sports', 'travel'
]

# dataset_args = {"name": "OUR_DATASET", "categories": categories, "datasetType": DatasetType.ORIGINAL}
# dataset_args = {"name": "OUR_DATASET", "categories": {}, "datasetType": DatasetType.FLATTEN}

if __name__ == '__main__':
    for i in imports:
        evaluate_fn = i["evaluation_fn"]
        dataset_args = i["dataset_args"]
        for embeddings in i["embeddings"]:
            emb_name = embeddings["embedding_name"]
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
                    config.model_name = model_name
                    config.emb_name = emb_name
                    config.test_type = test_label
                    config.dataset_type = str(dataset_args["dataset_type"])

                    dct_results_lst = []

                    for dataset in generate_dataset(test_label=test_label,
                                                    **dataset_args):
                        args = i["evaluation_fn_arg"]
                        args["dataset"] = dataset
                        args["embedding_model"] = embedding_model
                        args["classification_model"] = classification_model

                        results_dct = evaluate_fn(**args)
                        dct_results_lst.append(results_dct)
                        print(results_dct)
                        print()

                    results_dct = aggregate_results(dct_results_lst)

                    for k, v in results_dct.items():
                        wandb.log({k: v})
                    print_results(dataset_args["name"], model_name, emb_name, results_dct)
                    wandb.finish()
