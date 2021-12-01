import os
import json
import random

new_format_template = {

}

idx = 0
for subdir, dirs, files in os.walk("../data/zaloha/all/"):
    for filename in files:
        filepath = os.path.join(subdir, filename)

        if filepath.endswith(".json"):
            if os.path.isfile(filepath + ".annotated"):
                continue
            idx += 1
            print(idx)
            print(filepath)
            with open(filepath, "r") as f:
                data = json.load(f)

            for decision_point in data["decisionNodes"]:
                ood_sentences = []
                not_covered_sentences = []
                counter_questions = 0

                for questions in [data["utterances"][str(k)] for k, v in data["links"].items() if int(decision_point) in v]:
                    print(random.sample(questions, k=1))
                    counter_questions += 1

                if counter_questions <= 1:
                    continue

                print(" | ".join([x["intent"] for x in data["globalIntents"].values()]))
                for idx in data["links"][str(decision_point)]:
                    examples = data["intents"][str(idx)]
                    examples = examples["train"] + examples["val"] + examples["test"]
                    print(random.sample(examples, k=3 if len(examples) > 3 else len(examples)))
                print("Not covered sentences:")
                ncs_sentence = input(">>>")
                while ncs_sentence != "q":
                    not_covered_sentences.append(ncs_sentence)
                    ncs_sentence = input(">>>")
                data["intents"][str(idx)]["ncs"] = not_covered_sentences

                print("OOD sentences:")
                ood_sentence = input(">>>")
                while ood_sentence != "q":
                    ood_sentences.append(ood_sentence)
                    ood_sentence = input(">>>")
                data["intents"][str(idx)]["ood"] = ood_sentences

            with open(filepath + ".annotated", "w") as f:
                data = json.load(f)
