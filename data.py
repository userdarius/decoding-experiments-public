"""Data Loading Utilities."""

import os
import json
import hashlib
import datasets
from utils import split_dataset


def load_ds(dataset_name, seed, add_options=None, testsize=100):
    """Load dataset."""
    # user = os.environ['USER']
    user = "foodeei"

    train_dataset, validation_dataset = None, None
    if dataset_name == "squad":
        dataset = datasets.load_dataset("squad_v2")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"].select(list(range(testsize)))

    elif dataset_name == "svamp":
        dataset = datasets.load_dataset("ChilleD/SVAMP")
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"].select(list(range(testsize)))

        reformat = lambda x: {
            "question": x["Question"],
            "context": x["Body"],
            "type": x["Type"],
            "equation": x["Equation"],
            "id": x["ID"],
            "answers": {"text": [str(x["Answer"])]},
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "nq":
        dataset = datasets.load_dataset("nq_open")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"].select(list(range(testsize)))
        md5hash = lambda s: str(int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16))

        reformat = lambda x: {
            "question": x["question"] + "?",
            "answers": {"text": x["answer"]},
            "context": "",
            "id": md5hash(str(x["question"])),
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "trivia_qa":
        dataset = datasets.load_dataset("TimoImhof/TriviaQA-in-SQuAD-format")[
            "unmodified"
        ]
        dataset = dataset.train_test_split(test_size=testsize, seed=seed)
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

    elif dataset_name == "bioasq":
        # http://participants-area.bioasq.org/datasets/ we are using training 11b
        # could also download from here https://zenodo.org/records/7655130
        scratch_dir = os.getenv("SCRATCH_DIR", ".")
        path = f"{scratch_dir}/{user}/semantic_uncertainty/data/bioasq/training11b.json"
        with open(path, "rb") as file:
            data = json.load(file)

        questions = data["questions"]
        dataset_dict = {"question": [], "answers": [], "id": []}

        for question in questions:
            if "exact_answer" not in question:
                continue
            dataset_dict["question"].append(question["body"])
            if "exact_answer" in question:

                if isinstance(question["exact_answer"], list):
                    exact_answers = [
                        ans[0] if isinstance(ans, list) else ans
                        for ans in question["exact_answer"]
                    ]
                else:
                    exact_answers = [question["exact_answer"]]

                dataset_dict["answers"].append(
                    {
                        "text": exact_answers,
                        "answer_start": [0] * len(question["exact_answer"]),
                    }
                )
            else:
                dataset_dict["answers"].append(
                    {"text": question["ideal_answer"], "answer_start": [0]}
                )
            dataset_dict["id"].append(question["id"])

            dataset_dict["context"] = [None] * len(dataset_dict["id"])

        dataset = datasets.Dataset.from_dict(dataset_dict)

        # Split into training and validation set.
        dataset = dataset.train_test_split(test_size=testsize, seed=seed)
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]
    elif dataset_name == "mmlu":
        dataset = datasets.load_dataset("tasksource/mmlu")
    elif dataset_name == "gsm8k":
        dataset = datasets.load_dataset("gsm8k", "main")
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"].select(list(range(testsize)))

        def md5hash(s):
            return str(int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16))

        def reformat(x):
            return {
                "question": x["question"],
                "answers": {
                    "text": [x["answer"].split("####")[1].strip()]
                },  # Extract final answer
                "context": "",  # GSM8K doesn't have additional context
                "id": md5hash(str(x["question"])),
                "solution": x["answer"]
                .split("####")[0]
                .strip(),  # Store solution steps
            }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]
    else:
        raise ValueError

    return train_dataset, validation_dataset


### Load dataset ###
def get_dataset(dataset_name="trivia_qa", testsize=200):
    train_dataset, val_dataset = load_ds(dataset_name, 42, testsize)
    # Get indices of answerable and unanswerable questions and construct prompt.
    # only use on squad
    answerable_indices, unanswerable_indices = split_dataset(train_dataset)
    val_answerable, val_unanswerable = split_dataset(val_dataset)
    val_dataset = [val_dataset[i] for i in val_answerable]
    print(f"[train] {len(train_dataset)} | [val] {len(val_dataset)}")
    return train_dataset, val_dataset, answerable_indices, unanswerable_indices
