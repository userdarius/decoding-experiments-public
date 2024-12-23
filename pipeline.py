import gc
from tqdm import tqdm
import numpy as np
import random
from collections import defaultdict
import torch.nn.functional as F
from scipy.special import logsumexp
import pickle

from utils import *
from scores import *

from evaluate import load

metric = get_metric("squad")


def is_answerable(generation):
    if "reference" not in generation:
        return len(get_reference(generation)["answers"]["text"]) > 0
    return len(generation["reference"]["answers"]["text"]) > 0


def score_pipeline(
    base_gen_model,
    entailment_model,
    dataset,
    num_fewshot=200,
    num_generations=10,
    brief="",
    brief_always=True,
    return_indices=True,
    prompt="",
    save_path="result.pkl",
    p_true_few_shot_prompt="",
    use_context=False,
    cot=False,
):
    possible_indices = range(0, len(dataset))
    indices = random.sample(possible_indices, min(num_fewshot, len(dataset)))

    accuracies, generations, results_dict, p_trues = [], {}, {}, []
    entropies = defaultdict(list)
    embeddings, is_true, answerable = [], [], []

    it = 0
    for index in tqdm(indices):
        if (it + 1 % 10) == 0:
            gc.collect()
            torch.cuda.empty_cache()
        it += 1
        example = dataset[index]
        question, context = example["question"], example["context"]
        generations[example["id"]] = {"question": question, "context": context}
        correct_answer = example["answers"]["text"]
        if use_context:
            print("Context: ", context)
        print("Question: ", question)
        print("-------------------------")
        if cot:
            local_prompt = make_prompt(context, question, None, brief, True, cot=True)
        else:
            current_input = make_prompt(context, question, None, brief, True)
            local_prompt = prompt + current_input

        full_responses = []
        num_generations = 10 + 1
        for i in range(num_generations):
            if i == 0:
                temperature = 0.1
            else:
                temperature = 1.0
            predicted_answer, token_log_likelihoods, embedding = base_gen_model.predict(
                local_prompt, temperature
            )
            embedding = embedding.cpu() if embedding is not None else None

            # Only compute accuracy if question is answerable.
            compute_acc = True
            if correct_answer and compute_acc:
                acc = metric(predicted_answer, example, base_gen_model)
            else:
                acc = 0.0  # pylint: disable=invalid-name
            if i == 0:
                accuracies.append(acc)
                most_likely_answer_dict = {
                    "response": predicted_answer,
                    "token_log_likelihoods": token_log_likelihoods,
                    "embedding": embedding,
                    "accuracy": acc,
                }
                generations[example["id"]].update(
                    {
                        "most_likely_answer": most_likely_answer_dict,
                        "reference": get_reference(example),
                    }
                )
                # print('low-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                print("answer: ", example["answers"]["text"])
                print("low-t predicted_answer: ", predicted_answer)
            else:
                print(
                    "high-t prediction ".ljust(15) + str(i) + " : " + predicted_answer
                )
                # Aggregate predictions over num_generations.
                full_responses.append(
                    (predicted_answer, token_log_likelihoods, embedding, acc)
                )

        # Append all predictions for this example to `generations`.
        generations[example["id"]]["responses"] = full_responses
        # Compute P_true
        p_true = calculate_p_true(
            base_gen_model,
            question,
            most_likely_answer_dict["response"],
            [r[0] for r in full_responses],
            p_true_few_shot_prompt,
            hint=False,
        )
        p_trues.append(p_true)
        print("log p_true: ", p_true)
        print("p_true: ", np.exp(p_true))

        # Compute semantic entropy
        responses = [fr[0] for fr in full_responses]
        if is_answerable(example):
            acc = metric(most_likely_answer_dict["response"], example, None)
        else:
            acc = 0.0  # pylint: disable=invalid-name
        is_true.append(acc)
        answerable.append(is_answerable(example))
        embeddings.append(most_likely_answer_dict["embedding"])

        log_liks = [r[1] for r in full_responses]
        entropies["context_entails_response"].append(
            context_entails_response(context, responses, entailment_model)
        )
        responses = [f"{question} {r}" for r in responses]

        # Compute semantic ids.
        semantic_ids = get_semantic_ids(
            responses, model=entailment_model, strict_entailment=True, example=example
        )

        # result_dict['semantic_ids'].append(semantic_ids)

        # Compute entropy from frequencies of cluster assignments.
        entropies["cluster_assignment_entropy"].append(
            cluster_assignment_entropy(semantic_ids)
        )

        # Length normalization of generation probabilities.
        log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

        # Compute naive entropy.
        entropies["regular_entropy"].append(predictive_entropy(log_liks_agg))

        # Compute semantic entropy.
        log_likelihood_per_semantic_id = logsumexp_by_id(
            semantic_ids, log_liks_agg, agg="sum_normalized"
        )
        pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
        entropies["semantic_entropy"].append(pe)

        log_str = (
            f"semantic_ids: {semantic_ids}, avg_token_log_likelihoods: {log_liks_agg}"
        )
        print(log_str)
        print("Entropies:")
        for k, v in entropies.items():
            print(f"{k}: {v[-1]}")
        print(80 * "#")
    accuracy = np.mean(accuracies)
    print(f"Average accuracy: {accuracy}")

    if save_path is not None:
        results_base = {
            "accuracies": accuracies,
            "generations": generations,
            "results_dict": results_dict,
            "p_trues": p_trues,
            "entropies": entropies,
        }
        try:
            with open(save_path, "wb") as f:
                pickle.dump(results_base, f)
            print(f"Results saved at {save_path}!")
            print()
        except:
            pass
    if return_indices:
        return accuracy, generations, results_dict, p_trues, indices
    return accuracies, generations, results_dict, p_trues
