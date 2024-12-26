"""Utility functions."""

import os
import logging
import argparse
import pickle
import wandb

from contextlib import contextmanager
import signal
import time

from evaluate import load

BRIEF_PROMPTS = {
    "cot": "Let's think step by step: \n",
    "default": "Answer the following question as briefly as possible.\n",
    "chat": "Answer the following question in a single brief but complete sentence.\n",
    "summary": "Give a brief summary for the following paragraph.\n",
    "code": "Write a complete Python function implementation that solves the following problem.\n",
}


def split_dataset(dataset):
    """Get indices of answerable and unanswerable questions."""

    def clen(ex):
        return len(ex["answers"]["text"])

    answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    # union == full dataset
    assert set(answerable_indices) | set(unanswerable_indices) == set(
        range(len(dataset))
    )
    # no overlap
    assert set(answerable_indices) - set(unanswerable_indices) == set(
        answerable_indices
    )

    return answerable_indices, unanswerable_indices


def construct_fewshot_prompt_from_indices(
    dataset, example_indices, brief, brief_always, make_prompt
):
    """Given a dataset and indices, construct a fewshot prompt."""
    if not brief_always:
        prompt = brief
    else:
        prompt = ""

    for example_index in example_indices:

        example = dataset[example_index]
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]

        prompt = prompt + make_prompt(context, question, answer, brief, brief_always)

    return prompt


def save_pkl(obj, save_path):
    with open(f"{save_path}", "wb") as trg:
        pickle.dump(obj, trg)


def load_pkl(load_path):
    with open(f"{load_path}", "rb") as src:
        return pickle.load(src)


def get_reference(example):
    if "answers" not in example:
        example = example["reference"]
    answers = example["answers"]
    answer_starts = answers.get("answer_start", [])
    reference = {
        "answers": {"answer_start": answer_starts, "text": answers["text"]},
        "id": example["id"],
    }
    return reference


### LLM based metric
def model_based_metric(predicted_answer, example, model):
    if "answers" in example:
        correct_answers = example["answers"]["text"]
    elif "reference" in example:
        correct_answers = example["reference"]["answers"]["text"]
    else:
        raise ValueError

    prompt = f'We are assessing the quality of answers to the following question: {example["question"]}\n'
    if len(correct_answers) == 1:
        prompt += f"The expected answer is: {correct_answers[0]}.\n"
    else:
        prompt += (
            f"The following are expected answers to this question: {correct_answers}.\n"
        )

    prompt += f"The proposed answer is: {predicted_answer}\n"

    if len(correct_answers) == 1:
        prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
    else:
        prompt += "Within the context of the question, does the proposed answer mean the same as any of the expected answers?"

    prompt += " Respond only with yes or no.\nResponse:"

    if "gpt" in model.model_name.lower():
        predicted_answer = model.predict(prompt, 0.01)
    else:
        predicted_answer, _, _ = model.predict(prompt, 0.01)

    if "yes" in predicted_answer.lower():
        return 1.0
    elif "no" in predicted_answer.lower():
        return 0.0
    else:
        logging.warning("Redo llm check.")
        predicted_answer, _, _ = model.predict(prompt, 1)
        if "yes" in predicted_answer.lower():
            return 1.0
        elif "no" in predicted_answer.lower():
            return 0.0

        logging.warning("Answer neither no nor yes. Defaulting to no!")
        return 0.0


def llm_metric(predicted_answer, example, model):
    return model_based_metric(predicted_answer, example, model)


def get_gpt_metric(metric_name):

    model_name = "_".join(metric_name.split("_")[1:])

    class EntailmentGPT:
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, prompt, temperature):
            return oai.predict(prompt, temperature, model=self.model_name)

    gpt_model = EntailmentGPT(model_name)

    def gpt_metric(predicted_answer, example, model):
        del model
        return model_based_metric(predicted_answer, example, gpt_model)

    return gpt_metric


@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Execution timed out")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_humaneval_metric(predicted_answer, example, model=None):
    try:
        with timeout(5):  # 5 second timeout
            namespace = {}
            exec(predicted_answer, namespace)
            test_code = example["test_code"]
            test_namespace = namespace.copy()
            test_namespace["candidate"] = namespace[example["entry_point"]]
            exec(test_code, test_namespace)
            return 1.0
    except TimeoutError:
        logging.warning("Test execution timed out")
        return 0.0
    except Exception as e:
        logging.warning(f"Test failed: {str(e)}")
        return 0.0


def get_metric(metric):
    if metric == "squad":

        squad_metric = load("squad_v2")

        def metric(response, example, *args, **kwargs):
            # Compatibility with recomputation.
            if "id" in example:
                exid = example["id"]
            elif "id" in example["reference"]:
                exid = example["reference"]["id"]
            else:
                raise ValueError

            prediction = {
                "prediction_text": response,
                "no_answer_probability": 0.0,
                "id": exid,
            }
            results = squad_metric.compute(
                predictions=[prediction], references=[get_reference(example)]
            )
            return 1.0 if (results["f1"] >= 50.0) else 0.0

    # Reuses the globally active model for these.
    elif metric == "humaneval":
        metric = get_humaneval_metric
    elif metric == "llm":
        metric = llm_metric
    elif metric == "llm_gpt-3.5":
        metric = get_gpt_metric(metric)
    elif metric == "llm_gpt-4":
        metric = get_gpt_metric(metric)
    else:
        raise ValueError

    return metric


### Make prompt
def make_prompt(
    context, question, answer, brief, brief_always, use_context=False, cot=False
):
    prompt = ""
    if cot:
        prompt += f"Question: {question}\n"
        prompt += BRIEF_PROMPTS["cot"]
        return prompt
    if brief_always:
        prompt += brief
    if use_context and (context is not None):
        prompt += f"Context: {context}\n"
    prompt += f"Question: {question}\n"
    if answer:
        prompt += f"Answer: {answer}\n\n"
    else:
        prompt += "Answer:"
    return prompt


import random


def get_data_prompt(
    train_dataset, answerable_indices, brief=BRIEF_PROMPTS["default"], num_fewshot=5
):
    prompt_indices = random.sample(answerable_indices, num_fewshot)
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # Create Few-Shot prompt.
    prompt = construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, brief, True, make_prompt
    )

    return prompt, prompt_indices, remaining_answerable


def construct_few_shot_prompt(
    *,
    model,
    dataset,
    indices,
    prompt,
    brief,
    brief_always,
    make_prompt,
    num_generations,
    metric,
    cot=False,
):
    """Construct few shot prompt for p_true uncertainty metric."""

    # Call model n_shots many times.
    few_shot_prompt = []
    all_responses = dict()
    for it, i in enumerate(indices):
        prompt_candidate = []
        example = dataset[i]
        question = example["question"]
        context = example["context"]
        if it != 0:
            prompt_candidate += ["\n"]
        prompt_candidate += ["Question: " + question]
        prompt_candidate += ["\nBrainstormed Answers: "]
        current_question = make_prompt(context, question, None, brief, brief_always)
        local_prompt = prompt + current_question
        logging.info("P_TRUE >> Current Question: ".ljust(25) + current_question)

        responses = []
        for j in range(num_generations + 1):

            if j == 0:
                temperature = 0.1
            else:
                temperature = 1.0

            response, _, _ = model.predict(local_prompt, temperature)
            logging.info("P_TRUE >> Current Response: ".ljust(25) + response)

            responses.append(response)
            prompt_candidate += [f"{response.strip()} \n"]
            if j == 0:
                # Save most likely response and compute correctness metric for it.
                most_likely_response = response
                is_correct = metric(response, example, model)
                answers = [answer for answer in example["answers"]["text"]]
                logging.info(
                    "P_TRUE >> LOW-T >> true answer: ".ljust(35) + str(answers)
                )
                logging.info("P_TRUE >> LOW-T >> acc: ".ljust(35) + str(is_correct))

        all_responses[i] = dict(
            responses=responses,
            most_likely_response=most_likely_response,
            is_correct=is_correct,
        )

        prompt_candidate += ["Possible answer: " + most_likely_response + "\n"]
        prompt_candidate += ["Is the possible answer:\n"]
        prompt_candidate += ["A) True\n"]
        prompt_candidate += ["B) False\n"]
        prompt_candidate += ["The possible answer is:"]
        prompt_candidate += [" A" if is_correct else " B"]

        prompt_len = len(
            model.tokenizer.encode("".join(few_shot_prompt + prompt_candidate))
        )
        # At test time, get a maximum of `num_generations * model.token_limit` extra tokens
        # 200 buffer for question and 'Possible Answer'.
        max_input_len = prompt_len + num_generations * model.max_new_tokens + 200

        if max_input_len < model.token_limit:
            few_shot_prompt.extend(prompt_candidate)
        else:
            logging.warning("Cutting of p_true prompt at length %d.", it)
            break

    return "".join(few_shot_prompt), all_responses, it


metric = get_metric("squad")


def get_ptrue_prompt(
    model,
    train_dataset,
    answerable_indices,
    brief=BRIEF_PROMPTS["default"],
    num_fewshot=5,
    num_generations=10,
    prompt="",
):
    p_true_indices = random.sample(answerable_indices, num_fewshot)
    remaining_answerable = list(set(answerable_indices) - set(p_true_indices))
    p_true_few_shot_prompt, p_true_responses, len_p_true = construct_few_shot_prompt(
        model=model,
        dataset=train_dataset,
        indices=p_true_indices,
        prompt=prompt,
        brief=brief,
        brief_always=True,
        make_prompt=make_prompt,
        num_generations=num_generations,
        metric=metric,
    )
    return p_true_few_shot_prompt, p_true_responses, len_p_true
