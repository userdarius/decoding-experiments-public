import huggingface_hub

huggingface_hub.login(token="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")

import argparse

args_parser = argparse.ArgumentParser()
# DomainConfigArguments
args_parser.add_argument("--dataset", default="trivia_qa", type=str)
args_parser.add_argument(
    "--target_model_name", default="meta-llama/Llama-3.2-3B", type=str
)
args_parser.add_argument(
    "--draft_model_name", default="meta-llama/Llama-3.2-1B", type=str
)
args_parser.add_argument(
    "--save_dir", default="/scratch/homes/sfan/uncertainty/EE724/results", type=str
)
args_parser.add_argument("--prompt_brief", default="default", type=str)
args_parser.add_argument("--num_fewshot_data", default=100, type=int)
args_parser.add_argument("--num_fewshot_prompt", default=5, type=int)
args_parser.add_argument("--num_generations", default=10, type=int)
args_parser.add_argument("--model_max_new_tokens", default=10, type=int)
args_parser.add_argument("--run_base", action="store_true")
args_parser.add_argument("--run_spec", action="store_true")
args_parser.add_argument("--cot", action="store_true")
args_parser.add_argument(
    "--is_code_task", action="store_true", help="Whether this is a code generation task"
)
args_parser.add_argument(
    "--code_max_new_tokens",
    default=512,
    type=int,
    help="Maximum new tokens for code generation",
)

from pipeline import score_pipeline
from data import get_dataset
from model import get_models
from utils import split_dataset, get_data_prompt, get_ptrue_prompt, BRIEF_PROMPTS
import os
import gc
import torch
import time


def main():
    args = args_parser.parse_args()
    ## Define configs
    DATASET_NAME = args.dataset
    TARGET_MODEL_NAME = (
        args.target_model_name
    )  # ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B"]
    DRAFT_MODEL_NAME = args.draft_model_name
    quantize = True
    # quantize = (TARGET_MODEL_NAME=="meta-llama/Llama-3.1-8B")
    USE_CONTEXT = DATASET_NAME == "svamp"
    if args.dataset == "humaneval":
        MODEL_MAX_NEW_TOKENS = args.code_max_new_tokens
        USE_CONTEXT = True  # Always use context for code tasks to include test cases

    if args.cot:
        save_path_base = f"{args.save_dir}/{DATASET_NAME}-cot/results_base_TGT[{TARGET_MODEL_NAME.split('/')[-1]}]-{DATASET_NAME}.pkl"
        save_path_spec = f"{args.save_dir}/{DATASET_NAME}-cot/results_spec_TGT[{TARGET_MODEL_NAME.split('/')[-1]}]_DFT[{DRAFT_MODEL_NAME.split('/')[-1]}]-{DATASET_NAME}.pkl"

    else:
        save_path_base = f"{args.save_dir}/{DATASET_NAME}/results_base_TGT[{TARGET_MODEL_NAME.split('/')[-1]}]-{DATASET_NAME}.pkl"
        save_path_spec = f"{args.save_dir}/{DATASET_NAME}/results_spec_TGT[{TARGET_MODEL_NAME.split('/')[-1]}]_DFT[{DRAFT_MODEL_NAME.split('/')[-1]}]-{DATASET_NAME}.pkl"

    BRIEF = BRIEF_PROMPTS[args.prompt_brief]
    NUM_FEWSHOT_DATA = args.num_fewshot_data
    NUM_FEWSHOT_PROMPT = args.num_fewshot_prompt
    NUM_GEN = args.num_generations
    MODEL_MAX_NEW_TOKENS = args.model_max_new_tokens
    if args.cot:
        MODEL_MAX_NEW_TOKENS = 100

    train_dataset, val_dataset, answerable_indices, unanswerable_indices = get_dataset(
        DATASET_NAME
    )
    base_gen_model, entailment_model = get_models(
        TARGET_MODEL_NAME,
        draft_model_name=None,
        model_max_new_tokens=MODEL_MAX_NEW_TOKENS,
        quantize=quantize,
    )
    if args.cot:
        prompt = ""
    else:
        prompt, prompt_indices, remaining_answerable = get_data_prompt(
            train_dataset, answerable_indices, num_fewshot=NUM_FEWSHOT_PROMPT
        )
    p_true_few_shot_prompt, p_true_responses, len_p_true = get_ptrue_prompt(
        base_gen_model,
        train_dataset,
        answerable_indices,
        num_fewshot=5,
        num_generations=10,
        prompt=prompt,
    )
    if args.run_base:
        if os.path.exists(save_path_base):
            print(f"{save_path_base} already exists! skip evaluation.")
        else:
            tic = time.time()
            base_results = score_pipeline(
                base_gen_model,
                entailment_model,
                val_dataset,
                dataset_name=DATASET_NAME,
                num_fewshot=NUM_FEWSHOT_DATA,
                num_generations=NUM_GEN,
                brief=BRIEF,
                brief_always=True,
                return_indices=True,
                prompt=prompt,
                save_path=save_path_base,
                p_true_few_shot_prompt=p_true_few_shot_prompt,
                use_context=USE_CONTEXT,
            )
            toc = time.time()
            time_base = toc - tic
            print(f"Time for basic decoding with {TARGET_MODEL_NAME}: {time_base}")
            print("Evaluation on base generation finished.")

    if args.run_spec:
        if os.path.exists(save_path_spec):
            print(f"{save_path_spec} already exists! skip evaluation.")
        else:
            del base_gen_model, entailment_model
            gc.collect()
            torch.cuda.empty_cache()

            spec_gen_model, entailment_model = get_models(
                TARGET_MODEL_NAME,
                draft_model_name=DRAFT_MODEL_NAME,
                model_max_new_tokens=MODEL_MAX_NEW_TOKENS,
                quantize=quantize,
            )
            tic = time.time()
            spec_results = score_pipeline(
                spec_gen_model,
                entailment_model,
                val_dataset,
                dataset_name=DATASET_NAME,
                num_fewshot=NUM_FEWSHOT_DATA,
                num_generations=NUM_GEN,
                brief=BRIEF,
                brief_always=True,
                return_indices=True,
                prompt=prompt,
                save_path=save_path_spec,
                p_true_few_shot_prompt=p_true_few_shot_prompt,
                use_context=USE_CONTEXT,
            )
            toc = time.time()
            time_base = toc - tic
            print(f"Time for basic decoding with {TARGET_MODEL_NAME}: {time_base}")
            print("Evaluation on speculative decoding finished.")


if __name__ == "__main__":
    main()
