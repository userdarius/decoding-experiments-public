"""Implement HuggingfaceModel models."""

import copy
import logging
from collections import Counter
import torch
import os

import accelerate

from transformers import AutoTokenizer, DebertaV2TokenizerFast
from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download

from abc import ABC, abstractmethod
from typing import List, Text, Optional, Tuple
import re
from dataclasses import dataclass
import numpy as np

STOP_SEQUENCES = ["\n\n\n\n", "\n\n\n", "\n\n", "\n", "Question:", "Context:"]


class BaseModel(ABC):

    stop_sequences: List[Text]

    @abstractmethod
    def predict(self, input_data, temperature):
        pass

    @abstractmethod
    def get_p_true(self, input_data):
        pass


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""

    def __init__(self, stops, tokenizer, match_on="text", initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == "tokens":
            self.stops = [
                torch.tensor(self.tokenizer.encode(i)).to("cuda") for i in self.stops
            ]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        for stop in self.stops:
            if self.match_on == "text":
                generation = self.tokenizer.decode(
                    input_ids[0][self.initial_length :], skip_special_tokens=False
                )
                match = stop in generation
            elif self.match_on == "tokens":
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise ValueError("Invalid match_on value")
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter([".".join(i.split(".")[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer.
            raise ValueError(
                "More than one split layer.\n"
                f"Currently at layer {layer}.\n"
                f"In map: {device_map_in}\n"
                f"Out map: {device_map}\n"
            )

        logging.info(f"Split layer is {layer}.")

        device = None
        # Remove split for that layer.
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f"pop {name}")
                device = device_map.pop(name)

        if device is not None:
            device_map[layer] = device
        found_split = True

    return device_map


class HuggingfaceModel(BaseModel):
    """Hugging Face Model."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        if max_new_tokens is None:
            raise ValueError("max_new_tokens must be specified")
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name

        self.model_type = None
        if "llama" in model_name.lower():
            self.model_type = "llama"
            self.init_llama()
        # elif "gpt" in model_name.lower():
        #     self.model_type = "gpt"
        elif "falcon" in model_name.lower():
            self.model_type = "falcon"
            self.init_falcon()
        elif "mistral" in model_name.lower():
            self.model_type = "mistral"
            self.init_mistral()
        else:
            raise ValueError(f"Unknown model_type `{self.model_type}`.")
        if stop_sequences == "default":
            stop_sequences = STOP_SEQUENCES
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
        self.token_limit = 4096 if "llama-2" in self.model_name.lower() else 2048

    def init_llama(self):
        kwargs = {}
        eightbit = False
        if self.model_name.lower().endswith("-8bit"):
            kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            }
            self.model_name = self.model_name[: -len("-8bit")]
            eightbit = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            f"{self.model_name}", device_map="auto", token_type_ids=None
        )

        if (
            "1b" in self.model_name.lower()
            or "3b" in self.model_name.lower()
            or "7b" in self.model_name.lower()
            or "8b" in self.model_name.lower()
            or "13b" in self.model_name.lower()
        ) or eightbit:
            self.model = AutoModelForCausalLM.from_pretrained(
                f"{self.model_name}",
                device_map="auto",
                max_memory={0: "80GIB"},
                **kwargs,
            )
        elif "65b" in self.model_name.lower() or "70b" in self.model_name.lower():
            path = snapshot_download(
                repo_id=f"{self.model_name}",
                allow_patterns=["*.json", "*.model", "*.safetensors"],
                ignore_patterns=["pytorch_model.bin.index.json"],
            )
            config = AutoConfig.from_pretrained(f"{self.model_name}")
            with accelerate.init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(config)
            self.model.tie_weights()
            max_mem = 15 * 4686198491

            device_map = accelerate.infer_auto_device_map(
                self.model.model, max_memory={0: max_mem, 1: max_mem}, dtype="float16"
            )
            device_map = remove_split_layer(device_map)
            full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
            full_model_device_map["lm_head"] = 0

            self.model = accelerate.load_checkpoint_and_dispatch(
                self.model,
                path,
                device_map=full_model_device_map,
                dtype="float16",
                skip_keys="past_key_values",
            )
        else:
            raise ValueError

    def init_mistral(self):
        if self.model_name.endswith("-8bit"):
            kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            }
            self.model_name = self.model_name[: -len("-8bit")]
        if self.model_name.endswith("-4bit"):
            kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            }
            self.model_name = self.model_name[: -len("-4bit")]
        else:
            kwargs = {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            device_map="auto",
            token_type_ids=None,
            clean_up_tokenization_spaces=False,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            max_memory={0: "80GIB"},
            **kwargs,
        )

    def init_falcon(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            device_map="auto",
            token_type_ids=None,
            clean_up_tokenization_spaces=False,
        )

        kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
            )
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto",
            **kwargs,
        )

    # TODO: add speculative decoding
    def speculative_predict(
        self, input_data, temperature, draft_model=None, return_full=False
    ):
        if draft_model is None:
            draft_model = self.model
        return self.predict(input_data, temperature, return_full=return_full)

    def predict(self, input_data, temperature, return_full=False):

        # Implement prediction.
        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        if (
            "llama" in self.model_name.lower()
            or "falcon" in self.model_name.lower()
            or "mistral" in self.model_name.lower()
        ):
            if "token_type_ids" in inputs:  # Some HF models have changed.
                del inputs["token_type_ids"]
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList(
                [
                    StoppingCriteriaSub(
                        stops=self.stop_sequences,
                        initial_length=len(inputs["input_ids"][0]),
                        tokenizer=self.tokenizer,
                    )
                ]
            )
        else:
            stopping_criteria = None

        logging.debug("temperature: %f", temperature)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
            )

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                "Generation exceeding token limit %d > %d",
                len(outputs.sequences[0]),
                self.token_limit,
            )

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )
        input_data = self.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=True
        )

        if return_full:
            return full_answer

        # For some models, we need to remove the input_data from the answer.
        if full_answer.lstrip().startswith(input_data):
            input_data_offset = len(input_data)
        else:
            print("Full answer: ", full_answer)
            print("Length of full_answer: ", len(outputs.sequences[0]))
            print("Input data: ", input_data)
            print("Length of input_data: ", len(inputs["input_ids"][0]))
            print("input indices: ", inputs["input_ids"][0])
            print("answer indices: ", outputs.sequences[0])

            input_data_offset = 0
            # raise ValueError('Have not tested this in a while.')

        # Remove input from answer.
        answer = full_answer[input_data_offset:]

        # Remove stop_words from answer.
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            # if not all([stop not in sliced_answer for stop in self.stop_sequences]):
            #     error_msg = 'Error: Stop words not removed successfully!'
            #     error_msg += f'Answer: >{answer}< '
            #     error_msg += f'Sliced Answer: >{sliced_answer}<'
            #     if 'falcon' not in self.model_name.lower():
            #         raise ValueError(error_msg)
            #     else:
            #         logging.error(error_msg)

        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = sliced_answer.strip()

        # Get the number of tokens until the stop word comes up.
        # Note: Indexing with `stop_at` already excludes the stop_token.
        # Note: It's important we do this with full answer, since there might be
        # non-trivial interactions between the input_data and generated part
        # in tokenization (particularly around whitespaces.)
        token_stop_index = self.tokenizer(
            full_answer[: input_data_offset + stop_at], return_tensors="pt"
        )["input_ids"].shape[1]
        n_input_token = len(inputs["input_ids"][0])
        n_generated = token_stop_index - n_input_token

        if n_generated == 0:
            logging.warning(
                "Only stop_words were generated. For likelihoods and embeddings, taking stop word instead."
            )
            n_generated = 1

        # Get the last hidden state (last layer) and the last token's embedding of the answer.
        # Note: We do not want this to be the stop token.

        # outputs.hidden_state is a tuple of len = n_generated_tokens.
        # The first hidden state is for the input tokens and is of shape
        #     (n_layers) x (batch_size, input_size, hidden_size).
        # (Note this includes the first generated token!)
        # The remaining hidden states are for the remaining generated tokens and is of shape
        #    (n_layers) x (batch_size, 1, hidden_size).

        # Note: The output embeddings have the shape (batch_size, generated_length, hidden_size).
        # We do not get embeddings for input_data! We thus subtract the n_tokens_in_input from
        # token_stop_index to arrive at the right output.

        if "decoder_hidden_states" in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        if len(hidden) == 1:
            logging.warning(
                "Taking first and only generation for hidden! "
                "n_generated: %d, n_input_token: %d, token_stop_index %d, "
                "last_token: %s, generation was: %s",
                n_generated,
                n_input_token,
                token_stop_index,
                self.tokenizer.decode(outputs["sequences"][0][-1]),
                full_answer,
            )
            last_input = hidden[0]
        elif (n_generated - 1) >= len(hidden):
            # If access idx is larger/equal.
            logging.error(
                "Taking last state because n_generated is too large"
                "n_generated: %d, n_input_token: %d, token_stop_index %d, "
                "last_token: %s, generation was: %s, slice_answer: %s",
                n_generated,
                n_input_token,
                token_stop_index,
                self.tokenizer.decode(outputs["sequences"][0][-1]),
                full_answer,
                sliced_answer,
            )
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        # Then access last layer for input
        last_layer = last_input[-1]
        # Then access last token in input.
        last_token_embedding = last_layer[:, -1, :].cpu()

        # Get log_likelihoods.
        # outputs.scores are the logits for the generated token.
        # outputs.scores is a tuple of len = n_generated_tokens.
        # Each entry is shape (bs, vocabulary size).
        # outputs.sequences is the sequence of all tokens: input and generated.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        # Transition_scores[0] only contains the scores for the first generated tokens.

        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning("Taking first and only generation for log likelihood!")
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning("Generation interrupted by max_token limit.")

        if len(log_likelihoods) == 0:
            raise ValueError

        return sliced_answer, log_likelihoods, last_token_embedding

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input."""

        input_data += " A"
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors="pt").to(
            "cuda"
        )["input_ids"]
        # The computation of the negative log likelihoods follows:
        # https://huggingface.co/docs/transformers/perplexity.

        target_ids_true = tokenized_prompt_true.clone()
        # Set all target_ids except the last one to -100.
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(
                tokenized_prompt_true, labels=target_ids_true
            )

        loss_true = model_output_true.loss

        return -loss_true.item()


### Entailment Model ###
class BaseEntailment:
    def save_prediction_cache(self):
        pass


### Deberta-large (tuned on NLI datasets)
class EntailmentDeberta(BaseEntailment):
    def __init__(self, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli"
        ).to(self.device)

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(
            F.softmax(logits, dim=1)
        )  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        if os.environ.get("DEBERTA_FULL_LOG", False):
            logging.info("Deberta Input: %s -> %s", text1, text2)
            logging.info("Deberta Prediction: %s", prediction)

        return prediction


### Speculative Decoding ###
## Utils
import torch
from torch.nn import functional as F
import logging


# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float("-inf")
    return logits


def norm_logits(
    logits: torch.Tensor, temperature: float, top_k: float, top_p: float
) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


def sample(probs: torch.Tensor, num_samples: int = 1):
    """Sample from probability distribution with validation."""
    try:
        device = probs.device  # Get device from input tensor
        if torch.isnan(probs).any():
            raise ValueError("NaN values in probability distribution")
        if (probs == 0).all():
            raise ValueError("All zero probability distribution")

        idx_next = torch.multinomial(probs, num_samples=num_samples)
        if idx_next.item() == 0:
            probs[:, 0] = 0
            if probs.sum() > 0:
                probs = probs / probs.sum()
                idx_next = torch.multinomial(probs, num_samples=num_samples)
            else:
                idx_next = torch.randint(1, probs.size(-1), (1,), device=device)

        return idx_next.to(device)  # Ensure output is on correct device

    except Exception as e:
        logging.error(f"Sampling failed: {str(e)}")
        logging.error(f"Device: {probs.device}")
        raise RuntimeError(f"Sampling failed: {str(e)}")


def max_fn(x):
    """
    norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum


import torch
from typing import Optional

# from .utils import norm_logits, sample
from transformers.models.bloom.modeling_bloom import BloomForCausalLM


def _debug_show_kvcache(past_key_values):
    if past_key_values is None:
        return
    for elem in past_key_values:
        k, v = elem
        print(f"kv cache: k shape {k.shape}, v shape {v.shape}")
        break


class KVCacheModel:
    def __init__(
        self,
        model: torch.nn.Module,
        temperature: float = 1,
        top_k: int = 0,
        top_p: float = 0,
    ) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def _forward_with_kvcache(
        self, input_ids: torch.Tensor, use_debug=True
    ) -> torch.Tensor:
        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):
                self._prob_history[:, i, :] = norm_logits(
                    self._prob_history[:, i, :],
                    self._temperature,
                    self._top_k,
                    self._top_p,
                )
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]

            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)

            if use_debug:
                print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)

            outputs = self._model(
                last_input_id, past_key_values=self._past_key_values, use_cache=True
            )

            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)

            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(
                    not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p
                )

            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)

            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values

        return last_q

    def _generate_with_kvcache(
        self, prefix: torch.Tensor, gamma: int, use_debug=False
    ) -> torch.Tensor:
        """forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        for _ in range(gamma):
            q = self._forward_with_kvcache(x, use_debug)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output

    @torch.no_grad()
    def rollback(self, end_pos: int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)

            # Bloom is special one
            if isinstance(self._model, BloomForCausalLM):
                # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                k = k[:, :, :end_pos]
                v = v[:, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            else:
                # k, v (batch, head, seq, hidden_dim)
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)

        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]


# from .huggingface_models import HuggingfaceModel
import torch

# from .kvcache_model import KVCacheModel
# from .utils import sample, max_fn
from dataclasses import dataclass
from typing import Optional, Tuple
import logging


@dataclass
class SpeculativeOutput:
    sequences: torch.Tensor
    hidden_states: tuple
    logits: torch.Tensor
    scores: Optional[list[torch.Tensor]] = None
    past_key_values: Optional[Tuple[torch.Tensor]] = None
    decoder_hidden_states: Optional[tuple] = None


class SpeculativeSamplingModel(HuggingfaceModel):
    def __init__(
        self,
        approx_model_name,
        target_model_name,
        stop_sequences="default",
        max_new_tokens=20,
    ):
        # Initial setup logging
        logging.info("%s", f"\n{'='*50}")
        logging.info("Initializing SpeculativeSamplingModel:")
        logging.info("Target model: %s", target_model_name)
        logging.info("Approximation (Draft) model: %s", approx_model_name)
        logging.info("Max new tokens: %s", max_new_tokens)

        # Add 8-bit quantization suffix if not already present
        # if not target_model_name.endswith("-8bit"):
        #     target_model_name += "-8bit"
        #     logging.info(
        #         "Added 8-bit quantization to target model: %s", target_model_name
        #     )

        # if not approx_model_name.endswith("-8bit"):
        #     approx_model_name += "-8bit"
        #     logging.info(
        #         "Added 8-bit quantization to approximation model: %s", approx_model_name
        #     )

        logging.info("\nInitializing target model...")
        super().__init__(target_model_name, stop_sequences, max_new_tokens)
        logging.info("\nTarget Model Architecture:")
        logging.info("Model type: %s", type(self.model).__name__)
        logging.info(
            "Number of parameters: %s", sum(p.numel() for p in self.model.parameters())
        )
        logging.info("Layer structure:")

        for name, module in self.model.named_children():
            logging.info("  %s: %s", name, type(module).__name__)
            if hasattr(module, "config"):
                config = module.config
                logging.info("    Hidden size: %s", config.hidden_size)
                logging.info("    Number of layers: %s", config.num_hidden_layers)
                logging.info(
                    "    Number of attention heads: %s", config.num_attention_heads
                )
                logging.info("    Vocabulary size: %s", config.vocab_size)

        logging.info("\nInitializing approximation model...")
        self.approx_model = HuggingfaceModel(
            approx_model_name, stop_sequences, max_new_tokens
        )
        logging.info("\nApproximation Model Architecture:")
        logging.info("Model type: %s", type(self.approx_model.model).__name__)
        logging.info(
            "Number of parameters: %s",
            "%s" % f"{sum(p.numel() for p in self.approx_model.model.parameters()):,}",
        )
        logging.info("Layer structure:")
        for name, module in self.approx_model.model.named_children():
            logging.info("  %s: %s", name, type(module).__name__)
            if hasattr(module, "config"):
                config = module.config
                logging.info("    Hidden size: %s", config.hidden_size)
                logging.info("    Number of layers: %s", config.num_hidden_layers)
                logging.info(
                    "    Number of attention heads: %s", config.num_attention_heads
                )
                logging.info("    Vocabulary size: %s", config.vocab_size)

        # Log memory usage
        if torch.cuda.is_available():
            logging.info("\nGPU Memory Usage:")
            logging.info("Allocated: %.2f MB", torch.cuda.memory_allocated() / 1024**2)
            logging.info("Cached: %.2f MB", torch.cuda.memory_reserved() / 1024**2)

        self.gamma = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s", self.device)
        logging.info("Gamma (speculative tokens): %d", self.gamma)
        logging.info("%s", "=" * 50 + "\n")

        # Log model comparison if possible
        if hasattr(self.model, "config") and hasattr(self.approx_model.model, "config"):
            target_config = self.model.config
            approx_config = self.approx_model.model.config
            logging.info("\nModel Size Comparison:")
            logging.info("%-25s %15s %15s", "Metric", "Target", "Approximation")
            logging.info("%s", "-" * 60)
            logging.info(
                "%-25s %15s %15s",
                "Hidden size",
                target_config.hidden_size,
                approx_config.hidden_size,
            )
            logging.info(
                "%-25s %15s %15s",
                "Number of layers",
                target_config.num_hidden_layers,
                approx_config.num_hidden_layers,
            )
            logging.info(
                "%-25s %15s %15s",
                "Attention heads",
                target_config.num_attention_heads,
                approx_config.num_attention_heads,
            )
            logging.info(
                "%-25s %15s %15s",
                "Vocabulary size",
                target_config.vocab_size,
                approx_config.vocab_size,
            )

    def compute_transition_scores(self, sequences, scores, normalize_logits=True):
        """Compute transition scores similarly to HuggingFace's compute_transition_scores."""
        log_probs = []
        for logits in scores:
            # Normalize logits to probabilities
            if normalize_logits:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                log_probs_step = torch.nn.functional.log_softmax(logits, dim=-1)
            else:
                probs = logits
                log_probs_step = torch.log(logits)

            # Get the log probability of the selected tokens
            selected_tokens = sequences[:, -len(scores) :]
            batch_size = selected_tokens.shape[0]
            selected_log_probs = log_probs_step[
                torch.arange(batch_size, device=self.device),
                selected_tokens[:, -len(scores)],
            ]
            log_probs.append(selected_log_probs)

        return torch.stack(log_probs).T

    def predict(self, input_data: str, temperature: float, return_full: bool = False):
        """Generate text using speculative sampling with same interface as HuggingFaceModel."""
        logging.info("Starting prediction with temperature %s", temperature)
        logging.info("Input length: %d characters", len(input_data))

        # Tokenize input
        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        n_input_tokens = input_ids.size(1)
        logging.info("Input tokens: %d", n_input_tokens)

        # Setup pad token id like HuggingFaceModel
        if (
            "llama" in self.model_name.lower()
            or "falcon" in self.model_name
            or "mistral" in self.model_name.lower()
        ):
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        # Initialize KV caches with logging
        logging.info("Initializing KV caches...")
        approx_model_cache = KVCacheModel(
            self.approx_model.model, temperature, top_k=20, top_p=0.9
        )
        target_model_cache = KVCacheModel(self.model, temperature, top_k=20, top_p=0.9)
        logging.info("KV caches initialized")

        # Generate using speculative sampling
        outputs = SpeculativeOutput(
            sequences=input_ids.clone(),
            hidden_states=[],
            logits=torch.tensor([]),
            scores=[],
            decoder_hidden_states=[],
        )

        # Generation loop with logging
        generation_step = 0
        while outputs.sequences.shape[1] < n_input_tokens + self.max_new_tokens:
            generation_step += 1
            prefix_len = outputs.sequences.shape[1]
            logging.info("Generation step %d:", generation_step)
            logging.info("Current sequence length: %d", prefix_len)

            # Generate from approx model
            logging.info("Generating from approximation model...")
            x = approx_model_cache.generate(outputs.sequences, self.gamma)
            logging.info("Approximation model generated %d tokens", self.gamma)

            # Get target model probabilities
            logging.info("Getting target model probabilities...")
            _ = target_model_cache.generate(x, 1)

            n = prefix_len + self.gamma - 1
            accepted_tokens = 0

            # Accept/reject loop with logging
            for i in range(self.gamma):
                j = x[:, prefix_len + i]
                r = torch.rand(1, device=self.device)
                target_prob = target_model_cache._prob_history[:, prefix_len + i - 1, j]
                approx_prob = approx_model_cache._prob_history[:, prefix_len + i - 1, j]

                if r > target_prob / approx_prob:
                    logging.info(
                        "Token %d rejected (Target prob: %f, Approx prob: %f)",
                        i + 1,
                        target_prob.item(),
                        approx_prob.item(),
                    )
                    n = prefix_len + i - 1
                    break
                accepted_tokens += 1
                logging.info(
                    "Token %d accepted (Target prob: %f, Approx prob: %f)",
                    i + 1,
                    target_prob.item(),
                    approx_prob.item(),
                )

            outputs.sequences = x[:, : n + 1]
            approx_model_cache.rollback(n + 1)
            logging.info(
                "Accepted %d out of %d proposed tokens", accepted_tokens, self.gamma
            )

            if n < prefix_len + self.gamma - 1:
                # Rejection occurred, sample from target
                t = sample(
                    max_fn(
                        target_model_cache._prob_history[:, n, :]
                        - approx_model_cache._prob_history[:, n, :]
                    )
                ).to(
                    self.device
                )  # Ensure tensor is on correct device
                target_model_cache.rollback(n + 1)
            else:
                # All accepted, sample from target
                t = sample(target_model_cache._prob_history[:, -1, :]).to(
                    self.device
                )  # Ensure tensor is on correct device
                target_model_cache.rollback(n + 2)

            # Ensure both tensors are on the same device before concatenation
            outputs.sequences = torch.cat(
                (outputs.sequences.to(self.device), t.to(self.device)), dim=1
            )
            # Store hidden states and scores
            outputs.hidden_states = outputs.hidden_states + [
                target_model_cache._past_key_values
            ]
            outputs.decoder_hidden_states = outputs.decoder_hidden_states + [
                target_model_cache._past_key_values
            ]
            outputs.scores = outputs.scores + [
                target_model_cache._prob_history[:, -1, :]
            ]

        # Check token limit
        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                "Generation exceeding token limit %d > %d"
                % (len(outputs.sequences[0]), self.token_limit)
            )

        # Decode full answer

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )
        logging.info("Full generated text: %s", full_answer)

        if return_full:
            return full_answer

        # Process output and find correct offset
        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
            logging.info("Using direct input offset: %d", input_data_offset)
        else:
            # Try multiple ways to find the answer section
            content_start = full_answer.find("Answer:")
            if content_start != -1:
                input_data_offset = content_start
            else:
                for line in full_answer.split("\n"):
                    if line.strip().startswith("Answer:"):
                        input_data_offset = full_answer.find(line)
                        break
                else:
                    raise ValueError(
                        f"Cannot find answer content in text: {full_answer}"
                    )
            logging.info("Found answer at offset: %d", input_data_offset)

        # Extract answer portion and handle stop sequences
        answer = full_answer[input_data_offset:]
        logging.info("Extracted answer portion: %s", answer)

        stop_at = len(answer)
        sliced_answer = answer

        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                stop_idx = answer.find(stop)
                if stop_idx != -1:
                    stop_at = stop_idx
                    sliced_answer = answer[:stop_at]
                    break
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break

        sliced_answer = sliced_answer.strip()
        logging.info("Processed answer: %s", sliced_answer)

        # Calculate token counts based on the actual text portions
        input_portion = full_answer[:input_data_offset]
        generated_portion = full_answer[input_data_offset : input_data_offset + stop_at]

        # Get token counts for input and generated portions
        input_tokens = self.tokenizer(
            input_portion, return_tensors="pt", return_attention_mask=False
        )["input_ids"]

        full_tokens = self.tokenizer(
            input_portion + generated_portion,
            return_tensors="pt",
            return_attention_mask=False,
        )["input_ids"]

        # Calculate the actual number of generated tokens
        n_generated = len(full_tokens[0]) - len(input_tokens[0])
        logging.info(
            "Token counts - Full: %d, Input: %d, Generated: %d",
            len(full_tokens[0]),
            len(input_tokens[0]),
            n_generated,
        )

        if n_generated <= 0:
            logging.error("Token counting error:")
            logging.error("Input text: %s", input_portion)
            logging.error("Generated text: %s", generated_portion)
            logging.error(
                "Token counts - Full: %d, Input: %d",
                len(full_tokens[0]),
                len(input_tokens[0]),
            )
            raise ValueError(f"Invalid token count: {n_generated} tokens generated")

        # Handle hidden states
        if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states:
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        logging.info("Hidden states processing:")
        logging.info("n_generated: %d, hidden length: %d", n_generated, len(hidden))

        # Process hidden states and get embedding
        try:
            if n_generated - 1 >= len(hidden):
                logging.warning(
                    "Using last hidden state (index %d > length %d)",
                    n_generated - 1,
                    len(hidden),
                )
                last_input = hidden[-1]
            else:
                last_input = hidden[n_generated - 1]

            # Extract tensor from nested structure
            if isinstance(last_input, list) and len(last_input) > 0:
                first_tuple = last_input[0]
                if isinstance(first_tuple, tuple):
                    all_tensors = [
                        item for item in first_tuple if isinstance(item, torch.Tensor)
                    ]
                    if not all_tensors:
                        raise ValueError("No tensors found in tuple")
                    tensor = all_tensors[-1]
                else:
                    tensor = first_tuple
            else:
                tensor = last_input

            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor, got {type(tensor)}")

            last_token_embedding = tensor[0, -1, :].cpu()

        except Exception as e:
            logging.error("Error processing hidden states: %s", str(e))
            logging.error("Hidden type: %s", type(hidden))
            logging.error(
                "Hidden states info: length=%d, n_generated=%d",
                len(hidden),
                n_generated,
            )
            raise

        # Compute transition scores
        transition_scores = self.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        # Get log likelihoods
        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning("Only one log likelihood value available")
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == 0:
            raise ValueError("No log likelihoods computed")

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning("Generation reached max_new_tokens limit")

        return sliced_answer, log_likelihoods, last_token_embedding


### COT Model ###
@dataclass
class CoTOutput:
    sequences: torch.Tensor
    hidden_states: tuple
    logits: torch.Tensor
    scores: Optional[list[torch.Tensor]] = None
    past_key_values: Optional[Tuple[torch.Tensor]] = None
    decoder_hidden_states: Optional[tuple] = None
    reasoning_steps: List[str] = None


class ChainOfThoughtModel(HuggingfaceModel):
    def __init__(
        self,
        model_name: str,
        stop_sequences="default",
        max_new_tokens=20,
    ):
        super().__init__(model_name, stop_sequences, max_new_tokens)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_logging(model_name, max_new_tokens)

    def _setup_logging(self, model_name, max_new_tokens):
        logging.info("%s", f"\n{'='*50}")
        logging.info("Initializing ChainOfThoughtModel:")
        logging.info("Model: %s", model_name)
        logging.info("Max new tokens: %s", max_new_tokens)
        logging.info("\nModel Architecture:")
        logging.info("Model type: %s", type(self.model).__name__)
        logging.info(
            "Number of parameters: %s", sum(p.numel() for p in self.model.parameters())
        )

    def _get_next_token_logit(self, query):
        """Get logits for next token prediction."""
        inputs = self.tokenizer([query], return_tensors="pt").to(self.device)
        gen_out = self.model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return gen_out.scores[-1]

    def _get_token_path_prob(self, gen_out, num_append=1):
        """Calculate token path probabilities."""
        logits = gen_out.scores
        num_output = len(logits)
        output_ids = gen_out.sequences[0][-num_output - num_append :]
        path_prob = torch.stack([score[0].max() for score in logits])
        path_prob = torch.nn.functional.softmax(path_prob, dim=0)
        return output_ids, path_prob

    def _get_path_prob(self, gen_out, init_token_prob=None):
        """Calculate word-level path probabilities."""
        if init_token_prob is None:
            token_ids, probs = self._get_token_path_prob(gen_out, num_append=0)
        else:
            token_ids, probs = self._get_token_path_prob(gen_out)
            # Convert init_token_prob to tensor and match device
            init_prob_tensor = torch.tensor([init_token_prob]).to(probs.device)
            probs = torch.cat([init_prob_tensor, probs])

        word_probs = []
        ids = []
        current_n_tokens = 0
        word_prob = 0
        current_n_words = 0

        for token_id, prob in zip(token_ids, probs):
            ids.append(token_id)
            decode_seq = self.tokenizer.decode(ids)
            words = re.split(r" |\n|\.\|:", decode_seq)
            word = words[-1]

            if len(words) == current_n_words:
                word_prob += prob.item()  # Convert tensor to scalar
                current_n_tokens += 1
                word_probs[-1] = (word, word_prob / current_n_tokens)
            else:
                word_prob = prob.item()  # Convert tensor to scalar
                current_n_tokens = 1
                word_probs.append((word, word_prob / current_n_tokens))
                current_n_words += 1

        return word_probs

    def _get_follow_up_output(self, gen_out, follow_up_template, max_new_tokens=40):
        """Get follow-up completion with template."""
        construct_input = lambda new_ids: {
            "input_ids": new_ids,
            "attention_mask": torch.ones_like(new_ids),
        }
        output_ids = gen_out.sequences
        follow_up_ids = self.tokenizer(follow_up_template, return_tensors="pt")[
            "input_ids"
        ].to(self.device)
        new_ids = torch.cat([output_ids, follow_up_ids], dim=1)
        inputs = construct_input(new_ids)
        follow_up_out = self.model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            output_hidden_states=True,
        )
        return follow_up_out

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from generated text."""
        # Try to find explicit answer marker
        answer_markers = [
            "Therefore, ",
            "Thus, ",
            "So, ",
            "The answer is ",
            "Final answer: ",
        ]
        for marker in answer_markers:
            if marker in text:
                return text.split(marker)[-1].strip()

        # Fallback to last sentence
        sentences = text.split(".")
        return sentences[-1].strip()

    def get_last_path_probabilities(self):
        """Get word probabilities from the last prediction."""
        return getattr(self, "last_path_probs", None)

    def compute_answer_confidence(
        self, logits: torch.Tensor, answer_tokens: torch.Tensor
    ) -> float:
        """Compute confidence score for answer tokens as described in paper Section 2.2."""
        if not isinstance(logits, (list, tuple)):
            logits = [logits]

        confidence_scores = []

        for pos, logit in enumerate(logits):
            # Get probabilities for this position
            probs = torch.nn.functional.softmax(logit, dim=-1)[
                0
            ]  # Add [0] to get first batch

            # Sort probabilities in descending order
            sorted_probs, _ = torch.sort(probs, descending=True)

            # Calculate confidence as difference between top two probabilities
            # If there's only one non-zero probability, use 1.0 as confidence
            if len(sorted_probs) > 1:
                prob_diff = (sorted_probs[0] - sorted_probs[1]).item()
            else:
                prob_diff = 1.0

            confidence_scores.append(prob_diff)

        # Average over all positions
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        return 0.0

    def get_word_level_probs(
        self, token_ids: List[int], token_probs: List[float]
    ) -> List[Tuple[str, float]]:
        """Calculate word-level probabilities by aggregating token probabilities.

        Args:
            token_ids: List of token IDs in the sequence
            token_probs: Probability for each token

        Returns:
            List of (word, probability) tuples
        """
        word_probs = []
        current_word = []
        current_probs = []

        for token_id, prob in zip(token_ids, token_probs):
            token = self.tokenizer.decode([token_id])

            if token.startswith(" ") or not current_word:
                # New word boundary
                if current_word:
                    # Average probability for previous word
                    word = "".join(current_word)
                    avg_prob = sum(current_probs) / len(current_probs)
                    word_probs.append((word, avg_prob))
                    current_word = []
                    current_probs = []

            current_word.append(token.strip())
            current_probs.append(prob)

        # Handle last word
        if current_word:
            word = "".join(current_word)
            avg_prob = sum(current_probs) / len(current_probs)
            word_probs.append((word, avg_prob))

        return word_probs

    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract reasoning steps from generated text.

        Looks for:
        1. Numbered steps (e.g., "1.", "Step 1:", etc.)
        2. Bullet points or dashes
        3. Sequential keywords ("First", "Second", "Finally")
        4. Logical connectors starting new lines

        Args:
            text: Generated text to analyze

        Returns:
            List of extracted reasoning steps
        """
        steps = []
        lines = text.split("\n")
        current_step = ""

        # Keywords that indicate reasoning steps
        step_starters = [
            r"^\d+\.",  # Numbered steps like "1."
            r"^step\s+\d+:?",  # "Step 1:" format
            r"^(first|second|third|finally|next|then)[\s:]",  # Sequential keywords
            r"^\s*[-•]\s+",  # Bullet points or dashes
            r"^(therefore|thus|because|so|hence)",  # Logical connectors
        ]

        step_pattern = "|".join(step_starters)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line starts new step
            if re.match(step_pattern, line.lower()):
                if current_step:  # Save previous step if exists
                    steps.append(current_step.strip())
                current_step = line
            elif current_step:  # Continue current step
                current_step += " " + line
            else:  # Start first step if no clear marker
                current_step = line

        # Add final step
        if current_step:
            steps.append(current_step.strip())

        # Post-process steps
        processed_steps = []
        for step in steps:
            # Remove step numbers/markers
            step = re.sub(r"^\d+\.\s*", "", step)
            step = re.sub(r"^step\s+\d+:?\s*", "", step)
            step = re.sub(r"^[-•]\s+", "", step)

            # Clean up common prefixes
            for prefix in [
                "first",
                "second",
                "third",
                "finally",
                "next",
                "then",
                "therefore",
                "thus",
                "because",
                "so",
                "hence",
            ]:
                if step.lower().startswith(prefix):
                    step = step[len(prefix) :].strip()
                    if step.startswith(",") or step.startswith(":"):
                        step = step[1:].strip()

            if step:  # Only add non-empty steps
                processed_steps.append(step)

        # Handle special case for numerical calculations
        if not processed_steps:
            calculation_steps = re.findall(r"\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+", text)
            if calculation_steps:
                processed_steps.extend(calculation_steps)

        return processed_steps

    def is_cot_path(self, text: str) -> bool:
        """Detect if a generation path contains chain-of-thought reasoning.

        Looks for:
        - Step-by-step reasoning indicators
        - Mathematical operations/calculations
        - Logical connectors (therefore, because, so)
        - Numbered steps

        Args:
            text: Generated text to analyze

        Returns:
            bool: True if path shows CoT reasoning
        """
        # Look for numbered steps
        has_numbered_steps = bool(re.search(r"\d+\s*\.|step\s*\d+", text.lower()))

        # Look for reasoning keywords
        reasoning_indicators = [
            "therefore",
            "because",
            "so",
            "thus",
            "hence",
            "first",
            "second",
            "third",
            "finally",
        ]
        has_reasoning = any(
            indicator in text.lower() for indicator in reasoning_indicators
        )

        # Look for calculations
        has_calculations = bool(re.search(r"\d+\s*[\+\-\*\/]\s*\d+", text))

        return has_numbered_steps or has_calculations or has_reasoning

    def predict(self, input_data: str, temperature: float, return_full: bool = False):
        """Generate text using chain-of-thought reasoning with tree search decoding."""
        logging.info("Starting CoT prediction with temperature %s", temperature)
        logging.info("Input length: %d characters", len(input_data))

        # Format the CoT prompt
        if "Question:" not in input_data and "Q:" not in input_data:
            cot_input = f"Question: {input_data}\nLet's solve this step by step:\n1."
        else:
            cot_input = input_data

        inputs = self.tokenizer(cot_input, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids)
        inputs["attention_mask"] = attention_mask

        n_input_tokens = input_ids.size(1)
        logging.info("Input tokens: %d", n_input_tokens)

        # Get initial branching logits
        logits = self._get_next_token_logit(cot_input)
        k = 5  # Number of paths to explore
        top_k_tokens = logits[0].argsort()[-k:]
        top_k_probs = torch.nn.functional.softmax(logits[0][top_k_tokens], dim=0)

        best_response = None
        best_confidence = -float("inf")
        best_reasoning = None
        best_logits = None
        best_hidden = None

        # Basic generation parameters
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "output_scores": True,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": True if temperature > 0 else False,
            "temperature": temperature,
            "output_hidden_states": True,
        }

        # Explore each path
        for token, prob in zip(top_k_tokens, top_k_probs):
            try:
                # Generate initial reasoning chain
                new_query = cot_input + self.tokenizer.decode(token)
                new_inputs = self.tokenizer(new_query, return_tensors="pt").to(
                    self.device
                )

                chain_output = self.model.generate(**new_inputs, **gen_kwargs)

                # Generate answer after reasoning
                reasoning_text = self.tokenizer.decode(
                    chain_output.sequences[0], skip_special_tokens=True
                )
                follow_up = "\nTherefore, the answer is:"

                follow_up_ids = self.tokenizer(follow_up, return_tensors="pt")[
                    "input_ids"
                ].to(self.device)
                full_ids = torch.cat([chain_output.sequences, follow_up_ids], dim=1)

                answer_output = self.model.generate(
                    input_ids=full_ids,
                    attention_mask=torch.ones_like(full_ids),
                    **gen_kwargs,
                )

                full_output = self.tokenizer.decode(
                    answer_output.sequences[0], skip_special_tokens=True
                )

                # Extract reasoning and answer
                reasoning_steps = self._extract_reasoning_steps(full_output)
                answer = self._extract_answer(full_output)

                # Skip if no clear reasoning or answer
                if not reasoning_steps or not answer or len(answer.strip()) < 2:
                    continue

                # Calculate confidence based on logits
                if len(answer_output.scores) > 0:
                    answer_logits = answer_output.scores[
                        -len(self.tokenizer(answer)["input_ids"]) :
                    ]
                    confidence = self.compute_answer_confidence(
                        answer_logits,
                        self.tokenizer(answer, return_tensors="pt")["input_ids"][0],
                    )
                else:
                    confidence = 0.0

                if confidence > best_confidence:
                    best_response = answer
                    best_confidence = confidence
                    best_reasoning = reasoning_steps
                    best_logits = answer_output.scores
                    best_hidden = (
                        answer_output.hidden_states[-1]
                        if hasattr(answer_output, "hidden_states")
                        else None
                    )

            except Exception as e:
                logging.warning(f"Path generation failed: {e}")
                continue

        # Return fallback if no valid paths
        if best_response is None:
            logging.warning(
                "No valid reasoning paths found, falling back to direct generation"
            )
            outputs = self.model.generate(**inputs, **gen_kwargs)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = self._extract_answer(text)
            return answer, [0.0], torch.zeros(self.model.config.hidden_size)

        # Store reasoning steps
        self.last_reasoning_steps = best_reasoning

        # Calculate log likelihoods
        log_likelihoods = []
        if best_logits:
            probs = [
                torch.nn.functional.softmax(logit, dim=-1).max().item()
                for logit in best_logits
            ]
            log_likelihoods = [-sum(np.log(p) for p in probs if p > 0)]

        # Get embedding from hidden states
        if best_hidden is not None:
            embedding = best_hidden[0, -1, :].cpu()
        else:
            embedding = torch.zeros(self.model.config.hidden_size)

        return best_response, log_likelihoods, embedding


### Load Models ###
def init_model(model_name, model_max_new_tokens):
    mn = model_name
    if "llama" in mn.lower() or "falcon" in mn or "mistral" in mn.lower():
        model = HuggingfaceModel(
            mn, stop_sequences="default", max_new_tokens=model_max_new_tokens
        )
    else:
        raise ValueError(f"Unknown model_name `{mn}`.")
    return model


def init_speculative_model(target_model_name, approx_model_name, model_max_new_tokens):
    mn = target_model_name
    if "llama" in mn.lower() or "falcon" in mn or "mistral" in mn.lower():
        model = SpeculativeSamplingModel(
            target_model_name=target_model_name,
            approx_model_name=approx_model_name,
            stop_sequences="default",
            max_new_tokens=model_max_new_tokens,
        )
    else:
        raise ValueError(f"Unknown model_name `{mn}`.")
    return model


### Get models ###
MODEL_NAMES = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
]


def get_models(
    target_model_name, draft_model_name=None, model_max_new_tokens=20, quantize=False
):
    entailment_model = EntailmentDeberta()
    assert (
        target_model_name in MODEL_NAMES
    ), f"correct your TGT[{target_model_name}] or DFT[{draft_model_name}]!"
    if quantize:
        target_model_name = target_model_name + "-8bit"
    if draft_model_name is None:
        base_gen_model = init_model(
            model_name=target_model_name, model_max_new_tokens=model_max_new_tokens
        )
    else:
        assert draft_model_name in MODEL_NAMES
        if quantize:
            draft_model_name = draft_model_name + "-8bit"
        base_gen_model = init_speculative_model(
            target_model_name=target_model_name,
            approx_model_name=draft_model_name,
            model_max_new_tokens=model_max_new_tokens,
        )
    return base_gen_model, entailment_model
