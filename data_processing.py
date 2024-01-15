import re
import logging
from functools import partial
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments
)
from prompt_struct import *


#def load_training_dataset(path_or_dataset="databricks/databricks-dolly-15k"):
def load_training_dataset(path_or_dataset):
    dataset = load_dataset(path_or_dataset)["train"]
    def _add_text(rec):
        instruction = rec["instruction"]
        response = rec["response"]
        context = rec.get("context")
        if context:
            rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(
                instruction=instruction,
                response=response,
                input=context
            )
        else:
            rec["text"] = PROMPT_NO_INPUT_FORMAT.format(
                instruction=instruction,
                response=response
            )
        return rec
    dataset = dataset.map(_add_text)
    return dataset

def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

def preprocess_dataset(tokenizer, max_length,path_or_dataset):
    dataset = load_training_dataset(path_or_dataset)
    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "context", "response", "text", "category"],
    )

    # Make sure we don't have any truncated records, as this would mean the end keyword is missing.
    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
    dataset = dataset.shuffle()
    return dataset

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)
        labels = batch["labels"].clone()

        for i in range(len(examples)):
            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch