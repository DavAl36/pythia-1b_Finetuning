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

def load_tokenizer(pretrained_model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]}
    )
    return tokenizer

def load_model(pretrained_model_name_or_path, gradient_checkpointing):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True,
        use_cache=False if gradient_checkpointing else True
    )
    return model

def get_model_tokenizer(
    pretrained_model_name_or_path, gradient_checkpointing):
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(
        pretrained_model_name_or_path, gradient_checkpointing
    )
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer
