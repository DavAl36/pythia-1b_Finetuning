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
from data_processing import *
from model_loading import *




PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

def preprocess(tokenizer, instruction_text,device):
    prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(
        instruction=instruction_text
    )
    inputs = tokenizer(prompt_text, return_tensors="pt",).to(device)
    inputs["prompt_text"] = prompt_text
    inputs["instruction_text"] = instruction_text
    return inputs

def forward(model, tokenizer, model_inputs, max_length=200):
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", None)

    if input_ids.shape[1] == 0:
        input_ids = None
        attention_mask = None
        in_b = 1
    else:
        in_b = input_ids.shape[0]

    generated_sequence = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_length
    )

    out_b = generated_sequence.shape[0]
    generated_sequence = generated_sequence.reshape(
        in_b, out_b // in_b, *generated_sequence.shape[1:]
    )
    instruction_text = model_inputs.get("instruction_text", None)

    return {
        "generated_sequence": generated_sequence,
        "input_ids": input_ids, "instruction_text": instruction_text
    }

def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.

    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.

    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token

    Raises:
        ValueError: if more than one ID was generated

    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]

def postprocess(tokenizer, model_outputs, return_full_text=False):
    response_key_token_id = get_special_token_id(tokenizer, RESPONSE_KEY_NL)
    end_key_token_id = get_special_token_id(tokenizer, END_KEY)
    generated_sequence = model_outputs["generated_sequence"][0]
    instruction_text = model_outputs["instruction_text"]
    generated_sequence = generated_sequence.cpu().numpy().tolist() # https://stackoverflow.com/questions/53900910/typeerror-can-t-convert-cuda-tensor-to-numpy-use-tensor-cpu-to-copy-the-tens
    records = []

    #print(response_key_token_id, end_key_token_id)

    for sequence in generated_sequence:
        decoded = None

        try:
            response_pos = sequence.index(response_key_token_id)
        except ValueError:
            logger.warn(
                f"Could not find response key {response_key_token_id} in: {sequence}"
            )
            response_pos = None

        if response_pos:
            try:
                end_pos = sequence.index(end_key_token_id)
            except ValueError:
                logger.warning(
                    f"Could not find end key, the output is truncated!"
                )
                end_pos = None
            decoded = tokenizer.decode(
                sequence[response_pos + 1 : end_pos]).strip()

        # If True,append the decoded text to the original instruction.
        if return_full_text:
            decoded = f"{instruction_text}\n{decoded}"
        rec = {"generated_text": decoded}
        records.append(rec)
    return records



### name for model and tokenizer
INPUT_MODEL = "EleutherAI/pythia-1b"
#INPUT_MODEL = "databricks/dolly-v2-3b"

'''
_, tokenizer = get_model_tokenizer(
    pretrained_model_name_or_path=INPUT_MODEL,
    gradient_checkpointing=True
)
'''

logger = logging.getLogger("logger")

model_path = "/home/dav/Scrivania/LLM/finetuned"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = 'cpu'
print('Using device:', device)


model_finetuned = load_model(model_path, gradient_checkpointing=True).to(device)
tokenizer = load_tokenizer(model_path)


for i in range(0,10):
    print("######################################")
    print(str(i) + ")Write your question about Helicopters")
    text = str(input())
    #print("Write the max number of tokens")
    #n_tokens = int(input())
    pre_process_result = preprocess(tokenizer, text,device)
    model_result = forward(model_finetuned, tokenizer, pre_process_result,max_length=300)
    final_output = postprocess(tokenizer, model_result, False)
    print("Answer finetuned: " + str(final_output[0]['generated_text']))


