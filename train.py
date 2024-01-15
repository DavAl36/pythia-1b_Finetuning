from data_processing import *
from model_loading import *

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


train_path = "/home/dav/Scrivania/LLM/datasets/"
local_output_dir="/home/dav/Scrivania/LLM/finetuned"

logger = logging.getLogger("logger")


### name for model and tokenizer
INPUT_MODEL = "EleutherAI/pythia-1b"
#INPUT_MODEL = "databricks/dolly-v2-3b"



model, tokenizer = get_model_tokenizer(
    pretrained_model_name_or_path=INPUT_MODEL,
    gradient_checkpointing=True
)

# find max length in model configuration
conf = model.config
max_length = getattr(model.config, "max_position_embeddings", None)

processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length, path_or_dataset=train_path)
split_dataset = processed_dataset.train_test_split(test_size=0.3)

data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )


training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        use_cpu=True,##############
        fp16=False,
        bf16=False,
        learning_rate=1e-5,
        num_train_epochs=5,#5,
        deepspeed=None,
        gradient_checkpointing=True,
        logging_dir=f"{local_output_dir}/runs",
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=10,
        load_best_model_at_end=False,
        #report_to="tensorboard",
        disable_tqdm=True,
        remove_unused_columns=False,
        local_rank=2,
        warmup_steps=0,
    )

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=data_collator,
)


trainer.train()
trainer.save_model(output_dir=local_output_dir)

print("END TRAINING")


