import json
import copy
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
import numpy as np
import pandas as pd

from transformers.models.deberta_v2 import DebertaV2ForTokenClassification, DebertaV2TokenizerFast
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from transformers.data.data_collator import DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict, concatenate_datasets
import wandb

from data import CustomTokenizer
from model import ModelInit
from eval import MetricsComputer

TRAINING_MODEL_PATH = "microsoft/deberta-v3-large"
TRAINING_MAX_LENGTH = 3072
EVAL_MAX_LENGTH = 3072
CONF_THRESH = 0.9
LR = 2.5e-5
LR_SCHEDULER_TYPE = "linear"
NUM_EPOCHS = 3
BATCH_SIZE = 1
EVAL_BATCH_SIZE = 8
GRAD_ACCUMULATION_STEPS = 16 // BATCH_SIZE
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
AMP = True
FREEZE_EMBEDDING = False
FREEZE_LAYERS = 6
N_SPLITS = 4
NEGATIVE_RATIO = 0.3  # down sample ratio of negative samples in the training set
OUTPUT_DIR = "output"
Path(OUTPUT_DIR).mkdir(exist_ok=True)
TRAIN_DATA_PATH = "/root/workspace/pipelines/raw_data/kaggle/pii/train.json"
TEST_DATA_PATH = "/root/workspace/pipelines/raw_data/kaggle/pii/test.json"

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    fp16=AMP,
    learning_rate=LR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    report_to="none",
    eval_strategy="steps",
    eval_steps=50,
    eval_delay=100,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=1,
    logging_steps=10,
    metric_for_best_model="f5",
    greater_is_better=True,
    load_best_model_at_end=True,
    overwrite_output_dir=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
)

with Path(TRAIN_DATA_PATH).open("r") as f:
    original_data = json.load(f)

all_labels = [
    'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O'
]
id2label = {i: l for i, l in enumerate(all_labels)}
label2id = {v: k for k, v in id2label.items()}
target = [l for l in all_labels if l != "O"]

tokenizer = DebertaV2TokenizerFast.from_pretrained(TRAINING_MODEL_PATH)
train_encoder = CustomTokenizer(tokenizer=tokenizer, label2id=label2id, max_length=TRAINING_MAX_LENGTH)
eval_encoder = CustomTokenizer(tokenizer=tokenizer, label2id=label2id, max_length=EVAL_MAX_LENGTH)

ds = DatasetDict()

for key, data in zip(["original", "extra"], [original_data]): #, extra_data]):
    ds[key] = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [str(x["document"]) for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
        "provided_labels": [x["labels"] for x in data],
    })

model_init = ModelInit(
    TRAINING_MODEL_PATH,
    id2label=id2label,
    label2id=label2id,
    freeze_embedding=FREEZE_EMBEDDING,
    freeze_layers=FREEZE_LAYERS,
)

folds = [
    (
        np.array([i for i, d in enumerate(ds["original"]["document"]) if int(d) % N_SPLITS != s]),
        np.array([i for i, d in enumerate(ds["original"]["document"]) if int(d) % N_SPLITS == s])
    )
    for s in range(N_SPLITS)
]

negative_idxs = [i for i, labels in enumerate(ds["original"]["provided_labels"]) if not any(np.array(labels) != "O")]
exclude_indices = negative_idxs[int(len(negative_idxs) * NEGATIVE_RATIO):]

for fold_idx, (train_idx, eval_idx) in enumerate(folds):
    if fold_idx != 3:
        continue
    args.run_name = f"fold-{fold_idx}"
    args.output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
    original_ds = ds["original"].select([i for i in train_idx if i not in exclude_indices])
    # train_ds = concatenate_datasets([original_ds, ds["extra"]])
    train_ds = original_ds.map(train_encoder, num_proc=os.cpu_count())
    eval_ds = ds["original"].select(eval_idx)
    eval_ds = eval_ds.map(eval_encoder, num_proc=os.cpu_count())
    trainer = Trainer(
        args=args,
        model_init=model_init,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=MetricsComputer(eval_ds=eval_ds, label2id=label2id),
        data_collator=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16),
    )
    # break # delete this line to reproduce the result.
    trainer.train()
    eval_res = trainer.evaluate(eval_dataset=eval_ds)
    with open(os.path.join(args.output_dir, "eval_result.json"), "w") as f:
        json.dump(eval_res, f)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

