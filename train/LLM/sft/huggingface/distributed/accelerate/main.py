# from huggingface_hub import login

# login()
# Base Part
import torch

# Data Part
from datasets import load_dataset
from transformers import AutoTokenizer

# Model Part
from transformers import AutoModelForSequenceClassification

# Training Part
from transformers import TrainingArguments, Trainer

# Evaluation Part
import numpy as np
import evaluate

dataset = load_dataset("yelp_review_full")
dataset["train"] = dataset["train"].select(range(10000))  # Limit to 10,000 samples for faster training
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True, batch_size=10000)

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./test-optimizer",
    max_steps=1000,
    per_device_train_batch_size=4,
    logging_strategy="steps",
    # optim="apollo_adamw", ["apollo_adamw", "grokadamw", "adalomo", "schedule_free_radamw"]
    # lr_scheduler_type="constant",                                                            # if you want to use schedule-free optimizer
    # optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    # optim_args="proj=random,rank=1,scale=128.0,scale_type=tensor,update_proj_gap=200",       # if you want to use APOLLO-mini optimizer
    logging_steps=1,
    learning_rate=2e-5,
    save_strategy="no",
    run_name="optimizer-name",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)
# Huggingface Trainer #
trainer.train()

# Naive PyTorch Training Loop #
# from accelerate import Accelerator
# accelerator = Accelerator()

# model, optimizer, training_dataloader, scheduler = accelerator.prepare(
#     model, optimizer, training_dataloader, scheduler
# )

# for batch in training_dataloader:
#     optimizer.zero_grad()
#     inputs, targets = batch
#     inputs = inputs.to(device)
#     targets = targets.to(device)
#     outputs = model(inputs)
#     loss = loss_function(outputs, targets)
#     accelerator.backward(loss)
#       optimizer.step()
#       scheduler.step()

# TODO: Hyperparameter Tuning