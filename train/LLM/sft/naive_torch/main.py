from dotenv import load_dotenv
import os
import argparse
load_dotenv()
import math

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with SFT")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pre-trained model or model identifier from Hugging Face Hub.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training data file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate for AdamW optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Load training data
    train_dataset = load_dataset(args.train_path, split="train")  # Load your dataset here

    # Create DataLoader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        collate_fn=default_data_collator,
    )

    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = len(train_dataloader) * args.num_train_epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training loop
    model.train()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % args.logging_steps == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # Save the trained model
    model.save_pretrained(args.output_dir)