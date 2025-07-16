from datasets import load_dataset
from transformers import AutoTokenizer
dataset = load_dataset("HuggingFaceTB/smoltalk", name='all', split="train")

print("Number of samples in the dataset:", len(dataset))
print("First sample:", dataset[0])

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)

def tokenize_function(examples):
    return tokenizer(examples["input"], truncation=True, padding="max_length", max_length=16384)

def tokenize_chat_function(examples):
    return tokenizer()

tokenized_input_dataset = dataset.map(lambda x: {"input": tokenizer(x["input"], truncation=True, padding="max_length", max_length=16384)["input_ids"]}, batched=True, batch_size=10000, num_proc=16)
tokenized_output_dataset = dataset.map(lambda x: {"output": tokenizer(x["output"], truncation=True, padding="max_length", max_length=16384)["input_ids"]}, batched=True, batch_size=10000, num_proc=16)

# Calulate total number of tokens in the tokenized_dataset
total_input_tokens = sum(len(tokenized_input_dataset[i]["input"]) for i in range(len(tokenized_input_dataset)))
total_output_tokens = sum(len(tokenized_output_dataset[i]["output"]) for i in range(len(tokenized_output_dataset)))
print("Total input tokens:", total_input_tokens)
print("Total output tokens:", total_output_tokens)
