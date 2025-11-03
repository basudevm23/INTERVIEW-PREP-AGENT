# train_lora_tinyllama.py
import os
import json
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    # We'll convert to simple text pairs: "Q: ... A: ... Score: <score>"
    texts = [{"text": f"Q: {d['question']}\nA: {d['answer']}\nScore: {d['score']}"} for d in lines]
    return Dataset.from_list(texts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_rubric_pairs.jsonl")
    parser.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--out", default="./adapters/checkpoint-1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    print("Loading dataset...")
    ds = load_jsonl(args.data)
    print(ds[0])

    print("Loading tokenizer and base model (this may download files)...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    # tokenization
    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=256)
    tokenized = ds.map(tokenize, batched=True)

    # LoRA config for causal LM
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )

    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir="./adapters",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training (demo)...")
    trainer.train()

    print("Saving adapter to", args.out)
    model.save_pretrained(args.out)
    print("✅ LoRA demo training complete")

if __name__ == "__main__":
    main()
