import argparse
from tqdm import tqdm
import os
import random
import json
import logging
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
import wandb
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, prompt_template):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]
        output_text = item["output"]
        
        # Format using prompt template
        formatted_text = self.prompt_template.format(
            input=input_text,
            output=output_text
        )
        
        # Tokenize
        encoded = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels (only compute loss on the output part)
        labels = encoded["input_ids"].clone()
        
        # Prepare input portion mask - figure out where the output begins
        prompt_only = self.prompt_template.format(input=input_text, output="")
        prompt_len = len(self.tokenizer.encode(prompt_only)) - 1  # -1 because we don't want to mask the first output token
        
        # Set -100 for input portion (no loss)
        labels[0, :prompt_len] = -100
        
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        labels = labels.squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def load_jsonl_data(file_path):
    """
    Load data from a jsonl file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_prompt_template(task_type):
    """
    Returns an appropriate prompt template based on task_type.
    You can customize templates for each task type.
    """
    templates = {
        "math": "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>",
        "coding": "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>",
        "factual_knowledge": "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>",
        "creative_writing": "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    }
    
    # Default to factual_knowledge template if task_type not found
    return templates.get(task_type, templates["factual_knowledge"])

def prepare_datasets(args, tokenizer):
    """
    Prepare train, validation, and test datasets.
    """
    # Load from jsonl files
    train_data = load_jsonl_data(os.path.join(args.data_dir, args.task, "train.jsonl"))
    val_data = load_jsonl_data(os.path.join(args.data_dir, args.task, "validation.jsonl"))
    test_data = load_jsonl_data(os.path.join(args.data_dir, args.task, "test.jsonl"))
    
    # Extract task type from first example (assuming all examples have the same task type)
    task_type = train_data[0]["task_type"] if len(train_data) > 0 else args.task
    
    # Get appropriate prompt template for this task
    prompt_template = get_prompt_template(task_type)
    
    # If using data percentage < 100%, sample the data
    if args.percentage < 1.0:
        random.shuffle(train_data)
        train_data = train_data[:int(len(train_data) * args.percentage)]
    
    logger.info(f"Task: {task_type}, Train samples: {len(train_data)}, Val samples: {len(val_data)}, Test samples: {len(test_data)}")
    
    # Create datasets
    train_dataset = CustomDataset(train_data, tokenizer, args.max_token_length, prompt_template)
    val_dataset = CustomDataset(val_data, tokenizer, args.max_token_length, prompt_template)
    test_dataset = CustomDataset(test_data, tokenizer, args.max_token_length, prompt_template)
    
    return train_dataset, val_dataset, test_dataset, task_type

def compute_metrics(eval_preds):
    """
    Compute evaluation metrics.
    """
    logits, labels = eval_preds
    
    # Get predictions (argmax)
    predictions = np.argmax(logits, axis=-1)
    
    # Create a mask for valid labels (not -100)
    mask = labels != -100
    
    # Compute accuracy only on valid positions
    valid_preds = predictions[mask]
    valid_labels = labels[mask]
    
    accuracy = np.sum(valid_preds == valid_labels) / len(valid_labels) if len(valid_labels) > 0 else 0
    
    return {
        "accuracy": accuracy
    }

def generate_samples(args, model, tokenizer, test_dataset, task_type, num_samples=5):
    """
    Generate sample outputs for qualitative evaluation.
    """
    if not args.main_process:
        return
    
    model.eval()
    samples = []
    
    # Sample a few examples from test set
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    for i in indices:
        example = test_dataset.data[i]
        input_text = example["input"]
        
        # Format using prompt template
        prompt_template = get_prompt_template(task_type)
        prompt = prompt_template.format(input=input_text, output="")
        
        # Tokenize with attention mask
        encoded = tokenizer(
            prompt, 
            return_tensors="pt",
            return_attention_mask=True
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract just the generated part
        assistant_part = generated_text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0]
        
        # Log sample
        samples.append({
            "input": input_text,
            "expected": example["output"],
            "generated": assistant_part.strip()
        })
    
    # Log to wandb
    if args.use_wandb and args.main_process:
        for i, sample in enumerate(samples):
            wandb.log({f"sample_{i+1}/input": sample["input"],
                    f"sample_{i+1}/expected": sample["expected"],
                    f"sample_{i+1}/generated": sample["generated"]})
            
    else:
        for i, sample in enumerate(samples):
            logger.info(f"Sample {i+1}: Input: {sample['input']}, Expected: {sample['expected']}, Generated: {sample['generated']}")
    
    return samples

def seed_everything(seed):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="LoRA finetuning for multiple tasks")
    
    # Model and tokenizer arguments
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B")
    
    # Task and data arguments
    parser.add_argument("--task", type=str, required=True, help="Task name (math, coding, factual_knowledge, creative_writing)")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--output_dir", type=str, default="./output/")
    parser.add_argument("--cache_dir", type=str, default="./cache/")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_token_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--percentage", type=float, default=1.0, help="Percentage of training data to use")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    
    # Other arguments
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit mode")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit mode")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="lora-finetune")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    
    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_name = f"{args.task}_{timestamp}"
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up distributed training
    args.main_process = args.local_rank in [-1, 0]
    
    # Set random seed
    seed_everything(args.seed)
    
    # Initialize wandb if main process
    if args.use_wandb and args.main_process:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args)
        )
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, 
        trust_remote_code=True,
        use_fast=False
    )
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset, val_dataset, test_dataset, task_type = prepare_datasets(args, tokenizer)
    
    # LoRA config
    logger.info("Setting up LoRA config...")
    target_modules = args.target_modules.split(",")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Load base model
    logger.info(f"Loading model: {args.model}")
    model_kwargs = {
        "trust_remote_code": True,
        "cache_dir": args.cache_dir,
        "torch_dtype": torch.bfloat16,
    }
    
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        model_kwargs["quantization_config"] = {
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
    
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    
    # Prepare model for k-bit training if using quantization
    if args.load_in_8bit or args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA config
    model = get_peft_model(model, lora_config)
    
    if args.main_process:
        print_trainable_parameters(model)
    
    # Set up Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="wandb" if args.use_wandb else "none",
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # Training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    if args.main_process:
        logger.info("Saving model...")
        model.save_pretrained(os.path.join(args.output_dir, "final"))
    
    # Generate some samples for evaluation
    logger.info("Generating samples...")
    samples = generate_samples(args, model, tokenizer, test_dataset, task_type)
    
    # Finish wandb
    if args.use_wandb and args.main_process:
        wandb.finish()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()