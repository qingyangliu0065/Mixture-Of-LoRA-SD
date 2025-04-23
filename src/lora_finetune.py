import argparse
import os
import random
import json
import logging
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
import wandb


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_max_chat_length(data, tokenizer):
    system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    max_len = 0
    for item in data:
        input_text = item["input"]
        output_text = item["output"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
        input_ids = tokenizer(chat_text, truncation=False)["input_ids"]
        max_len = max(max_len, len(input_ids))
    
    return max_len, max_len  # Return the same value for both max_len and padded_len


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]
        output_text = item["output"]

        # Build the message format expected by Qwen
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )

        # Prompt-only message (to determine where labels start)
        prompt_only = messages[:-1]
        prompt_text = self.tokenizer.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True)
        prompt_len = len(self.tokenizer(prompt_text, truncation=True, max_length=self.max_length)["input_ids"])

        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        # Print the name of the parameter and whether it is trainable
        # print(name, param.requires_grad)
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def load_jsonl_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_datasets(args, tokenizer):
    train_data = load_jsonl_data(os.path.join(args.data_dir, args.task, "train.jsonl"))
    val_data = load_jsonl_data(os.path.join(args.data_dir, args.task, "validation.jsonl"))

    if args.debug:
        train_data = train_data[:200]
        val_data = val_data[:50]
    
    # Shuffle the data
    random.shuffle(train_data)
    logger.info(f"Task: {args.task}, Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # Compute max token length from training set if not provided
    max_raw_len, _ = get_max_chat_length(train_data, tokenizer)
    max_allowed_len = 6144

    if max_raw_len > max_allowed_len:
        logger.warning(f"Max token length ({max_raw_len}) exceeds limit → truncating to {max_allowed_len}")
        args.max_token_length = max_allowed_len
    else:
        args.max_token_length = max_raw_len
    logger.info(f"Final max token length used: {args.max_token_length}")
    
    # Build datasets
    train_dataset = CustomDataset(train_data, tokenizer, args.max_token_length)
    val_dataset = CustomDataset(val_data, tokenizer, args.max_token_length)
    
    return train_dataset, val_dataset


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
    parser.add_argument("--cache_dir", type=str, default="./.cache/")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_token_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--debug", action="store_true", help="Use a small subset of data for fast testing")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    
    # Other arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="lora-finetune")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16"
    )
    args.device = accelerator.device
    logger.info(f"Using device: {args.device}")
    
    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_name = args.run_name or f"{args.task}_{timestamp}"
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Only log on main process
    args.main_process = accelerator.is_local_main_process
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
    if args.main_process:
        logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, 
        trust_remote_code=True,
        use_fast=False, 
        cache_dir=args.cache_dir
    )
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare datasets
    if args.main_process:
        logger.info("Preparing datasets...")
    train_dataset, val_dataset = prepare_datasets(args, tokenizer)
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )

    # LoRA config
    if args.main_process:
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
    if args.main_process:
        logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map=args.device
    )

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if args.main_process:
            logger.info("Gradient checkpointing enabled")
    
    # Apply LoRA config
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    if args.main_process:
        print_trainable_parameters(model)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-08
    )
    
    # Set up learning rate scheduler
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(0.03 * num_training_steps)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps
    )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # Training loop
    if args.main_process:
        logger.info("Starting training...")
    model.train()
    completed_steps = 0
    best_eval_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Progress bar
        if args.main_process:
            train_progress_bar = tqdm(
                total=len(train_dataloader),
                desc=f"Epoch {epoch+1}/{args.epochs}",
                disable=not args.main_process
            )
        for step, batch in enumerate(train_dataloader):
            # Forward pass
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

            # Update progress bar
            if args.main_process:
                train_progress_bar.update(1)
                train_progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log training loss
            if args.main_process and completed_steps % args.logging_steps == 0:
                accelerator.print(f"Epoch: {epoch}, Step: {completed_steps}, Loss: {loss.item()}")
                if args.use_wandb:
                    wandb.log({"train_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=completed_steps)
            
            # Evaluation
            if completed_steps % args.eval_steps == 0:
                model.eval()
                eval_loss = 0
                eval_steps = 0

                # Create evaluation progress bar
                if args.main_process:
                    eval_progress_bar = tqdm(
                        total=len(eval_dataloader),
                        desc="Evaluation",
                        disable=not args.main_process
                    )
                
                for eval_batch in eval_dataloader:
                    with torch.no_grad():
                        outputs = model(**eval_batch)
                    eval_loss += outputs.loss.item()
                    eval_steps += 1

                    # Update evaluation progress bar
                    if args.main_process:
                        eval_progress_bar.update(1)
                
                eval_loss = eval_loss / eval_steps
                
                if args.main_process:
                    # Close evaluation progress bar
                    eval_progress_bar.close()
                    accelerator.print(f"Evaluation at step {completed_steps}: Loss: {eval_loss}")
                    if args.use_wandb:
                        wandb.log({"eval_loss": eval_loss}, step=completed_steps)
                
                # Save best model
                if args.main_process and eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        os.path.join(args.output_dir, "best_model"),
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
                    accelerator.print(f"Best model saved with loss: {best_eval_loss}")
                
                # Save checkpoint
                if completed_steps % args.save_steps == 0:
                    if args.main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            os.path.join(args.output_dir, f"checkpoint-{completed_steps}"),
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )
                        accelerator.print(f"Checkpoint saved at step {completed_steps}")
                
                model.train()

        # Close training progress bar
        if args.main_process:
            train_progress_bar.close()
    
    # Finish wandb
    if args.use_wandb and args.main_process:
        wandb.finish()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()