"""
A very simple script to check the inference output of the finetuned model.
Checked the model is correctly working. The inference can be done in a single GPU.
"""

import argparse
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path", type=str, default="./weights/")
    parser.add_argument("--task", type=str, default="math")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--cache_dir", type=str, default="./.cache/")
    parser.add_argument("--debug", action="store_true", help="Use a small subset of data for fast testing")
    args = parser.parse_args()

    # Load tokenizer
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

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load adapter
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model = model.to("cuda")
    model.eval()

    # Load data
    test_data = []
    with open(os.path.join(args.data_dir, args.task, "test.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line))
    if args.debug:
        test_data = test_data[:10]

    system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    # Inference
    for item in test_data:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["input"]}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
            )
            response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            print("Input:", item["input"])
            print("Output:", response)
            print("-" * 50)


if __name__ == "__main__":
    main()