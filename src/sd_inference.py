import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
import os
from tqdm import tqdm
import sys

# Switch to parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.speculative_decoding import monkey_patch_for_debugging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--task", type=str, required=True, help="Task name (math, coding, factual_knowledge, creative_writing)")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--cache_dir", type=str, default="./.cache/")
    parser.add_argument("--lora_path", type=str, default="./weights/")
    parser.add_argument("--type", type=str, default="baseline")
    parser.add_argument("--results_dir", type=str, default="./results/")
    parser.add_argument("--debug", action="store_true", help="Sample a small subset of data for fast testing")
    parser.add_argument("--num_assistant_tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    results_file = os.path.join(args.results_dir, f"{args.task}_results.jsonl")
    debug_log_file = os.path.join(args.results_dir, f"{args.task}_debug_log.txt")

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
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
    print(f"Loading base model: {args.assistant_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.assistant_model,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load the larger target model
    print(f"Loading target model: {args.target_model}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    target_model.eval()

    # Load adapter
    if args.type == "validation":
        # Directly use the adapter corresponding to the task
        draft_model = PeftModel.from_pretrained(base_model, args.lora_path)
    elif args.type == "baseline":
        # Use the base model as the draft model
        draft_model = base_model
    elif args.type == "routing":
        # Use the router to select the optimal adapters
        # TODO: Implement this; Consider using the top-1 adapter, or the top-2 merged adapters
        pass
    else:
        raise ValueError(f"Invalid type: {args.type}")

    draft_model = draft_model.to("cuda")
    draft_model.eval()

    # Load validation set data
    val_data = []
    with open(os.path.join(args.data_dir, args.task, "validation.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            val_data.append(json.loads(line))

    if args.debug:
        val_data = val_data[:3]

    system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    # Init results
    results = []
    metrics = {
        "total_tokens": 0, 
        "total_prompt_tokens": 0,
        "total_accepted_tokens": 0,
        "total_speculated_tokens": 0,
        "total_time": 0,
        "total_rounds": 0
    }
    with open(debug_log_file, "w") as f:
        f.write(f"=== Speculative Decoding Debug Log for Task: {args.task} ===\n")

    # Inference
    for idx, item in enumerate(tqdm(val_data, desc=f"Inference on {args.task}")):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["input"]}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.cuda()
        attention_mask = inputs.attention_mask.cuda()
        prompt_len = input_ids.shape[-1]


        # setup debugging hooks
        debug_setup = monkey_patch_for_debugging(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            spec_length=args.num_assistant_tokens
        )
        tracker = debug_setup["tracker"]

        try:
            outputs = target_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                assistant_model=draft_model,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=512,
                do_sample=True,
                temperature=args.temperature,
            )

            output_ids = outputs.sequences[0]
            generated_text = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
            
            debug_output = tracker.get_debug_output()
            stats = tracker.get_statistics()

            # Print debug info for the current example
            print(f"\n\n=== Example {idx+1}/{len(val_data)} ===")
            print(debug_output)
            
            # Write debug info to log file
            with open(debug_log_file, "a") as f:
                f.write(f"\n\n=== Example {idx+1}/{len(val_data)} ===\n")
                f.write(debug_output)
                f.write("\n" + "-"*80 + "\n")
            
            # Store the result
            result = {
                "input": item["input"],
                "output": generated_text,
                "prompt_tokens": prompt_len,
                "new_tokens": len(output_ids) - prompt_len,
                "target_model_calls": stats["total_rounds"],
                "avg_accept_length": stats["average_accept_length"],
                "acceptance_rate": stats["overall_acceptance_rate"],
                "spec_length": args.num_assistant_tokens,
                "generation_time": stats["generation_time"],
                "tokens_per_second": stats["tokens_per_second"]
            }

            # Update metrics
            metrics["total_tokens"] += len(output_ids)
            metrics["total_prompt_tokens"] += prompt_len
            metrics["total_accepted_tokens"] += stats["total_accepted_tokens"]
            metrics["total_speculated_tokens"] += stats["total_speculated_tokens"]
            metrics["total_time"] += stats["generation_time"] if stats["generation_time"] else 0
            metrics["total_rounds"] += stats["total_rounds"]
            
            # Add to results list
            results.append(result)
            
            # Write result to output file
            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")

        except Exception as e:
            print(f"Error generating for example {idx}: {e}")

        finally:
            # Restore original methods
            debug_setup["restore"]()

    # Print final metrics
    num_examples = len(results)
    if num_examples > 0:
        overall_metrics = {
            "task": args.task,
            "num_examples": num_examples,
            "avg_prompt_tokens": metrics["total_prompt_tokens"] / num_examples,
            "avg_new_tokens": (metrics["total_tokens"] - metrics["total_prompt_tokens"]) / num_examples,
            "avg_target_model_calls": metrics["total_rounds"] / num_examples,
            "avg_accept_length": metrics["total_accepted_tokens"] / metrics["total_rounds"] if metrics["total_rounds"] > 0 else 0,
            "overall_acceptance_rate": metrics["total_accepted_tokens"] / metrics["total_speculated_tokens"] if metrics["total_speculated_tokens"] > 0 else 0,
            "avg_tokens_per_second": (metrics["total_tokens"] - metrics["total_prompt_tokens"]) / metrics["total_time"] if metrics["total_time"] > 0 else 0,
            "speculation_length": args.num_assistant_tokens,
            "temperature": args.temperature
        }
        
        # Save overall metrics
        metrics_file = os.path.join(args.results_dir, f"{args.task}_overall_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(overall_metrics, f, indent=2)
        
        # Print overall metrics
        print("\n" + "="*50)
        print(f"Overall Metrics for {args.task} ({num_examples} examples)")
        print(f"Average prompt tokens: {overall_metrics['avg_prompt_tokens']:.2f}")
        print(f"Average new tokens: {overall_metrics['avg_new_tokens']:.2f}")
        print(f"Average target model calls: {overall_metrics['avg_target_model_calls']:.2f}")
        print(f"Average accept length: {overall_metrics['avg_accept_length']:.2f}")
        print(f"Overall acceptance rate: {overall_metrics['overall_acceptance_rate']:.2%}")
        print(f"Average tokens per second: {overall_metrics['avg_tokens_per_second']:.2f}")
        print("="*50)
    else:
        print("No successful generations to compute metrics.")



if __name__ == "__main__":
    main()