# https://huggingface.co/blog/dynamic_speculation_lookahead
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import json
from tqdm import tqdm
import time

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run baseline evaluation with dynamic speculation")
parser.add_argument("--cache_dir", type=str, default="/ocean/projects/cis240042p/sliang6/hf_cache", help="Hugging Face cache directory")
parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the evaluation data")
parser.add_argument("--result_dir", type=str, default="results", help="Directory to store results")
parser.add_argument("--task", type=str, choices=["coding", "creative_writing", "factual_knowledge", "math", "all"], 
                    default="all", help="Category of data to evaluate")
parser.add_argument("--assistant_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Assistant model checkpoint")
parser.add_argument("--target_model", type=str, default="Qwen/Qwen2.5-14B-Instruct", help="Target model checkpoint")
parser.add_argument("--num_assistant_tokens", type=int, default=20, help="Number of tokens assistant model generates each time")
parser.add_argument("--assistant_token_schedule", type=str, default="constant", choices=["constant", "heuristic", "heuristic_transient"], help="Assistant token schedule")
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
parser.add_argument("--assistant_temperature", type=float, default=0, help="Sampling temperature for assistant model")
parser.add_argument("--debug", action="store_true", help="Debug mode")
parser.add_argument("--run_all_tasks", action="store_true", help="Run all tasks with a single model load")

args = parser.parse_args()

# Set environment variables
# os.environ['HF_HOME'] = args.cache_dir

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

checkpoint = args.target_model
assistant_checkpoint = args.assistant_model

print(f"Loading tokenizer from: {checkpoint}")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
print(f"Tokenizer was loaded.")
print(f"Loading model from: {checkpoint}")
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, 
    torch_dtype=torch.float16,
    device_map="auto")
print(f"Model was loaded.")
print(f"Loading assistant model from: {assistant_checkpoint}")
assistant_model = AutoModelForCausalLM.from_pretrained(
    assistant_checkpoint, 
    torch_dtype=torch.float16,
    device_map="auto")
print(f"Assistant model was loaded.")
# Set models to evaluation mode
model.eval()
assistant_model.eval()

assistant_model.generation_config.num_assistant_tokens = args.num_assistant_tokens
assistant_model.generation_config.num_assistant_tokens_schedule = args.assistant_token_schedule
assistant_model.generation_config.assistant_confidence_threshold = 0
assistant_model.generation_config.temperature = args.assistant_temperature

# Count forward calls
call_counter = {"n_calls": 0}
orig_forward = model.forward

def counting_forward(*args, **kwargs):
    call_counter["n_calls"] += 1
    return orig_forward(*args, **kwargs)

model.forward = counting_forward

# Evaluation loop
base_path = Path(args.data_dir)
output_dir = Path(args.result_dir)
output_dir.mkdir(exist_ok=True)

# Determine which categories to evaluate
if args.run_all_tasks or args.task == "all":
    categories = ["coding", "creative_writing", "factual_knowledge", "math"]
else:
    categories = [args.task]

# Process each category
for category in categories:
    print(f"\n{'='*50}")
    print(f"Evaluating category: {category}")
    print(f"{'='*50}")
    
    test_file = base_path / category / "test.jsonl"
    if not test_file.exists():
        print(f"Test file not found for {category}, skipping.")
        continue

    category_results = []
    total_accepts = 0
    total_examples = 0
    total_wall_time = 0.0

    with open(test_file, "r") as f:
        lines = f.readlines()
    
    if args.debug:
        lines = lines[:5]

    for line in tqdm(lines, desc=f"Evaluating category: {category}"):
        entry = json.loads(line)
        prompt = entry.get("prompt") or entry.get("input") or entry.get("text")
        if not prompt:
            continue

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[-1]

        call_counter["n_calls"] = 0  # reset for each example

        start_time = time.time()
        outputs = model.generate(
            **inputs,
            assistant_model=assistant_model,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=200,
            temperature=args.temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
        end_time = time.time()

        wall_time = end_time - start_time
        total_wall_time += wall_time

        total_len = outputs.sequences.shape[-1]
        new_tokens = total_len - prompt_len
        num_calls = call_counter["n_calls"]
        avg_accept = new_tokens / num_calls if num_calls > 0 else 0
        decoded_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        total_accepts += avg_accept
        total_examples += 1

        category_results.append({
            "category": category,
            "prompt": prompt,
            "output": decoded_text,
            "new_tokens": new_tokens,
            "num_calls": num_calls,
            "avg_accept": round(avg_accept, 2),
            "wall_time_sec": round(wall_time, 4),
        })

    # Create summary for this category
    category_summary = {
        "num_examples": total_examples,
        "avg_accept_length": round(total_accepts / total_examples, 2) if total_examples > 0 else 0.0,
        "avg_wall_time_sec": round(total_wall_time / total_examples, 4) if total_examples > 0 else 0.0
    }

    # Save results for this category to a separate file
    output_data = {
        "results": category_results,
        "summary": {category: category_summary},
        "config": {
            "target_model": args.target_model,
            "assistant_model": args.assistant_model,
            "num_assistant_tokens": args.num_assistant_tokens,
            "temperature": args.temperature
        }
    }
    
    # Name the output file as 'baseline_<category>_<assistant_token_schedule>_<debug>.json'
    output_path = output_dir / f"baseline_{category}_{args.assistant_token_schedule}_{'debug' if args.debug else ''}.json"
    with open(output_path, "w") as out_f:
        json.dump(output_data, out_f, indent=2)

    print(f"\nResults for category '{category}' saved to: {output_path}")

print(f"\nAll evaluations completed. Output files are named: baseline_<category>_<assistant_token_schedule>_<debug>.json")