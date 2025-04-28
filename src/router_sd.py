"""
Load the router and use it to route the input to the correct task.
Can run with validation or test set.
This is a simple script to test the performance of the router on the validation set.
"""

import argparse
import os
import sys
import json
from tqdm import tqdm
import time
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
# Switch to parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.router import EmbeddingSimilarityRouter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model", type=str, default="jinaai/jina-embeddings-v2-base-code")
    parser.add_argument("--cache_dir", type=str, default="./cache/")
    parser.add_argument("--router_dir", type=str, default="./weights/")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--lora_dir", type=str, default="./weights/")
    parser.add_argument("--output_dir", type=str, default="./output/")
    parser.add_argument("--assistant_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--task", type=str, default="math")
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument("--target_temperature", type=float, default=1.0)
    parser.add_argument("--router_strategy", type=str, default="max", choices=["max", "merge"])
    parser.add_argument("--debug", action="store_true", help="Use a small subset of data for fast testing")
    args = parser.parse_args()
    
    NUM_ASSISTANT_TOKENS = [15, 15, 15, 15]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Laod embedding model
    embedding_model = AutoModel.from_pretrained(
        args.embedding_model, 
        trust_remote_code=True
    )
    print(f"Embedding model loaded")
    # Load router
    if (not os.path.exists(os.path.join(args.router_dir, "router_config.json"))) or (not os.path.exists(os.path.join(args.router_dir, "router_task_centroids.pt"))):
        raise ValueError("Router not found. Please run router_create.py first.")
    
    router = EmbeddingSimilarityRouter.load(args.router_dir, embedding_model)
    router.temperature = args.router_temperature
    

    # Load data
    test_data = []
    with open(os.path.join(args.data_dir, args.task, "test.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            test_data.append(example['input'])

    if args.debug:
        test_data = test_data[:5]

    # Route data
    task_weights, similarity = router.compute_routing_weights(test_data)
    print(task_weights)
    # Find the task with the highest weight; task_weights is a tensor of shape (num_data, num_tasks)
    # Loop through each row of task_weights
    # During inference, if we consider merging multiple adapters, we can use this method to find the second highest weight
    
    pred_max_categories = torch.argmax(task_weights, dim=1)
    num_per_category = {}
    for i, task in enumerate(router.tasks):
        num_per_category[task] = pred_max_categories.count(i)
    
    print(num_per_category)
    
    if args.router_strategy == "merge":
        pred_merge_categories = []
        for i in range(task_weights.shape[0]): 
            max_weight_idx = pred_max_categories[i]
            threshold = 0.7 * task_weights[i, max_weight_idx]
            candidate_tasks = [max_weight_idx, -1]
            second_max_weight = 0
            for j in range(task_weights.shape[1]):
                if task_weights[i, j] > threshold and task_weights[i, j] > second_max_weight:
                    second_max_weight = task_weights[i, j]
                    candidate_tasks[1] = j
            pred_merge_categories.append(candidate_tasks)
    
    
    # load the target model
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"Target model loaded")
    # load the base assistant model
    assistant_model = AutoModelForCausalLM.from_pretrained(
        args.assistant_model, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"Assistant model loaded")
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model, 
        trust_remote_code=True
    )
    print(f"Tokenizer loaded")
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # speculative inference loop
    target_model.eval()
    
    # Count forward calls
    call_counter = {"n_calls": 0}
    orig_forward = target_model.forward

    def counting_forward(*args, **kwargs):
        call_counter["n_calls"] += 1
        return orig_forward(*args, **kwargs)

    target_model.forward = counting_forward
    
    results = []
    total_accepts = 0
    total_examples = 0
    total_wall_time = 0.0
    
    if args.router_strategy == "max":
        for i in tqdm(range(task_weights.shape[0]), desc="Speculative inference loop"):
            # load the lora model
            pred_task_idx = pred_max_categories[i]
            pred_task = router.tasks[pred_task_idx]
            lora_path = os.path.join(args.lora_dir, f"{pred_task}_7b_1e-5_2epoch", "best_model")
            lora_model = PeftModel.from_pretrained(assistant_model, lora_path)
            lora_model.eval()
            lora_model.to("cuda")
            
            lora_model.generation_config.num_assistant_tokens = NUM_ASSISTANT_TOKENS[pred_task_idx]
            lora_model.generation_config.num_assistant_tokens_schedule = "constant"
            lora_model.generation_config.assistant_confidence_threshold = 0
            lora_model.generation_config.temperature = 0
            
            # generate the response
            prompt = test_data[i]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.shape[-1]

            call_counter["n_calls"] = 0  # reset for each example

            start_time = time.time()
            outputs = target_model.generate(
                **inputs,
                assistant_model=lora_model,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=200,
                temperature=args.target_temperature,
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

            results.append({
                "category": args.task,
                "pred_category": pred_task,
                "prompt": prompt,
                "output": decoded_text,
                "new_tokens": new_tokens,
                "num_calls": num_calls,
                "avg_accept": round(avg_accept, 2),
                "wall_time_sec": round(wall_time, 4),
            })
            
            # delete the lora model
            del lora_model
            
    # Create summary for this category
    category_summary = {
        "num_examples": total_examples,
        "num_examples_per_task": num_per_category,
        "avg_accept_length": round(total_accepts / total_examples, 2) if total_examples > 0 else 0.0,
        "avg_wall_time_sec": round(total_wall_time / total_examples, 4) if total_examples > 0 else 0.0
    }
    
    output_data = {
        "results": results,
        "summary": {args.task: category_summary},
        "config": {
            "target_model": args.target_model,
            "assistant_model": args.assistant_model,
            "num_assistant_tokens": NUM_ASSISTANT_TOKENS,
            "target_temperature": args.target_temperature,
            "router_strategy": args.router_strategy,
            "task": args.task
        }
    }
    
    # Name the output file as 'router_<strategy>_<category>_<num_assistant_tokens>.json'
    output_path = os.path.join(args.output_dir, f"router_{args.router_strategy}_{args.task}_{NUM_ASSISTANT_TOKENS[0]}.json")
    with open(output_path, "w") as out_f:
        json.dump(output_data, out_f, indent=2)
        
    print(f"\nResults for category '{args.task}' saved to: {output_path}")

if __name__ == "__main__":
    main()