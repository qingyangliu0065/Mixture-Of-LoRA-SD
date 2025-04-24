"""
Load the router and use it to route the input to the correct task.
Can run with validation or test set.
This is a simple script to test the performance of the router on the validation set.
"""

import argparse
import os
import sys
import json
import torch
from transformers import AutoModel

# Switch to parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.router import EmbeddingSimilarityRouter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model", type=str, default="jinaai/jina-embeddings-v2-base-code")
    parser.add_argument("--cache_dir", type=str, default="./.cache/")
    parser.add_argument("--router_dir", type=str, default="./weights/")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--task", type=str, default="math")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true", help="Use a small subset of data for fast testing")
    args = parser.parse_args()

    # Laod embedding model
    embedding_model = AutoModel.from_pretrained(
        args.embedding_model, 
        trust_remote_code=True, 
        cache_dir=args.cache_dir,
    )

    # Load router
    if (not os.path.exists(os.path.join(args.router_dir, "router_config.json"))) or (not os.path.exists(os.path.join(args.router_dir, "router_task_centroids.pt"))):
        raise ValueError("Router not found. Please run router_create.py first.")
    
    router = EmbeddingSimilarityRouter.load(args.router_dir, embedding_model)
    router.temperature = args.temperature
    print(router.tasks)

    # Load data
    val_data = []
    with open(os.path.join(args.data_dir, args.task, "validation.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            val_data.append(example['input'])

    if args.debug:
        val_data = val_data[:20]

    # Route data
    task_weights, similarity = router.compute_routing_weights(val_data)
    print(task_weights)
    # Find the task with the highest weight; task_weights is a tensor of shape (num_data, num_tasks)
    # Loop through each row of task_weights
    # During inference, if we consider merging multiple adapters, we can use this method to find the second highest weight
    for i in range(task_weights.shape[0]):
        max_weight = torch.max(task_weights[i])
        threshold = 0.7 * max_weight
        for j in range(task_weights.shape[1]):
            if task_weights[i, j] > threshold:
                print(f"Also consider {j}")
    

if __name__ == "__main__":
    main()