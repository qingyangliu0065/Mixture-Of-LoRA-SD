"""
Create a router using the training data.
"""

import argparse
import os
import json
import sys
from transformers import AutoModel
import torch

# Switch to parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.router import EmbeddingSimilarityRouter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model", type=str, default="jinaai/jina-embeddings-v2-base-code")
    parser.add_argument("--cache_dir", type=str, default="./.cache/")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--output_dir", type=str, default="./weights/")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--similarity_metric", type=str, default="cosine")
    args = parser.parse_args()

    # Load embedding model
    embedding_model = AutoModel.from_pretrained(
        args.embedding_model, 
        trust_remote_code=True, 
        cache_dir=args.cache_dir,
    )

    # Define tasks
    tasks = ["math", "coding", "factual_knowledge", "creative_writing"]

    # Create router
    router = EmbeddingSimilarityRouter(
        embedding_model,
        tasks,
        temperature=args.temperature,
        top_k=args.top_k,
        similarity_metric=args.similarity_metric,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load task examples from training data
    task_examples = {}
    for task in tasks:
        task_examples[task] = []
        with open(os.path.join(args.data_dir, task, "train.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                task_examples[task].append(example["input"])
    print(task_examples.keys())
    
    # Compute task centroids
    router.compute_task_centroids(task_examples)

    # Save router
    router.save(args.output_dir)


if __name__ == "__main__":
    main()