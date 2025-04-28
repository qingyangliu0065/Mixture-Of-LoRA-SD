#!/bin/bash
#SBATCH --job-name=sd-router-coding-heuristic-transient
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=80G
#SBATCH --partition=general
#SBATCH --output=.slurm_logs/sd-router-coding-heuristic-transient.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qliu3@andrew.cmu.edu

# Set executable permissions when running this script for the first time:
# chmod +x run_router_sd.sh

export NCCL_P2P_DISABLE=1

# Base directory
CACHE_DIR="./cache"
DATA_DIR="./data"
WEIGHTS_DIR="./weights"
OUTPUT_DIR="./output"

# Default parameters
TASK="coding"
ROUTER_STRATEGY="max_base"
TARGET_TEMPERATURE=0.3
TARGET_MODEL="Qwen/Qwen2.5-14B-Instruct"
ASSISTANT_MODEL="Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL="jinaai/jina-embeddings-v2-base-code"
ASSISTANT_TOKEN_SCHEDULE="heuristic_transient"

# Run the command
python ./src/router_sd.py \
    --task $TASK \
    --router_strategy $ROUTER_STRATEGY \
    --target_temperature $TARGET_TEMPERATURE \
    --embedding_model $EMBEDDING_MODEL \
    --cache_dir $CACHE_DIR \
    --router_dir $WEIGHTS_DIR \
    --data_dir $DATA_DIR \
    --lora_dir $WEIGHTS_DIR \
    --output_dir $OUTPUT_DIR \
    --assistant_model $ASSISTANT_MODEL \
    --target_model $TARGET_MODEL \
    --assistant_token_schedule $ASSISTANT_TOKEN_SCHEDULE \
