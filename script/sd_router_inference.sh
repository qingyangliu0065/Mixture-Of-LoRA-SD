#!/bin/bash

# Set executable permissions when running this script for the first time:
# chmod +x run_router_sd.sh

# Base directory
CACHE_DIR="./cache"
DATA_DIR="./data"
WEIGHTS_DIR="./weights"
OUTPUT_DIR="./output"

# Default parameters
TASK="math"
ROUTER_STRATEGY="max"  # Only using max strategy for now
TARGET_TEMPERATURE=0.3
TARGET_MODEL="Qwen/Qwen2.5-14B-Instruct"
ASSISTANT_MODEL="Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL="jinaai/jina-embeddings-v2-base-code"

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
    --debug