#!/bin/bash

# Set executable permissions when running this script for the first time:
# chmod +x sd_baseline_inference.sh

# Base directory
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CACHE_DIR="$BASE_DIR/cache"
DATA_DIR="$BASE_DIR/data"
RESULT_DIR="$BASE_DIR/output"

# Create output directory if it doesn't exist
mkdir -p $RESULT_DIR

# Default parameters
NUM_ASSISTANT_TOKENS=20
TEMPERATURE=0.7
TARGET_MODEL="Qwen/Qwen2.5-14B-Instruct"
ASSISTANT_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Tasks to run
TASKS=("math" "coding" "creative_writing" "factual_knowledge")

# Uncomment to run all tasks at once
# TASKS=("all")

# Loop through each task
for task in "${TASKS[@]}"; do
    echo "Running baseline for task: $task"
    
    # Run the command
    python $BASE_DIR/src/baseline.py \
        --cache_dir $CACHE_DIR \
        --data_dir $DATA_DIR \
        --result_dir $RESULT_DIR \
        --task $task \
        --assistant_model "$ASSISTANT_MODEL" \
        --target_model "$TARGET_MODEL" \
        --num_assistant_tokens $NUM_ASSISTANT_TOKENS \
        --temperature $TEMPERATURE \
        --debug
    
    echo "Completed task: $task"
    echo "------------------------------------"
done

echo "All tasks completed. Output files are named: baseline_<task>_<num_assistant_tokens>.json" 