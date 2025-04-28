#!/bin/bash
#SBATCH --job-name=sd-baseline-factual-knowledge
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=80G
#SBATCH --partition=general
#SBATCH --output=.slurm_logs/sd-baseline-factual-knowledge.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qliu3@andrew.cmu.edu

# Set executable permissions when running this script for the first time:
# chmod +x sd_baseline_inference.sh

export NCCL_P2P_DISABLE=1

CACHE_DIR="./cache"
DATA_DIR="./data"
RESULT_DIR="./output"

# Default parameters
NUM_ASSISTANT_TOKENS=15
TEMPERATURE=0.3
TARGET_MODEL="Qwen/Qwen2.5-14B-Instruct"
ASSISTANT_MODEL="Qwen/Qwen2.5-7B-Instruct"
SPECIFIC_TASK="factual_knowledge"

echo "Running evaluation for specific task: $SPECIFIC_TASK"

python ./src/baseline.py \
    --cache_dir $CACHE_DIR \
    --data_dir $DATA_DIR \
    --result_dir $RESULT_DIR \
    --task $SPECIFIC_TASK \
    --assistant_model "$ASSISTANT_MODEL" \
    --target_model "$TARGET_MODEL" \
    --num_assistant_tokens $NUM_ASSISTANT_TOKENS \
    --temperature $TEMPERATURE \

echo "Completed task: $SPECIFIC_TASK"
echo "Output saved to: baseline_${SPECIFIC_TASK}_${NUM_ASSISTANT_TOKENS}.json"


# # Run all tasks with a single model load (more efficient)
# echo "Running evaluation for all tasks with a single model load"

# python $BASE_DIR/src/baseline.py \
#     --cache_dir $CACHE_DIR \
#     --data_dir $DATA_DIR \
#     --result_dir $RESULT_DIR \
#     --run_all_tasks \
#     --assistant_model "$ASSISTANT_MODEL" \
#     --target_model "$TARGET_MODEL" \
#     --num_assistant_tokens $NUM_ASSISTANT_TOKENS \
#     --temperature $TEMPERATURE \
    
# echo "Completed all tasks"
# echo "Output files are named: baseline_<task>_${NUM_ASSISTANT_TOKENS}.json"