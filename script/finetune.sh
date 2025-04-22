#!/bin/bash
#SBATCH --job-name=7b_1e-5_2epoch
#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000:8
#SBATCH --partition=general
#SBATCH --output=.slurm_logs/7b_1e-5_2epoch.out
#SBATCH --time=02-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xinyuel4@andrew.cmu.edu

TASKS=("math" "coding" "factual_knowledge" "creative_writing")

# Loop through each task
for TASK in "${TASKS[@]}"; do
  echo "Starting training for task: $TASK"

  accelerate launch ./src/lora_finetune.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --tokenizer "Qwen/Qwen2.5-7B-Instruct" \
    --task "$TASK" \
    --data_dir ./data/ \
    --output_dir ./output/ \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --epochs 2 \
    --lr 1e-5 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --use_wandb \
    --wandb_entity "irisiris" \
    --wandb_project "lora_${TASK}" \
    --run_name "${TASK}_7b_1e-5_2epoch" \
    --debug

    echo "Completed training for task: $TASK"
  echo "========================================"
done