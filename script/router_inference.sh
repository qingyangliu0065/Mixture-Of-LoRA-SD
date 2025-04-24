TASKS=("math" "coding" "factual_knowledge" "creative_writing")

for TASK in "${TASKS[@]}"; do
    echo "Running router inference for task: $TASK"
    python ./src/router_inference.py \
        --router_dir "./weights/" \
        --data_dir "./data/" \
        --task "$TASK" \
        --temperature 1 \
        --debug
done