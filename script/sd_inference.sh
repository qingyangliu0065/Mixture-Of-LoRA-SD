python ./src/sd_inference.py \
    --assistant_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --target_model "Qwen/Qwen2.5-7B-Instruct" \
    --tokenizer "Qwen/Qwen2.5-7B-Instruct" \
    --task "creative_writing" \
    --type "baseline" \
    --debug \
    --num_assistant_tokens 5 \
    --temperature 0.7