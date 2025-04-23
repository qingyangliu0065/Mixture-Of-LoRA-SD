python ./src/lora_inference.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --tokenizer "Qwen/Qwen2.5-7B-Instruct" \
    --lora_path "./output/math_7b_1e-5_2epoch/best_model" \
    --task "math" \
    --debug