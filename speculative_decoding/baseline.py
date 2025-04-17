# https://huggingface.co/blog/dynamic_speculation_lookahead

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompt = "Alice and Bob"
#checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
#assistant_checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
#checkpoint = "Qwen/Qwen2.5-14B-Instruct"
#checkpoint = "Qwen/Qwen2.5-3B-Instruct"
checkpoint = "Qwen/Qwen2.5-1.5B-Instruct"
assistant_checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
prompt_len = inputs.input_ids.shape[-1]  # number of prompt tokens :contentReference[oaicite:2]{index=2}

model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint).to(device)

# # pick a constant draft length (e.g. 20 tokens)
# assistant_model.generation_config.num_assistant_tokens = 20  
# # force it to always generate that many (rather than growing/shrinking based on the last iteration)
# assistant_model.generation_config.num_assistant_tokens_schedule = "constant"
# # if you don’t want early‐stopping on low confidence, set threshold to 0
# assistant_model.generation_config.assistant_confidence_threshold = 0

# By default (i.e. if you don’t touch these), Transformers will run in “dynamic” speculation mode with its built‑in heuristic for lookahead. Adjusting num_assistant_tokens and num_assistant_tokens_schedule lets you explicitly set a fixed draft length per iteration.


# 3. Monkey‐patch to count forward calls
call_counter = {"n_calls": 0}
orig_forward = model.forward

def counting_forward(*args, **kwargs):
    call_counter["n_calls"] += 1
    return orig_forward(*args, **kwargs)

model.forward = counting_forward  # override :contentReference[oaicite:3]{index=3}

# 4. Generate text
outputs = model.generate(
    **inputs,
    assistant_model=assistant_model,
    return_dict_in_generate=True,   # keep rich output :contentReference[oaicite:4]{index=4}
    output_scores=True,             # include scores/stats flags :contentReference[oaicite:5]{index=5}
    max_new_tokens=200,
    pad_token_id=tokenizer.eos_token_id,
)

# 5. Compute average accept length
total_len = outputs.sequences.shape[-1]                # total tokens in output :contentReference[oaicite:6]{index=6}
new_tokens = total_len - prompt_len                    # new tokens generated 
num_calls = call_counter["n_calls"]                    # spec rounds 
avg_accept = new_tokens / num_calls                    # average draft tokens accepted :contentReference[oaicite:7]{index=7}

print(f"Total new tokens: {new_tokens}")
print(f"Number of target-model calls: {num_calls}")
print(f"Average accept length: {avg_accept:.2f} tokens")
texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
print(texts)  # since your batch size is 1