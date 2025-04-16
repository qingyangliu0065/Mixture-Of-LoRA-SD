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

model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint).to(device)

outputs = model.generate(**inputs, assistant_model=assistant_model, tokenizer=tokenizer, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))