import torch
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class SpeculativeDecoder:
    """
    A speculative decoding implementation with built-in debugging and visualization.
    This implementation shows exactly which tokens are accepted and rejected.
    """
    def __init__(
        self, 
        tokenizer,
        draft_model,
        target_model,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        max_new_tokens: int = 100,
        spec_length: int = 5,
        debug: bool = True
    ):
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.target_model = target_model
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.spec_length = spec_length
        self.debug = debug
        
        # Set models to evaluation mode
        self.draft_model.eval()
        self.target_model.eval()
        
        # Stats tracking
        self.stats = {
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "draft_forwards": 0,
            "target_forwards": 0,
            "total_rounds": 0,
            "generation_time": 0,
        }
        
        # Debug info
        self.rounds = []
        
    def get_next_token(self, model, input_ids, temperature, top_k, top_p):
        """Generate a single token using the model."""
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            logits = logits / max(temperature, 1e-8)
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits[0, :] = torch.full_like(logits[0, :], float('-inf'))
                logits[0, top_k_indices[0]] = top_k_logits[0]
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = 0  # Keep the top token
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            return next_token, probs
    
    def generate_draft_tokens(self, input_ids):
        """Generate tokens using the draft model."""
        draft_input_ids = input_ids.clone()
        draft_tokens = []
        draft_probs = []
        
        for _ in range(self.spec_length):
            self.stats["draft_forwards"] += 1
            next_token, probs = self.get_next_token(
                self.draft_model, 
                draft_input_ids, 
                self.temperature, 
                self.top_k, 
                self.top_p
            )
            draft_input_ids = torch.cat([draft_input_ids, next_token], dim=1)
            draft_tokens.append(next_token[0].item())
            draft_probs.append(probs[0, next_token[0].item()].item())
        
        return draft_tokens, draft_probs
    
    def verify_tokens(self, input_ids, draft_tokens, draft_probs):
        """Verify tokens from the draft model with the target model."""
        current_input_ids = input_ids.clone()
        
        # Run target model on the prefix to get initial probabilities
        self.stats["target_forwards"] += 1
        with torch.no_grad():
            outputs = self.target_model(input_ids=current_input_ids)
            target_probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
        
        accepted_tokens = []
        rejected_idx = None
        target_token = None
        accepted_probs = []
        
        # Verify each token
        for i, (token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
            # Check the probability ratio for this token
            token_tensor = torch.tensor([[token]], device=input_ids.device)
            target_prob = target_probs[0, token].item()  # Fixed indexing
            ratio = min(1.0, target_prob / (draft_prob + 1e-8))
            
            # Accept or reject based on probability ratio
            r = np.random.random()
            if r < ratio:
                # Accept the token
                accepted_tokens.append(token)
                accepted_probs.append(target_prob)
                
                # Update for next iteration
                current_input_ids = torch.cat([current_input_ids, token_tensor], dim=1)
                self.stats["target_forwards"] += 1
                with torch.no_grad():
                    outputs = self.target_model(input_ids=current_input_ids)
                    target_probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
            else:
                # Reject - sample a new token from the target model
                rejected_idx = i
                self.stats["target_forwards"] += 1
                with torch.no_grad():
                    outputs = self.target_model(input_ids=current_input_ids)
                    next_token = torch.multinomial(torch.softmax(outputs.logits[:, -1, :], dim=-1), num_samples=1)
                    target_token = next_token[0, 0].item()
                break
        
        # If all tokens were accepted, generate one more from the target model
        if rejected_idx is None:
            self.stats["target_forwards"] += 1
            with torch.no_grad():
                outputs = self.target_model(input_ids=current_input_ids)
                next_token = torch.multinomial(torch.softmax(outputs.logits[:, -1, :], dim=-1), num_samples=1)
                target_token = next_token[0, 0].item()
        
        # Update stats
        self.stats["accepted_tokens"] += len(accepted_tokens)
        if rejected_idx is not None:
            self.stats["rejected_tokens"] += 1
        
        return accepted_tokens, rejected_idx, target_token
    
    def record_round(self, draft_tokens, accepted_count, rejected_token=None, target_token=None):
        """Record debugging information for a round of speculative decoding."""
        if not self.debug:
            return
        
        # Create token strings for better readability
        draft_text = self.tokenizer.decode(draft_tokens)
        
        round_info = {
            "round": len(self.rounds) + 1,
            "draft_tokens": draft_tokens,
            "draft_text": draft_text,
            "accepted_count": accepted_count,
            "total_count": len(draft_tokens),
            "acceptance_rate": accepted_count / len(draft_tokens) if draft_tokens else 0,
        }
        
        if rejected_token is not None and accepted_count < len(draft_tokens):
            round_info["rejected_token"] = rejected_token
            round_info["rejected_text"] = self.tokenizer.decode([rejected_token])
        
        if target_token is not None:
            round_info["target_token"] = target_token
            round_info["target_text"] = self.tokenizer.decode([target_token])
        
        self.rounds.append(round_info)
    
    def print_round_info(self, round_info):
        """Print debugging information for a round."""
        if not self.debug:
            return
            
        r = round_info
        print(f"\n==== Round {r['round']} ====")
        print(f"Draft: \"{r['draft_text']}\"")
        print(f"Accepted: {r['accepted_count']}/{r['total_count']} tokens ({r['acceptance_rate']:.2%})")
        
        if "rejected_token" in r and r["accepted_count"] < r["total_count"]:
            print(f"Rejected: {r['rejected_text']} -> Target chose: {r['target_text']}")
        elif "target_token" in r:
            print(f"All tokens accepted! Target continuation: {r['target_text']}")
    
    def get_debug_summary(self):
        """Generate a summary of the debugging information."""
        if not self.debug or not self.rounds:
            return "No debugging information available."
        
        output = []
        output.append(f"===== Speculative Decoding Summary =====")
        output.append(f"Total rounds: {self.stats['total_rounds']}")
        output.append(f"Total accepted tokens: {self.stats['accepted_tokens']}")
        output.append(f"Total rejected tokens: {self.stats['rejected_tokens']}")
        
        avg_acceptance = self.stats["accepted_tokens"] / (self.stats["accepted_tokens"] + self.stats["rejected_tokens"]) if (self.stats["accepted_tokens"] + self.stats["rejected_tokens"]) > 0 else 0
        output.append(f"Overall acceptance rate: {avg_acceptance:.2%}")
        
        avg_tokens_per_round = self.stats["accepted_tokens"] / self.stats["total_rounds"] if self.stats["total_rounds"] > 0 else 0
        output.append(f"Average accepted tokens per round: {avg_tokens_per_round:.2f}")
        
        speedup = (self.stats["accepted_tokens"] + self.stats["rejected_tokens"]) / self.stats["target_forwards"] if self.stats["target_forwards"] > 0 else 0
        output.append(f"Effective speedup factor: {speedup:.2f}x")
        
        if self.stats["generation_time"] > 0:
            tokens_per_sec = (self.stats["accepted_tokens"] + self.stats["rejected_tokens"]) / self.stats["generation_time"]
            output.append(f"Tokens per second: {tokens_per_sec:.2f}")
        
        output.append("\n===== Round Details =====")
        for r in self.rounds:
            output.append(f"\nRound {r['round']}: {r['accepted_count']}/{r['total_count']} tokens accepted ({r['acceptance_rate']:.2%})")
            output.append(f"  Draft: \"{r['draft_text']}\"")
            
            if "rejected_token" in r and r["accepted_count"] < r["total_count"]:
                output.append(f"  Rejection at position {r['accepted_count'] + 1}: '{r['rejected_text']}' -> '{r['target_text']}'")
            elif "target_token" in r:
                output.append(f"  All tokens accepted! Target continuation: {r['target_text']}")
        
        return "\n".join(output)
    
    def generate(self, prompt):
        """Generate text using speculative decoding."""
        start_time = time.time()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.draft_model.device)
        input_prompt_len = input_ids.shape[1]
        
        # Generate tokens up to max_new_tokens
        generated_tokens = input_ids.clone()
        tokens_generated = 0
        
        with tqdm(total=self.max_new_tokens, desc="Generating") as pbar:
            while tokens_generated < self.max_new_tokens:
                # Generate draft tokens
                draft_tokens, draft_probs = self.generate_draft_tokens(generated_tokens)
                
                # Verify tokens with target model
                accepted_tokens, rejected_idx, target_token = self.verify_tokens(
                    generated_tokens, draft_tokens, draft_probs
                )
                
                # Record debugging information
                self.stats["total_rounds"] += 1
                if rejected_idx is not None:
                    rejected_token = draft_tokens[rejected_idx]
                    self.record_round(draft_tokens, len(accepted_tokens), rejected_token, target_token)
                else:
                    self.record_round(draft_tokens, len(accepted_tokens), None, target_token)
                
                # Print round info if debugging
                if self.debug and len(self.rounds) > 0:
                    self.print_round_info(self.rounds[-1])
                
                # Add tokens to the generated sequence
                for token in accepted_tokens:
                    generated_tokens = torch.cat([
                        generated_tokens, 
                        torch.tensor([[token]], device=generated_tokens.device)
                    ], dim=1)
                
                # Add the target token
                generated_tokens = torch.cat([
                    generated_tokens, 
                    torch.tensor([[target_token]], device=generated_tokens.device)
                ], dim=1)
                
                # Update progress
                new_tokens = len(accepted_tokens) + 1
                tokens_generated += new_tokens
                pbar.update(new_tokens)
        
        # Record generation time
        self.stats["generation_time"] = time.time() - start_time
        
        # Convert tokens to text
        generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        new_text = generated_text[len(prompt_text):]
        
        return generated_text, new_text


if __name__ == "__main__":
    # Initialize models and tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
        use_fast=False, 
        cache_dir="../.cache"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Initializing draft model...")
    draft_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True,
        cache_dir="../.cache",
        torch_dtype=torch.float16,
        device_map="auto"
    ).to("cuda")
    draft_model.eval()

    print("Initializing target model...")
    target_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
        cache_dir="../.cache",
        torch_dtype=torch.float16,
        device_map="auto"
    ).to("cuda")
    target_model.eval()

    # Create the speculative decoder
    decoder = SpeculativeDecoder(
        tokenizer=tokenizer,
        draft_model=draft_model,
        target_model=target_model,
        spec_length=5,  # Number of tokens to speculate
        max_new_tokens=100,  # Maximum new tokens to generate
        debug=True  # Enable detailed debugging
    )

    # Generate text
    prompt = "When you die, you appear in a room with two buttons: Heaven and Hell. You do n't know which is which. So you press both at the same time."
    full_text, new_text = decoder.generate(prompt)

    # Print debugging summary
    print(decoder.get_debug_summary())
    
    # Print the final generation
    print("\n===== Final Generation =====")
    print(f"Prompt: {prompt}")
    print(f"Generated: {new_text}")
    print(f"Full text: {full_text}")