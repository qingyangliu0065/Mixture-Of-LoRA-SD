import time
import torch


class SDDebuggingTracker:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.reset()


    def reset(self):
        self.rounds = []
        self.current_prompt = None
        self.current_output = None
        self.start_time = None
        self.end_time = None


    def record_start(self, prompt_ids):
        self.reset()
        self.current_prompt = prompt_ids
        self.start_time = time.time()


    def record_round(self, speculated_ids, accepted_count, total_count, rejected_id=None, target_id=None):
        round_info = {
            "speculated_ids": speculated_ids.tolist() if isinstance(speculated_ids, torch.Tensor) else speculated_ids,
            "speculated_text": self.tokenizer.decode(speculated_ids),
            "accepted_count": accepted_count,
            "total_count": total_count,
            "acceptance_rate": accepted_count / total_count if total_count > 0 else 0,
        }

        if rejected_id is not None:
            round_info["rejected_id"] = rejected_id
            round_info["rejected_token"] = self.tokenizer.decode([rejected_id])
        
        if target_id is not None:
            round_info["target_id"] = target_id
            round_info["target_token"] = self.tokenizer.decode([target_id])
        
        self.rounds.append(round_info)


    def record_end(self, output_ids):
        self.current_output = output_ids
        self.end_time = time.time()


    def get_statistics(self):
        total_speculated = sum(round_info["total_count"] for round_info in self.rounds)
        total_accepted = sum(round_info["accepted_count"] for round_info in self.rounds)
        
        stats = {
            "total_rounds": len(self.rounds),
            "total_speculated_tokens": total_speculated,
            "total_accepted_tokens": total_accepted,
            "average_accept_length": total_accepted / len(self.rounds) if self.rounds else 0,
            "overall_acceptance_rate": total_accepted / total_speculated if total_speculated > 0 else 0,
            "generation_time": self.end_time - self.start_time if self.end_time and self.start_time else None,
            "tokens_per_second": len(self.current_output) / (self.end_time - self.start_time) if self.end_time and self.start_time else None,
        }

        return stats


    def get_debug_output(self):
        if not self.rounds:
            return "No speculative rounds recorded."

        output = []

        # Print the prompt
        prompt_text = self.tokenizer.decode(self.current_prompt)
        output.append(f"Input: {prompt_text}")

        # Show each round
        for i, round_info in enumerate(self.rounds):
            spec_text = round_info["speculated_text"]
            accepted = round_info["accepted_count"]
            total = round_info["total_count"]
            
            round_header = f"Round {i+1}: {accepted}/{total} tokens accepted ({round_info['acceptance_rate']:.2%})"
            output.append(f"\n{round_header}")
            output.append(f"  Speculated: \"{spec_text}\"")
            
            if accepted < total and "rejected_token" in round_info:
                output.append(f"  Rejection at position {accepted+1}: '{round_info['rejected_token']}' → '{round_info['target_token']}'")
        
        # Show final output
        output_text = self.tokenizer.decode(self.current_output)
        output.append(f"\nFinal output: {output_text}")

        # Show statistics
        stats = self.get_statistics()
        output.append(f"\nStatistics:")
        output.append(f"  Total rounds: {stats['total_rounds']}")
        output.append(f"  Total accepted tokens: {stats['total_accepted_tokens']}")
        output.append(f"  Average accept length: {stats['average_accept_length']:.2f}")
        output.append(f"  Overall acceptance rate: {stats['overall_acceptance_rate']:.2%}")
        
        if stats['generation_time'] is not None:
            output.append(f"  Generation time: {stats['generation_time']:.2f} seconds")
            output.append(f"  Tokens per second: {stats['tokens_per_second']:.2f}")
        
        return "\n".join(output)


def restore_original_methods(target_model, draft_model, orig_target_forward, orig_target_generate, orig_draft_generate):
    target_model.forward = orig_target_forward
    target_model.generate = orig_target_generate
    draft_model.generate = orig_draft_generate


def monkey_patch_for_debugging(target_model, draft_model, tokenizer, spec_length):
    tracker = SDDebuggingTracker(tokenizer)

    # Draft model config
    draft_model.generation_config.num_assistant_tokens = spec_length
    draft_model.generation_config.num_assistant_tokens_schedule = "constant"
    draft_model.generation_config.assistant_confidence_threshold = 0

    # Store original methods
    orig_target_forward = target_model.forward
    orig_target_generate = target_model.generate
    orig_draft_generate = draft_model.generate

    # Counter for target model forward passes
    call_counter = {"n_calls": 0, "current_round_accepted": 0, "current_round_total": 0}
    draft_speculated_tokens = [None]
    
    def target_generate_hook(*args, **kwargs):
        if "input_ids" in kwargs:
            tracker.record_start(kwargs["input_ids"][0])

        call_counter["n_calls"] = 0
        result = orig_target_generate(*args, **kwargs)
        if hasattr(result, "sequences"):
            tracker.record_end(result.sequences[0])
        
        return result
    

    def target_forward_hook(*args, **kwargs):
        call_counter["n_calls"] += 1

        if call_counter["current_round_total"] > 0:
            tracker.record_round(
                speculated_ids=draft_speculated_tokens[0],
                accepted_count=call_counter["current_round_accepted"],
                total_count=call_counter["current_round_total"]
            )

        # Reset for next round
        call_counter["current_round_accepted"] = 0
        call_counter["current_round_total"] = 0

        # Call original forward
        return orig_target_forward(*args, **kwargs)


    def draft_generate_hook(*args, **kwargs):
        result = orig_draft_generate(*args, **kwargs)

        if hasattr(result, "sequences"):
            if "input_ids" in kwargs:
                prompt_length = kwargs["input_ids"].shape[1]
                draft_speculated_tokens[0] = result.sequences[0, prompt_length:]
                call_counter["current_round_total"] = len(draft_speculated_tokens[0])
            
        return result


    # Apply the hooks
    target_model.forward = target_forward_hook
    target_model.generate = target_generate_hook
    draft_model.generate = draft_generate_hook

    return {
        "tracker": tracker,
        "call_counter": call_counter,
        "draft_speculated_tokens": draft_speculated_tokens,  # Pass this back in case it's needed
        "restore": lambda: restore_original_methods(
            target_model, draft_model, 
            orig_target_forward, orig_target_generate, orig_draft_generate
        )
    }