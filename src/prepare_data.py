import datasets
import random
import os
import json
import argparse
import ftfy
from typing import Dict, List
from transformers import AutoTokenizer

class DatasetPreparator:
    def __init__(self, output_dir: str = 'data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

        self.dataset_configs = {
            'math': {
                'name': 'TIGER-Lab/MathInstruct',
                'subset': None,
                'train_split': 'train[:90%]',
                'test_split': 'train[90%:]',
                'task_type': 'math'
            },
            'coding': {
                'name': 'sahil2801/CodeAlpaca-20k',
                'subset': None,
                'train_split': 'train[:90%]',
                'test_split': 'train[90%:]',
                'task_type': 'coding'
            },
            'factual_knowledge': {
                'name': 'akoksal/LongForm',
                'subset': None,
                'train_split': 'train',
                'test_split': 'validation',
                'task_type': 'factual_knowledge'
            },
            'creative_writing': {
                'name': 'euclaise/writingprompts',
                'subset': None,
                'train_split': 'train',
                'test_split': 'validation',
                'task_type': 'creative_writing'
            }
        }

        self.split_counts = {
            'train': 5000,
            'validation': 500,
            'test': 500
        }

    def load_dataset(self, config: Dict, split: str) -> datasets.Dataset:
        try:
            print(f"Loading {config['name']} - {split} split...")
            dataset_split = config.get(f'{split}_split', 'train')

            if config['subset']:
                dataset = datasets.load_dataset(config['name'], config['subset'], split=dataset_split)
            else:
                dataset = datasets.load_dataset(config['name'], split=dataset_split)

            print(f"Loaded {len(dataset)} examples")
            return dataset
        except Exception as e:
            print(f"Error loading {config['name']}: {e}")
            return None

    def clean_text(self, text: str) -> str:
        # Use ftfy to fix text encoding issues and normalize Unicode characters
        text = ftfy.fix_text(text)
        
        # Split by space and reconnect to normalize whitespace
        text = ' '.join(text.split())
        
        # Handle specific quote cases
        text = text.replace('``', '"').replace("''", '"')  # LaTeX-style quotes
        text = text.replace('"', '"').replace('"', '"')  # Smart quotes
        text = text.replace('"', '"').replace('"', '"')  # More smart quotes
        text = text.replace('"', '"').replace('"', '"')  # Even more smart quotes
        text = text.replace('"', '"').replace('"', '"')  # Additional quote variants
        text = text.replace("'", "'")  # Single quotes
        
        # Handle contractions with spaces
        text = text.replace(" ' ", "'")  # Fix spaces around apostrophes
        text = text.replace(" '", "'")   # Fix leading space before apostrophe
        text = text.replace("' ", "'")   # Fix trailing space after apostrophe
        
        # Handle specific contractions
        text = text.replace("'s", "'s")  # Fix possessive apostrophes
        text = text.replace("'m", "'m")  # Fix I'm
        text = text.replace("'ve", "'ve")  # Fix we've, they've, etc.
        text = text.replace("'ll", "'ll")  # Fix will contractions
        text = text.replace("'d", "'d")  # Fix would/had contractions
        text = text.replace("'re", "'re")  # Fix are contractions
        text = text.replace("'t", "'t")  # Fix is not contractions
        
        # Fix spacing around punctuation
        text = text.replace(" .", ".")
        text = text.replace(" ,", ",")
        text = text.replace(" ?", "?")
        text = text.replace(" !", "!")
        text = text.replace(" ;", ";")
        text = text.replace(" :", ":")
        
        return text

    def clean_writing_prompt(self, prompt: str) -> str:
        # Remove [XX] pattern from the beginning
        prompt = prompt.strip()
        if prompt.startswith('[') and ']' in prompt:
            prompt = prompt[prompt.index(']') + 1:].strip()
        return self.clean_text(prompt)

    def prepare_examples(self, dataset: datasets.Dataset, task_type: str) -> List[Dict]:
        examples = []

        if task_type == 'math':
            for item in dataset:
                question = item.get('instruction')
                answer = item.get('output')
                if question and answer:
                    # Count tokens in the output
                    num_tokens = len(self.tokenizer.encode(answer))
                    if num_tokens > 64:
                        examples.append({
                            'input': question,
                            'output': answer,
                            'task_type': task_type
                        })

        elif task_type == 'coding':
            for item in dataset:
                instruction = item.get('instruction')
                output = item.get('output')
                if instruction and output:
                    num_tokens = len(self.tokenizer.encode(output))
                    if num_tokens > 64:
                        examples.append({
                            'input': instruction,
                            'output': output,
                            'task_type': task_type
                        })

        elif task_type == 'factual_knowledge':
            for item in dataset:
                input_text = item.get('input')
                output_text = item.get('output')
                if input_text and output_text:
                    num_tokens = len(self.tokenizer.encode(output_text))
                    if num_tokens > 64:
                        examples.append({
                            'input': input_text,
                            'output': output_text,
                            'task_type': task_type
                        })

        elif task_type == 'creative_writing':
            # First select a subset of data
            selected_items = []
            for item in dataset:
                prompt = item.get('prompt')
                story = item.get('story')
                if prompt and story:
                    # Quick length check before cleaning
                    if len(story) > 100:  # Basic length check to filter out very short stories
                        selected_items.append(item)
            
            # Shuffle and take a subset
            random.shuffle(selected_items)
            max_items = 10000  # Limit the number of items to process
            selected_items = selected_items[:max_items]
            
            # Now clean and process the selected items
            for item in selected_items:
                prompt = item.get('prompt')
                story = item.get('story')
                
                # Clean the texts
                prompt = self.clean_writing_prompt(prompt)
                story = self.clean_text(story)
                
                num_tokens = len(self.tokenizer.encode(story))
                if num_tokens > 64:
                    examples.append({
                        'input': prompt,
                        'output': story,
                        'task_type': task_type
                    })

        return examples

    def create_splits(self, examples: List[Dict], for_eval: bool = False) -> Dict[str, List[Dict]]:
        # No need to filter by length here since we already did it in prepare_examples
        random.shuffle(examples)

        if for_eval:
            # For evaluation data (from validation split), split into validation and test
            total_eval = len(examples)
            val_size = min(500, total_eval // 2)  # Take first half for validation
            test_size = min(500, total_eval - val_size)  # Take second half for test
            
            val = examples[:val_size]
            test = examples[val_size:val_size + test_size]
            return {'validation': val, 'test': test}
        else:
            # For training, take up to the specified number of examples
            train = examples[:min(self.split_counts['train'], len(examples))]
            return {'train': train}

    def save_splits(self, splits: Dict[str, List[Dict]], task_type: str):
        task_dir = os.path.join(self.output_dir, task_type)
        os.makedirs(task_dir, exist_ok=True)

        for split_name, split_data in splits.items():
            file_path = os.path.join(task_dir, f'{split_name}.jsonl')
            with open(file_path, 'w', encoding='utf-8') as f:
                for example in split_data:
                    json.dump(example, f, ensure_ascii=False)
                    f.write('\n')
            print(f"Saved {len(split_data)} examples to {file_path}")

    def prepare_all_datasets(self, task_name: str = None):
        if task_name:
            if task_name not in self.dataset_configs:
                print(f"Error: Task '{task_name}' not found. Available tasks: {list(self.dataset_configs.keys())}")
                return
            configs = {task_name: self.dataset_configs[task_name]}
        else:
            configs = self.dataset_configs

        for task_name, config in configs.items():
            print(f"\nPreparing {task_name} dataset...")

            # Handle training data from train split
            train_dataset = self.load_dataset(config, 'train')
            if train_dataset:
                train_examples = self.prepare_examples(train_dataset, config['task_type'])
                train_splits = self.create_splits(train_examples, for_eval=False)
                self.save_splits(train_splits, config['task_type'])

            # Handle evaluation data from validation split
            eval_dataset = self.load_dataset(config, 'validation')
            if eval_dataset:
                eval_examples = self.prepare_examples(eval_dataset, config['task_type'])
                eval_splits = self.create_splits(eval_examples, for_eval=True)
                self.save_splits(eval_splits, config['task_type'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare datasets for training')
    parser.add_argument('--task', type=str, default=None,
                      help='Specific task to prepare (math, coding, factual_knowledge, creative_writing). '
                           'If not specified, all tasks will be prepared.')
    args = parser.parse_args()

    random.seed(42)
    preparator = DatasetPreparator()
    preparator.prepare_all_datasets(args.task)
