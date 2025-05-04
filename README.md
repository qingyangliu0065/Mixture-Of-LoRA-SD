# Mixture-Of-LoRA-SD

A framework for efficient fine-tuning and inference using LoRA (Low-Rank Adaptation) with a router-based approach for task-specific model selection.

## Overview

This project implements a mixture-of-experts approach using LoRA adapters for efficient fine-tuning and inference of large language models. The system uses a router to dynamically select the most appropriate LoRA adapter for a given input, enabling efficient multi-task learning and inference.

## Features

- LoRA fine-tuning for efficient model adaptation
- Router-based task selection mechanism
- Support for multiple tasks and domains
- Efficient inference with dynamic adapter selection
- Integration with popular language models
- Weights & Biases integration for experiment tracking


## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Mixture-Of-LoRA-SD.git
cd Mixture-Of-LoRA-SD
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Fine-tuning with LoRA

```bash
bash script/lora_finetune.sh
```

### Baseline Inference

```bash
bash script/sd_baseline_inference.sh
```

### Router-based Inference

```bash
bash script/sd_router_inference.sh
```