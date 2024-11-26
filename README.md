# Fine-Tuning Guide

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Example Commands](#example-commands)
- [Data Formats](#data-formats)
  - [Custom Data](#custom-data)
  - [OpenLLM Bench Data](#openllm-bench-data)
  - [Dictionary Custom Data](#dictionary-custom-data)
- [Output](#output)
- [Troubleshooting](#troubleshooting)

## Requirements

- Python 3.12
- [Poetry](https://python-poetry.org/) for dependency management
- CUDA-enabled GPU

## Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd fine-tune-llm-qa
   ```

2. **Install Poetry:**

   If you haven't installed Poetry yet, follow the [official installation guide](https://python-poetry.org/docs/#installation). It's awesome.

   You can also install it via the command line:

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install Dependencies:**

   Use Poetry to install the project dependencies.

   ```bash
   poetry install
   ```

   This command will create a virtual environment automatically and install all required packages as specified in the `pyproject.toml` file.

4. **Activate the Virtual Environment:**

   Activate the Poetry-managed virtual environment to ensure that all commands use the correct dependencies.

   ```bash
   poetry shell
   ```

5. **Verify Installation:**

   Run the help command to ensure everything is set up correctly.

   ```bash
   python fine_tune_and_evaluate/pipeline.py --help
   ```

## Usage

The script is executed via the command line with various arguments to customize the fine-tuning process.

### Command-Line Arguments

- **Required Arguments:**
  - `--model_name`: Name or path of the pre-trained Hugging Face model (e.g., `gpt2`, `EleutherAI/gpt-j-6B`).
  - `--data_path`: Path to the QA dataset JSON file.
  - `--data_type`: Type of the input data. Choices:
    - `openllm_bench`
    - `custom`
    - `dict_custom`

- **Conditional Arguments (Required for `openllm_bench` and `dict_custom`):**
  - `--task_key`: Key of the task in the OpenLLM Bench or dict_custom JSON data (e.g., `mmlu`).
  - `--task_names`: List of evaluation task names (e.g., `mmlu hellaswag`).

- **Optional Arguments:**
  - `--output_dir`: Directory to save the fine-tuned model and tokenizer. *(Default: `./fine-tuned-llm-QA`)*
  - `--batch_size`: Batch size per device during training and evaluation. *(Default: `2`)*
  - `--num_epochs`: Number of training epochs. *(Default: `3`)*
  - `--learning_rate`: Learning rate for the optimizer. *(Default: `5e-5`)*
  - `--max_prompt_length`: Maximum token length for prompts. *(Default: `512`)*
  - `--max_completion_length`: Maximum token length for completions. *(Default: `128`)*
  - `--seed`: Random seed for reproducibility. *(Default: `42`)*
  - `--device`: Device to run training and evaluation on (`cuda` or `cpu`). *(Default: `cuda` if available, else `cpu`)*

### Example Commands

#### 1. Fine-Tuning with Custom Data

Fine-tune the model using a custom QA dataset where each JSON entry contains `prompt` and `completion` fields.

```bash
python fine_tune_and_evaluate/pipeline.py \
  --model_name gpt2 \
  --data_path /path/to/custom_data.json \
  --data_type custom \
  --output_dir ./fine-tuned-gpt2-custom \
  --batch_size 4 \
  --num_epochs 5 \
  --learning_rate 3e-5
```

#### 2. Fine-Tuning with OpenLLM Bench Data

Fine-tune the model using OpenLLM Bench data. Requires specifying the task key and task names for evaluation.

```bash
python fine_tune_and_evaluate/pipeline.py \
  --model_name distilgpt2 \
  --data_path data/data_QA.json \
  --data_type openllm_bench \
  --task_key leaderboard_bbh_boolean_expressions \
  --task_names bbh \
  --output_dir ./fine-tuned-distilgpt2-openllm \
  --batch_size 2 \
  --num_epochs 1 \
  --learning_rate 5e-5
```

#### 3. Fine-Tuning with Dictionary Custom Data

Fine-tune the model using dictionary-based custom data. Similar to OpenLLM Bench data, requires task key and task names.

```bash
python fine_tune_and_evaluate/pipeline.py \
  --model_name facebook/opt-1.3b \
  --data_path /path/to/dict_custom_data.json \
  --data_type dict_custom \
  --task_key custom_task \
  --task_names custom_task1,custom_task2 \
  --output_dir ./fine-tuned-opt-dict-custom \
  --batch_size 8 \
  --num_epochs 4 \
  --learning_rate 2e-5
```

## Data Formats

### Custom Data

For `data_type` set to `custom`, the JSON file should be a list of objects, each containing `prompt` and `completion` fields.

**Example:**

```json
[
  {
    "prompt": "What is the capital of France?",
    "completion": "The capital of France is Paris."
  },
  {
    "prompt": "Explain the theory of relativity.",
    "completion": "The theory of relativity, developed by Albert Einstein, encompasses two interrelated theories: special relativity and general relativity..."
  }
]
```

### OpenLLM Bench Data

For `data_type` set to `openllm_bench`, the JSON file should contain tasks with corresponding questions (`Qs`) and answers (`As`) under the specified `task_key`.

**Example:**

```json
{
  "mmlu": {
    "Qs": {
      "1": "What is the largest planet in our solar system?",
      "2": "Who wrote '1984'?"
    },
    "As": {
      "1": "Jupiter is the largest planet in our solar system.",
      "2": "George Orwell wrote '1984'."
    }
  },
  "hellaswag": {
    "Qs": { ... },
    "As": { ... }
  }
}
```

### Dictionary Custom Data

For `data_type` set to `dict_custom`, the structure is similar to OpenLLM Bench but allows for custom task configurations.

**Example:**

```json
{
  "custom_task": {
    "Qs": {
      "1": "Define photosynthesis.",
      "2": "What is the powerhouse of the cell?"
    },
    "As": {
      "1": "Photosynthesis is the process by which green plants...",
      "2": "The mitochondria is known as the powerhouse of the cell."
    }
  }
}
```

## Output

- **Fine-Tuned Model:** Saved in the specified `--output_dir`. This includes both the model and tokenizer.
- **Logs:** A log file named `fine_tune_llm.log` is created in the current directory.

## Troubleshooting

If you encounter issues during installation or execution, consider the following steps:

1. **Check Python Version:**
   Ensure you are using Python 3.12.

   ```bash
   python --version
   ```

2. **Poetry Environment:**
   Make sure the Poetry virtual environment is activated.

   ```bash
   poetry shell
   ```

3. **Dependency Issues:**
   If dependencies fail to install, try updating Poetry and clearing caches.

   ```bash
   poetry self update
   poetry cache clear pypi --all
   poetry install
   ```
