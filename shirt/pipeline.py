import argparse
import os
import json
import subprocess
import sys
import logging
from typing import List, Tuple, Union
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def install_packages():
    """
    Install required packages if they are not already installed.
    """
    try:
        import transformers
        import datasets
        import lm_eval
        logger.info("Required packages are already installed.")
    except ImportError as e:
        logger.warning(f"Missing package detected: {e.name}. Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "transformers", "datasets", "scikit-learn",
            "pandas", "torch", "accelerate", "lm-evaluation-harness[all]"
        ])
        logger.info("All packages are installed.")


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a Hugging Face LLM on a custom QA dataset or OpenLLM Bench data and evaluate it using LM Evaluation Harness."
    )
    
    # Required arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the pre-trained Hugging Face model (e.g., 'gpt2', 'EleutherAI/gpt-j-6B')."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the QA dataset JSON file."
    )
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        choices=["openllm_bench", "custom", "dict_custom"],
        help="Type of the input data: 'openllm_bench', 'custom', or 'dict_custom'."
    )
    
    # Conditional arguments for OpenLLM Bench and dict_custom
    parser.add_argument(
        "--finetuning_task_key",
        type=str,
        required=False,
        help="Key of the task in the OpenLLM Bench or dict_custom JSON data (e.g., 'mmlu') that we're finetuning on. Required for 'openllm_bench' and 'dict_custom' data types."
    )
    parser.add_argument(
        "--task_names",
        type=str,
        required=False,
        nargs="+",
        help="List of evaluation task names (e.g., mmlu hellaswag). Required for 'openllm_bench' and 'dict_custom' data types."
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fine-tuned-llm-QA",
        help="Directory to save the fine-tuned model and tokenizer."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size per device during training and evaluation."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=512,
        help="Maximum token length for prompts."
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=128,
        help="Maximum token length for completions."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run training and evaluation on ('cuda' or 'cpu')."
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on data_type
    if args.data_type in ["openllm_bench", "dict_custom"]:
        if not args.finetuning_task_key:
            parser.error("--finetuning_task_key is required when --data_type is 'openllm_bench' or 'dict_custom'.")
        if not args.task_names:
            parser.error("--task_names is required when --data_type is 'openllm_bench' or 'dict_custom'.")
    elif args.data_type == "custom":
        if args.finetuning_task_key or args.task_names:
            parser.error("--finetuning_task_key and --task_names should not be provided when --data_type is 'custom'.")
    
    logger.debug(f"Parsed arguments: {args}")
    return args


def load_and_preprocess_data(
    data_path: str,
    data_type: str,
    finetuning_task_key: Union[str, None],
    max_prompt_length: int,
    max_completion_length: int
) -> Tuple[Dataset, Dataset]:
    """
    Load and preprocess the QA dataset based on the data type.

    Args:
        data_path (str): Path to the QA dataset JSON file.
        data_type (str): Type of the input data ('openllm_bench', 'custom', or 'dict_custom').
        finetuning_task_key (str, optional): Key of the task in the OpenLLM Bench or dict_custom data.
        max_prompt_length (int): Maximum token length for prompts.
        max_completion_length (int): Maximum token length for completions.

    Returns:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
    """
    logger.info("Loading JSON data from %s", data_path)
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if data_type in ["openllm_bench", "dict_custom"]:
        logger.debug("Processing data_type: %s with finetuning_task_key: %s", data_type, finetuning_task_key)
        if finetuning_task_key not in data:
            logger.error("Task key '%s' not found in the provided %s data.", finetuning_task_key, data_type)
            raise ValueError(f"Task key '{finetuning_task_key}' not found in the provided {data_type} data.")
        
        Qs = data[finetuning_task_key].get('Qs', {})
        As = data[finetuning_task_key].get('As', {})

        
        if not Qs or not As:
            logger.error("No questions or answers found under task key '%s'.", finetuning_task_key)
            raise ValueError(f"No questions or answers found under task key '{finetuning_task_key}'.")
    
        logger.debug("Creating DataFrame from Qs and As.")
        df = pd.DataFrame({
            "question": Qs,
            "answers": As
        })
        
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        logger.info("Data split into training and validation sets.")
    
    elif data_type == "custom":
        logger.debug("Processing custom data_type.")
        if not isinstance(data, list):
            logger.error("For 'custom' data_type, the JSON file should contain a list of examples.")
            raise ValueError("For 'custom' data_type, the JSON file should contain a list of examples.")
        
        # Expecting each example to have 'prompt' and 'completion' fields
        examples = []
        for idx, example in enumerate(data):
            if not isinstance(example, dict):
                logger.error("Example at index %d is not a JSON object.", idx)
                raise ValueError(f"Example at index {idx} is not a JSON object.")
            if 'prompt' not in example or 'completion' not in example:
                logger.error("Example at index %d is missing 'prompt' or 'completion' fields.", idx)
                raise ValueError(f"Example at index {idx} is missing 'prompt' or 'completion' fields.")
            examples.append({
                'prompt': example['prompt'],
                'completion': example['completion']
            })
        
        df = pd.DataFrame(examples)
        
        # Split into train and validation sets
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        logger.info("Custom data split into training and validation sets.")
    
    else:
        logger.error("Unsupported data_type: %s", data_type)
        raise ValueError(f"Unsupported data_type: {data_type}")
    
    # Convert to Hugging Face Datasets
    if data_type == "custom":
        logger.debug("Converting training and validation DataFrames to Hugging Face Datasets.")
        train_dataset = Dataset.from_pandas(train_df[['prompt', 'completion']])
        val_dataset = Dataset.from_pandas(val_df[['prompt', 'completion']])
    else:
        logger.debug("Renaming columns and converting DataFrames to Hugging Face Datasets.")
        train_dataset = Dataset.from_pandas(train_df[['question', 'answers']].rename(columns={'question': 'prompt', 'answers': 'completion'}))
        val_dataset = Dataset.from_pandas(val_df[['question', 'answers']].rename(columns={'question': 'prompt', 'answers': 'completion'}))
    
    logger.info("Datasets loaded and preprocessed successfully.")
    return train_dataset, val_dataset


def tokenize_data(
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_prompt_length: int,
    max_completion_length: int
) -> Tuple[Dataset, Dataset]:
    """
    Tokenize the datasets.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        tokenizer (AutoTokenizer): Hugging Face tokenizer.
        max_prompt_length (int): Maximum token length for prompts.
        max_completion_length (int): Maximum token length for completions.

    Returns:
        tokenized_train (Dataset): Tokenized training dataset.
        tokenized_val (Dataset): Tokenized validation dataset.
    """
    logger.info("Starting tokenization of datasets.")
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        logger.debug("Tokenizer has no pad token. Adding pad token '[PAD]'.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    def tokenize_function(examples):
        # The model will learn to generate 'completion' given 'prompt'
        inputs = examples['prompt']
        targets = examples['completion']

        # Encode inputs (prompts)
        input_encodings = tokenizer(
            inputs,
            truncation=True,
            max_length=max_prompt_length,
            padding='max_length'
        )

        # Encode targets (answers)
        target_encodings = tokenizer(
            text_target=targets,
            truncation=True,
            max_length=max_completion_length,
            padding='max_length'
        )

        # Set the labels; use -100 to ignore padding tokens in loss computation
        labels = target_encodings['input_ids']
        labels = [
            [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
            for label_seq in labels
        ]

        # Combine inputs and labels
        input_encodings['labels'] = labels

        return input_encodings

    # Apply tokenization
    logger.debug("Tokenizing training dataset.")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    logger.debug("Tokenizing validation dataset.")
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    # Set the format for PyTorch
    logger.debug("Setting dataset format to PyTorch tensors.")
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    logger.info("Tokenization completed successfully.")
    return tokenized_train, tokenized_val

def fine_tune_model(
    model_name: str,
    tokenizer: AutoTokenizer,
    tokenized_train: Dataset,
    tokenized_val: Dataset,
    output_dir: str,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    device: str,
    seed: int
) -> Trainer:
    """
    Fine-tune the model using Hugging Face Trainer.

    Args:
        model_name (str): Name or path of the pre-trained model.
        tokenizer (AutoTokenizer): Hugging Face tokenizer.
        tokenized_train (Dataset): Tokenized training dataset.
        tokenized_val (Dataset): Tokenized validation dataset.
        output_dir (str): Directory to save the fine-tuned model.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size per device.
        num_epochs (int): Number of training epochs.
        device (str): Device to run training on ('cuda' or 'cpu').
        seed (int): Random seed for reproducibility.

    Returns:
        trainer (Trainer): Trained Hugging Face Trainer object.
    """
    logger.info("Setting random seed for reproducibility.")

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    
    logger.info("Loading pre-trained model: %s", model_name)
    # Load pre-trained model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.gradient_checkpointing_enable()
    
    # Resize token embeddings if a new pad token was added
    if tokenizer.pad_token is not None and tokenizer.pad_token_id != model.config.pad_token_id:
        logger.debug("Resizing token embeddings to accommodate new pad token.")
        model.resize_token_embeddings(len(tokenizer))
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        save_strategy='epoch',
        load_best_model_at_end=True,
        seed=seed,
        fp16=True if device.startswith("cuda") else False,
        report_to="none"  # no reporting to wandb rn
    )
    logger.debug("TrainingArguments configured: %s", training_args)
    
    # Define Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, 
    )
    
    logger.info("Initializing Hugging Face Trainer.")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    
    # Fine-Tune the Model
    logger.info("Starting model fine-tuning.")

    trainer.train()
    logger.info("Model fine-tuning completed.")
    
    # Save the fine-tuned model
    logger.info("Saving the fine-tuned model to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Fine-tuned model and tokenizer saved successfully.")
    
    return trainer


def evaluate_model_with_lm_eval(
    model_path: str,
    task_names: List[str],
    device: str,
    batch_size: int,
    output_dir: str
):
    """
    Evaluate the fine-tuned model using LM Evaluation Harness.

    Args:
        model_path (str): Path to the fine-tuned model.
        task_names (List[str]): List of evaluation task names.
        device (str): Device to run evaluation on ('cuda' or 'cpu').
        batch_size (int): Batch size for evaluation.
        output_dir (str): Directory to save evaluation results.
    """
    logger.info("Starting model evaluation with LM Evaluation Harness.")
    
    # Ensure lm-evaluation-harness is installed
    try:
        import lm_eval
        logger.info("lm-evaluation-harness is already installed.")
    except ImportError:
        logger.warning("lm-evaluation-harness is not installed. Installing now...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "lm-evaluation-harness[all]"
        ])
        logger.info("lm-evaluation-harness installed successfully.")
    
    lm_eval_command = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", ",".join(task_names),
        "--trust_remote_code",
        "--device", device,
        "--batch_size", str(batch_size),
        "--log_samples",
        "--output_path", os.path.join(output_dir, "lm_eval_results.json"),
    ]

    logger.info("Running LM Evaluation Harness with the following command:")
    logger.info(" ".join(lm_eval_command))

    result = subprocess.run(lm_eval_command, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error("LM Evaluation Harness encountered an error:")
        logger.error(result.stderr)
    else:
        logger.info("LM Evaluation Harness completed successfully.")
        logger.info("Evaluation Results:")
        logger.info(result.stdout)
        
        # Save stdout and stderr to files
        stdout_path = os.path.join(output_dir, "lm_eval_results_stdout.txt")
        stderr_path = os.path.join(output_dir, "lm_eval_results_stderr.txt")
        with open(stdout_path, 'w') as f:
            f.write(result.stdout)
        with open(stderr_path, 'w') as f:
            f.write(result.stderr)
        logger.info("Evaluation results saved to %s and %s", stdout_path, stderr_path)


def main():
    install_packages()
    
    args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Output directory set to: %s", args.output_dir)
    
    logger.info("Loading and preprocessing data...")
    train_dataset, val_dataset = load_and_preprocess_data(
        data_path=args.data_path,
        data_type=args.data_type,
        finetuning_task_key=args.finetuning_task_key if args.data_type in ["openllm_bench", "dict_custom"] else None,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length
    )
    
    logger.info("Initializing tokenizer for model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    logger.info("Tokenizing datasets...")
    tokenized_train, tokenized_val = tokenize_data(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length
    )
    
    logger.info("Starting model fine-tuning...")
    trainer = fine_tune_model(
        model_name=args.model_name,
        tokenizer=tokenizer,
        tokenized_train=tokenized_train,
        tokenized_val=tokenized_val,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
        seed=args.seed
    )
    logger.info("Model fine-tuning completed.")

    #if args.data_type in ["openllm_bench", "dict_custom"]:
    #    logger.info("Starting model evaluation with LM Evaluation Harness...")
    #    evaluate_model_with_lm_eval(
    #        model_path=args.output_dir,
    #        task_names=args.task_names,
    #        device=args.device,
    #        batch_size=args.batch_size,
    #        output_dir=args.output_dir
    #    )
    #    logger.info("Model evaluation completed.")
    #else:
    #    logger.info("Skipping evaluation as data_type is 'custom'. To evaluate, please use 'openllm_bench' or 'dict_custom' data types.")


if __name__ == "__main__":
    main()
