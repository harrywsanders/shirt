from pipeline import *
from definitions import *
import os
import torch
import json

os.environ["WANDB_PROJECT"]="" ## ADD THIS
os.environ["WANDB_API_KEY"] = "" ## ADD THIS
# or, run "wandb login" in the terminal. 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

def validate_json_structure(file_path, finetuning_task_key):
    with open(file_path, 'r') as f:
        data = json.load(f)
    if finetuning_task_key not in data:
        raise ValueError(f"Key '{finetuning_task_key}' not found in {file_path}.")
    if 'Qs' not in data[finetuning_task_key] or 'As' not in data[finetuning_task_key]:
        raise ValueError(f"'Qs' or 'As' missing under '{finetuning_task_key}' in {file_path}.")

for n_target in ns_target:
    for target_task in benchmarks:
        for ft_method in ft_methods:
            task_name = target_task.replace("leaderboard_", "")
            data_path = f"../ft_data/{ft_method}_n-target={n_target}_n-aux={n_aux}_{target_task}.json"
            validate_json_structure(data_path, 'ft_data')
            run_name = f"{ft_method}_n_target={n_target}_n_aux={n_aux}_{task_name}"
            command = (
                f"python pipeline.py "
                f"--model_name {model_name} "
                f"--data_path ../ft_data/{ft_method}_n-target={n_target}_n-aux={n_aux}_{target_task}.json "
                f"--data_type openllm_bench "
                f"--finetuning_task_key ft_data "
                f"--task_names {task_name} "
                f"--output_dir ../ft_models/{ft_method}_{n_target}_{n_aux}_{task_name} "
                f"--batch_size {batch_size} "
                f"--num_epochs {n_epochs} "
                f"--learning_rate 5e-5 "
                f"--wandb_project shirt-ft-tracking "
                f"--wandb_run_name {run_name}" 
            )
            os.system(command)
