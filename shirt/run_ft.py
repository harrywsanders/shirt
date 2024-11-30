from definitions import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for n_target in ns_target:
    for target_task in benchmarks:
        for ft_method in ft_methods:
            command = f"python pipeline.py   --model_name {model_name}   --data_path ../ft_data/{ft_method}_n-target={n_target}_n-aux={n_aux}_{target_task}.json   --data_type openllm_bench   --finetuning_task_key ft_data --task_names {target_task.replace("leaderboard_","")}   --output_dir ../ft_models/{ft_method}_{n_target}_{n_aux}_{target_task.replace("leaderboard_","")}   --batch_size {batch_size}   --num_epochs {n_epochs}   --learning_rate 5e-5"
            os.system(command)