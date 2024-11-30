from pipeline import *
from definitions import *
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

for n_target in ns_target:
    for target_task in benchmarks:

        evaluate_model_with_lm_eval(
                            model_path=model_name,
                            task_names=[target_task],
                            device=device,
                            batch_size=batch_size,
                            output_dir=f"../evals/nonft_{n_target}_{n_aux}_{target_task.replace("leaderboard_","")}"
                        )
        
        for ft_method in ft_methods:

            evaluate_model_with_lm_eval(
                            model_path=f"../ft_models/{ft_method}_{n_target}_{n_aux}_{target_task.replace("leaderboard_","")}",
                            task_names=[target_task],
                            device=device,
                            batch_size=batch_size,
                            output_dir=f"../evals/{ft_method}_{n_target}_{n_aux}_{target_task.replace("leaderboard_","")}"
                        )

