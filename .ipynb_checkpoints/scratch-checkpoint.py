import pandas as pd
from fine_tune_and_evaluate.pipeline import load_and_preprocess_data, tokenize_data, fine_tune_model, evaluate_model_with_lm_eval

data = pd.read_json('data/data_QA.json')