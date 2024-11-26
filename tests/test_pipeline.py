import os
import json
import pytest
from unittest.mock import patch, MagicMock
from transformers import AutoTokenizer, AutoModelForCausalLM
from fine_tune_and_evaluate.pipeline import (
    load_and_preprocess_data,
    tokenize_data,
    fine_tune_model,
    evaluate_model_with_lm_eval
)
from datasets import Dataset
import torch
import subprocess


@pytest.fixture
def dummy_data_path(tmp_path):
    """
    Create a dummy QA JSON file in MMLU format for testing.
    """
    data = {"mmlu_custom": {"Qs":{"0":"Q: The symmetric group $S_n$ has $\nactorial{n}$ elements, hence it is not true that $S_{10}$ has 10 elements.\nFind the characteristic of the ring 2Z.\nA. 0\nB. 30\nC. 3\nD. 10\nE. 12\nF. 50\nG. 2\nH. 100\nI. 20\nJ. 5\nAnswer: A\n\nLet V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p'(x) = d\/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?\nA. ST + TS is the identity map of V onto itself.\nB. TS = 0\nC. ST = 1\nD. ST - TS = 0\nE. ST = T\nF. ST = 0\nG. ST = TS\nH. ST - TS is the identity map of V onto itself.\nI. TS = T\nJ. ST = S\nAnswer: H\n\nLet A be the set of all ordered pairs of integers (m, n) such that 7m + 12n = 22. What is the greatest negative number in the set B = {m + n : (m, n) \\in A}?\nA. -5\nB. 0\nC. -3\nD. -7\nE. -4\nF. -6\nG. -1\nH. -2\nI. -9\nJ. N\/A\nAnswer: E\n\nA tank initially contains a salt solution of 3 grams of salt dissolved in 100 liters of water. A salt solution containing 0.02 grams of salt per liter of water is sprayed into the tank at a rate of 4 liters per minute. The sprayed solution is continually mixed with the salt solution in the tank, and the mixture flows out of the tank at a rate of 4 liters per minute. If the mixing is instantaneous, how many grams of salt are in the tank after 100 minutes have elapsed?\nA. 3 + e^-2\nB. 2 - e^-4\nC. 2 - e^-2\nD. 3 + e^-4\nE. 2 + e^-3\nF. 2 - e^-3\nG. 3 - e^-2\nH. 2 + e^-2\nI. 2 + e^-4\nJ. 2\nAnswer: I\n\nA total of 30 players will play basketball at a park. There will be exactly 5 players on each team. Which statement correctly explains how to find the number of teams needed?\nA. Multiply 5 by 5 to find 25 teams.\nB. Divide 30 by 5 to find 6 teams.\nC. Add 5 to 30 to find 35 teams.\nD. Subtract 30 from 5 to find -25 teams.\nE. Divide 5 by 30 to find 0.1667 teams.\nF. Add 5 to 30 then divide by 2 to find 17.5 teams.\nG. N\/A\nH. N\/A\nI. N\/A\nJ. N\/A\nAnswer: B\n\nIf the exhaust gases contain 0.22% of NO by weight, calculate (i) The minimum value of \\delta allowable if a NO reduction rate of0.032 lb\/ft^2hr is to be achieved. (ii) The corresponding minimum allowable value of K. Use the following data. The gases are at: T = 1200\u00b0F P = 18.2psia Average molecular wt. of the gases = 30.0 Effective rate constant K = 760 ft\/hr Diffusion coefficientD_(_1)m = 4.0 ft^2\/hr\nA. \\delta = 0.0028 ft, K = 55 lb\/ft^2hr\nB. \\delta = 0.0040 ft, K = 45 lb\/ft^2hr\nC. \\delta = 0.0050 ft, K = 60 lb\/ft^2hr\nD. \\delta = 0.0032 ft, K = 47 lb\/ft^2hr\nE. \\delta = 0.0045 ft, K = 44 lb\/ft^2hr\nF. \\delta = 0.0024 ft, K = 49 lb\/ft^2hr\nG. \\delta = 0.0032 ft, K = 50 lb\/ft^2hr\nH. \\delta = 0.0018 ft, K = 42 lb\/ft^2hr\nI. \\delta = 0.0026 ft, K = 47 lb\/ft^2hr\nJ. \\delta = 0.0035 ft, K = 52 lb\/ft^2hr\nAnswer:\nA: ","1":"Q: The symmetric group $S_n$ has $\nactorial{n}$ elements, hence it is not true that $S_{10}$ has 10 elements.\nFind the characteristic of the ring 2Z.\nA. 0\nB. 30\nC. 3\nD. 10\nE. 12\nF. 50\nG. 2\nH. 100\nI. 20\nJ. 5\nAnswer: A\n\nLet V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p'(x) = d\/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?\nA. ST + TS is the identity map of V onto itself.\nB. TS = 0\nC. ST = 1\nD. ST - TS = 0\nE. ST = T\nF. ST = 0\nG. ST = TS\nH. ST - TS is the identity map of V onto itself.\nI. TS = T\nJ. ST = S\nAnswer: H\n\nLet A be the set of all ordered pairs of integers (m, n) such that 7m + 12n = 22. What is the greatest negative number in the set B = {m + n : (m, n) \\in A}?\nA. -5\nB. 0\nC. -3\nD. -7\nE. -4\nF. -6\nG. -1\nH. -2\nI. -9\nJ. N\/A\nAnswer: E\n\nA tank initially contains a salt solution of 3 grams of salt dissolved in 100 liters of water. A salt solution containing 0.02 grams of salt per liter of water is sprayed into the tank at a rate of 4 liters per minute. The sprayed solution is continually mixed with the salt solution in the tank, and the mixture flows out of the tank at a rate of 4 liters per minute. If the mixing is instantaneous, how many grams of salt are in the tank after 100 minutes have elapsed?\nA. 3 + e^-2\nB. 2 - e^-4\nC. 2 - e^-2\nD. 3 + e^-4\nE. 2 + e^-3\nF. 2 - e^-3\nG. 3 - e^-2\nH. 2 + e^-2\nI. 2 + e^-4\nJ. 2\nAnswer: I\n\nA total of 30 players will play basketball at a park. There will be exactly 5 players on each team. Which statement correctly explains how to find the number of teams needed?\nA. Multiply 5 by 5 to find 25 teams.\nB. Divide 30 by 5 to find 6 teams.\nC. Add 5 to 30 to find 35 teams.\nD. Subtract 30 from 5 to find -25 teams.\nE. Divide 5 by 30 to find 0.1667 teams.\nF. Add 5 to 30 then divide by 2 to find 17.5 teams.\nG. N\/A\nH. N\/A\nI. N\/A\nJ. N\/A\nAnswer: B\n\nAccording to Hobbes, the definition of injustice is _____.\nA. failure to abide by a contract\nB. disregard for societal norms\nC. acting against the welfare of others\nD. disobedience to parental authority\nE. disobedience to God's law\nF. acting against one's own self-interest\nG. failure to follow the rule of law\nH. failure to respect inherent rights\nI. failure to uphold moral duties\nJ. disobedience to a sovereign\nAnswer:\nA: ","2":"Q: The symmetric group $S_n$ has $\nactorial{n}$ elements, hence it is not true that $S_{10}$ has 10 elements.\nFind the characteristic of the ring 2Z.\nA. 0\nB. 30\nC. 3\nD. 10\nE. 12\nF. 50\nG. 2\nH. 100\nI. 20\nJ. 5\nAnswer: A\n\nLet V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p'(x) = d\/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?\nA. ST + TS is the identity map of V onto itself.\nB. TS = 0\nC. ST = 1\nD. ST - TS = 0\nE. ST = T\nF. ST = 0\nG. ST = TS\nH. ST - TS is the identity map of V onto itself.\nI. TS = T\nJ. ST = S\nAnswer: H\n\nLet A be the set of all ordered pairs of integers (m, n) such that 7m + 12n = 22. What is the greatest negative number in the set B = {m + n : (m, n) \\in A}?\nA. -5\nB. 0\nC. -3\nD. -7\nE. -4\nF. -6\nG. -1\nH. -2\nI. -9\nJ. N\/A\nAnswer: E\n\nA tank initially contains a salt solution of 3 grams of salt dissolved in 100 liters of water. A salt solution containing 0.02 grams of salt per liter of water is sprayed into the tank at a rate of 4 liters per minute. The sprayed solution is continually mixed with the salt solution in the tank, and the mixture flows out of the tank at a rate of 4 liters per minute. If the mixing is instantaneous, how many grams of salt are in the tank after 100 minutes have elapsed?\nA. 3 + e^-2\nB. 2 - e^-4\nC. 2 - e^-2\nD. 3 + e^-4\nE. 2 + e^-3\nF. 2 - e^-3\nG. 3 - e^-2\nH. 2 + e^-2\nI. 2 + e^-4\nJ. 2\nAnswer: I\n\nA total of 30 players will play basketball at a park. There will be exactly 5 players on each team. Which statement correctly explains how to find the number of teams needed?\nA. Multiply 5 by 5 to find 25 teams.\nB. Divide 30 by 5 to find 6 teams.\nC. Add 5 to 30 to find 35 teams.\nD. Subtract 30 from 5 to find -25 teams.\nE. Divide 5 by 30 to find 0.1667 teams.\nF. Add 5 to 30 then divide by 2 to find 17.5 teams.\nG. N\/A\nH. N\/A\nI. N\/A\nJ. N\/A\nAnswer: B\n\nAn input signal v(t) =I\\delta(t) (impulse function) is passed through a filter having function (1 - e-j\\omega\\tau)\/j\\omegawhere \\tau is the stretchedumpulsewidth. Determine the output of the filter.\nA. v_o(t) = I [\u03b4(t) * e^(-t\/\u03c4)]\nB. v_o(t) = I [e^(-j\u03c9t) - u(t - \u03c4)]\nC. v_o(t) = I [1 - e^(-j\u03c9\u03c4)]\nD. v_o(t) = I [u(t) \/ (u(t) + u(t - \u03c4))]\nE. v_o(t) = I [u(t) \/ u(t - \tau)]\nF. v_o(t) = I [u(t) + u(t - \tau)]\nG. v_o(t) = I [u(t) * u(t - \tau)]\nH. v_o(t) = I [u(t) - u(t - \tau)]\nI. v_o(t) = I [\u03b4(t) - e^(-t\/\u03c4)]\nJ. v_o(t) = I [sin(\u03c9t) * u(t - \u03c4)]\nAnswer:\nA: ","3":"Q: The symmetric group $S_n$ has $\nactorial{n}$ elements, hence it is not true that $S_{10}$ has 10 elements.\nFind the characteristic of the ring 2Z.\nA. 0\nB. 30\nC. 3\nD. 10\nE. 12\nF. 50\nG. 2\nH. 100\nI. 20\nJ. 5\nAnswer: A\n\nLet V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p'(x) = d\/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?\nA. ST + TS is the identity map of V onto itself.\nB. TS = 0\nC. ST = 1\nD. ST - TS = 0\nE. ST = T\nF. ST = 0\nG. ST = TS\nH. ST - TS is the identity map of V onto itself.\nI. TS = T\nJ. ST = S\nAnswer: H\n\nLet A be the set of all ordered pairs of integers (m, n) such that 7m + 12n = 22. What is the greatest negative number in the set B = {m + n : (m, n) \\in A}?\nA. -5\nB. 0\nC. -3\nD. -7\nE. -4\nF. -6\nG. -1\nH. -2\nI. -9\nJ. N\/A\nAnswer: E\n\nA tank initially contains a salt solution of 3 grams of salt dissolved in 100 liters of water. A salt solution containing 0.02 grams of salt per liter of water is sprayed into the tank at a rate of 4 liters per minute. The sprayed solution is continually mixed with the salt solution in the tank, and the mixture flows out of the tank at a rate of 4 liters per minute. If the mixing is instantaneous, how many grams of salt are in the tank after 100 minutes have elapsed?\nA. 3 + e^-2\nB. 2 - e^-4\nC. 2 - e^-2\nD. 3 + e^-4\nE. 2 + e^-3\nF. 2 - e^-3\nG. 3 - e^-2\nH. 2 + e^-2\nI. 2 + e^-4\nJ. 2\nAnswer: I\n\nA total of 30 players will play basketball at a park. There will be exactly 5 players on each team. Which statement correctly explains how to find the number of teams needed?\nA. Multiply 5 by 5 to find 25 teams.\nB. Divide 30 by 5 to find 6 teams.\nC. Add 5 to 30 to find 35 teams.\nD. Subtract 30 from 5 to find -25 teams.\nE. Divide 5 by 30 to find 0.1667 teams.\nF. Add 5 to 30 then divide by 2 to find 17.5 teams.\nG. N\/A\nH. N\/A\nI. N\/A\nJ. N\/A\nAnswer: B\n\n This is a form of targeted advertising, on websites, with advertisements selected and served by automated systems based on the content displayed to the user.\nA. Social media marketing.\nB. Display advertising.\nC. Mobile advertising.\nD. Search engine marketing.\nE. Contextual advertising.\nF. Email advertising.\nG. Direct marketing.\nH. Affiliate marketing.\nI. Interactive marketing.\nJ. Internet advertising.\nAnswer:\nA: ","4":"Q: The symmetric group $S_n$ has $\nactorial{n}$ elements, hence it is not true that $S_{10}$ has 10 elements.\nFind the characteristic of the ring 2Z.\nA. 0\nB. 30\nC. 3\nD. 10\nE. 12\nF. 50\nG. 2\nH. 100\nI. 20\nJ. 5\nAnswer: A\n\nLet V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p'(x) = d\/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?\nA. ST + TS is the identity map of V onto itself.\nB. TS = 0\nC. ST = 1\nD. ST - TS = 0\nE. ST = T\nF. ST = 0\nG. ST = TS\nH. ST - TS is the identity map of V onto itself.\nI. TS = T\nJ. ST = S\nAnswer: H\n\nLet A be the set of all ordered pairs of integers (m, n) such that 7m + 12n = 22. What is the greatest negative number in the set B = {m + n : (m, n) \\in A}?\nA. -5\nB. 0\nC. -3\nD. -7\nE. -4\nF. -6\nG. -1\nH. -2\nI. -9\nJ. N\/A\nAnswer: E\n\nA tank initially contains a salt solution of 3 grams of salt dissolved in 100 liters of water. A salt solution containing 0.02 grams of salt per liter of water is sprayed into the tank at a rate of 4 liters per minute. The sprayed solution is continually mixed with the salt solution in the tank, and the mixture flows out of the tank at a rate of 4 liters per minute. If the mixing is instantaneous, how many grams of salt are in the tank after 100 minutes have elapsed?\nA. 3 + e^-2\nB. 2 - e^-4\nC. 2 - e^-2\nD. 3 + e^-4\nE. 2 + e^-3\nF. 2 - e^-3\nG. 3 - e^-2\nH. 2 + e^-2\nI. 2 + e^-4\nJ. 2\nAnswer: I\n\nA total of 30 players will play basketball at a park. There will be exactly 5 players on each team. Which statement correctly explains how to find the number of teams needed?\nA. Multiply 5 by 5 to find 25 teams.\nB. Divide 30 by 5 to find 6 teams.\nC. Add 5 to 30 to find 35 teams.\nD. Subtract 30 from 5 to find -25 teams.\nE. Divide 5 by 30 to find 0.1667 teams.\nF. Add 5 to 30 then divide by 2 to find 17.5 teams.\nG. N\/A\nH. N\/A\nI. N\/A\nJ. N\/A\nAnswer: B\n\nWith capital fixed at one unit with 1, 2, 3 units of labor added in equal successive units, production of the output increases from 300 (1 unit of labor), to 350 (2 units of labor) to 375 (3 units of labor). Which of the following is a correct interpretation?\nA. This is long run constant returns to scale.\nB. This is short run decreasing returns to scale.\nC. This is long run increasing returns to scale.\nD. This is long run decreasing returns to scale.\nE. This is short run increasing returns to scale.\nF. This is long run diminishing marginal productivity.\nG. This is the effect of labor saturation.\nH. This is due to labor specialization efficiency.\nI. This is short run diminishing marginal productivity.\nJ. This is short run constant returns to scale.\nAnswer:\nA: "},"As":{"0":"D","1":"A","2":"H","3":"E","4":"I"},"__index_level_0__":{"0":11660,"1":10642,"2":11795,"3":427,"4":7306}}}
    file_path = tmp_path / "dummy_data_QA.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path

@pytest.fixture
def dummy_output_dir(tmp_path):
    """
    Create a temporary directory for model outputs.
    """
    return tmp_path / "fine-tuned-llm-QA"

def test_load_and_preprocess_data(dummy_data_path):
    """
    Test loading and preprocessing of data in MMLU format.
    """
    train_dataset, val_dataset = load_and_preprocess_data(
        data_path=str(dummy_data_path),
        task_key="mmlu_custom",
        max_prompt_length=512,
        data_type="dict_custom",
        max_completion_length=1  # Answers are single letters
    )
    
    assert isinstance(train_dataset, Dataset)
    assert isinstance(val_dataset, Dataset)
    assert len(train_dataset) == 4  # 4 questions, no duplicates
    assert 'prompt' in train_dataset.column_names
    assert 'completion' in train_dataset.column_names
    
    # Check format of a sample
    sample = train_dataset[0]
    assert sample['prompt'].startswith("Q: ")
    assert sample['completion'] in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

def test_tokenize_data(dummy_data_path):
    """
    Test tokenization of data in MMLU format.
    """
    train_dataset, val_dataset = load_and_preprocess_data(
        data_path=str(dummy_data_path),
        task_key="mmlu_custom",
        data_type="dict_custom",
        max_prompt_length=512,
        max_completion_length=1
    )

    untokenized_sample = train_dataset[0]
    assert isinstance(untokenized_sample['prompt'], str)
    assert isinstance(untokenized_sample['completion'], str)

    
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenized_train, tokenized_val = tokenize_data(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        max_prompt_length=512,
        max_completion_length=1
    )
    
    assert 'input_ids' in tokenized_train.column_names
    assert 'attention_mask' in tokenized_train.column_names
    assert 'labels' in tokenized_train.column_names
    
    # Check a sample
    sample = tokenized_train[0]
    
    # Adjust assertions to check for torch.Tensor types
    assert isinstance(sample['input_ids'], torch.Tensor), f"Expected input_ids to be a torch.Tensor, got {type(sample['input_ids'])}"
    assert isinstance(sample['attention_mask'], torch.Tensor), f"Expected attention_mask to be a torch.Tensor, got {type(sample['attention_mask'])}"
    assert isinstance(sample['labels'], torch.Tensor), f"Expected labels to be a torch.Tensor, got {type(sample['labels'])}"
    
    # Convert tensors to lists for label comparison if necessary
    expected_label = tokenizer.encode(untokenized_sample['completion'], add_special_tokens=False)[0]
    # Assuming labels are shifted for language modeling, where the last token is the label
    # and others are -100
    labels = sample['labels'].tolist()
    assert labels[:-1] == [-100] * (len(labels) - 1), "Labels except the last token should be -100"
    assert labels[-1] == expected_label, f"Expected last label to be {expected_label}, got {labels[-1]}"

@pytest.mark.slow
def test_fine_tune_model(dummy_data_path, dummy_output_dir):
    """
    Test the fine-tuning process with a small model and MMLU-formatted dataset.
    """
    train_dataset, val_dataset = load_and_preprocess_data(
        data_path=str(dummy_data_path),
        task_key="mmlu_custom",
        data_type="dict_custom",
        max_prompt_length=512,
        max_completion_length=1
    )
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenized_train, tokenized_val = tokenize_data(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        max_prompt_length=512,
        max_completion_length=1
    )
    
    # Fine-tune the model
    trainer = fine_tune_model(
        model_name="distilgpt2",
        tokenizer=tokenizer,
        tokenized_train=tokenized_train,
        tokenized_val=tokenized_val,
        output_dir=str(dummy_output_dir),
        learning_rate=5e-5,
        batch_size=1,
        num_epochs=1,
        device="cpu",  
        seed=42
    )
    
    trainer.save_model(str(dummy_output_dir))

    # Check if model is saved
    assert os.path.exists(dummy_output_dir), f"Output directory {dummy_output_dir} does not exist."
    assert os.path.exists(dummy_output_dir / "model.safetensors"), "model.safetensors was not found in the output directory."
    assert os.path.exists(dummy_output_dir / "config.json"), "config.json was not found in the output directory."
    assert os.path.exists(dummy_output_dir / "tokenizer.json"), "tokenizer.json was not found in the output directory."


@patch('fine_tune_and_evaluate.pipeline.subprocess.run')
@pytest.mark.slow
def test_evaluate_model_with_lm_eval(mock_subprocess_run, dummy_output_dir, tmp_path):
    """
    Test the evaluation process by mocking the LM Evaluation Harness call.
    """
    # Define a side effect function to simulate directory creation and file writing
    def mock_run_side_effect(*args, **kwargs):
        # Simulate the creation of the output directory
        os.makedirs(dummy_output_dir, exist_ok=True)
        
        # Simulate creation of expected output files
        with open(dummy_output_dir / "evaluation_result.json", 'w') as f:
            json.dump({"result": "success"}, f)
        
        # Simulate writing to stdout and stderr files in tmp_path
        with open(tmp_path / "lm_eval_results_stdout.txt", 'w') as f:
            f.write("Evaluation completed successfully.\n")
        with open(tmp_path / "lm_eval_results_stderr.txt", 'w') as f:
            f.write("")
        with open(tmp_path / "lm_eval_results.json", 'w') as f:
            json.dump({"result": "success"}, f)
        
        # Return a mock subprocess.CompletedProcess object
        return subprocess.CompletedProcess(args, 0, "Evaluation completed successfully.\n", "")
    
    # Assign the side effect to the mock
    mock_subprocess_run.side_effect = mock_run_side_effect
    
    # Call the function under test
    evaluate_model_with_lm_eval(
        model_path=str(dummy_output_dir),
        task_names=["mmlu_custom"],
        device="cpu",
        batch_size=1,
        output_dir=str(tmp_path)
    )
    
    # Now the directory should exist, and you can list its contents
    print(f"DIR CONTENTS {os.listdir(dummy_output_dir)}")
    
    # Proceed with your assertions
    assert os.path.exists(tmp_path), f"Output directory {tmp_path} does not exist."
    assert os.path.exists(tmp_path / "lm_eval_results.json"), "lm_eval_results.json was not created."
    assert os.path.exists(tmp_path / "lm_eval_results_stdout.txt"), "lm_eval_results_stdout.txt was not created."
    assert os.path.exists(tmp_path / "lm_eval_results_stderr.txt"), "lm_eval_results_stderr.txt was not created."
    
    # Check the contents of the stdout file
    with open(tmp_path / "lm_eval_results_stdout.txt", 'r') as f:
        stdout = f.read()
    assert "Evaluation completed successfully." in stdout, "Expected success message not found in stdout."


@patch('fine_tune_and_evaluate.pipeline.subprocess.run')
@pytest.mark.slow
def test_end_to_end(mock_subprocess_run, dummy_data_path, dummy_output_dir, tmp_path):
    """
    An end-to-end test that runs the entire pipeline with dummy data.
    """
    # Mock subprocess.run side effects
    def mock_run_side_effect(*args, **kwargs):
        os.makedirs(tmp_path, exist_ok=True)
        
        # Create expected evaluation files
        with open(tmp_path / "lm_eval_results.json", 'w') as f:
            json.dump({"results": "success"}, f)
        with open(tmp_path / "lm_eval_results_stdout.txt", 'w') as f:
            f.write("Evaluation completed successfully.\n")
        with open(tmp_path / "lm_eval_results_stderr.txt", 'w') as f:
            f.write("")
        
        # Simulate subprocess.CompletedProcess
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Evaluation completed successfully.\n"
        mock_result.stderr = ""
        return mock_result
    
    mock_subprocess_run.side_effect = mock_run_side_effect
    
    # Load and preprocess data
    train_dataset, val_dataset = load_and_preprocess_data(
        data_path=str(dummy_data_path),
        task_key="mmlu_custom",
        data_type="dict_custom",
        max_prompt_length=512,
        max_completion_length=1
    )
    
    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenized_train, tokenized_val = tokenize_data(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        max_prompt_length=512,
        max_completion_length=1
    )
    
    # Fine-tune the model
    trainer = fine_tune_model(
        model_name="distilgpt2",
        tokenizer=tokenizer,
        tokenized_train=tokenized_train,
        tokenized_val=tokenized_val,
        output_dir=str(dummy_output_dir),
        learning_rate=5e-5,
        batch_size=1,
        num_epochs=1,
        device="cpu",
        seed=42
    )
    
    # Save the model
    trainer.save_model(str(dummy_output_dir))
    
    # Run evaluation
    evaluate_model_with_lm_eval(
        model_path=str(dummy_output_dir),
        task_names=["mmlu_custom"],
        device="cpu",
        batch_size=1,
        output_dir=str(tmp_path)
    )
    
    # Assert evaluation results
    assert os.path.exists(tmp_path / "lm_eval_results.json"), "lm_eval_results.json was not created."
    assert os.path.exists(tmp_path / "lm_eval_results_stdout.txt"), "lm_eval_results_stdout.txt was not created."
    assert os.path.exists(tmp_path / "lm_eval_results_stderr.txt"), "lm_eval_results_stderr.txt was not created."
    
    with open(tmp_path / "lm_eval_results_stdout.txt", 'r') as f:
        stdout = f.read()
    assert "Evaluation completed successfully." in stdout, "Expected success message not found in stdout."