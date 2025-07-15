
import subprocess
import os
import json

def test_optuna_hyperparameter_search():
    """
    Test function to run the Optuna hyperparameter search script
    and check for successful execution and correct outputs.
    """
    print("Running Optuna hyperparameter search test...")

    # Clean up any previous results
    for fname in ["best_params.json", "optimization_history.html", "param_importances.html", "hyperparameter_slices.html"]:
        if os.path.exists(fname):
            os.remove(fname)

    # Run the script with a temporary configuration
    result = subprocess.run([
        "python", "../main/optimize_hyperparameters.py",
        "--file_list_data", "data_list.txt",
        "--n_trials_Bayesian_optimization", "2",  # keep trials small for test
        "--total_epochs_trial", "2",               # also reduce epochs
        "--batch_size", "4",
        "--lr_min", "1e-3",
        "--lr_max", "1e-2",
        "--mlp_hidden_min", "32",
        "--mlp_hidden_max", "64",
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        print("Optuna search failed with stderr:")
        print(result.stderr)
        raise RuntimeError("Optuna search test failed")

    # Verify output files
    assert os.path.exists("best_params.json"), "best_params.json not created"
    assert os.path.exists("optimization_history.html"), "optimization_history.html not created"

    with open("best_params.json") as f:
        params = json.load(f)
        required_keys = ["n_layers", "n_units", "dropout_rate", "learning_rate"]
        for key in required_keys:
            assert key in params, f"Missing {key} in best_params.json"

    print("Optuna hyperparameter search test passed.")
