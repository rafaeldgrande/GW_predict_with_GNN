
import subprocess
import os
import json
import shutil

def test_optuna_hyperparameter_search():
    """
    Test function to run the Optuna hyperparameter search script
    and check for successful execution and correct outputs.
    """
    print("Running Optuna hyperparameter search test...")

    # Clean up any previous results
    for fname in ["best_params_M2.json", "best_params_M3.json",
                  "opt_history_M2.html", "opt_history_M3.html",
                  "param_importances_M2.html", "param_importances_M3.html"]:
        if os.path.exists(fname):
            os.remove(fname)
    
    # Clean up logs directory if it exists
    if os.path.exists("logs"):
        shutil.rmtree("logs")

    # Run the script with a temporary configuration
    result = subprocess.run([
        "python", "../main/optimize_hyperparameters.py",
        "--train_list", "data_list.txt",
        "--dev_list", "data_list.txt",
        "--n_trials", "2",
        "--epochs_per_trial", "2",
        "--batch_size", "4",
        "--n_layers_min", "2",
        "--n_layers_max", "3",
        "--hidden_dims_min", "32",
        "--hidden_dims_max", "64",
        "--hidden_dims_step", "32",
        "--dropout_rate_min", "0.1",
        "--dropout_rate_max", "0.3",
        "--mlp_hidden_min", "32",
        "--mlp_hidden_max", "64",
        "--no_mlflow",
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        print("Optuna search failed with stderr:")
        print(result.stderr)
        raise RuntimeError("Optuna search test failed")

    # Verify output files (--study all runs M2 and M3 separately)
    assert os.path.exists("best_params_M2.json"), "best_params_M2.json not created"
    assert os.path.exists("best_params_M3.json"), "best_params_M3.json not created"

    # Verify logs directory and files are created
    assert os.path.exists("logs"), "logs directory not created"
    log_files = [f for f in os.listdir("logs")
                 if (f.startswith("optuna_M2_") or f.startswith("optuna_M3_")) and f.endswith(".log")]
    assert len(log_files) > 0, "No log files created in logs directory"

    with open("best_params_M2.json") as f:
        params = json.load(f)
        # Check for required base parameters
        required_keys = ["n_layers", "dropout_rate", "learning_rate", "mlp_hidden_dim"]
        for key in required_keys:
            assert key in params, f"Missing {key} in best_params.json"
        
        # Check for layer-specific hidden dimensions
        n_layers = params["n_layers"]
        for i in range(n_layers):
            layer_key = f"hidden_dim_layer_{i}"
            assert layer_key in params, f"Missing {layer_key} in best_params.json"
        
        # Verify parameter ranges are reasonable
        assert 2 <= params["n_layers"] <= 3, f"n_layers should be 2-3, got {params['n_layers']}"
        assert 0.1 <= params["dropout_rate"] <= 0.3, f"dropout_rate should be 0.1-0.3, got {params['dropout_rate']}"
        assert 1e-3 <= params["learning_rate"] <= 1e-1, f"learning_rate should be 1e-3 to 1e-1, got {params['learning_rate']}"
        assert 32 <= params["mlp_hidden_dim"] <= 64, f"mlp_hidden_dim should be 32-64, got {params['mlp_hidden_dim']}"
        
        print(f"Best parameters found: {params}")

    print("Optuna hyperparameter search test passed.")
