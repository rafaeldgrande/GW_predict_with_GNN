
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
    for fname in ["best_params.json", "optimization_history.html", "param_importances.html", "hyperparameter_slices.html"]:
        if os.path.exists(fname):
            os.remove(fname)
    
    # Clean up logs directory if it exists
    if os.path.exists("logs"):
        shutil.rmtree("logs")

    # Run the script with a temporary configuration
    result = subprocess.run([
        "python", "../main/optimize_hyperparameters.py",
        "--file_list_data", "data_list.txt",
        "--n_trials_Bayesian_optimization", "2",  # keep trials small for test
        "--total_epochs_trial", "2",               # also reduce epochs
        "--batch_size", "4",
        "--lr_min", "1e-3",
        "--lr_max", "1e-2",
        "--n_layers_min", "2",
        "--n_layers_max", "3",
        "--hidden_dims_min", "32",
        "--hidden_dims_max", "64",
        "--hidden_dims_step", "32",
        "--dropout_rate_min", "0.1",
        "--dropout_rate_max", "0.3",
        "--mlp_hidden_min", "32",
        "--mlp_hidden_max", "64",
        "--log_level", "WARNING",  # Reduce log verbosity for test
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        print("Optuna search failed with stderr:")
        print(result.stderr)
        raise RuntimeError("Optuna search test failed")

    # Verify output files
    assert os.path.exists("best_params.json"), "best_params.json not created"
    assert os.path.exists("optimization_history.html"), "optimization_history.html not created"
    assert os.path.exists("param_importances.html"), "param_importances.html not created"
    assert os.path.exists("hyperparameter_slices.html"), "hyperparameter_slices.html not created"
    
    # Verify logs directory and files are created
    assert os.path.exists("logs"), "logs directory not created"
    log_files = [f for f in os.listdir("logs") if f.startswith("optuna_optimization_") and f.endswith(".log")]
    assert len(log_files) > 0, "No log files created in logs directory"

    with open("best_params.json") as f:
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
        assert 1e-3 <= params["learning_rate"] <= 1e-2, f"learning_rate should be 1e-3 to 1e-2, got {params['learning_rate']}"
        assert 32 <= params["mlp_hidden_dim"] <= 64, f"mlp_hidden_dim should be 32-64, got {params['mlp_hidden_dim']}"
        
        print(f"Best parameters found: {params}")

    print("Optuna hyperparameter search test passed.")
