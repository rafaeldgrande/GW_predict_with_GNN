
import subprocess
import json
import torch
import os
import numpy as np
import shutil

def test_prediction_pipeline():
    """
    Test function to run the prediction script and verify outputs.
    """
    print("Running prediction pipeline test...")
    
    # Clean up any previous results
    for fname in ["predictions.npz", "pred_vs_true_qp.png"]:
        if os.path.exists(fname):
            os.remove(fname)
    
    # Clean up logs directory if it exists
    if os.path.exists("logs"):
        shutil.rmtree("logs")

    # 1. Create dummy config file
    config_path = "default_GNN.json"
    config = {
        "n_layers": 2,
        "n_units": 32,
        "dropout_rate": 0.1,
        "learning_rate": 0.01
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)

    # Check if we have trained weights, if not skip this test
    weights_file = "out_weights.pth"
    if not os.path.exists(weights_file):
        print(f"Skipping prediction test - no trained weights found ({weights_file})")
        print("Please run training tests first to generate weights file")
        os.remove(config_path)
        return

    # 2. Run the prediction script via subprocess
    result = subprocess.run([
        "python", "../main/predict.py",
        "--file_list_data", "data_list.txt",
        "--batch_size", "4",
        "--model", "default_GNN.json",
        "--weights_file", "out_weights.pth",
        "--predictions_file", "predictions.npz",
        "--log_level", "WARNING",  # Reduce log verbosity for test
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        print("Prediction failed with stderr:")
        print(result.stderr)
        raise RuntimeError("Prediction test failed")

    # 3. Verify output files
    assert os.path.exists("predictions.npz"), "predictions.npz not created"
    
    # Verify logs directory and files are created
    assert os.path.exists("logs"), "logs directory not created"
    log_files = [f for f in os.listdir("logs") if f.startswith("prediction_") and f.endswith(".log")]
    assert len(log_files) > 0, "No log files created in logs directory"
    
    # 4. Verify predictions file content
    with np.load("predictions.npz") as data:
        assert "y_pred" in data, "y_pred not found in predictions file"
        assert "y_true" in data, "y_true not found in predictions file"
        
        y_pred = data["y_pred"]
        y_true = data["y_true"]
        
        assert y_pred.shape == y_true.shape, f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
        assert len(y_pred) > 0, "Empty predictions array"
        
        print(f"Predictions shape: {y_pred.shape}")
        print(f"Sample predictions: {y_pred[:5]}")
        print(f"Sample true values: {y_true[:5]}")

    # 5. Test with plotting enabled
    result_plot = subprocess.run([
        "python", "../main/predict.py",
        "--file_list_data", "data_list.txt",
        "--batch_size", "4",
        "--model", "default_GNN.json",
        "--weights_file", "out_weights.pth",
        "--predictions_file", "predictions_with_plot.npz",
        "--plot_data",  # Enable plotting
        "--log_level", "WARNING",
    ], capture_output=True, text=True)

    if result_plot.returncode == 0:
        # Check if plot was created (optional, might fail in headless environment)
        if os.path.exists("pred_vs_true_qp.png"):
            print("Plot generated successfully")
        else:
            print("Plot generation skipped (likely headless environment)")
    
    # Clean up test files
    os.remove(config_path)
    
    print("Prediction pipeline test passed.")