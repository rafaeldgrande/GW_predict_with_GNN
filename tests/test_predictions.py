
import subprocess
import json
import torch

def test_training_pipeline():
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

    data_list_path = "data_list.txt"

    # 3. Run the training script via subprocess
    result = subprocess.run([
        "python", "../main/predict.py",
        "--file_list_data", "data_list.txt",
        "--batch_size", "4",
        "--model", "default_GNN.json",
        "--weights_file", "weights.pth",
        "--predictions_file", "predictions.npy",
        "--plot_data", "False",
    ], capture_output=True, text=True)

    print(result.stdout)
    print(result.stderr)