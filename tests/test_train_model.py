
import os
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
        "python", "../main/train_model.py",
        "--epochs_train_model", "2",
        "--model", "default_GNN.json",
        "--file_list_data", "data_list.txt",
        "--batch_size", "4",
        "--plot_data", "False",
    ], capture_output=True, text=True)

    print(result.stdout)
    print(result.stderr)

    # 4. Check outputs
    assert result.returncode == 0, "Script crashed"
    assert os.path.exists("weights.pth"), "Model not saved"
    assert torch.load("weights.pth"), "Model file is empty or corrupt"
    
def test_training_with_pretrained_weights():
    result = subprocess.run([
        "python", "../main/train_model.py",
        "--epochs_train_model", "2",
        "--model", "default_GNN.json",
        "--file_list_data", "data_list.txt",
        "--batch_size", "4",
        "--plot_data", "False",
        "--use_pre_trained_weights", "True",
        "--weights_file", "weights.pth"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0