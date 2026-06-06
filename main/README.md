# main — GNN Training and Prediction Scripts

This directory contains all scripts for hyperparameter optimization, model training, and prediction. The two support modules (`gnn_def.py`, `utils.py`) are imported by the scripts and not run directly.

Run all scripts from this directory. Each script accepts `--help` for the full argument list.

---

## Support modules (not run directly)

### `gnn_def.py` — Model definitions

Defines all GNN architectures and factory functions:

| Class / Function | Description |
|---|---|
| `AngleGNNLayer` | Single message-passing layer with distance-weighted and angle-based messages |
| `AngleGNN` | Full model: N layers of `AngleGNNLayer` + BatchNorm + Dropout + GlobalMeanPool + MLP readout |
| `GNNOptunaModel` | Baseline model using `NNConv` (distances only, no angles) |
| `create_gnn_model_from_params_variable_dims` | Builds `AngleGNN` from a params dict (used by `train_model.py`) |
| `create_gnn_model_for_hyperparameter_search_variable_dims` | Builds `AngleGNN` from an Optuna trial (used by `optimize_hyperparameters.py`) |

### `utils.py` — Utilities

| Function | Description |
|---|---|
| `load_dataset` | Reads a list of HDF5 files and returns a list of PyG `Data` objects |
| `load_gnn_samples_from_h5` | Parses one HDF5 file into graph samples |
| `build_edge_index_with_pbc` | Builds edges and angle triplets using ASE neighbor list with PBC (cutoff 3.0 Å) |
| `split_data_set` | Splits dataset into train / dev / val loaders (80/10/10 by default) |
| `evaluate_model` | Runs inference on a DataLoader, returns `(y_pred, y_true)` arrays |
| `plot_mae_loss` | Saves `loss_mae.png` (train/val loss and MAE vs epoch) |
| `plot_prediction_vs_true_val` | Saves `pred_vs_true_qp.png` (parity plot) |
| `set_seed` | Sets Python / NumPy / PyTorch / CUDA seeds for reproducibility |

---

## `optimize_hyperparameters.py` — Bayesian hyperparameter search

Runs an Optuna TPE hyperparameter search. Each trial trains the model for a fixed number of epochs and reports the validation MAE. Saves the best hyperparameters to `best_params.json` and logs everything to MLflow.

### Usage

```bash
python optimize_hyperparameters.py \
    --file_list_data data_list.txt \
    --use_angles --use_distances \
    --n_trials_Bayesian_optimization 1000 \
    --total_epochs_trial 40 \
    --mlflow_experiment_name GNN_GW_hypersearch \
    --run_description "Full model: distances + angles"
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--use_angles` / `--no_use_angles` | `True` | Include bond angles as edge features |
| `--use_distances` / `--no_use_distances` | `True` | Include interatomic distances as edge features |
| `--n_trials_Bayesian_optimization` | `10` | Number of Optuna trials |
| `--total_epochs_trial` | `10` | Training epochs per trial |
| `--n_layers_min` / `--n_layers_max` | `2` / `4` | Search range for number of GNN layers |
| `--hidden_dims_min` / `--hidden_dims_max` | `32` / `128` | Search range for hidden layer widths |
| `--hidden_dims_step` | `32` | Step size for hidden layer widths |
| `--mlp_hidden_min` / `--mlp_hidden_max` | `64` / `256` | Search range for readout MLP hidden size |
| `--dropout_rate_min` / `--dropout_rate_max` | `0.1` / `0.5` | Search range for dropout rate |
| `--lr_min` / `--lr_max` | `1e-3` / `1e-1` | Search range for Adam learning rate (log-uniform) |
| `--batch_size` | `32` | Batch size |
| `--split_train` | `0.8` | Train fraction (remaining split equally into dev/val) |
| `--seed` | `42` | Random seed for Optuna sampler and PyTorch |
| `--file_list_data` | `data_list.txt` | Text file listing HDF5 data files (one per line) |
| `--log_level` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--mlflow_experiment_name` | `GNN_GW_hypersearch` | MLflow experiment name |
| `--mlflow_tracking_uri` | `sqlite:///mlflow.db` | MLflow backend. Use `sqlite:///mlflow.db` for local runs or `sqlite:////absolute/path/mlflow.db` for a shared database |
| `--run_description` | `''` | Free-text note stored as an MLflow tag to describe this run |

### Outputs

| File | Description |
|------|-------------|
| `best_params.json` | Best hyperparameters found |
| `optimization_history.html` | Optuna optimization history plot |
| `param_importances.html` | Optuna hyperparameter importance plot |
| `hyperparameter_slices.html` | Optuna slice plot |
| `logs/optuna_optimization_<timestamp>.log` | Detailed log file |
| MLflow run | All params, `trial_mae` curve, and `best_mae` metric logged to the database |

---

## `train_model.py` — Full model training

Trains the `AngleGNN` model with early stopping and a learning rate scheduler. Reads hyperparameters from a JSON file (typically `best_params.json` produced by `optimize_hyperparameters.py`). Logs training curves per epoch to MLflow.

### Usage

```bash
python train_model.py \
    --model best_params.json \
    --file_list_data data_list.txt \
    --epochs_train_model 500 \
    --patience_training 30 \
    --output_weights_file out_weights.pth \
    --plot_data \
    --mlflow_experiment_name GNN_GW_training
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model_GNN.json` | JSON file with model hyperparameters (produced by `optimize_hyperparameters.py`) |
| `--file_list_data` | `data_list.txt` | Text file listing HDF5 data files (one per line) |
| `--epochs_train_model` | `10` | Maximum number of training epochs |
| `--patience_training` | `20` | Early stopping patience (epochs without improvement on val MAE) |
| `--patience_learning_rate` | `10` | Epochs without improvement before reducing the learning rate (factor 0.8) |
| `--batch_size` | `32` | Batch size |
| `--split_train` | `0.8` | Train fraction |
| `--seed` | `42` | Random seed |
| `--output_weights_file` | `out_weights.pth` | Where to save the best model weights |
| `--use_pre_trained_weights` | `False` | Load weights from `--weights_file` before training (fine-tuning) |
| `--weights_file` | `weights.pth` | Pre-trained weights file (only used if `--use_pre_trained_weights`) |
| `--plot_data` | `False` | If set, saves `loss_mae.png` and `pred_vs_true_qp.png` |
| `--log_level` | `INFO` | Logging verbosity |
| `--mlflow_experiment_name` | `GNN_GW_corrections` | MLflow experiment name |
| `--mlflow_tracking_uri` | `sqlite:///mlflow.db` | MLflow backend URI |

### Outputs

| File | Description |
|------|-------------|
| `out_weights.pth` (or custom name) | Best model weights (saved at best validation MAE epoch) |
| `loss_mae.png` | Train/val loss and MAE curves (if `--plot_data`) |
| `pred_vs_true_qp.png` | Parity plot on validation set (if `--plot_data`) |
| `logs/train_model_<timestamp>.log` | Detailed log file |
| MLflow run | Per-epoch `train_loss`, `train_mae`, `val_loss`, `val_mae`, `learning_rate`; final `best_val_mae`, `final_mae`, `final_rmse`; weights and plots as artifacts |

---

## `predict.py` — Prediction on new data

Loads a trained model and runs inference on a new dataset. Produces both a NumPy archive of predictions and a BerkeleyGW-compatible `eqp_from_GNN.dat` output file.

### Usage

```bash
python predict.py \
    --model best_params.json \
    --weights_file out_weights.pth \
    --file_list_data new_data_list.txt \
    --predictions_file predictions.npz \
    --plot_data
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `default_GNN.json` | JSON file with model hyperparameters |
| `--weights_file` | `best_model.pth` | Trained model weights |
| `--file_list_data` | `data_list.txt` | Text file listing HDF5 data files to run inference on |
| `--predictions_file` | `predictions.npz` | Output file for predictions (always written as `.npz` with `y_pred` and `y_true` keys) |
| `--batch_size` | `32` | Batch size for inference |
| `--seed` | `42` | Random seed |
| `--plot_data` | `False` | If set, saves `pred_vs_true_qp.png` |
| `--log_level` | `INFO` | Logging verbosity |

### Outputs

| File | Description |
|------|-------------|
| `predictions.npz` | NumPy archive with `y_pred` and `y_true` arrays (eV) |
| `eqp_from_GNN.dat` | Two-column text file: `Edft, Eqp_GNN` (eV), one row per state $(n,k)$. Compatible with BerkeleyGW `eqp.dat` format |
| `pred_vs_true_qp.png` | Parity plot (if `--plot_data`) |
| `logs/prediction_<timestamp>.log` | Detailed log file |
