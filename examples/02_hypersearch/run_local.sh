#!/bin/bash
# Hyperparameter search with Optuna + MLflow tracking
# Run from the examples/ directory

set -e

MAINDIR="../../main"

# Quick test: 20 trials, 10 epochs (runs in ~5 min on CPU)
python $MAINDIR/optimize_hyperparameters.py \
    --file_list_data data/data_list_opt_study.txt \
    --use_angles --use_distances \
    --n_trials_Bayesian_optimization 20 \
    --total_epochs_trial 10 \
    --n_layers_min 2 --n_layers_max 5 \
    --hidden_dims_min 16 --hidden_dims_max 128 --hidden_dims_step 16 \
    --lr_min 0.001 --lr_max 0.1 \
    --dropout_rate_min 0.0 --dropout_rate_max 0.25 \
    --split_train 0.8 \
    --mlp_hidden_min 32 --mlp_hidden_max 128 \
    --mlflow_tracking_uri sqlite:///mlflow.db \
    --mlflow_experiment_name GNN_GW_hypersearch \
    --run_description "Example: dist+angles, 20 trials"

echo "Done. Best params saved to best_params.json"
echo "View results: mlflow ui --backend-store-uri sqlite:///mlflow.db"
