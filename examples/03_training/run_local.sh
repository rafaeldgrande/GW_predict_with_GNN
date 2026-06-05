#!/bin/bash
# Full model training with early stopping and MLflow tracking
# Run from the examples/ directory

set -e

MAINDIR="../../main"

python $MAINDIR/train_model.py \
    --model 03_training/best_params.json \
    --file_list_data data/data_list.txt \
    --epochs_train_model 300 \
    --patience_training 30 \
    --patience_learning_rate 10 \
    --batch_size 32 \
    --split_train 0.8 \
    --output_weights_file 03_training/out_weights.pth \
    --plot_data \
    --mlflow_tracking_uri sqlite:///mlflow.db \
    --mlflow_experiment_name GNN_GW_training

echo "Done."
echo "Weights: 03_training/out_weights.pth"
echo "Plots:   loss_mae.png  pred_vs_true_qp.png"
echo "View:    mlflow ui --backend-store-uri sqlite:///mlflow.db"
