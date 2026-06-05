#!/bin/bash
# Apply a trained model to new structures
# Run from the examples/ directory

set -e

MAINDIR="../../main"

python $MAINDIR/predict.py \
    --model 03_training/best_params.json \
    --weights_file 03_training/out_weights.pth \
    --file_list_data data/data_list.txt \
    --predictions_file 04_prediction/predictions.npz \
    --plot_data

echo "Done."
echo "Predictions: 04_prediction/predictions.npz  (keys: y_pred, y_true)"
echo "BerkeleyGW format: eqp_from_GNN.dat         (columns: Edft, Eqp_GNN in eV)"
