
python ../main/optimize_hyperparameters.py \
    --batch_size 32 \
    --lr_min 1e-4 \
    --lr_max 1e-2 \
    --mlp_hidden_min 32 \
    --mlp_hidden_max 64 \
    --file_list_data data_list.txt \
    --split_train 0.8 \
    --n_trials_Bayesian_optimization 20 \
    --total_epochs_trial 30 \
    --n_layers_min=1 \
    --n_layers_max=5 \
    --hidden_dims_min=16 \
    --hidden_dims_max=256 \
    --hidden_dims_step=16 \
    --dropout_rate_min=0.0 \
    --dropout_rate_max=0.5 \