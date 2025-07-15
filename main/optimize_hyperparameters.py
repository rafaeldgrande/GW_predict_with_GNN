# Loads your dataset
# Runs Optuna for N trials
# Saves best hyperparameters to best_params.json or best_params.pkl
# Optionally saves training plots

from gnn_def import *
from utils import *
import argparse
import logging

import optuna
import optuna.visualization as vis
from torch_geometric.loader import DataLoader

# Configure Optuna logging level (optional)
optuna.logging.set_verbosity(optuna.logging.INFO)  # Can be DEBUG, INFO, WARNING, ERROR

def custom_callback(study, trial):
    """
    Custom callback function to customize Optuna output logging.
    This function is called after each trial completion.
    """
    current_value = trial.value
    best_value = study.best_value
    best_trial_number = study.best_trial.number
    
    # Get total number of trials from study user attributes (set during optimization)
    total_trials = study.user_attrs.get('total_trials', '?')
    
    # Custom formatted output
    print(f"Trial {trial.number + 1} of {total_trials} | MAE: {current_value:.6f} | "
          f"Best MAE: {best_value:.6f} (Trial {best_trial_number + 1})")
    
    # Show parameter summary in a nicer format
    params = trial.params
    param_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                           for k, v in params.items()])
    print(f"   Parameters: {param_str}")
    
    # Show improvement status
    if trial.number > 0:
        improvement = best_value < study.trials[trial.number-1].value if trial.number > 0 else False
        status = "NEW BEST!" if improvement and trial.number == best_trial_number else " Continue"
        print(f"    {status}")
    
    print("-" * 80)

def objective(trial):
    # Build the model with Optuna suggestions
    """
    Objective function for Optuna hyperparameter optimization.

    This function builds and trains a GNN model using hyperparameters suggested by Optuna.
    The model's performance is evaluated on a validation set, and the mean absolute error (MAE)
    is returned as the objective value for Optuna to minimize.

    Args:
        trial: An Optuna trial object used to suggest hyperparameters.

    Returns:
        float: The mean absolute error (MAE) of the model on the validation set.
    """

    model = create_gnn_model_for_hyperparameter_search(
        trial,
        input_dim=input_dim,
        params={
            "n_layers_min": n_layers_min,
            "n_layers_max": n_layers_max,
            "hidden_dims_min": hidden_dims_min,
            "hidden_dims_max": hidden_dims_max,
            "hidden_dims_step": hidden_dims_step,
            "dropout_rate_min": dropout_rate_min,
            "dropout_rate_max": dropout_rate_max,
            "mlp_hidden_min": mlp_hidden_min,  # Example value, adjust as needed
            "mlp_hidden_max": mlp_hidden_max,  # Example value, adjust as needed
        },
        use_distances=use_distances,
        use_angles=use_angles,
    ).to(device)

    # Suggest and set up optimizer
    lr = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    # Training loop
    
    pre_avg_loss = 0.0
    for epoch in range(total_epochs_trial):
        model.train()
        total_loss, total_samples = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs

        avg_loss = total_loss / total_samples
        if epoch == 0 or (epoch+1) % 5 == 0 or epoch == total_epochs_trial - 1:
            variation_avg_loss = avg_loss - pre_avg_loss
            if epoch > 1:
                print(f"Epoch {epoch+1}/{total_epochs_trial} | Avg Train Loss: {avg_loss:.4f} | Variation of avg loss: {variation_avg_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{total_epochs_trial} | Avg Train Loss: {avg_loss:.4f}")
        pre_avg_loss = avg_loss

    # Validation
    model.eval()
    val_mae, val_samples = 0.0, 0
    with torch.no_grad():
        for batch in dev_loader:
            batch = batch.to(device)
            pred = model(batch)
            val_mae     += F.l1_loss(pred, batch.y.view(-1), reduction="sum").item()
            val_samples += batch.num_graphs

    return val_mae / val_samples


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="GNN training with Optuna hyperparameter optimization")

    parser.add_argument('--use_angles', action='store_true', default=True, help='Use angle features in the GNN')
    parser.add_argument('--no_use_angles', dest='use_angles', action='store_false', help='Do not use angle features')
    parser.add_argument('--use_distances', action='store_true', default=True, help='Use distance features in the GNN')
    parser.add_argument('--no_use_distances', dest='use_distances', action='store_false', help='Do not use distance features')
    parser.add_argument('--n_trials_Bayesian_optimization', type=int, default=10, help='Number of Optuna trials')
    parser.add_argument('--total_epochs_trial', type=int, default=10, help='Epochs per Optuna trial')
    parser.add_argument('--lr_min', type=float, default=1e-3, help='Minimum learning rate')
    parser.add_argument('--lr_max', type=float, default=1e-1, help='Maximum learning rate')
    parser.add_argument('--n_layers_min', type=int, default=2, help='Minimum number of GNN layers')
    parser.add_argument('--n_layers_max', type=int, default=4, help='Maximum number of GNN layers')
    parser.add_argument('--hidden_dims_min', type=int, default=32, help='Minimum number of hidden units per layer')
    parser.add_argument('--hidden_dims_max', type=int, default=128, help='Maximum number of hidden units per layer')
    parser.add_argument('--hidden_dims_step', type=int, default=32, help='Step size for hidden units per layer')
    parser.add_argument('--dropout_rate_min', type=float, default=0.1, help='Minimum dropout rate')
    parser.add_argument('--dropout_rate_max', type=float, default=0.5, help='Maximum dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--file_list_data', type=str, default='data_list.txt', help='File containing the list of data files to be read')
    parser.add_argument('--split_train', type=float, default=0.8, help="Fraction of the data that will be used to train model. The rest will be used in evaluation and development (50%, 50%)")
    parser.add_argument('--mlp_hidden_min', type=int, default=64, help='Minimum hidden units for MLP layer')
    parser.add_argument('--mlp_hidden_max', type=int, default=256, help='Maximum hidden units for MLP layer')
    
    args = parser.parse_args()
    
    use_angles = args.use_angles
    use_distances = args.use_distances
    n_trials_Bayesian_optimization = args.n_trials_Bayesian_optimization
    total_epochs_trial = args.total_epochs_trial
    lr_min = args.lr_min
    lr_max = args.lr_max
    n_layers_min = args.n_layers_min
    n_layers_max = args.n_layers_max
    hidden_dims_min = args.hidden_dims_min
    hidden_dims_max = args.hidden_dims_max
    hidden_dims_step = args.hidden_dims_step
    dropout_rate_min = args.dropout_rate_min
    dropout_rate_max = args.dropout_rate_max
    batch_size = args.batch_size
    seed = args.seed
    file_list_data = args.file_list_data
    split_train = args.split_train
    mlp_hidden_min = args.mlp_hidden_min
    mlp_hidden_max = args.mlp_hidden_max
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print if using GPU or CPU
    print(f"Using device: {device}")
    
    # load data
    dataset, input_dim = load_dataset(file_list_data)
    
    # split dataset
    train_loader, dev_loader, val_loader = split_data_set(dataset, batch_size, split_ratio=split_train)

    # Bayesian optimization
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    # Set total trials for the callback to display progress
    study.set_user_attr('total_trials', n_trials_Bayesian_optimization)
    
    # Option 1: Use custom callback with default Optuna logging
    study.optimize(objective, n_trials=n_trials_Bayesian_optimization, callbacks=[custom_callback])
    
    # Option 2: Disable default Optuna logging and use only custom callback
    # optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress default messages
    # study.optimize(objective, n_trials=n_trials_Bayesian_optimization, callbacks=[custom_callback])

    print("Best MAE:", study.best_value)
    print("Best hyperparameters:", study.best_trial.params)
    
    # save json file with best hyperparameters
    with open("best_params.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    
    fig1 = vis.plot_optimization_history(study)
    fig2 = vis.plot_param_importances(study)
    fig3 = vis.plot_slice(study)

    fig1.write_html("optimization_history.html")
    fig2.write_html("param_importances.html")
    fig3.write_html("hyperparameter_slices.html")