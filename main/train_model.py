
# Loads your dataset
# Loads best_params.json
# Builds model
# Trains using early stopping
# Saves best_model.pth, training logs, and plots

from gnn_def import *
from utils import *
from torch_geometric.loader import DataLoader
import copy
import matplotlib.pyplot as plt
import json
import argparse


def train_model_with_patience(model, train_loader, val_loader, optimizer, loss_fn, device, patience_learning_rate, epochs=10, patience=20):
    print('Starting training...')
    train_loss_list = []
    train_mae_list = []
    val_loss_list = []
    val_mae_list = []
    
    best_val_mae = float('inf')
    best_epoch = -1
    patience_counter = 0
    best_model_weights = copy.deepcopy(model.state_dict())  # init

    model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=patience_learning_rate, min_lr=1e-6
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        total_train_mae = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y.view(-1))
            mae = F.l1_loss(pred, batch.y.view(-1))

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch.num_graphs
            total_train_mae += mae.item() * batch.num_graphs

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_train_mae = total_train_mae / len(train_loader.dataset)
        train_loss_list.append(avg_train_loss)
        train_mae_list.append(avg_train_mae)

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        total_val_mae = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_loss = loss_fn(pred, batch.y.view(-1))
                val_mae = F.l1_loss(pred, batch.y.view(-1))
                total_val_loss += val_loss.item() * batch.num_graphs
                total_val_mae += val_mae.item() * batch.num_graphs

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        avg_val_mae = total_val_mae / len(val_loader.dataset)
        scheduler.step(avg_val_loss)
        val_loss_list.append(avg_val_loss)
        val_mae_list.append(avg_val_mae)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:03d} of {epochs} | Train Loss: {avg_train_loss:.4f} | Train MAE: {avg_train_mae:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f} | patience_counter {patience_counter} | Current LR: {current_lr:.2e}")

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())  # Save best model
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} because patience counter reached {patience_counter}")
            break

    model.load_state_dict(best_model_weights)  # Restore best model weights
    print(f'Finished this training. Best val MAE: {best_val_mae:.4f} at epoch {best_epoch}')

    return train_loss_list, train_mae_list, val_loss_list, val_mae_list, model, best_val_mae, best_epoch

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="GNN training with Optuna hyperparameter optimization")
    
    parser.add_argument('--epochs_train_model', type=int, default=10, help='Epochs to train the best model')
    parser.add_argument('--model', type=str, default='model_GNN.json', help='File with hyperparameters of the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--file_list_data', type=str, default='data_list.txt', help='File containing the list of data files to be read')
    parser.add_argument('--plot_data', type=bool, default=True, help="Plot mae, loss vs epochs and scatter plot y_pred vs y_true")
    parser.add_argument('--use_pre_trained_weights', type=bool, default=False, help="Use pre-trained weights from file weigths.pt")
    parser.add_argument('--weights_file', type=str, default='weights.pth', help="Weights file to be loaded")
    parser.add_argument('--patience_learning_rate', type=int, default=10, help="Patience for learning rate scheduler")
    parser.add_argument('--patience_training', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--output_weights_file', type=str, default='out_weights.pth', help='Output weights file')
    parser.add_argument('--split_train', type=float, default=0.8, help="Fraction of the data that will be used to train model. The rest will be used in evaluation and development (50%, 50%)")
    
    args = parser.parse_args()

    epochs_train_model = args.epochs_train_model
    model_name = args.model
    file_list_data = args.file_list_data
    seed = args.seed
    batch_size = args.batch_size
    plot_data = args.plot_data
    use_pre_trained_weights = args.use_pre_trained_weights
    weights_file = args.weights_file
    patience_learning_rate = args.patience_learning_rate
    patience_training = args.patience_training
    output_weights_file = args.weights_file
    split_train = args.split_train
    
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print if using GPU or CPU
    print(f"Using device: {device}")

    # Load dataset
    
    dataset, input_dim = load_dataset(file_list_data)
    
    # Split dataset
    train_loader, dev_loader, val_loader = split_data_set(dataset, batch_size, split_ratio=split_train)
    
    # Load model config and initialize
    params_NN = load_params_NN(model_name)
    model = create_gnn_model_from_params(params_NN, input_dim)    
    
    # Create optimizer and loss function
    lr = params_NN["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    if use_pre_trained_weights:
        model.load_state_dict(torch.load(weights_file, map_location=device))

    # train model
    train_loss, train_mae, val_loss, val_mae, trained_model, best_val_mae, best_epoch = train_model_with_patience(model, 
                                                                                                          train_loader, 
                                                                                                          val_loader, 
                                                                                                          optimizer, 
                                                                                                          loss_fn, device, 
                                                                                                          patience_learning_rate=patience_learning_rate,
                                                                                                          epochs=epochs_train_model,
                                                                                                          patience=patience_training)
    
    torch.save(trained_model.state_dict(), output_weights_file)
    print(f"Weights saved to {output_weights_file}")
    
    # ---- Evaluate on val_loader ----
    y_pred, y_true = evaluate_model(trained_model, val_loader, device)
    
    if plot_data:
        plot_mae_loss(train_loss, train_mae, val_loss, val_mae)
        plot_prediction_vs_true_val(y_pred, y_true)
        
    print('Finished!')