
# Loads best_model.pth
# Loads new dataset (or validation/test set)
# Evaluates or predicts
# Saves predictions.npy or pred_vs_true.png

from gnn_def import *
from utils import *
import argparse
from torch_geometric.loader import DataLoader
import json


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="GNN training with Optuna hyperparameter optimization")
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--file_list_data', type=str, default='data_list.txt', help='File containing the list of data files to be read')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--model', type=str, default='default_GNN.json', help='Model configuration file')
    parser.add_argument('--weights_file', type=str, default='best_model.pth', help='File containing the model weights')
    parser.add_argument('--predictions_file', type=str, default='predictions.npy', help='File to save predictions')
    parser.add_argument('--plot_data', type=bool, default=False, help='Plot predictions vs true values')
    
    args = parser.parse_args()
    
    file_list_data = args.file_list_data
    batch_size = args.batch_size
    seed = args.seed
    model_name = args.model
    weights_file = args.weights_file
    predictions_file = args.predictions_file
    plot_data = args.plot_data
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print if using GPU or CPU
    print(f"Using device: {device}")
    
    # load dataset
    dataset, input_dim = load_dataset(file_list_data)
    all_data_loader = DataLoader(dataset, batch_size=batch_size)
    
    # load hyperparameters
    params_NN = load_params_NN(model_name)
    model = create_gnn_model_from_params(params_NN, input_dim)
    
    print(f"Model loaded from {model_name}")
    
    # load weights
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model = model.to(device)
    
    # evaluate
    y_pred, y_true = evaluate_model(model, all_data_loader, device)
    
    # Save predictions
    np.savetxt(predictions_file, {"y_pred": y_pred, "y_true": y_true})
    print(f"Predictions saved to {predictions_file}")
    
    if plot_data:
        plot_prediction_vs_true_val(y_pred, y_true)
        
    print('Finished!')