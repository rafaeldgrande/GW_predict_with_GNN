
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
import logging
import os
import time
from datetime import datetime, timedelta

# Configure logging
def setup_logging(log_level=logging.INFO):
    """Setup logging configuration for the training process."""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/train_model_{timestamp}.log'
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger, log_filename

# Initialize logging
logger, log_file = setup_logging()


def train_model_with_patience(model, train_loader, val_loader, optimizer, loss_fn, device, patience_learning_rate, epochs=10, patience=20):
    logger.info('Starting training...')
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

    logger.info(f"Training configuration: {epochs} epochs, patience={patience}, lr_patience={patience_learning_rate}")

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

        # Log epoch details
        logger.info(f"Epoch {epoch:03d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train MAE: {avg_train_mae:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f} | patience_counter {patience_counter} | Current LR: {current_lr:.2e}")
        print(f"Epoch {epoch:03d} of {epochs} | Train Loss: {avg_train_loss:.4f} | Train MAE: {avg_train_mae:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f} | patience_counter {patience_counter} | Current LR: {current_lr:.2e}")

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())  # Save best model
            patience_counter = 0
            logger.info(f"New best model found at epoch {epoch}: Val MAE = {best_val_mae:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch} because patience counter reached {patience_counter}")
            print(f"Early stopping at epoch {epoch} because patience counter reached {patience_counter}")
            break

    model.load_state_dict(best_model_weights)  # Restore best model weights
    logger.info(f'Finished training. Best val MAE: {best_val_mae:.4f} at epoch {best_epoch}')
    print(f'Finished this training. Best val MAE: {best_val_mae:.4f} at epoch {best_epoch}')

    return train_loss_list, train_mae_list, val_loss_list, val_mae_list, model, best_val_mae, best_epoch

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="GNN model training with early stopping")
    
    parser.add_argument('--epochs_train_model', type=int, default=10, help='Epochs to train the best model')
    parser.add_argument('--model', type=str, default='model_GNN.json', help='File with hyperparameters of the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--file_list_data', type=str, default='data_list.txt', help='File containing the list of data files to be read')
    parser.add_argument('--plot_data', action='store_true', help="Plot mae, loss vs epochs and scatter plot y_pred vs y_true")
    parser.add_argument('--use_pre_trained_weights', action='store_true', help="Use pre-trained weights from file weights.pt")
    parser.add_argument('--weights_file', type=str, default='weights.pth', help="Weights file to be loaded")
    parser.add_argument('--patience_learning_rate', type=int, default=10, help="Patience for learning rate scheduler")
    parser.add_argument('--patience_training', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--output_weights_file', type=str, default='out_weights.pth', help='Output weights file')
    parser.add_argument('--split_train', type=float, default=0.8, help="Fraction of the data that will be used to train model. The rest will be used in evaluation and development (50%, 50%)")
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    
    args = parser.parse_args()

    # Update logger level if specified
    if hasattr(args, 'log_level'):
        log_level = getattr(logging, args.log_level.upper())
        logging.getLogger().setLevel(log_level)
        logger.setLevel(log_level)

    # Log all arguments
    logger.info("=" * 60)
    logger.info("MODEL TRAINING STARTED")
    logger.info("=" * 60)
    logger.info(f"Arguments: {vars(args)}")

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
    output_weights_file = args.output_weights_file
    split_train = args.split_train
    
    start_time = time.time()
    start_datetime = datetime.now()
    logger.info(f"Training started at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    set_seed(seed)
    logger.info(f"Random seed set to: {seed}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Log file: {log_file}")
    logger.info(f"Using device: {device}")

    try:
        # Load dataset
        logger.info(f"Loading dataset from: {file_list_data}")
        dataset, input_dim = load_dataset(file_list_data)
        logger.info(f"Dataset loaded - Size: {len(dataset)}, Input dimension: {input_dim}")
        
        # Split dataset
        logger.info(f"Splitting dataset - Train ratio: {split_train}, Batch size: {batch_size}")
        train_loader, dev_loader, val_loader = split_data_set(dataset, batch_size, split_ratio=split_train)
        logger.info(f"Data split completed - Train batches: {len(train_loader)}, Dev batches: {len(dev_loader)}, Val batches: {len(val_loader)}")
        
        # Load model config and initialize
        logger.info(f"Loading model configuration from: {model_name}")
        params_NN = load_params_NN(model_name)
        logger.info(f"Model parameters loaded: {params_NN}")
        model = create_gnn_model_from_params(params_NN, input_dim)
        logger.info(f"Model created with input dimension: {input_dim}")
        
        # Create optimizer and loss function
        lr = params_NN["learning_rate"]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        logger.info(f"Optimizer and loss function created. Learning rate: {lr}")
        
        if use_pre_trained_weights:
            logger.info(f"Loading pre-trained weights from: {weights_file}")
            model.load_state_dict(torch.load(weights_file, map_location=device))
            logger.info("Pre-trained weights loaded successfully")

        # train model
        logger.info("Starting model training with patience and early stopping")
        train_loss, train_mae, val_loss, val_mae, trained_model, best_val_mae, best_epoch = train_model_with_patience(model, 
                                                                                                              train_loader, 
                                                                                                              val_loader, 
                                                                                                              optimizer, 
                                                                                                              loss_fn, device, 
                                                                                                              patience_learning_rate=patience_learning_rate,
                                                                                                              epochs=epochs_train_model,
                                                                                                              patience=patience_training)
        
        logger.info(f"Training completed. Best val MAE: {best_val_mae:.4f} at epoch {best_epoch}")
        
        # Save model weights
        torch.save(trained_model.state_dict(), output_weights_file)
        logger.info(f"Model weights saved to: {output_weights_file}")
        print(f"Weights saved to {output_weights_file}")
        
        # ---- Evaluate on val_loader ----
        logger.info("Evaluating trained model on validation set")
        y_pred, y_true = evaluate_model(trained_model, val_loader, device)
        
        # Calculate final metrics
        final_mae = np.mean(np.abs(y_pred - y_true))
        final_mse = np.mean((y_pred - y_true) ** 2)
        final_rmse = np.sqrt(final_mse)
        
        logger.info(f"Final evaluation metrics:")
        logger.info(f"  - MAE: {final_mae:.6f}")
        logger.info(f"  - MSE: {final_mse:.6f}")
        logger.info(f"  - RMSE: {final_rmse:.6f}")
        
        print(f"Final evaluation metrics:")
        print(f"  MAE: {final_mae:.6f}")
        print(f"  MSE: {final_mse:.6f}")
        print(f"  RMSE: {final_rmse:.6f}")
        
        if plot_data:
            logger.info("Generating training plots")
            try:
                plot_mae_loss(train_loss, train_mae, val_loss, val_mae)
                plot_prediction_vs_true_val(y_pred, y_true)
                logger.info("Training plots generated successfully")
                print("Plots generated: loss_mae.png, pred_vs_true_qp.png")
            except Exception as e:
                logger.error(f"Failed to generate plots: {str(e)}")
                print(f"Failed to generate plots: {str(e)}")
        
        # Calculate total duration
        total_duration = time.time() - start_time
        end_datetime = datetime.now()
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {str(timedelta(seconds=int(total_duration)))}")
        logger.info(f"Best epoch: {best_epoch}, Best val MAE: {best_val_mae:.4f}")
        
        print(f"Training completed in {str(timedelta(seconds=int(total_duration)))}")
        print('Finished!')
        
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error(f"Training failed after {total_duration:.2f}s: {str(e)}")
        logger.error(f"Arguments used: {vars(args)}")
        print(f"Training failed: {str(e)}")
        raise e
    
    logger.info("=" * 60)
    logger.info("SCRIPT COMPLETED")
    logger.info("=" * 60)