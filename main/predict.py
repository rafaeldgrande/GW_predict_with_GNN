
# Loads best_model.pth
# Loads new dataset (or validation/test set)
# Evaluates or predicts
# Saves predictions.npy or pred_vs_true.png

from gnn_def import *
from utils import *
import argparse
import logging
import os
from torch_geometric.loader import DataLoader
import json
import time
from datetime import datetime, timedelta

# Configure logging
def setup_logging(log_level=logging.INFO):
    """Setup logging configuration for the prediction process."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/prediction_{timestamp}.log'
    
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Create custom formatter for INFO messages (more readable)
    class InfoFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                return record.getMessage()
            else:
                # Keep full format for non-INFO messages (DEBUG, WARNING, ERROR)
                return f'{record.asctime} - {record.name} - {record.levelname} - {record.getMessage()}'
    
    # Create file handler with full format for all levels
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Create console handler with simplified format for INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(InfoFormatter())
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
    )
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger, log_filename

# Initialize logging
logger, log_file = setup_logging()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="GNN prediction and evaluation")
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--file_list_data', type=str, default='data_list.txt', help='File containing the list of data files to be read')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--model', type=str, default='default_GNN.json', help='Model configuration file')
    parser.add_argument('--weights_file', type=str, default='best_model.pth', help='File containing the model weights')
    parser.add_argument('--predictions_file', type=str, default='predictions.npy', help='File to save predictions')
    parser.add_argument('--plot_data', action='store_true', help='Plot predictions vs true values')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    
    args = parser.parse_args()
    
    # Update logger level if specified
    if hasattr(args, 'log_level'):
        log_level = getattr(logging, args.log_level.upper())
        logging.getLogger().setLevel(log_level)
        logger.setLevel(log_level)
    
    # Log all arguments
    logger.info("=" * 60)
    logger.info("PREDICTION AND EVALUATION STARTED")
    logger.info("=" * 60)
    logger.info(f"Arguments: {vars(args)}")
    
    start_time = time.time()
    start_datetime = datetime.now()
    
    file_list_data = args.file_list_data
    batch_size = args.batch_size
    seed = args.seed
    model_name = args.model
    weights_file = args.weights_file
    predictions_file = args.predictions_file
    plot_data = args.plot_data
    
    logger.info(f"Prediction started at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Using seed: {seed}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print if using GPU or CPU
    print(f"Using device: {device}")
    print(f"Log file: {log_file}")
    logger.info(f"Using device: {device}")
    
    try:
        # load dataset
        logger.info(f"Loading dataset from: {file_list_data}")
        dataset, input_dim = load_dataset(file_list_data)
        logger.info(f"Dataset loaded - Size: {len(dataset)}, Input dimension: {input_dim}")
        
        all_data_loader = DataLoader(dataset, batch_size=batch_size)
        logger.info(f"Data loader created with batch size: {batch_size}, Total batches: {len(all_data_loader)}")
        
        # load hyperparameters
        logger.info(f"Loading model configuration from: {model_name}")
        params_NN = load_params_NN(model_name)
        logger.info(f"Model parameters loaded: {params_NN}")
        
        model = create_gnn_model_from_params(params_NN, input_dim)
        logger.info(f"Model created with input dimension: {input_dim}")
        
        print(f"Model loaded from {model_name}")
        
        # load weights
        logger.info(f"Loading model weights from: {weights_file}")
        model.load_state_dict(torch.load(weights_file, map_location=device))
        model = model.to(device)
        logger.info(f"Model weights loaded and moved to device: {device}")
        
        # evaluate
        logger.info("Starting model evaluation")
        eval_start_time = time.time()
        y_pred, y_true = evaluate_model(model, all_data_loader, device)
        eval_duration = time.time() - eval_start_time
        
        logger.info(f"Evaluation completed in {eval_duration:.2f}s")
        logger.info(f"Predictions shape: {y_pred.shape}, True values shape: {y_true.shape}")
        
        # Calculate evaluation metrics
        mae = np.mean(np.abs(y_pred - y_true))
        mse = np.mean((y_pred - y_true) ** 2)
        rmse = np.sqrt(mse)
        
        logger.info(f"Evaluation metrics:")
        logger.info(f"  - MAE: {mae:.6f}")
        logger.info(f"  - MSE: {mse:.6f}")
        logger.info(f"  - RMSE: {rmse:.6f}")
        
        print(f"Evaluation metrics:")
        print(f"  MAE: {mae:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        
        # Save predictions
        logger.info(f"Saving predictions to: {predictions_file}")
        np.savez(predictions_file, y_pred=y_pred, y_true=y_true)
        logger.info(f"Predictions saved successfully")
        print(f"Predictions saved to {predictions_file}")
        
        if plot_data:
            logger.info("Generating prediction vs true value plot")
            try:
                plot_prediction_vs_true_val(y_pred, y_true)
                logger.info("Plot generated successfully")
                print("Plot saved: pred_vs_true_qp.png")
            except Exception as e:
                logger.error(f"Failed to generate plot: {str(e)}")
                print(f"Failed to generate plot: {str(e)}")
        
        # Calculate total duration
        total_duration = time.time() - start_time
        end_datetime = datetime.now()
        
        logger.info("=" * 60)
        logger.info("PREDICTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {str(timedelta(seconds=int(total_duration)))}")
        logger.info(f"Evaluation duration: {eval_duration:.2f}s")
        
        print(f"Prediction completed in {str(timedelta(seconds=int(total_duration)))}")
        print('Finished!')
        
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error(f"Prediction failed after {total_duration:.2f}s: {str(e)}")
        logger.error(f"Arguments used: {vars(args)}")
        print(f"Prediction failed: {str(e)}")
        raise e
    
    logger.info("=" * 60)
    logger.info("SCRIPT COMPLETED")
    logger.info("=" * 60)