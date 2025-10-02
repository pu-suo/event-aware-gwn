import argparse
import yaml
import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Make sure to place the 'lib' and 'model' directories in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.utils import load_dataset, load_graph_data, masked_mae_loss, masked_rmse_loss, masked_mape_loss
from model.gwn_model import AttentionModulatedGWN

def calculate_metrics(preds, labels, horizons):
    """
    Calculate and print MAE, RMSE, and MAPE for given prediction horizons.
    """
    mae = masked_mae_loss(preds, labels, null_val=0.0)
    rmse = masked_rmse_loss(preds, labels, null_val=0.0)
    mape = masked_mape_loss(preds, labels, null_val=0.0)
    print(f"Overall Metrics | MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")

    for i in horizons:
        pred_h = preds[:, i, :]
        label_h = labels[:, i, :]
        mae_h = masked_mae_loss(pred_h, label_h, null_val=0.0)
        rmse_h = masked_rmse_loss(pred_h, label_h, null_val=0.0)
        mape_h = masked_mape_loss(pred_h, label_h, null_val=0.0)
        print(f"Horizon {((i + 1) * 5):>2} min | MAE: {mae_h:.4f}, RMSE: {rmse_h:.4f}, MAPE: {mape_h:.4f}")

def main(args):
    """
    Main function to evaluate the trained model on the test set.
    """
    # --- Configuration and Setup ---
    try:
        with open(args.config_filename) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config_filename}'")
        sys.exit(1)

    # Updated device selection logic
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading test data...")
    dataloaders, scaler = load_dataset(
        config['data']['dataset_dir'],
        config['data']['batch_size'],
        config['model']['seq_length'],
        config['model']['horizon']
    )
    
    # --- Graph Loading ---
    adj_mx = load_graph_data(config['data']['graph_pkl_filename'])
    supports = [adj_mx]

    # --- Model Initialization and Loading Checkpoint ---
    print("Initializing model and loading checkpoint...")
    model = AttentionModulatedGWN(
        supports=supports,
        num_nodes=config['model']['num_nodes'],
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['horizon'],
        seq_length=config['model']['seq_length'],
        event_feature_dim=config['model']['event_feature_dim'],
        node_embedding_dim=config['model']['node_embedding_dim'],
        d_k=config['model']['d_k']
    ).to(device)

    checkpoint_path = os.path.join(config['train']['log_dir'], 'best_model.pth')
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Model checkpoint loaded from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at '{checkpoint_path}'. Please train the model first.")
        sys.exit(1)

    model.eval()
    
    # --- Inference on Test Set ---
    print("Running inference on the test set...")
    predictions = []
    labels = []
    
    test_loader_tqdm = tqdm(dataloaders['test'], desc="Evaluating")
    with torch.no_grad():
        for batch in test_loader_tqdm:
            x, x_events, y = batch
            x = x.to(device)
            x_events = x_events.to(device)
            y = y.to(device)
            
            output = model(x, x_events) # Shape: (B, Horizon, N)
            y = y.squeeze(-1)           # Shape: (B, Horizon, N)

            predictions.append(output)
            labels.append(y)

    # Concatenate all batches
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Inverse transform to get real traffic speeds
    predictions_rescaled = scaler.inverse_transform(predictions)
    labels_rescaled = scaler.inverse_transform(labels)
    
    # --- Overall Evaluation ---
    print("\n--- Overall Test Set Performance ---")
    horizons_to_eval = [2, 5, 11] # 15, 30, 60 minutes
    calculate_metrics(predictions_rescaled, labels_rescaled, horizons_to_eval)

    # --- Event-Specific Evaluation ---
    print("\n--- Event-Specific Test Set Performance ---")
    # Load event data to identify event windows
    events_df = pd.read_csv(os.path.join(config['data']['dataset_dir'], 'events.csv'))
    events_df['event_start'] = pd.to_datetime(events_df['event_start'])
    
    # Get the timestamps corresponding to our test set predictions
    test_timestamps = dataloaders['test'].dataset.timestamps
    
    event_indices = []
    for i, ts in enumerate(test_timestamps):
        ts_datetime = pd.to_datetime(ts)
        for _, event in events_df.iterrows():
            # Define event window: 3 hours before to 4 hours after start
            window_start = event['event_start'] - pd.Timedelta(hours=3)
            window_end = event['event_start'] + pd.Timedelta(hours=4)
            if window_start <= ts_datetime <= window_end:
                event_indices.append(i)
                break # Move to the next timestamp once it's marked as part of an event

    if not event_indices:
        print("No event windows found in the test set. Skipping event-specific evaluation.")
    else:
        # Filter predictions and labels for event windows
        event_predictions = predictions_rescaled[event_indices]
        event_labels = labels_rescaled[event_indices]
        print(f"Found {len(event_indices)} time steps within event windows.")
        calculate_metrics(event_predictions, event_labels, horizons_to_eval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the Event-Aware Graph WaveNet model.")
    parser.add_argument('--config_filename', default='./config.yaml', type=str,
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()
    main(args)
