import argparse
import yaml
import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm

from lib.utils import load_dataset, load_graph_data, masked_mae_loss
from model.gwn_model import AttentionModulatedGWN

def main(args):
    """
    Main function to orchestrate the model training and validation process.
    """
    # --- Configuration and Setup ---
    try:
        with open(args.config_filename) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config_filename}'")
        sys.exit(1)

    if config is None:
        print(f"Error: Configuration file '{args.config_filename}' is empty or invalid. Please check the file.")
        sys.exit(1)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # --- Data Loading ---
    print("Loading data...")
    dataloaders, scaler = load_dataset(
        config['data']['dataset_dir'],
        config['data']['batch_size'],
        config['model']['seq_length'],
        config['model']['horizon']
    )
    
    # --- Graph Loading ---
    try:
        adj_mx = load_graph_data(config['data']['graph_pkl_filename'])
        supports = [adj_mx]
        print("Adjacency matrix loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # --- Model Initialization ---
    print("Initializing model...")
    model = AttentionModulatedGWN(
        supports=supports,
        num_nodes=config['model']['num_nodes'],
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        seq_length=config['model']['seq_length'], 
        event_feature_dim=config['model']['event_feature_dim'],
        node_embedding_dim=config['model']['node_embedding_dim'],
        d_k=config['model']['d_k']
    ).to(device)

    # --- Optimizer and Loss Function ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['base_lr'],
        weight_decay=config['train']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        patience=config['train']['lr_patience'],
        factor=config['train']['lr_decay_ratio'],
        verbose=False
    )

    loss_fn = masked_mae_loss
    
    if not os.path.exists(config['train']['log_dir']):
        os.makedirs(config['train']['log_dir'])
    
    best_val_loss = float('inf')
    
    # --- Training and Validation Loop ---
    print("Starting training...")
    for epoch in range(1, config['train']['epochs'] + 1):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        train_loader_tqdm = tqdm(dataloaders['train'], desc=f"Epoch {epoch} Training")
        for i, batch in enumerate(train_loader_tqdm):
            x, x_events, y = batch
            x = x.to(device)
            x_events = x_events.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            
            output = model(x, x_events)
            
            y = y.squeeze(-1)
            
            loss = loss_fn(output, y)
            loss.backward()
            
            if config['train']['clip_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_grad_norm'])
            
            optimizer.step()
            total_loss += loss.item()
            
            train_loader_tqdm.set_postfix(loss=f'{total_loss / (i + 1):.4f}')

        train_loss = total_loss / len(dataloaders['train'])
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_loader_tqdm = tqdm(dataloaders['val'], desc=f"Epoch {epoch} Validation")
            for i, batch in enumerate(val_loader_tqdm):
                x, x_events, y = batch
                x = x.to(device)
                x_events = x_events.to(device)
                y = y.to(device)
                
                output = model(x, x_events)
                y = y.squeeze(-1)
                
                loss = loss_fn(output, y)
                val_loss += loss.item()

                val_loader_tqdm.set_postfix(loss=f'{val_loss / (i + 1):.4f}')

        val_loss /= len(dataloaders['val'])
        
        if scheduler.get_last_lr()[0] > config['train']['base_lr'] * 1e-4:
             scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{config['train']['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config['train']['log_dir'], 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved. Model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='./config.yaml', type=str,
                        help='Configuration filename for training the model.')
    args = parser.parse_args()
    main(args)

