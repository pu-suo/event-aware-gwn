import pickle
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

class StandardScaler:
    """
    Standard scaler for Z-score normalization.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        Applies Z-score normalization.
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Reverts the Z-score normalization.
        """
        return (data * self.std) + self.mean

def load_graph_data(pkl_filename):
    """
    Loads the adjacency matrix from the specified .pkl file.
    Args:
        pkl_filename (str): Path to the .pkl file.
    Returns:
        np.ndarray: The adjacency matrix.
    """
    try:
        with open(pkl_filename, 'rb') as f:
            # [sensor_ids, sensor_id_to_idx, adj_mx]
            _ , _, adj_mx = pickle.load(f, encoding='latin1')
        return adj_mx
    except FileNotFoundError:
        print(f"Error: Adjacency matrix file not found at {pkl_filename}")
        return None
    except Exception as e:
        print(f"Error loading graph data: {e}")
        return None

def generate_sequences(data, seq_length, horizon):
    """
    Generates input/output sequences from time series data.
    Args:
        data (np.ndarray): The input data tensor.
        seq_length (int): The number of time steps in the input sequence.
        horizon (int): The number of time steps in the output sequence.
    Returns:
        np.ndarray: A tensor of input sequences.
        np.ndarray: A tensor of output sequences.
    """
    total_len = len(data)
    sequences = []
    for i in range(total_len - seq_length - horizon + 1):
        start = i
        mid = i + seq_length
        end = mid + horizon
        sequences.append((data[start:mid], data[mid:end]))
    
    x, y = zip(*sequences)
    return np.array(x), np.array(y)

# -----------------------------------------------------------------
# VVVVVV    THIS IS THE MODIFIED load_dataset FUNCTION     VVVVVV
# -----------------------------------------------------------------

def load_dataset(dataset_dir, batch_size, seq_length=12, horizon=12, test_batch_size=64):
    """
    Loads and prepares the PEMS-BAY dataset, including traffic and event data.
    
    *** MODIFIED VERSION: ***
    This version performs a random 70/10/20 shuffle split on all generated
    sequences to ensure events are distributed across train, val, and test sets.
    
    Args:
        dataset_dir (str): Directory where the data files are located.
        batch_size (int): Batch size for the training and validation dataloaders.
        seq_length (int): History length.
        horizon (int): Prediction horizon length.
        test_batch_size (int): Batch size for the test dataloader.
    Returns:
        dict: A dictionary containing dataloaders for train, val, and test sets.
        StandardScaler: The scaler object fit on the (original) training data.
    """
    # Load traffic data
    traffic_path = os.path.join(dataset_dir, 'PEMS-BAY.csv')
    traffic_df = pd.read_csv(traffic_path, index_col=0)
    traffic_df.index = pd.to_datetime(traffic_df.index)
    traffic_data = traffic_df.values
    timestamps = traffic_df.index.values

    # Load event features
    events_path = os.path.join(dataset_dir, 'event_features.npz')
    event_data = np.load(events_path)['features']

    # Ensure data aligns
    assert traffic_data.shape[0] == event_data.shape[0], "Mismatch in timestamps between traffic and event data."

    # --- NEW SPLIT LOGIC ---
    
    # 1. Get stats for scaler from the *original* 70% train split to prevent leakage
    num_samples = traffic_data.shape[0]
    num_train_orig = int(num_samples * 0.7)
    train_traffic_for_scaler = traffic_data[:num_train_orig]
    
    # 2. Initialize scaler and transform ALL data
    scaler = StandardScaler(mean=train_traffic_for_scaler.mean(), std=train_traffic_for_scaler.std())
    traffic_data_scaled = scaler.transform(traffic_data)

    # 3. Generate sequences from the *entire* dataset
    print("Generating sequences from all data...")
    x_all, y_all = generate_sequences(traffic_data_scaled, seq_length, horizon)
    x_events_all, _ = generate_sequences(event_data, seq_length, horizon)
    
    # Align all timestamps as well (these are the END timestamps of the target window)
    timestamps_all_seq = timestamps[seq_length + horizon - 1:]

    # 4. Create shuffled indices for all sequences
    num_seq_samples = len(x_all)
    indices = np.arange(num_seq_samples)
    # The seed is set in train.py, so this shuffle will be reproducible
    np.random.shuffle(indices) 

    # 5. Split the *shuffled indices* into 70/10/20
    num_train_seq = int(num_seq_samples * 0.7)
    num_val_seq = int(num_seq_samples * 0.1)

    train_indices = indices[:num_train_seq]
    val_indices = indices[num_train_seq : num_train_seq + num_val_seq]
    test_indices = indices[num_train_seq + num_val_seq:]

    # 6. Create final datasets using the shuffled indices
    x_train, y_train = x_all[train_indices], y_all[train_indices]
    x_train_events = x_events_all[train_indices]

    x_val, y_val = x_all[val_indices], y_all[val_indices]
    x_val_events = x_events_all[val_indices]

    x_test, y_test = x_all[test_indices], y_all[test_indices]
    x_test_events = x_events_all[test_indices]
    
    # Get the timestamps corresponding to the test set indices
    test_timestamps_seq = timestamps_all_seq[test_indices]

    # --- END NEW SPLIT LOGIC ---

    # Convert to PyTorch Tensors
    # Shape needs to be (B, T, N, C) for the model
    x_train = torch.from_numpy(x_train).float().unsqueeze(-1)
    y_train = torch.from_numpy(y_train).float().unsqueeze(-1)
    x_train_events = torch.from_numpy(x_train_events).float()

    x_val = torch.from_numpy(x_val).float().unsqueeze(-1)
    y_val = torch.from_numpy(y_val).float().unsqueeze(-1)
    x_val_events = torch.from_numpy(x_val_events).float()

    x_test = torch.from_numpy(x_test).float().unsqueeze(-1)
    y_test = torch.from_numpy(y_test).float().unsqueeze(-1)
    x_test_events = torch.from_numpy(x_test_events).float()

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(x_train, x_train_events, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = TensorDataset(x_val, x_val_events, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    test_dataset = TensorDataset(x_test, x_test_events, y_test)
    # Attach timestamps to the test_dataset for event-specific evaluation
    test_dataset.timestamps = test_timestamps_seq
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    print(f"Data loaded and split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test sequences.")
    
    return dataloaders, scaler

# -----------------------------------------------------------------
# ^^^^^^  THE REST OF THE FILE REMAINS UNCHANGED   ^^^^^^
# -----------------------------------------------------------------

def masked_mae_loss(preds, labels):
    """
    Masked Mean Absolute Error.
    """
    mask = (labels != 0).float()
    mask /= torch.mean(mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss[torch.isnan(loss)] = 0
    return torch.mean(loss)

def masked_rmse_loss(preds, labels):
    """
    Masked Root Mean Squared Error.
    """
    mask = (labels != 0).float()
    mask /= torch.mean(mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss[torch.isnan(loss)] = 0
    return torch.sqrt(torch.mean(loss))

def masked_mape_loss(preds, labels, epsilon=1e-6):
    """
    Masked Mean Absolute Percentage Error.
    """
    mask = (labels != 0).float()
    mask /= torch.mean(mask)
    # Add epsilon to the denominator for numerical stability
    loss = torch.abs((preds - labels) / (labels + epsilon))
    loss = loss * mask
    loss[torch.isnan(loss)] = 0
    return torch.mean(loss)
