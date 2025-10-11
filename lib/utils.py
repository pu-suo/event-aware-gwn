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

def load_dataset(dataset_dir, batch_size, seq_length=12, horizon=12, test_batch_size=64):
    """
    Loads and prepares the PEMS-BAY dataset, including traffic and event data.
    Args:
        dataset_dir (str): Directory where the data files are located.
        batch_size (int): Batch size for the training and validation dataloaders.
        seq_length (int): History length.
        horizon (int): Prediction horizon length.
        test_batch_size (int): Batch size for the test dataloader.
    Returns:
        dict: A dictionary containing dataloaders for train, val, and test sets.
        StandardScaler: The scaler object fit on the training data.
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

    # Data splitting (70% train, 10% val, 20% test)
    num_samples = traffic_data.shape[0]
    num_train = int(num_samples * 0.7)
    num_val = int(num_samples * 0.1)
    
    train_traffic, train_events = traffic_data[:num_train], event_data[:num_train]
    val_traffic, val_events = traffic_data[num_train:num_train + num_val], event_data[num_train:num_train + num_val]
    test_traffic, test_events = traffic_data[num_train + num_val:], event_data[num_train + num_val:]
    test_timestamps = timestamps[num_train + num_val:]

    # Normalization
    scaler = StandardScaler(mean=train_traffic.mean(), std=train_traffic.std())
    train_traffic_scaled = scaler.transform(train_traffic)
    val_traffic_scaled = scaler.transform(val_traffic)
    test_traffic_scaled = scaler.transform(test_traffic)

    # Generate sequences for each split
    x_train, y_train = generate_sequences(train_traffic_scaled, seq_length, horizon)
    x_train_events, _ = generate_sequences(train_events, seq_length, horizon)
    
    x_val, y_val = generate_sequences(val_traffic_scaled, seq_length, horizon)
    x_val_events, _ = generate_sequences(val_events, seq_length, horizon)

    x_test, y_test = generate_sequences(test_traffic_scaled, seq_length, horizon)
    x_test_events, _ = generate_sequences(test_events, seq_length, horizon)
    
    # Slice timestamps to align with the generated sequences
    test_timestamps_seq = test_timestamps[seq_length + horizon - 1:]

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    val_dataset = TensorDataset(x_val, x_val_events, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    test_dataset = TensorDataset(x_test, x_test_events, y_test)
    # Attach timestamps to the test_dataset for event-specific evaluation
    test_dataset.timestamps = test_timestamps_seq
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    return dataloaders, scaler

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