import numpy as np
from scipy import signal
import torch

def compute_psd(signal_data, fs=64, nperseg=256):
    if torch.is_tensor(signal_data):
        signal_data = signal_data.numpy()
    
    if signal_data.ndim == 1:
        signal_data = signal_data.reshape(1, -1)
    
    psd_list = []
    for sig in signal_data:
        f, psd = signal.welch(sig, fs=fs, nperseg=nperseg)
        psd_list.append(psd)
    
    return f, np.mean(psd_list, axis=0)

def get_weights(psd, lambda_reg=0.001):
    weights = 1.0 / np.sqrt(psd + lambda_reg)
    weights = weights / np.sqrt(np.mean(weights**2))
    return weights

def normalize_signal(x):
    return (x - x.mean()) / x.std()

def split_clients(data, labels, num_clients=10):
    n = len(data) // num_clients
    client_data = []
    client_labels = []
    
    for i in range(num_clients):
        start = i * n
        end = start + n if i < num_clients - 1 else len(data)
        client_data.append(data[start:end])
        client_labels.append(labels[start:end])
    
    return client_data, client_labels
