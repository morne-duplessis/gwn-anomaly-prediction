import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import linalg
import os

def transform_(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-5
        data = (data - norm_statistic['min']) / scale
        data = np.clip(data, 0.0, 1.0)
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = (data - mean) / std
        norm_statistic['std'] = std
    return data


def inverse_transform_(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-8
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data

# def get_node_count_from_data(datafile):
#     # data_file = os.path.join('data', datafile + '.csv')
#     data_file = os.path.join('data', datafile + '')
#     data = pd.read_csv(data_file).values
#     return data.shape[1]

import os
import numpy as np
import pandas as pd

def get_node_count_from_data(datafile: str):
    """
    Infer the number of nodes from a dataset file (csv, npy, or npz).

    Parameters
    ----------
    datafile : str
        Filename of dataset (with extension), e.g. 'pems04.csv', 'pems04.npy', 'pems04.npz'

    Returns
    -------
    int
        Number of nodes (second dimension in array).
    """
    data_path = os.path.join("data", datafile)

    if datafile.endswith(".csv"):
        data = pd.read_csv(data_path).values
    elif datafile.endswith(".npy"):
        data = np.load(data_path)          # shape (time, nodes, features)
    elif datafile.endswith(".npz"):
        loaded = np.load(data_path)
        # assume key 'data' or first array inside
        if "data" in loaded:
            data = loaded["data"]
        else:
            data = loaded[list(loaded.keys())[0]]
    else:
        raise ValueError(f"Unsupported file format: {datafile}")

    # expect shape (time, nodes, features) or (time, nodes)
    if data.ndim == 3:
        return data.shape[1]
    elif data.ndim == 2:
        return data.shape[1]
    else:
        raise ValueError(f"Unexpected data shape {data.shape} for {datafile}")


def correlation_adjacency_matrix2(dataset):
    return pd.read_csv(dataset).corr().to_numpy()

def correlation_adjacency_matrix(dataset):
    """
    Builds correlation adjacency matrix from a dataset file.
    Supports CSV, NPY, and NPZ files.
    """
    if dataset.endswith(".csv"):
        df = pd.read_csv(dataset)
        return df.corr().to_numpy()

    elif dataset.endswith(".npy"):
        data = np.load(dataset)  # shape (T, N, F)
        # Flatten time dimension before computing correlations
        T, N, F = data.shape
        reshaped = data.reshape(T, N * F)
        df = pd.DataFrame(reshaped)
        return df.corr().to_numpy()

    elif dataset.endswith(".npz"):
        npzfile = np.load(dataset)
        # assume main array is first entry
        data = npzfile[npzfile.files[0]]
        T, N, F = data.shape
        reshaped = data.reshape(T, N * F)
        df = pd.DataFrame(reshaped)
        return df.corr().to_numpy()

    else:
        raise ValueError(f"Unsupported dataset format for adjacency: {dataset}")


def symmetric_adjacency(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asymmetric_adjacency(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(row_sum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L = calculate_normalized_laplacian(adj)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    identity = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - identity
    return L.astype(np.float32).todense()


def process_data2(data, window_size, horizon):
    """
    Transforms a two-dimensional input (N x T) into a four-dimensional dataset,
    where N is the number of nodes and T is the steps.

    Yaguang Li, Rose Yu, Cyrus Shahabi, and Yan Liu. 2018.
    Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.

    Parameters
    ----------
    data : numpy.ndarray
        Input dataset
    window_size : int
        Input sequence length
    horizon : int
        Output sequence length

    Returns
    -------
    numpy.ndarray
    """
    x_offsets = np.sort(np.concatenate((np.arange(-(window_size - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, (horizon + 1), 1))
    samples, nodes = data.shape[0], data.shape[1]
    data = np.expand_dims(data, axis=-1)
    data = np.concatenate([data], axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

def process_data(data, window_size, horizon):
    """
    Converts raw traffic data into sequences for Graph WaveNet.

    Parameters
    ----------
    data : np.ndarray
        Input data, shape (time_steps, num_nodes, num_features)
        If single-feature, can be (time_steps, num_nodes)
    window_size : int
        Number of past timesteps used as input
    horizon : int
        Number of future timesteps to predict

    Returns
    -------
    x : np.ndarray
        Input sequences, shape (num_samples, num_features, num_nodes, window_size)
    y : np.ndarray
        Target sequences, shape (num_samples, horizon, num_nodes, num_features)
    """
    # Ensure 3D: (time, nodes, features)
    if data.ndim == 2:
        data = np.expand_dims(data, axis=-1)

    T, N, F = data.shape

    x_offsets = np.arange(-(window_size - 1), 1, 1)
    y_offsets = np.arange(1, horizon + 1)

    x_list, y_list = [], []

    min_t = abs(x_offsets[0])
    max_t = T - abs(y_offsets[-1])

    for t in range(min_t, max_t):
        x_seq = data[t + x_offsets, :, :]  # (window_size, N, F)
        y_seq = data[t + y_offsets, :, :]  # (horizon, N, F)

        x_list.append(x_seq)
        y_list.append(y_seq)

    x = np.stack(x_list, axis=0) 
    y = np.stack(y_list, axis=0)

    # [batch, features, nodes, window_size] for GWN
    x = x.transpose(0, 1, 2, 3)
    return x, y


# def process_data(data, window_size, horizon, num_features=3):
#     """
#     Works with multiple features per node.
#     data shape: (timesteps, num_nodes * num_features)
#     """
#     print("Raw data shape:", data.shape)        # (timesteps, total_features)
#     print("Total elements:", data.size)

#     num_features = 3
#     num_nodes = data.shape[1] // num_features
#     print("Assumed num_nodes:", num_nodes)
#     print("Expected elements:", data.shape[0] * num_nodes * num_features)
#     num_nodes = data.shape[1] // num_features
#     # data = data.reshape(data.shape[0], num_nodes, num_features)

#     x_offsets = np.arange(-(window_size - 1), 1, 1)
#     y_offsets = np.arange(1, horizon + 1, 1)

#     x, y = [], []
#     min_t = abs(min(x_offsets))
#     max_t = data.shape[0] - abs(max(y_offsets))
#     for t in range(min_t, max_t):
#         x.append(data[t + x_offsets, ...])
#         y.append(data[t + y_offsets, ...])
#     x = np.stack(x, axis=0)  # (num_samples, window_size, num_nodes, num_features)
#     y = np.stack(y, axis=0)  # (num_samples, horizon, num_nodes, num_features)

#     return x, y



def process_adjacency_matrix(adj_data, adj_type):
    """
    Preprocesses a Graph WaveNet adjacency matrix

    Parameters
    ----------
    adj_data : str
        File containing adjacency matrix data
    adj_type : str
        Adjacency matrix transformation type

    Returns
    -------
    [numpy.ndarray]
    """
    adj = correlation_adjacency_matrix(adj_data)
    if adj_type == "scaled_laplacian":
        adj = [calculate_scaled_laplacian(adj)]
    elif adj_type == "normalized_laplacian":
        adj = [calculate_normalized_laplacian(adj).astype(np.float32).todense()]
    elif adj_type == "symmetric_adjacency" or adj_type == "transition":
        adj = [symmetric_adjacency(adj)]
    elif adj_type == "double_transition":
        adj = [asymmetric_adjacency(adj), asymmetric_adjacency(np.transpose(adj))]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error
    return adj


def process_classification_data(data, labels, window_size, horizon):
    """
    Transforms time-series data into windows and assigns classification labels.

    Parameters
    ----------
    data : numpy.ndarray
        Input data, shape (time_steps, ...)
    labels : numpy.ndarray
        Array of class labels corresponding to each step in `data`, shape (time_steps,)
    window_size : int
        Number of past timesteps used as input
    horizon : int
        Steps ahead to predict the class for

    Returns
    -------
    x : numpy.ndarray
        Input sequences, shape (num_samples, num_features, num_nodes, window_size)
    y : numpy.ndarray
        Target labels, shape (num_samples,)
    """
    T = data.shape[0]

    x_offsets = np.arange(-(window_size - 1), 1, 1)
    y_offsets = np.arange(1, horizon + 1)
    
    x_list, y_list = [], []

    min_t = abs(x_offsets[0])
    max_t = T - abs(y_offsets[-1])

    for t in range(min_t, max_t):
        x_seq = data[t + x_offsets, ...]
        
        label_seq = labels[t + y_offsets] 
        
        x_list.append(x_seq)
        y_list.append(label_seq)

    if len(x_list) == 0:
        raise ValueError("No valid samples found. Check your data and label dimensions.")
        
    x = np.stack(x_list, axis=0)  
    y = np.stack(y_list, axis=0)
    
    return x, y