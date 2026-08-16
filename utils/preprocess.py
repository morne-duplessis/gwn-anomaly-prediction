from user_preprocessing.utils import process_adjacency_matrix, process_data
import user_preprocessing

import os
import json
import numpy as np
import pandas as pd

from user_preprocessing.loader import load_dataset

from torch.utils.data import DataLoader, TensorDataset
import user_preprocessing

def create_walk_forward_folds(train_data, num_folds, batch_size, window_type='sliding'):
    """
    Splits the processed training data into multiple walk-forward folds.

    Parameters:
    - train_data (tuple): (x_train, y_train) numpy arrays.
    - num_folds (int): The number of splits.
    - batch_size (int): Batch size for DataLoaders.
    - window_type (str): 'sliding' (default) or 'expanding'.
    """
    x_train, y_train = train_data

    if num_folds <= 0:
        print("num_folds <= 0, yielding full dataset for training with no validation.")
        full_train_loader = user_preprocessing.loader.CustomSimpleDataLoader(x_train, y_train, batch_size)
        yield full_train_loader, None
        return

    x_chunks = np.array_split(x_train, num_folds)
    y_chunks = np.array_split(y_train, num_folds)

    for i in range(num_folds):
        train_x_chunk, train_y_chunk = x_chunks[i], y_chunks[i]
        
        print(f"\n  Fold {i+1}/{num_folds}:")
        print(f"    Training on fold {i+1} ({len(train_x_chunk)} samples)")
        train_loader = user_preprocessing.loader.CustomSimpleDataLoader(train_x_chunk, train_y_chunk, batch_size)

        valid_loader = None
        if i + 1 < num_folds:
            valid_x_chunk, valid_y_chunk = x_chunks[i+1], y_chunks[i+1]
            print(f"    Validating on fold {i+2} ({len(valid_x_chunk)} samples)")
            valid_loader = user_preprocessing.loader.CustomSimpleDataLoader(valid_x_chunk, valid_y_chunk, batch_size)
        else:
            print("    Last fold, no validation set.")

        yield train_loader, valid_loader

def _split_features_labels(arr: np.ndarray, label_idx: int):
    """
    From a possibly mixed-type 2D array:
      - picks label column by index (supports negative)
      - keeps only numeric-convertible feature columns (excludes label)
    Returns:
      X: float32 (T, F_num), y: float32 (T,)
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    T, F_total = arr.shape

    if label_idx < 0:
        label_idx = F_total + label_idx
    if not (0 <= label_idx < F_total):
        raise ValueError(f"label_index {label_idx} out of range for F_total={F_total}")

    y_series = pd.to_numeric(pd.Series(arr[:, label_idx]), errors='coerce').ffill().bfill().fillna(0)
    y = y_series.values.astype(np.float32)

    feats = []
    for j in range(F_total):
        if j == label_idx:
            continue
        s = pd.to_numeric(pd.Series(arr[:, j]), errors='coerce')
        if s.notna().any():
            s = s.ffill().bfill().fillna(0)
            feats.append(s.values.astype(np.float32)[:, None])
    if not feats:
        raise ValueError("No numeric feature columns remain after dropping non-numeric and label columns.")
    X = np.concatenate(feats, axis=1)
    return X, y


def _make_label_windows(args, y_all: np.ndarray, window_size: int, horizon: int, sample_count: int, n_nodes: int = 1):
    """
    Build label windows aligned with process_data's indexing and tile across nodes:
      t runs from window_size .. window_size + sample_count - 1
      y[t : t + horizon] -> shape (H, n_nodes)
    Returns: (S, H, N, 1) float32
    """
    starts = np.arange(window_size, window_size + sample_count, dtype=np.int64)
    S = sample_count
    H = int(horizon)
    N = int(n_nodes)
    y_out = np.zeros((S, H, N, 1), dtype=np.float32)

    if  args.get("startAtT"):
        print("Note: Using startAtT=True, labels start from t (not t+1).")
        
    for i, t in enumerate(starts):
        if not args.get("startAtT", False):
            y_slice = y_all[t:t + H]  # (H,)
        else:
            y_slice = y_all[t - 1:t - 1 + H]  # if you want to predict t+0
        y_out[i, :, :, 0] = np.tile(y_slice[:, None], (1, N))
    return y_out

def preprocess(self, args, datafile, result_file, **kwargs):
    """
    LSTM preprocessing.
    If args.walk_forward is True, it returns a generator of (train_loader, valid_loader) pairs.
    Otherwise, it returns a single (train_loader, None, test_loader).
    """

    # If window_size is passed by the experiment manager (sampled value), use it.
    # Otherwise, fall back to the default in args.
    
    window_size =args.get("window_size")

    print(f"Using window_size={window_size} for LSTM preprocessing.")
    horizon = int(args.get("horizon"))
    label_index = int(args.get("label_index", -1))  # default last col

    train_arr, _, test_arr = load_dataset(
        datafile, args.get("train_length"), 0, args.get("test_length"))

    if len(train_arr) == 0:
        raise Exception('Cannot organize enough training data')

    X_train, y_train_all = _split_features_labels(train_arr, label_index)
    X_test,  y_test_all  = _split_features_labels(test_arr,  label_index)

    # Choose the node axis depending on requested model:
    # - GWN's start_conv is a fixed nn.Conv2d(in_channels=in_dim, ...) with no
    #   reshaping in forward(), so it needs each feature column delivered as its
    #   own node up front: (T, N=features, F=1)
    # - Every other model (LSTM, TCN, MLP, STGNN) expects the whole feature vector
    #   as a single node's input_size: (T, N=1, F=features). STGNN in particular
    #   squeezes out that N=1 axis itself and reinterprets F as its node axis
    #   inside forward() (models/stgnn_classification.py), so it must keep
    #   receiving this layout rather than the GWN one.
    model_name = args.get("model", "").upper()
    if model_name == "GWN":
        X_train_3d = X_train[:, :, None]  # (T, F, 1)
        X_test_3d  = X_test[:, :, None]   # (T, F, 1)
    else:
        X_train_3d = X_train[:, None, :]  # (T, 1, F)
        X_test_3d  = X_test[:, None, :]   # (T, 1, F)
    x_train, _ = process_data(X_train_3d, window_size, horizon)
    x_test,  _ = process_data(X_test_3d,  window_size, horizon)

    n_nodes_train = int(1 if x_train.ndim < 3 else (x_train.shape[2] if x_train.ndim == 4 else 1))
    y_train = _make_label_windows(args, y_train_all, window_size, horizon, sample_count=x_train.shape[0], n_nodes=n_nodes_train)
    y_test  = _make_label_windows(args, y_test_all,  window_size, horizon, sample_count=x_test.shape[0],  n_nodes=n_nodes_train)

    # Explicitly transpose to (B, F, N, T) assuming input is (B, T, N, F)
    x_train = x_train.transpose(0, 3, 2, 1)
    x_test  = x_test.transpose(0, 3, 2, 1)

    # Explicitly transpose to (B, 1, N, H) assuming input is (B, H, N, 1)
    y_train = y_train.transpose(0, 3, 2, 1)
    y_test  = y_test.transpose(0, 3, 2, 1)

    if model_name == "GWN":
        args["in_dim"] = int(x_train.shape[1])    # F: feature-channel count (1 per node)
        args["node_cnt"] = int(x_train.shape[2])  # N: number of nodes
    elif model_name == "LSTM":
        args["node_cnt"] = int(x_train.shape[1])  # F: LSTM treats node_cnt as input_size
    # TCN/MLP/STGNN keep whatever node_cnt/in_dim was supplied via CLI, since they
    # read it directly (STGNN and TCN reinterpret it as their own channel/node axis).

  # Save normalization stats like user_preprocess (use x_train stats)
    if args.get("norm_method") == 'z_score':
        train_mean = np.mean(x_train, axis=0)
        train_std = np.std(x_train, axis=0)
        norm_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.get("norm_method") == 'min_max':
        train_min = np.min(x_train, axis=0)
        train_max = np.max(x_train, axis=0)
        norm_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        norm_statistic = None

    if norm_statistic is not None:
        os.makedirs(result_file, exist_ok=True)
        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
            json.dump(norm_statistic, f)

    # Per-channel standardization: mean/std over samples and time, keep (1,F,N,1) for broadcast
    eps = 1e-6
    ch_mean = x_train.mean(axis=(0, 3), keepdims=True)             # (1, F, N, 1)
    ch_std = x_train.std(axis=(0, 3), keepdims=True) + eps         # (1, F, N, 1)
    x_train_n = (x_train - ch_mean) / ch_std
    x_test_n = (x_test - ch_mean) / ch_std



    class _IdentityScaler:
        def transform(self, x): return x
    train_scaler = _IdentityScaler()

    # If walk_forward is enabled, return a generator of fold loaders
    if args.get('walk_forward', False):
        num_folds = args.get('num_folds', 3)
        batch_size = args.get('batch_size', 32)
        window_type = args.get('walk_forward_window', 'sliding')
        
        fold_generator = create_walk_forward_folds(
            (x_train_n, y_train), num_folds, batch_size, window_type=window_type
        )
        
        test_loader = user_preprocessing.loader.CustomSimpleDataLoader(x_test_n, y_test, batch_size)
        
        return fold_generator, None, test_loader, train_scaler, train_scaler
    
    train_loader = user_preprocessing.loader.CustomSimpleDataLoader(x_train_n, y_train, args.get("batch_size"))
    test_loader  = user_preprocessing.loader.CustomSimpleDataLoader(x_test_n,  y_test,  args.get("batch_size"))
    valid_loader = None

    return train_loader, valid_loader, test_loader, train_scaler, train_scaler