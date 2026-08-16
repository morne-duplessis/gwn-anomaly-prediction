import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as torch_data

from user_preprocessing.utils import transform_


class CustomStandardScaler:
    """
    A Z-score normalisation scaler for higher-dimensional inputs
    """

    def __init__(self, mean, std):
        """
        Parameters
        ----------
        mean : float
            Raw data mean
        std : float
            Raw data standard deviation
        """
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        Scales the raw data using the Z-score method

        Parameters
        ----------
        data : pandas.DataFrame
            Data to scale

        Returns
        -------
        pandas.DataFrame
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Performs the inverse operation to return the denormalize the scaled data

        Parameters
        ----------
        data : pandas.DataFrame
            Data to transform

        Returns
        -------
        pandas.DataFrame
        """
        return (data * self.std) + self.mean


class CustomSimpleDataLoader(object):
    """
    A custom data loader suitable for four-dimensional input features. Used by GWN and MTGNN.
    """

    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return wrapper()


class ForecastDataset(torch_data.Dataset):
    """
    Represents a StemGNN and LSTM dataset for training, validation and testing.
    """

    def __init__(self, df, window_size, horizon, normalize_method=None, norm_statistic=None, interval=1):
        self.window_size = window_size
        self.interval = interval
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.norm_statistic = norm_statistic
        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data = df
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()
        if normalize_method:
            self.data = transform_(self.data, normalize_method, norm_statistic)

    def __getitem__(self, index):
        hi = self.x_end_idx[index]
        lo = hi - self.window_size
        train_data = self.data[lo: hi]
        target_data = self.data[hi:hi + self.horizon]
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        return x, y

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx


def load_dataset2(dataset, train_length, valid_length, test_length):
    """
    Performs the inverse operation to return the denormalize the scaled data

    Parameters
    ----------
    dataset : pandas.DataFrame
        Raw dataset to partition
    train_length : int
        Train set relative size
    valid_length : int
        Validation set relative size
    test_length : int
        Test set relative size

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    data_file = os.path.join('data', dataset + '.csv')
    data = pd.read_csv(data_file).values

    train_ratio = train_length / (train_length + valid_length + test_length)
    valid_ratio = valid_length / (train_length + valid_length + test_length)
    train_data = data[:int(train_ratio * len(data))]
    valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
    test_data = data[int((train_ratio + valid_ratio) * len(data)):]
    return train_data, valid_data, test_data

def load_dataset(dataset, train_length, valid_length, test_length):
    """
    Loads dataset from .csv, .npy, or .npz file and splits into train/valid/test.

    If the CSV contains a 'source' column, it is used to define predefined splits:
      - train:  {'train', 'training', 'train_split'}
      - valid:  {'valid', 'validation', 'val', 'valid_split', 'validation_split'}
      - test:   {'test', 'testing', 'test_split'}
    The 'source' column is dropped from the returned arrays.

    Parameters
    ----------
    dataset : str
        Dataset name or filename (without extension if CSV in 'data/' folder).
    train_length : int
        Train set relative size.
    valid_length : int
        Validation set relative size.
    test_length : int
        Test set relative size.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """

    # if no extension, assume CSV inside ./data/
    if not any(dataset.endswith(ext) for ext in [".csv", ".npy", ".npz"]):
        data_file = os.path.join("data", dataset + ".csv")
    else:
        data_file = os.path.join("data", dataset)

    if data_file.endswith(".csv"):
        df = pd.read_csv(data_file)

        # If 'source' column exists, use predefined splits
        if "source" in df.columns:
            src = df["source"].astype(str).str.strip().str.lower()
            train_tokens = {"train", "training", "train_split"}
            valid_tokens = {"valid", "validation", "val", "valid_split", "validation_split"}
            test_tokens  = {"test", "testing", "test_split"}

            train_df = df[src.isin(train_tokens)]
            valid_df = df[src.isin(valid_tokens)]
            test_df  = df[src.isin(test_tokens)]

            # Drop 'source' column before returning values
            train_data = train_df.drop(columns=["source"], errors="ignore").values
            valid_data = valid_df.drop(columns=["source"], errors="ignore").values
            test_data  = test_df.drop(columns=["source"], errors="ignore").values
            return train_data, valid_data, test_data

        # Fallback: no 'source' column -> use ratio-based split
        data = df.values

    elif data_file.endswith(".npy"):
        data = np.load(data_file)
    elif data_file.endswith(".npz"):
        npzfile = np.load(data_file)
        # expect npz to already contain train/valid/test
        if all(k in npzfile for k in ["train", "valid", "test"]):
            return npzfile["train"], npzfile["valid"], npzfile["test"]
        else:
            data = npzfile[npzfile.files[0]]
    else:
        raise ValueError(f"Unsupported file format: {data_file}")

    total = train_length + valid_length + test_length
    train_ratio = train_length / total
    valid_ratio = valid_length / total

    n = len(data)
    train_end = int(train_ratio * n)
    valid_end = int((train_ratio + valid_ratio) * n)

    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    return train_data, valid_data, test_data

def load_dataset_and_labels(dataset, label_file, train_length, valid_length, test_length):
    """
    Loads dataset and corresponding labels (e.g., from CSV) and splits them into train/valid/test.
    Assumes labels align exactly with the dataset rows.
    """
    train_data, valid_data, test_data = load_dataset(dataset, train_length, valid_length, test_length)
    
    if not any(label_file.endswith(ext) for ext in [".csv", ".npy", ".npz"]):
        label_file = os.path.join("data", label_file + ".csv")
    else:
        label_file = os.path.join("data", label_file)

    if not isinstance(label_file, str) or not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")
        
    if label_file.endswith(".csv"):
        labels = pd.read_csv(label_file).values.squeeze()
    elif label_file.endswith(".npy"):
        labels = np.load(label_file).squeeze()
    else:
        raise ValueError(f"Unsupported label file format: {label_file}")

    total_data_len = len(train_data) + len(valid_data) + len(test_data)
    if len(labels) != total_data_len:
         print(f"Warning: Ensure label sequence length ({len(labels)}) matches data sequence length ({total_data_len}).")

    train_end = len(train_data)
    valid_end = train_end + len(valid_data)

    train_labels = labels[:train_end]
    valid_labels = labels[train_end:valid_end]
    test_labels = labels[valid_end:]

    return train_data, valid_data, test_data, train_labels, valid_labels, test_labels