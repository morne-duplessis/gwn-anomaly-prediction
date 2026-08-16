import os
import numpy as np
import pandas as pd


def to_long_df(array, name, in_dim=1, horizon=None):
    """
    Convert (n_samples, n_nodes, horizon) or (n_samples, n_nodes, horizon, features)
    into a long dataframe: time, sensor_id, horizon, feature, value.
    Handles flattened last dim = horizon*in_dim if provided.
    """
    arr = np.asarray(array)
    if arr.ndim == 3:
        n_samples, n_nodes, last = arr.shape
        if horizon is not None and in_dim > 1 and last == horizon * in_dim:
            arr4 = arr.reshape(n_samples, n_nodes, horizon, in_dim)
            n_features = in_dim
        else:
            horizon = last
            arr4 = arr.reshape(n_samples, n_nodes, horizon, 1)
            n_features = 1
    elif arr.ndim == 4:
        n_samples, n_nodes, horizon, n_features = arr.shape
        arr4 = arr
    else:
        raise ValueError(f"Unsupported shape {arr.shape}")

    times = np.repeat(np.arange(n_samples), n_nodes * horizon * n_features)
    sensor_ids = np.tile(np.repeat(np.arange(n_nodes), horizon * n_features), n_samples)
    horizons = np.tile(np.repeat(np.arange(horizon), n_features), n_samples * n_nodes)
    features = np.tile(np.arange(n_features), n_samples * n_nodes * horizon)
    values = arr4.reshape(-1)

    return pd.DataFrame({
        "time": times,
        "sensor_id": sensor_ids,
        "horizon": horizons,
        "feature": features,
        name: values
    })


def test_lstm_model(self, test_loader, scaler, args, result_train_file):
    """
    Test routine specialized for LSTM baseline (no adjacency export).
    Expects manager providing:
        - has_model()
        - load_model(path)
        - validate_model(...)
    """
    if not self.has_model():
        self.load_model(result_train_file)

    # Run validation in test mode
    performance_metrics, y_pred, y_true = self.validate_model(
        test_loader,
        args.get("device"),
        args.get("norm_method"),
        args.get("horizon"),
        scaler=scaler,
        return_preds=True
    )

    # Assemble results
    test_frame = pd.DataFrame([performance_metrics])
    results = {
        "test": test_frame,
        "y_pred": to_long_df(
            y_pred,
            "y_pred",
            in_dim=int(args.get("in_dim")),
            horizon=int(args.get("horizon"))
        ),
        "y_true": to_long_df(
            y_true,
            "y_true",
            in_dim=int(args.get("in_dim")),
            horizon=int(args.get("horizon"))
        )
    }

    mae, mape, rmse = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse']
    print(f"LSTM Test Performance: MAPE: {mape*100:5.2f} | MAE: {mae:5.2f} | RMSE: {rmse:5.2f}")
    return results