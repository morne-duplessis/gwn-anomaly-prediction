import os
import pandas as pd
import re
from datetime import datetime
from utils.results import ExperimentResultSet

def append_run_to_csv(result, config, param_names=None):
    """
    Appends a single run's result to a CSV file immediately after completion.
    Creates the file and header if it doesn't exist.
    """
    args = config.get_training_params().get('args', {})
    csv_path = args.get('csv_path')

    metrics = result.get_dataframes()["test"].iloc[-1].to_dict()

    for p in param_names:
        if metrics.get(p) is None:
            metrics[p] = args.get(p)

    metrics["loss_fn"] = args.get("loss_fn")
    metrics["alpha1"] = args.get("alpha1")

    for key, value in metrics.items():
        if isinstance(value, list):
            metrics[key] = str(value)

    df_row = pd.DataFrame(metrics, index=[0])
    header = not os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode='a', header=header, index=False)
    print(f"--> Appended run results to {csv_path}")


def create_output_dirs(args):
    result_train_file = os.path.join('output', args.model, args.dataset, str(
        args.window_size), str(args.horizon), 'train')
    baseline_train_file = os.path.join('output', 'lstm', args.dataset, str(
        args.window_size), str(args.horizon), 'train')
    if not os.path.exists(result_train_file):
        os.makedirs(result_train_file)
    if not os.path.exists(baseline_train_file):
        os.makedirs(baseline_train_file)

    return result_train_file

class ModelFileNotFoundError(FileNotFoundError):
    """
    Error raised when model file cannot be found
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor for ModelFileNotFoundError
        """
        super().__init__(*args, **kwargs)

def export_runs_to_csv(param_names, csv_path, args=None, result_set=None):
    """
    Build a CSV where each row is a run:
      - hyperparameters from param_names
      - test split metrics only (no train/valid)
    Loads results from the results.pickle in the same directory as csv_path.
    """

    # Ensure loss_fn is always included in the output
    if param_names is None:
        param_names = []
    else:
        param_names = list(param_names)
        
    if "loss_fn" not in param_names:
        param_names.append("loss_fn")
    if "alpha1" not in param_names:
        param_names.append("alpha1")
    if "onset_boost" not in param_names:
        param_names.append("onset_boost")

    base_dir = os.path.dirname(csv_path)
    os.makedirs(base_dir, exist_ok=True)

    if result_set is None:
        try:
            result_set = ExperimentResultSet.load_from(base_dir, "results.pickle")
        except Exception as e:
            print(f"Warning: failed to load results from {base_dir}/results.pickle: {e}")
            return

    def _try_extract_cfg(label, result):
        cfg = {}
        # Check standard attributes
        for attr in ("config", "configuration", "params", "hyperparameters"):
            if hasattr(result, attr):
                val = getattr(result, attr)
                if isinstance(val, dict):
                    cfg = val
                    break
                try:
                    cfg = dict(val)
                    break
                except Exception:
                    pass
        # Fallback: parse from label
        if not cfg and param_names:
            print("Falling back to regex parsing of label for hyperparameters.")
            for p in param_names:
                if p in cfg:
                    continue
                m = re.search(rf"{p}\s*[:=@]\s*([0-9.eE+\-\[\],_ ]+|\w+)", str(label))
                if m:
                    v = m.group(1)
                    # Try to evaluate list/numeric
                    try:
                        v_eval = eval(v)
                        cfg[p] = v_eval
                    except Exception:
                        cfg[p] = v
        # Convert list/tuple hyperparameters to string
        for k, v in cfg.items():
            if isinstance(v, (list, tuple)):
                # flatten single-element tuple
                if isinstance(v, tuple) and len(v) == 1 and isinstance(v[0], list):
                    v = v[0]
                cfg[k] = "_".join(map(str, v))
        return cfg

    def _is_numeric(x):
        return isinstance(x, (int, float)) or (hasattr(x, "item") and isinstance(x.item(), (int, float)))

    rows = []
    results_dict = result_set.get_results() if hasattr(result_set, "get_results") else {}

    for label, result in results_dict.items():
        cfg_full = _try_extract_cfg(label, result)
        cfg = {k: cfg_full.get(k, None) for k in (param_names or [])}

        # Fallback: if loss_fn is missing in config, grab it from args
        if cfg.get("loss_fn") is None and args is not None and hasattr(args, "loss_fn"):
            cfg["loss_fn"] = args.loss_fn
        if cfg.get("alpha1") is None and args is not None and hasattr(args, "alpha1"):
            cfg["alpha1"] = args.alpha1
        if cfg.get("onset_boost") is None and args is not None and hasattr(args, "onset_boost"):
            cfg["onset_boost"] = args.onset_boost
        # if cfg.get("dataset_name") is None and args is not None:
        #     cfg["dataset_name"] = args.dataset

        frames = result.get_dataframes() if hasattr(result, "get_dataframes") else {}
        df = frames.get("test")
        if df is None or len(df) == 0:
            continue

        run_ids = set()
        if isinstance(df.index, pd.MultiIndex) and ("run" in (df.index.names or [])):
            run_ids |= set(df.index.get_level_values("run").unique().tolist())
        if not run_ids:
            run_ids = {None}

        for run_id in sorted(run_ids, key=lambda x: str(x)):
            sub = df
            if isinstance(df.index, pd.MultiIndex) and ("run" in (df.index.names or [])) and run_id is not None:
                try:
                    sub = df.xs(run_id, level="run")
                except Exception:
                    continue
            if len(sub) == 0:
                continue
            if "epoch" in sub.columns:
                sub = sub.sort_values("epoch")
            last_row = sub.iloc[-1]

            epoch_value = cfg.get("epoch", args.epoch)

            metrics = {}
            for col, val in last_row.items():
                if col in ("epoch", "step", "iteration"):
                    continue
                if _is_numeric(val):
                    try:
                        metrics[col] = float(val if isinstance(val, (int, float)) else val.item())
                    except Exception:
                        pass

            row_label = f"{label}/run={run_id}" if run_id is not None else str(label)
            metrics["run"] = run_id
            # metrics["epoch"] = getattr(result, "epoch", args.epoch)
            metrics["epoch"] = epoch_value
            rows.append({"label": row_label, **cfg, **metrics})

    all_param_set = set(param_names or [])
    all_metric_cols = sorted({k for r in rows for k in r.keys()} - all_param_set - {"label"})
    ordered_cols = ["label"] + list(param_names or []) + all_metric_cols

    df_out = pd.DataFrame(rows)
    for c in ordered_cols:
        if c not in df_out.columns:
            df_out[c] = None
    df_out = df_out[ordered_cols]

    if args.dataset is not None: 
        dataset_name_for_file = os.path.basename(args.dataset)  # remove any path
        dataset_name_for_file = os.path.splitext(dataset_name_for_file)[0]  # remove .csv if present
        filename = f"runs_summary_%Y%m%d_%H%M%S_{args.model}_{dataset_name_for_file}.csv"
    else: 
        filename = "runs_summary_%Y%m%d_%H%M%S.csv"

    csv_path = os.path.join(base_dir, datetime.now().strftime(filename))

    # csv_path = os.path.join(base_dir, datetime.now().strftime("runs_summary_%Y%m%d_%H%M%S.csv"))

    df_out.to_csv(csv_path, index=False)
    print(f"Saved test run summary CSV to: {csv_path}")