# Lossfunc

Event/anomaly detection experiments comparing loss functions (BCE, hinge, squared hinge, focal, and weighted variants of each) across three models (LSTM, MLP, TCN, STGNN).

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `torch` to the CPU/CUDA-agnostic version used in development; install a build matching your own CUDA setup if needed (see the comment in that file).

## Running an experiment

```bash
python3 main.py --model LSTM --dataset lossfunc_SWATds.csv --horizon 1 --device cuda:0 --node_cnt 26
```

Key flags (see [config.py](config.py) for the full list):

- `--model`: `LSTM`, `MLP`, `TCN`, or `STGNN`
- `--dataset`: base name (no `data/` prefix, no `.csv` extension) of a file under `data/`
- `--loss_fn`: which loss function to train with (see below)
- `--run_pipeline`: run a fresh experiment grid (default `True`); set to `False` to instead load and re-aggregate a previously saved `results/<model>/results.pickle`
- `--walk_forward`: use walk-forward validation instead of a single train/valid/test split

Each run explores a small hyperparameter grid (defined in `main.py` via `ConfigSpace`; note the epoch choices there are currently hardcoded rather than driven by `--epoch`) and writes:
- `results/<model>/results.pickle` — the raw experiment results
- `results/<model>/runs_summary_*.csv` — a flattened per-run summary (via `utils/utils.py:export_runs_to_csv`)

## Loss functions

Implemented in [losses.py](losses.py) and dispatched by name in `utils/train.py`/`utils/validation.py` via `--loss_fn`:

- `bce`, `pbce` — (weighted) binary cross-entropy variants
- `hinge`, `squared_hinge` — hinge-loss variants, plus weighted (`w...`) forms
- `wbce`, `wpbce`, `whinge`, `wsquared_hinge` — label/positive-weighted variants (accept `--alpha0`/`--alpha1`)
- `focal` — focal loss (`--focal_alpha`, `--focal_gamma`)

## Project layout

```
main.py                  Entry point: builds the ConfigSpace grid and runs the experiment
config.py                Command-line argument definitions
model_managers.py        classificationManager: wires preprocessing/train/validate/test together
losses.py                Loss function implementations
models/                  Model architectures (LSTM, MLP, TCN, STGNN)
utils/                   Training/validation/testing loops, CSV export, and a small
                         vendored experiment-tracking framework (results.py, manager.py,
                         experiment.py, pickl_mixin.py)
user_preprocessing/      Vendored data loading/preprocessing utilities
data/                    Datasets (CSV)
```