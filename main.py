from model_managers import classificationManager
from models.lstm import LSTM
from models.mlp import MLP
from models.tcn import TCN
from models.stgnn_classification import STGNN
from models.gwn_classification import GraphWaveNetClassifier
from utils.utils import export_runs_to_csv, create_output_dirs
from config import argparse_setup, makeExperimentConfig

from utils.experiment import ExperimentManager, ExperimentConfigManager
from utils.results import ExperimentResultSet


import argparse
import os
import pandas as pd
import re
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from datetime import datetime

import torch


def main():

    args = argparse_setup()
    save_dir = os.path.join("results", args.model)
    if args.dataset is not None:
        dataset_name_for_file = os.path.basename(args.dataset)  # remove any path
        dataset_name_for_file = os.path.splitext(dataset_name_for_file)[0]  # remove .csv if present
        filename = f"runs_summary_%Y%m%d_%H%M%S_{args.model}_{dataset_name_for_file}.csv"
    else:
        filename = "runs_summary_%Y%m%d_%H%M%S.csv"

    args.csv_path = os.path.join(save_dir, datetime.now().strftime(filename))

    result_train_file = create_output_dirs(args)

    # ===============================
    # ConfigSpace
    # ===============================
    cs = CS.ConfigurationSpace(seed=1234)

    # ---- constants
    dataset_name = CS.Constant(
        "dataset_name",
        args.dataset,
        meta={"config": "preprocess"},
    )

    # epoch as a CategoricalHyperparameter
    epoch_list = [30]
    epoch = CSH.CategoricalHyperparameter("epoch", choices=epoch_list, default_value=epoch_list[0], meta={"config": "train"})

    # ---- losses
    # Available options: wbce, wpbce, whinge, wsquared_hinge, focal, bce, hinge, squared_hinge, pbce
    lossfns = [args.loss_fn,]

    lossfn = CSH.CategoricalHyperparameter(
        "loss_fn",
        choices=lossfns,
        default_value=lossfns[0],
        meta={"config": "preprocess"},
    )

    model_type = CS.Constant("model_type", args.model, meta={"config": "model"})

    # ---- model params
    window_sizes = [24]
    window_size = CSH.CategoricalHyperparameter(
        "window_size",
        choices=window_sizes,
        default_value=window_sizes[0],
        meta={"config": "model"},
    )

    print("Using window size: ", args.window_size)

    # ---------------------------
    # Model-specific configs
    # ---------------------------
    model = args.model.upper()

    if model == "TCN":
        hidden_sizes_layers = [
            [512, 1024],
            [512, 1024, 2048],
            [256, 512, 1024, 2048],
            [64, 128, 256, 512, 1024, 2048],
        ]
        lr_list = [1e-6, 1e-7]


    elif model in ["LSTM", "MLP"]:
        hidden_sizes_layers = [
            [256, 128],
            # [250, 150, 100],
            # [512, 256, 128],
            # [1024, 512, 256, 128],
        ]
        lr_list = [1e-3]
    
    elif model == "STGNN":
        print("Using hidden sizes from argparse: ", args.channels)
        hidden_sizes_layers = [
            # [args.channels,],
            [32],
            [64],
            [128],
            [256],
            [512],
        ]
        lr_list = [1e-3, 1e-4]

    else:
        hidden_sizes_layers = [[1]]
        lr_list = [1e-4]

    print(f"[CONFIG] Model: {model}")
    print(f"[CONFIG] Hidden configs: {hidden_sizes_layers}")
    print(f"[CONFIG] LR configs: {lr_list}")

    # ---------------------------
    # ConfigSpace params
    # ---------------------------
    hidden_sizes = CSH.CategoricalHyperparameter(
        "hidden_sizes",
        choices=hidden_sizes_layers,
        default_value=hidden_sizes_layers[0],
        meta={"config": "model"},
    )

    lr = CSH.CategoricalHyperparameter(
        "lr",
        choices=lr_list,
        default_value=lr_list[0],
        meta={"config": "train"},
    )
    
    # ---- optional params
    alpha1 = CSH.CategoricalHyperparameter(
        "alpha1", choices=[50], meta={"config": "train"}
    )


    focal_alpha = CSH.CategoricalHyperparameter(
        "focal_alpha", choices=[0.1, 0.25, 0.5, 0.9], meta={"config": "train"}
    )
    focal_gamma = CSH.CategoricalHyperparameter(
        "focal_gamma", choices=[3,], meta={"config": "train"}
    )

    # ===============================
    # Add always-present parameters
    # ===============================
    cs.add([
        lr,
        dataset_name,
        model_type,
        epoch,
        lossfn,
        hidden_sizes,
        window_size,
    ])

    # ===============================
    # Conditional parameters
    # ===============================
    from ConfigSpace.conditions import EqualsCondition, InCondition

    weighted_losses = [
        "wbce", "wpbce", "whinge", "wsquared_hinge",
    ]

    active_weighted = [l for l in lossfns if l in weighted_losses]

    if active_weighted:
        cs.add(alpha1)
        cs.add_condition(InCondition(alpha1, lossfn, active_weighted))


    if "focal" in lossfns:
        cs.add([focal_alpha, focal_gamma])
        cs.add_condition(EqualsCondition(focal_alpha, lossfn, "focal"))
        cs.add_condition(EqualsCondition(focal_gamma, lossfn, "focal"))


    # ===============================
    # Grid
    # ===============================
    grid = {
        "lr": len(lr.choices),
        "dataset_name": 1,
        "epoch": len(epoch.choices),
        "model_type": 1,
        "loss_fn": len(lossfn.choices),
        "hidden_sizes": len(hidden_sizes.choices),
        "window_size": len(window_size.choices),
        # "window_size": 1,
    }

    if active_weighted:
        grid["alpha1"] = len(alpha1.choices)


    if "focal" in lossfns:
        grid["focal_alpha"] = len(focal_alpha.choices)
        grid["focal_gamma"] = len(focal_gamma.choices)

    experiment_config = {
        "config_space": cs,
        "grid": grid,
        "runs": 5,
    }

    param_names = sorted({hp.name for hp in cs.get_hyperparameters()} | set(experiment_config.get("grid", {}).keys()))

    # Pipeline and model configuration
    if args.model == 'LSTM':
        print('Training Model: ', args.model)
        pipeline_config = {
            "model": {
                "meta": {
                    "type": LSTM,
                    "manager": classificationManager
                },
                "params": dict(
                    args= vars(args),
                    node_cnt=args.node_cnt,
                    horizon=args.horizon,
                    out_dim=1,
                    dropout=args.dropout_rate,
                    autoregressive=True,
                    layer_norm=True,
                )
            },
            "preprocess": {
                "params": dict(
                    args=vars(args), datafile=args.dataset, result_file=result_train_file
                )
            },
            "train": {
                "params": dict(
                    args=vars(args), result_file=result_train_file, param_names=param_names
                )
            },
            "test": {
                "params": dict(
                    args=vars(args), result_train_file=result_train_file
                )
            }
        }
    elif args.model == 'MLP':
        print('Training Model: ', args.model)
        pipeline_config = {
            "model": {
                "meta": {
                    "type": MLP,
                    "manager": classificationManager
                },
                "params": dict(
                    node_cnt=args.node_cnt,
                    horizon=args.horizon,
                    out_dim=1,
                    dropout=args.dropout_rate,
                    args=vars(args)
                )
            },
            "preprocess": {
                "params": dict(
                    args=vars(args), datafile=args.dataset, result_file=result_train_file
                )
            },
            "train": {
                "params": dict(
                    args=vars(args), result_file=result_train_file, param_names=param_names
                )
            },
            "test": {
                "params": dict(
                    args=vars(args), result_train_file=result_train_file
                )
            }
        }

    elif args.model == 'TCN':
        print('Training Model: ', args.model)
        pipeline_config = {
            "model": {
                "meta": {
                    "type": TCN,
                    "manager": classificationManager
                },
                "params": dict(
                    args=vars(args)
                )
            },
            "preprocess": {
                "params": dict(
                    args=vars(args), datafile=args.dataset, result_file=result_train_file
                )
            },
            "train": {
                "params": dict(
                    args=vars(args), result_file=result_train_file, param_names=param_names
                )
            },
            "test": {
                "params": dict(
                    args=vars(args), result_train_file=result_train_file
                )
            }
        }

    elif args.model == 'STGNN':
        print('Training Model: ', args.model)
        pipeline_config = {
            "model": {
                "meta": {
                    "type": STGNN,
                    "manager": classificationManager
                },
                "params": dict(
                    args=vars(args)
                )
            },
            "preprocess": {
                "params": dict(
                    args=vars(args), datafile=args.dataset, result_file=result_train_file
                )
            },
            "train": {
                "params": dict(
                    args=vars(args), result_file=result_train_file, param_names=param_names
                )
            },
            "test": {
                "params": dict(
                    args=vars(args), result_train_file=result_train_file
                )
            }
        }

    elif args.model == 'GWN':
        print('Training Model: ', args.model)
        pipeline_config = {
            "model": {
                "meta": {
                    "type": GraphWaveNetClassifier,
                    "manager": classificationManager
                },
                "params": dict(device=args.device, node_cnt=args.node_cnt, dropout=args.dropout_rate,
                            gcn_bool=args.gcn_bool, adapt_adj=args.adapt_adj,
                            out_dim=args.horizon, in_dim=args.in_dim,
                            hidden_sizes=args.hidden_sizes,
                            residual_channels=args.channels, dilation_channels=args.channels,
                            skip_channels=args.channels * 8, end_channels=args.channels * 16,
                            num_classes=args.num_classes, classification_type=args.classification_type
                )
            },
            "preprocess": {
                "params": dict(
                    args=vars(args), datafile=args.dataset, result_file=result_train_file
                )
            },
            "train": {
                "params": dict(
                    args=vars(args), result_file=result_train_file, param_names=param_names
                )
            },
            "test": {
                "params": dict(
                    args=vars(args), result_train_file=result_train_file
                )
            }
        }

    print(f"Configuration Space:\n{cs}")

    raw_results = None

    if args.run_pipeline:
        exp_config = ExperimentConfigManager(
            pipeline_config, experiment_config)
        experiment_manager = ExperimentManager(exp_config)

        param_names = sorted({hp.name for hp in cs.get_hyperparameters()} | set(experiment_config.get("grid", {}).keys()))
        raw_results = experiment_manager.run_experiments()

        # Save results
        save_dir = os.path.join("results", args.model)
        raw_results.save_to(save_dir, "results.pickle")

        export_runs_to_csv(
            param_names=param_names,
            csv_path=os.path.join(save_dir, "runs_summary.csv"),
            args=args,
            result_set = raw_results
        )
    else:
        # Load results
        raw_results = ExperimentResultSet.load_from(
            f"results/{args.model}", "results.pickle")

    # Format results
    training_results = {
        label: result.aggregate(group_by="epoch", which=[
                                "train", "valid"], join=True).get_dataframes()
        for label, result in raw_results.get_results().items()
    }

if __name__ == '__main__':
    main()
