import argparse
import os
import torch

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def str2bool(v):
    """
    Converts a string argument to the boolean equivalent

    Parameters
    ----------
    v : Union[bool, str]
        Command line argument value

    Returns
    -------
    bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def argparse_setup():
    """
    Setup command line argument parser

    Returns
    -------
    argparse.Namespace
        A Namespace containing all the command line argument values
    """

    # TODO Organize parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_pipeline', type=str2bool, default=True)
    parser.add_argument('--model', type=str, default='GWN')
    parser.add_argument('--dataset', type=str, default='METR-LA')
    parser.add_argument('--window_size', type=int, default=24)
    parser.add_argument('--train_length', type=float, default=6)
    parser.add_argument('--valid_length', type=float, default=2)
    parser.add_argument('--test_length', type=float, default=2)
    parser.add_argument('--norm_method', type=str, default='z_score')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--early_stop', type=str2bool, default=False)
    parser.add_argument('--node_cnt', type=int, default=None)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[250, 150])

    parser.add_argument('--exponential_decay_step', type=int, default=5)
    parser.add_argument('--validate_freq', type=int, default=1)
    
    # Classification args
    parser.add_argument('--classify_threshold', type=float, default=0.3, help='decision threshold for positive class')
    parser.add_argument('--auto_threshold', type=str2bool, default=False, help='sweep threshold on validation/test to maximize F1')
    parser.add_argument('--classify_eval_h0_only', type=str2bool, default=True, help='evaluate only horizon 0 to avoid duplicate labels')
    parser.add_argument('--onset_weight', type=float, default=1.0, help='extra weight on onset positions in loss (1.0 = off)')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes for classification task')
    parser.add_argument('--pos_weight', type=str, default=None, help='"auto" or a float for BCE positive class weight')
    parser.add_argument('--classify_node_reduce', type=str, default='mean', help='how to reduce node-level to graph-level logits: mean, max, min, sum')
    parser.add_argument('--alpha0', type=float, default=1.0, help='negative-class weight for WBCE')
    parser.add_argument('--alpha1', type=float, default=1.0, help='positive-class weight for WBCE')
    parser.add_argument('--onset_boost', type=float, default=1.0, help='weight for onset positions in loss')
    parser.add_argument('--pre_split', type=str2bool, default=True, help='whether the data is pre-split into train/test sets')
    parser.add_argument('--loss_fn', type=str, default='wbce') #, choices=['wbce','bce','hinge','squared_hinge'])
    parser.add_argument('--wbce_norm', type=str, default='mean', choices=['weightsum','numel','sum','mean'])
    parser.add_argument('--hinge_margin', type=float, default=2.0, help='margin for hinge loss')
    parser.add_argument('--walk_forward', type=str2bool, default=False, help='Enable walk-forward validation.')
    parser.add_argument('--num_folds', type=int, default=3, help='Number of folds for walk-forward validation.')
    parser.add_argument('--walk_forward_window', type=str, default='sliding', choices=['expanding', 'sliding'], help='Type of window for walk-forward validation.')
    parser.add_argument('--fold_valid_size', type=float, default=0.3, help='Proportion (float) for the validation set of each walk-forward fold.')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='gamma parameter for focal loss')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='alpha parameter for focal loss')
    parser.add_argument('--tversky_alpha', type=float, default=0.8, help='alpha parameter for Tversky loss')
    parser.add_argument('--tversky_beta', type=float, default=0.5, help='beta parameter for Tversky loss')


    # GWN arguments
    parser.add_argument('--adj_data', type=str2bool, default=False)
    parser.add_argument('--adj_type', type=str, default='double_transition')
    parser.add_argument('--device', type=str,
                        default=('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--gcn_bool', type=str2bool, default=True)
    parser.add_argument('--horizon', type=int, default=5)
    # Run from t+0
    parser.add_argument('--startAtT', type=str2bool, default=False)
    parser.add_argument('--apt_only', type=str2bool, default=True)
    parser.add_argument('--adapt_adj', type=str2bool, default=True)
    parser.add_argument('--random_adj', type=str2bool, default=True)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--in_dim', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--clip', type=int, default=None, help='value to clip gradient by')
    parser.add_argument('--classification_type', type=str, default='graph', help='graph or node classification')
    parser.add_argument('--csv_path', type=str)

    # User argument configuration
    args = parser.parse_args()

    return args

def makeExperimentConfig(args):
    cs = CS.ConfigurationSpace(seed=1234)

    # ---- constants
    dataset_name = CS.Constant(
        "dataset_name",
        args.dataset,
        meta={"config": "preprocess"},
    )

    # epoch as a CategoricalHyperparameter
    epoch_list = [5, 10]
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
    window_sizes = [6, 12, 24, 36, 48]
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
            [250, 150],
            [250, 150, 100],
        ]
        lr_list = [1e-3]

    elif model == "STGNN":
        print("Using hidden sizes from argparse: ", args.channels)
        hidden_sizes_layers = [
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
    cs = CS.ConfigurationSpace(seed=1234)

    # ---- constants
    dataset_name = CS.Constant(
        "dataset_name",
        args.dataset,
        meta={"config": "train", "config": "preprocess"},
    )
    
    # epoch = CS.Constant("epoch", 10, meta={"config": "train"})

    #epoch as a CategoricalHyperparameter with 5 10 and 20
    epoch_list = [5,10]#, 30]#,20]
    epoch = CSH.CategoricalHyperparameter("epoch", choices=epoch_list, default_value=epoch_list[0], meta={"config": "train"})

    # ---- losses
    # lossfns = ["bce", "hinge", "squared_hinge", "pbce"]

    lossfns = ["lwbce", "wpbce", "whinge", "wsquared_hinge", "focal", "bce", "hinge", "squared_hinge", "pbce"]
    lossfns = [args.loss_fn,]

    lossfn = CSH.CategoricalHyperparameter(
        "loss_fn",
        choices=lossfns,
        default_value=lossfns[0],
        meta={"config": "train", "config": "preprocess"},
    )

    model_type = CS.Constant("model_type", args.model, meta={"config": "model"})

    # ---- model params
    window_sizes = [6, 12, 24, 36, 48]
    # window_sizes = [args.window_size,]
    window_size = CSH.CategoricalHyperparameter(
        "window_size",
        choices=window_sizes,
        default_value=window_sizes[0],
        meta={"config": "preprocess", "config": "model"},
    )

    # Use the value passed via argparse
    # window_size = CSH.Constant(
    #     "window_size",
    #     args.window_size,
    #     meta={"config": "preprocess", "config": "model"},
    # )

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
            [250, 150],
            [250, 150, 100],
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
        "alpha1", choices=[25.0, 30.0, 35.0, 40.0, 45.0, 10,20,50,80,100,], meta={"config": "train"}
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


    return experimentConfig