import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import *
import torch
import pandas as pd
import numpy as np

def train_model(self, train_loader, valid_loader, scaler, args, result_file, progress_callback=None, model=None, optimizer=None, paramnames=None):
    """
    Trains a graph neural network model.
    If a model and optimizer are passed, it continues training them.
    Otherwise, it creates a new model instance.
    """
    device = args.get("device")
    task = args.get("task")
    print("Loss Function in Train.py:", args.get("loss_fn"))
    if model is None:
        print("Creating new model instance for training.")
        model = self.model
        model.to(device)
    else:
        print("Continuing training on existing model instance.")
        self.model = model

    if optimizer is None:
        if args.get("optimizer") == 'RMSProp':
            optimizer = torch.optim.RMSprop(params=model.parameters(), lr=args.get("lr"), eps=1e-08)
        elif args.get("optimizer") == 'SGD':
            optimizer = torch.optim.SGD(params=model.parameters(), lr=args.get("lr"), weight_decay=args.get("weight_decay"))
        elif args.get("optimizer") == 'Adagrad':
            optimizer = torch.optim.Adagrad(params=model.parameters(), lr=args.get("lr"), weight_decay=args.get("weight_decay"))
        elif args.get("optimizer") == 'Adadelta':
            optimizer = torch.optim.Adadelta(params=model.parameters(), lr=args.get("lr"), weight_decay=args.get("weight_decay"))
        else:
            optimizer = torch.optim.Adam(params=model.parameters(), lr=args.get("lr"), weight_decay=args.get("weight_decay"))

    use_scheduler = (getattr(args, "decay_rate", 1.0) < 0.999) and (getattr(args, "exponential_decay_step", 0) > 0)
    if use_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.get("decay_rate"))

    
    thr = float(args.get("classify_threshold"))
    alpha0 = float(args.get("alpha0"))
    alpha1 = float(args.get("alpha1"))
    if alpha0 <= 0 or alpha1 <= 0:
        raise ValueError("alpha0 and alpha1 must be > 0")
    loss_name = str(args.get("loss_fn")).lower()

    best_validate_score = -np.inf
    validate_score_non_decrease_count = 0

    valid_frame = pd.DataFrame({
        "epoch": pd.Series(dtype=int),
        "loss": pd.Series(dtype=float),
        "acc": pd.Series(dtype=float),
        "precision": pd.Series(dtype=float),
        "recall": pd.Series(dtype=float),
        "f1": pd.Series(dtype=float),
    })
    train_frame = pd.DataFrame({
        "epoch": pd.Series(dtype=int),
        "loss": pd.Series(dtype=float),
        "acc": pd.Series(dtype=float),
        "precision": pd.Series(dtype=float),
        "recall": pd.Series(dtype=float),
        "f1": pd.Series(dtype=float),
    })

    def _classification_step(inputs, target):
        x = torch.as_tensor(inputs, device=device)   # (B,F,N,T)
        y = torch.as_tensor(target, device=device)   # (B,1,N,H)

        logits = self.model(x)                    # (B,1,N,H)

        y_bin = (y > 0.5).float()

        if loss_name == "wbce":
            loss = wbce_logits(logits, y_bin, alpha0=alpha0, alpha1=alpha1, normalize=args.get("wbce_norm"))
        elif loss_name == "owbce":
            loss = onset_wbce_logits(logits, y_bin, alpha0=alpha0, alpha1=alpha1, onset_boost=args.get("onset_boost", 100.0), reduction="weightsum")
        elif loss_name == "bce":
            loss = bce_logits(logits, y_bin, reduction="mean")
        elif loss_name in ("hinge", "hinge_new", "hinge-new"):
            loss = hinge_logits(logits, y_bin, margin=args.get("hinge_margin"), reduction="mean")
        elif loss_name in ("squared_hinge", "squared-hinge", "sq_hinge", "squared_hinge_new", "squared-hinge-new"):
            loss = squared_hinge_logits(logits, y_bin, margin=float(args.get("hinge_margin")), reduction="mean")
        elif loss_name in ("whinge", "whinge_new", "whinge-new"):
            loss = weighted_hinge_logits(logits, y_bin, alpha0=alpha0, alpha1=alpha1, margin=args.get("hinge_margin"), reduction="mean")
        elif loss_name in ("owhinge", "owhinge_new", "owhinge-new"):
            loss = onset_weighted_hinge_logits(logits, y_bin, alpha0=alpha0, alpha1=alpha1, margin=args.get("hinge_margin"), onset_boost=args.get("onset_boost", 100.0), reduction="weightsum")
        elif loss_name in ("wsquared_hinge", "wsquared-hinge", "wsq_hinge", "squared_whinge", "squared-whinge", "sq_whinge", "wsquared_hinge_new", "wsquared-hinge-new"):
            loss = weighted_squared_hinge_logits(logits, y_bin, alpha0=alpha0, alpha1=alpha1, margin=args.get("hinge_margin"), reduction="mean")
        elif loss_name in ("owsquared_hinge", "owsquared_hinge_new", "owsquared-hinge-new"):
            loss = onset_weighted_squared_hinge_logits(logits, y_bin, alpha0=alpha0, alpha1=alpha1, margin=args.get("hinge_margin"), onset_boost=args.get("onset_boost", 100.0), reduction="weightsum")
        elif loss_name == "focal":
            loss = focal_logits(logits, y_bin, gamma=args.get("focal_gamma"), alpha1=args.get("focal_alpha"), reduction="mean")
        elif loss_name == "pbce":
            loss = pbce_logits(logits, y_bin, epsilon=1.0, reduction="mean")
        elif loss_name == "wpbce":
            loss = weighted_pbce_logits(logits, y_bin, epsilon=2.0, alpha0=alpha0, alpha1=alpha1, reduction="mean")
        elif loss_name == "owpbce":
            loss = onset_weighted_pbce_logits(logits, y_bin, epsilon=2.0, alpha0=alpha0, alpha1=alpha1, onset_boost=args.get("onset_boost", 100.0), reduction="weightsum")
        elif loss_name == "dice":
            loss = dice_loss(logits, y_bin, reduction="mean")
        elif loss_name == "wdice":
            loss = weighted_dice_loss(logits, y_bin, alpha0=alpha0, alpha1=alpha1, reduction="mean")
        elif loss_name == "tversky":
            loss = tversky_loss(logits, y_bin, alpha=args.get("tversky_alpha"), beta=args.get("tversky_beta"))
        else:
            raise ValueError(f"Unknown loss_fn='{loss_name}'")

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        return loss, probs, preds, y_bin, logits

    @torch.no_grad()
    def _validate_classification(dataloader):
        if dataloader is None:
            return None
        return self.validate_model(dataloader, device=args.get("device"), args=args)

    for epoch in range(args.get("epoch")):
        self.model.train()
        if hasattr(train_loader, "shuffle"):
            train_loader.shuffle()

        if progress_callback is not None:
            progress_callback((epoch + 1) / args.get("epoch"), message=f"epoch {epoch+1}/{args.get('epoch')}")
        else:
            cb = getattr(self, "_progress_callback", None)
            if cb is not None:
                cb((epoch + 1) / args.get("epoch"), message=f"epoch {epoch+1}/{args.get('epoch')}")

        loss_total = 0.0
        for i, (inputs, target) in enumerate(train_loader.get_iterator()):
            inputs = torch.as_tensor(inputs, device=device)  # (B,F,N,T)
            target = torch.as_tensor(target, device=device)  # (B,1,N,H)

            self.model.zero_grad()
            
            loss, _, _, y_bin, logits = _classification_step(inputs, target)

            loss.backward()
            # if args.get("clip") is not None:
            #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.get("clip"))
            optimizer.step()
            loss_total += loss.item()

        #self.save_model(result_file, epoch)
        # self.save_model(result_file, epoch)

        if use_scheduler and (epoch + 1) % args.get("exponential_decay_step") == 0:
            lr_scheduler.step()

        if (epoch + 1) % args.get("validate_freq") == 0:
            is_best = False
            
            performance_metrics_train = _validate_classification(train_loader)

            performance_metrics = _validate_classification(valid_loader) if valid_loader is not None else None

            entry_train = pd.DataFrame({"epoch": epoch, **performance_metrics_train}, index=[0])
            train_frame = entry_train if train_frame.empty else pd.concat([train_frame, entry_train], ignore_index=True)

            if performance_metrics is not None:
                entry = pd.DataFrame({"epoch": epoch, **performance_metrics}, index=[0])
                valid_frame = entry if valid_frame.empty else pd.concat([valid_frame, entry], ignore_index=True)

                current = performance_metrics["f1"]
                if current > best_validate_score + 1e-12:
                    best_validate_score = current
                    is_best = True
                    validate_score_non_decrease_count = 0
                else:
                    validate_score_non_decrease_count += 1
            else:
                validate_score_non_decrease_count = 0

            if is_best:
                self.save_model(result_file)

        if args.get("early_stop") and valid_loader is not None and validate_score_non_decrease_count >= args.get("early_stop_step") :
            break

    return dict(train=train_frame, valid=valid_frame)
