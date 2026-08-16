import torch
from losses import *

@torch.no_grad()
def validate_model(self, data_loader, device,args):
    self.model.eval()

    thr_default = args.get("classify_threshold")
    auto_thr = bool(args.get("auto_threshold")) if args is not None else False
    alpha0 = float(args.get("alpha0")) if args is not None else 1.0
    alpha1 = float(args.get("alpha1")) if args is not None else 1.0
    loss_name = str(args.get("loss_fn")).lower() if args is not None else "wbce"
    hinge_margin = float(args.get("hinge_margin")) if args is not None else 1.0
    pos_weight_arg = args.get("pos_weight") if args is not None else None
    wbce_norm = str((args.get("wbce_norm")).lower()) if args is not None else "mean"
    focal_gamma = float(args.get("focal_gamma")) if args is not None else 2.0
    focal_alpha1 = float(args.get("focal_alpha"))#float(args.get("focal_alpha1")) if args is not None else 0.25
    focal_alpha0 = None#args.get("focal_alpha0") if args is not None else None
    pbce_eps = 1.0#float(args.get("pbce_epsilon")) if args is not None else 1.0

    total_loss, num_batches = 0.0, 0
    all_probs, all_targets = [], []
    h0_probs_list, h0_targets_list = [], []
    H = -1

    def _assert_batch_shapes(x, y):
        assert x.dim() == 4 and y.dim() == 4, f"expected 4D tensors, got {tuple(x.shape)}, {tuple(y.shape)}"
        Bx, Fx, Nx, Tx = x.shape
        By, Cy, Ny, Hy = y.shape
        assert Cy == 1, f"target channel must be 1, got {Cy}"
        assert Bx == By and Nx == Ny, "batch/node mismatch between x and y"

    for bi, (inputs, target) in enumerate(data_loader.get_iterator()):
        x = torch.as_tensor(inputs, device=device, dtype=torch.float32)   # (B,F,N,T)
        y = torch.as_tensor(target, device=device, dtype=torch.float32)   # (B,1,N,H)
        if bi == 0:
            _assert_batch_shapes(x, y)
            H = y.shape[-1]

        logits = self.model(x)                       # (B,1,N,H)
        y_bin = (y > 0.5).float()
        assert logits.shape == y_bin.shape, f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(y_bin.shape)}"

        # compute chosen loss
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

        total_loss += float(loss.item())
        num_batches += 1

        probs = torch.sigmoid(logits)                                     # (B,1,N,H)
        all_probs.append(probs.detach().cpu())
        all_targets.append(y_bin.detach().cpu())

        # collect horizon-0 for onset metrics: (B,1,N,1) -> flatten
        h0_probs = probs[:, :, :, 0].detach().reshape(-1).cpu()
        h0_tgts = y_bin[:, :, :, 0].detach().reshape(-1).cpu()
        h0_probs_list.append(h0_probs)
        h0_targets_list.append(h0_tgts)

    if num_batches == 0:
        return dict(loss=float("nan"), acc=float("nan"), precision=0.0, recall=0.0, f1=0.0,
                    threshold=thr_default, onset_precision=0.0, onset_recall=0.0, onset_f1=0.0, onset_support=0)

    # all_probs is a list of (B, 1, N, H) tensors. Concatenate along batch dim.
    # Resulting shape: (S, 1, N, H) where S is total samples
    probs_all = torch.cat(all_probs, 0)
    targets_all = torch.cat(all_targets, 0)

    # Flatten across all dimensions for overall metrics
    probs_flat = probs_all.reshape(-1)
    t_flat = targets_all.reshape(-1)

    # Base metrics across all horizons/elements
    def metrics_at_threshold(probs_in, targets_in, th):
        p = (probs_in >= th).to(torch.int)
        t = targets_in.to(torch.int)
        tp = int(((p == 1) & (t == 1)).sum().item())
        tn = int(((p == 0) & (t == 0)).sum().item())
        fp = int(((p == 1) & (t == 0)).sum().item())
        fn = int(((p == 0) & (t == 1)).sum().item())
        total = len(p)
        acc = (tp + tn) / max(1, total)
        precision = tp / max(1, (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = tp / max(1, (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        return acc, precision, recall, f1

    chosen_thr = thr_default
    if auto_thr:
        grid = torch.linspace(0.01, 0.99, steps=99).tolist()
        best_f1, best_thr = -1.0, chosen_thr
        for th in grid:
            _, _, _, f1_tmp = metrics_at_threshold(probs_flat, t_flat, th)
            if f1_tmp > best_f1:
                best_f1 = f1_tmp
                best_thr = th
        chosen_thr = best_thr

    acc, precision, recall, f1 = metrics_at_threshold(probs_flat, t_flat, chosen_thr)
    avg_loss = total_loss / max(1, num_batches)

    # --- Per-horizon metrics ---
    results = {}
    for h in range(H):
        # Get predictions and targets for horizon h, then flatten
        probs_h = probs_all[:, :, :, h].reshape(-1)
        targets_h = targets_all[:, :, :, h].reshape(-1)
        h_acc, h_prec, h_rec, h_f1 = metrics_at_threshold(probs_h, targets_h, chosen_thr)
        results[f'h{h}_acc'] = h_acc
        results[f'h{h}_precision'] = h_prec
        results[f'h{h}_recall'] = h_rec
        results[f'h{h}_f1'] = h_f1

    # --- Onset metrics ---
    def onset_metrics(probs_in, targets_in, th):
        """
        Calculates onset metrics. Onset is when target changes from 0 to 1.
        Assumes inputs are (S, N) where S is samples and N is nodes.
        The temporal dependency is along the S dimension.
        """
        if targets_in.numel() < 2:
            return 0.0, 0.0, 0.0, 0

        gt_curr = targets_in[1:, :]
        gt_prev = targets_in[:-1, :]
        onset_gt = ((gt_curr == 1) & (gt_prev == 0))
        support_on = int(onset_gt.sum().item())

        pred_pos = (probs_in >= th).to(torch.int)[1:, :]

        tp_on = int(((pred_pos == 1) & onset_gt).sum().item())
        fp_on = int(((pred_pos == 1) & ~onset_gt).sum().item())
        fn_on = int(((pred_pos == 0) & onset_gt).sum().item())

        precision = tp_on / max(1, (tp_on + fp_on))
        recall = tp_on / max(1, (tp_on + fn_on))
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        return precision, recall, f1, support_on

    # Average onset metrics across all horizons
    # Reshape to (S, N, H) and then average probs over H
    avg_probs_on = probs_all.squeeze(1).mean(dim=-1) # (S, N)
    # For targets, we need to define what an "average" onset means.
    # A common approach is to check for any onset across the horizon.
    # targets_all is (S, 1, N, H). Squeeze to (S, N, H)
    targets_on = targets_all.squeeze(1) # (S, N, H)
    # Any transition from 0 to 1 in the sequence S for any horizon H
    curr_targets_on = targets_on[1:, :, :]
    prev_targets_on = targets_on[:-1, :, :]
    any_onset_gt = ((curr_targets_on == 1) & (prev_targets_on == 0)).any(dim=-1) # (S-1, N)

    # We need to adjust the calculation for the averaged case.
    # Let's use the targets from h=0 for the average onset calculation as a proxy.
    avg_onset_prec, avg_onset_rec, avg_onset_f1, avg_onset_sup = onset_metrics(
        avg_probs_on, targets_on[..., 0], chosen_thr
    )

    # Per-horizon onset metrics
    for h in range(H):
        probs_h = probs_all[:, 0, :, h] # (S, N)
        targets_h = targets_all[:, 0, :, h] # (S, N)
        p, r, f1_h, s = onset_metrics(probs_h, targets_h, chosen_thr)
        results[f'h{h}_onset_prec'] = p
        results[f'h{h}_onset_rec'] = r
        results[f'h{h}_onset_f1'] = f1_h
        results[f'h{h}_onset_support'] = s


    # Onset metrics at horizon 0 (positive if y_t=1 and y_{t-1}=0)
    # This is now calculated and stored in results['h0_onset_...']
    # For backward compatibility, we can retrieve it for the base results dict.
    onset_precision = results.get('h0_onset_prec', 0.0)
    onset_recall = results.get('h0_onset_rec', 0.0)
    onset_f1 = results.get('h0_onset_f1', 0.0)
    onset_support = results.get('h0_onset_support', 0)


    base_results = dict(
        loss=avg_loss,
        acc=acc, precision=precision, recall=recall, f1=f1, threshold=chosen_thr,
        avg_onset_prec=avg_onset_prec, avg_onset_rec=avg_onset_rec, avg_onset_f1=avg_onset_f1, avg_onset_support=avg_onset_sup
    )
    base_results.update(results)
    return base_results