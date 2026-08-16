import torch
import torch.nn.functional as F


def _assert_same_shape(logits, targets):
    assert logits.shape == targets.shape, f"loss shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}"


def bce_logits(logits, targets, reduction="mean", weight=None, pos_weight=None):
    _assert_same_shape(logits, targets)

    """
    Standard Binary Cross Entropy with logits.
    Thin wrapper around torch.nn.functional.binary_cross_entropy_with_logits.

    Args:
      logits: raw model outputs
      targets: binary targets {0,1}, same shape as logits
      reduction: 'none' | 'mean' | 'sum'
      weight: optional per-element weight tensor (broadcastable to logits)
      pos_weight: optional scalar or tensor for positive class weighting

    Returns:
      Scalar loss (if reduction != 'none') or per-element loss tensor.
    """
    return F.binary_cross_entropy_with_logits(
        logits, targets, weight=weight, pos_weight=pos_weight, reduction=reduction
    )


def wbce_logits(logits, targets, alpha0=1.0, alpha1=1.0, normalize="mean", element_weight=None):
    _assert_same_shape(logits, targets)

    """
    Label-Weighted BCE with logits: (1/X) * sum_i a_c * BCE_i, a_c = a1 if y=1 else a0.
    Optionally multiply by an extra per-element weight mask (element_weight).
    """
    loss_elts = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    a0 = torch.as_tensor(alpha0, device=logits.device, dtype=loss_elts.dtype)
    a1 = torch.as_tensor(alpha1, device=logits.device, dtype=loss_elts.dtype)
    class_w = torch.where(targets > 0, a1, a0)
    if element_weight is not None:
        ew = torch.as_tensor(element_weight, device=logits.device, dtype=loss_elts.dtype)
        weighted = loss_elts * class_w * ew
        denom = class_w * ew
    else:
        weighted = loss_elts * class_w
        denom = class_w

    return weighted.mean()


def onset_wbce_logits(
    logits,
    targets,
    alpha0=1.0,
    alpha1=1.0,
    onset_boost=100.0,
    reduction="weightsum",
    element_weight=None,
):
    _assert_same_shape(logits, targets)

    """
    Onset Weighted BCE with logits.

    Original:
        BCE = BCEWithLogits

    Added:
        onset weighting for anomaly start positions
        (first 1 after a 0)

    Final:
        L = onset_weight * class_weight * BCE

    Args:
        logits: raw model logits
        targets: binary labels
        alpha0: negative class weight
        alpha1: positive class weight
        onset_boost: extra weight for onset positions
        reduction: 'none' | 'mean' | 'sum' | 'weightsum'
        element_weight: optional per-element multiplier
    """

    _assert_same_shape(logits, targets)

    targets = targets.float()

    # ---------------------------------------------------------
    # BCE
    # ---------------------------------------------------------
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    # ---------------------------------------------------------
    # Class weighting
    # ---------------------------------------------------------
    a0 = torch.as_tensor(
        alpha0,
        device=logits.device,
        dtype=losses.dtype
    )

    a1 = torch.as_tensor(
        alpha1,
        device=logits.device,
        dtype=losses.dtype
    )

    class_w = torch.where(targets > 0, a1, a0)

    # ---------------------------------------------------------
    # Onset detection
    # onset = current == 1 AND previous == 0
    # ---------------------------------------------------------
    prev = torch.zeros_like(targets)

    # shift targets right
    prev[:, 1:] = targets[:, :-1]

    onset_mask = (
        (targets > 0) &
        (prev == 0)
    ).float()

    # onset multiplier
    onset_w = 1.0 + onset_boost * onset_mask

    total_w = class_w * onset_w

    # Optional external element weights
    if element_weight is not None:
        ew = torch.as_tensor(
            element_weight,
            device=logits.device,
            dtype=losses.dtype
        )

        total_w = total_w * ew

    weighted = losses * total_w

    if reduction == "sum":
        return weighted.sum()

    elif reduction == "weightsum":
        d = torch.clamp(total_w.sum(), min=1.0)
        return weighted.sum() / d

    elif reduction == "mean":
        return weighted.mean()

    elif reduction == "none":
        return weighted

    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

def pbce_logits(logits, targets, epsilon=1.0, reduction="mean", element_weight=None):
    _assert_same_shape(logits, targets)
    """
    Poly BCE (PolyLoss for BCE):
      CE = BCEWithLogits
      p = sigmoid(logits); p_t = p     if y=1 else 1-p
      PBCE = CE + e * (1 - p_t)

    Args:
      epsilon: e coefficient for the poly term
      reduction: 'none' | 'mean' | 'sum' | 'weightsum'
      element_weight: optional per-element multiplier
    """
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = torch.where(targets > 0, p, 1.0 - p)
    poly = (1.0 - p_t)
    loss_elts = ce + float(epsilon) * poly

    if element_weight is not None:
        ew = torch.as_tensor(element_weight, device=logits.device, dtype=loss_elts.dtype)
        loss_elts = loss_elts * ew
        denom = ew
    else:
        denom = None

    if reduction == "mean":
        return loss_elts.mean()


def weighted_pbce_logits(logits, targets, epsilon=1.0, alpha0=1.0, alpha1=1.0, reduction="mean", element_weight=None):
    _assert_same_shape(logits, targets)
    """
    Weighted Poly BCE (PolyLoss for BCE):
      CE = BCEWithLogits
      p = sigmoid(logits); p_t = p if y=1 else 1-p
      PBCE = CE + e * (1 - p_t)
      L = ? a_c * PBCE ?

    Args:
      epsilon: e coefficient for the poly term
      alpha0: weight for negative class
      alpha1: weight for positive class
      reduction: 'none' | 'mean' | 'sum' | 'weightsum'
      element_weight: optional per-element multiplier
    """
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = torch.where(targets > 0, p, 1.0 - p)
    poly = (1.0 - p_t)
    loss_elts = ce + float(epsilon) * poly

    a0 = torch.as_tensor(alpha0, device=logits.device, dtype=loss_elts.dtype)
    a1 = torch.as_tensor(alpha1, device=logits.device, dtype=loss_elts.dtype)
    class_w = torch.where(targets > 0, a1, a0)

    if element_weight is not None:
        ew = torch.as_tensor(element_weight, device=logits.device, dtype=loss_elts.dtype)
        weighted = loss_elts * class_w * ew
        denom_w = class_w * ew
    else:
        weighted = loss_elts * class_w
        denom_w = class_w

    if reduction == "mean":
        return weighted.mean()

def onset_weighted_pbce_logits(
    logits,
    targets,
    epsilon=1.0,
    alpha0=1.0,
    alpha1=1.0,
    onset_boost=100.0,
    reduction="weightsum",
    element_weight=None,
):
    _assert_same_shape(logits, targets)

    """
    Onset Weighted Poly BCE (PBCE) with logits.

    Original:
        CE = BCEWithLogits
        p_t = p if y=1 else 1-p
        PBCE = CE + e * (1 - p_t)

    Added:
        onset weighting for anomaly start positions
        (first 1 after a 0)

    Args:
        logits: raw model logits
        targets: binary labels
        epsilon: poly loss coefficient
        alpha0: negative class weight
        alpha1: positive class weight
        onset_boost: extra weight for onset positions
        reduction: 'none' | 'mean' | 'sum' | 'weightsum'
        element_weight: optional per-element multiplier
    """

    targets = targets.float()

    ce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    p = torch.sigmoid(logits)

    p_t = torch.where(
        targets > 0,
        p,
        1.0 - p
    )

    poly = (1.0 - p_t)

    losses = ce + float(epsilon) * poly

    a0 = torch.as_tensor(
        alpha0,
        device=logits.device,
        dtype=losses.dtype
    )

    a1 = torch.as_tensor(
        alpha1,
        device=logits.device,
        dtype=losses.dtype
    )

    class_w = torch.where(targets > 0, a1, a0)

    prev = torch.zeros_like(targets)

    # shift targets right
    prev[:, 1:] = targets[:, :-1]

    onset_mask = (
        (targets > 0) &
        (prev == 0)
    ).float()

    onset_w = 1.0 + onset_boost * onset_mask

    total_w = class_w * onset_w

    # Optional external element weights
    if element_weight is not None:
        ew = torch.as_tensor(
            element_weight,
            device=logits.device,
            dtype=losses.dtype
        )

        total_w = total_w * ew

    weighted = losses * total_w

    if reduction == "sum":
        return weighted.sum()

    elif reduction == "weightsum":
        d = torch.clamp(total_w.sum(), min=1.0)
        return weighted.sum() / d

    elif reduction == "mean":
        return weighted.mean()

    elif reduction == "none":
        return weighted

    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

def focal_logits(logits, targets, alpha1, gamma=2.0, alpha0=None, reduction="mean", element_weight=None):
    _assert_same_shape(logits, targets)
    """
    Binary Focal Loss (with logits) from RetinaNet:
      FL = - a_t (1 - p_t)^? log(p_t)
    where p_t = s(logit) if y=1 else 1-s(logit).
    Implemented as modulation of BCEWithLogits for numerical stability.

    Args:
      gamma: focusing parameter
      alpha1: positive-class weight (default 0.25).
      alpha0: negative-class weight (default 1 - alpha1 if alpha1 in [0,1], else 1.0).
      reduction: 'none' | 'mean' | 'sum' | 'weightsum'
      element_weight: optional per-element weight multiplier
    """
    # Per-element BCE
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # p = sigmoid(logits); p_t = p for y=1 else 1-p
    p = torch.sigmoid(logits)
    p_t = torch.where(targets > 0, p, 1.0 - p)

    # Alpha for classes
    if alpha0 is None:
        if 0.0 <= float(alpha1) <= 1.0:
            alpha0_t = 1.0 - float(alpha1)
        else:
            alpha0_t = 1.0
    else:
        alpha0_t = float(alpha0)
    a1 = torch.as_tensor(alpha1, device=logits.device, dtype=bce.dtype)
    a0 = torch.as_tensor(alpha0_t, device=logits.device, dtype=bce.dtype)
    alpha_t = torch.where(targets > 0, a1, a0)

    # Modulation
    mod = (1.0 - p_t).clamp(min=1e-6).pow(gamma)
    loss_elts = alpha_t * mod * bce

    if element_weight is not None:
        ew = torch.as_tensor(element_weight, device=logits.device, dtype=bce.dtype)
        loss_elts = loss_elts * ew
        denom = (alpha_t * ew)
    else:
        denom = alpha_t

    if reduction == "mean":
        return loss_elts.mean()
    else:
        return loss_elts


def hinge_logits(logits, targets, margin=1.0, reduction="mean"):
    _assert_same_shape(logits, targets)

    """
    Binary hinge loss on raw logits:
      loss_i = max(0, margin - y_i * f_i)
    where y_i in {-1, +1} and f_i are the raw logits (no sigmoid squashing).

    Args:
      logits: raw scores, arbitrary real values
      targets: binary targets in {0,1}, same shape as logits
      margin: hinge margin
      reduction: 'none' | 'mean' | 'sum'

    Returns:
      Scalar loss (if reduction != 'none') or per-element loss tensor.
    """

    y_pm = torch.where(targets > 0, 1.0, -1.0).to(
        dtype=logits.dtype, device=logits.device
    )

    losses = F.relu(margin - y_pm * logits)

    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "none":
        return losses
    else:
        return losses

def weighted_hinge_logits(
    logits,
    targets,
    margin,
    alpha0=1.0,
    alpha1=1.0,
    reduction="mean",
    element_weight=None
):
    _assert_same_shape(logits, targets)

    """
    Class-weighted hinge loss on raw logits:
      L = sum_i a_c * max(0, m - y_i * f_i)  (reduced per `reduction`)
    where y_i in {-1, +1}, f_i are raw logits, a_c is the class weight.

    reduction:
      - "mean": (1/n) sum a_c * hinge
      - "sum":  sum a_c * hinge
      - "weightsum": sum a_c * hinge / sum a_c (scale-invariant)
    """

    # Convert labels to {-1, +1}
    y_pm = torch.where(
        targets > 0,
        1.0,
        -1.0
    ).to(dtype=logits.dtype, device=logits.device)

    # Raw-logit hinge
    violation = margin - y_pm * logits
    losses = torch.clamp(violation, min=0.0)

    # Class weights
    a0 = torch.as_tensor(alpha0, device=logits.device, dtype=losses.dtype)
    a1 = torch.as_tensor(alpha1, device=logits.device, dtype=losses.dtype)
    class_w = torch.where(targets > 0, a1, a0)

    if element_weight is not None:
        ew = torch.as_tensor(
            element_weight,
            device=logits.device,
            dtype=losses.dtype
        )

        weighted = losses * class_w * ew
        denom_w = class_w * ew
    else:
        weighted = losses * class_w
        denom_w = class_w

    if reduction == "sum":
        return weighted.sum()

    elif reduction == "weightsum":
        d = torch.clamp(denom_w.sum(), min=1.0)
        return weighted.sum() / d

    elif reduction == "mean":
        return weighted.mean()

    else:
        return weighted

def onset_weighted_hinge_logits(
    logits,
    targets,
    margin=1.0,
    alpha0=1.0,
    alpha1=1.0,
    onset_boost=100.0,
    reduction="weightsum",
    element_weight=None,
):
    _assert_same_shape(logits, targets)

    """
    Onset Weighted Hinge Loss on raw logits.

    Original:
        L = a_c * max(0, m - y * f)

    where:
        f = raw logits
        y in {-1, +1}

    Added:
        onset weighting for anomaly start positions
        (first 1 after a 0)

    Args:
        logits: raw model logits
        targets: binary labels
        margin: hinge margin
        alpha0: negative class weight
        alpha1: positive class weight
        onset_boost: extra weight for onset positions
        reduction: 'none' | 'mean' | 'sum' | 'weightsum'
        element_weight: optional per-element multiplier
    """

    # ---------------------------------------------------------
    # Labels -> {-1, +1}
    # ---------------------------------------------------------
    y_pm = torch.where(
        targets > 0,
        1.0,
        -1.0
    ).to(dtype=logits.dtype, device=logits.device)

    # ---------------------------------------------------------
    # Raw-logit hinge
    # ---------------------------------------------------------
    violation = margin - y_pm * logits
    losses = torch.clamp(violation, min=0.0)

    # ---------------------------------------------------------
    # Class weighting
    # ---------------------------------------------------------
    a0 = torch.as_tensor(
        alpha0,
        device=logits.device,
        dtype=losses.dtype
    )

    a1 = torch.as_tensor(
        alpha1,
        device=logits.device,
        dtype=losses.dtype
    )

    class_w = torch.where(
        targets > 0,
        a1,
        a0
    )

    # ---------------------------------------------------------
    # Onset detection
    # ---------------------------------------------------------
    prev = torch.zeros_like(targets)
    prev[:, 1:] = targets[:, :-1]

    onset_mask = (
        (targets > 0) &
        (prev == 0)
    ).float()

    onset_w = 1.0 + onset_boost * onset_mask

    # ---------------------------------------------------------
    # Total weight
    # ---------------------------------------------------------
    total_w = class_w * onset_w

    if element_weight is not None:
        ew = torch.as_tensor(
            element_weight,
            device=logits.device,
            dtype=losses.dtype
        )
        total_w = total_w * ew

    # ---------------------------------------------------------
    # Apply weighting
    # ---------------------------------------------------------
    weighted = losses * total_w

    # ---------------------------------------------------------
    # Reduction
    # ---------------------------------------------------------
    if reduction == "sum":
        return weighted.sum()

    elif reduction == "weightsum":
        d = torch.clamp(total_w.sum(), min=1.0)
        return weighted.sum() / d

    elif reduction == "mean":
        return weighted.mean()

    elif reduction == "none":
        return weighted

    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

def squared_hinge_logits(logits, targets, margin=1.0, reduction="mean"):
    _assert_same_shape(logits, targets)

    """
    Binary squared hinge loss on raw logits:
      loss_i = max(0, margin - y_i * f_i)^2
    where y_i in {-1, +1} and f_i are the raw logits (no sigmoid squashing).

    Args:
      logits: raw scores
      targets: binary targets in {0,1}, same shape as logits
      margin: hinge margin
      reduction: 'none' | 'mean' | 'sum'

    Returns:
      Scalar loss (if reduction != 'none') or per-element loss tensor.
    """

    y_pm = torch.where(targets > 0, 1.0, -1.0).to(
        dtype=logits.dtype, device=logits.device
    )

    losses = F.relu(margin - y_pm * logits).pow(2)

    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "none":
        return losses
    else:
        return losses

def weighted_squared_hinge_logits(
    logits,
    targets,
    margin=1.0,
    alpha0=1.0,
    alpha1=1.0,
    reduction="mean",
    element_weight=None
):
    _assert_same_shape(logits, targets)

    """
    Class-weighted squared hinge loss on raw logits:
      L = sum_i a_c * max(0, m - y_i * f_i)^2  (reduced per `reduction`)
    Same reductions as weighted_hinge_logits.
    """

    # Convert labels to {-1, +1} exactly like TF
    y_pm = torch.where(targets > 0, 1.0, -1.0).to(dtype=logits.dtype, device=logits.device)

    # RAW logits (this is the key fix vs the old 2*sigmoid(z)-1 version)
    violation = margin - y_pm * logits
    losses = torch.clamp(violation, min=0.0).pow(2)

    # Class weights (same logic as TF)
    a0 = torch.as_tensor(alpha0, device=logits.device, dtype=losses.dtype)
    a1 = torch.as_tensor(alpha1, device=logits.device, dtype=losses.dtype)
    class_w = torch.where(targets > 0, a1, a0)

    # Element-wise weighting (unchanged extension)
    if element_weight is not None:
        ew = torch.as_tensor(element_weight, device=logits.device, dtype=losses.dtype)
        weighted = losses * class_w * ew
        denom_w = class_w * ew
    else:
        weighted = losses * class_w
        denom_w = class_w

    # Reductions
    if reduction == "sum":
        return weighted.sum()

    elif reduction == "weightsum":
        d = torch.clamp(denom_w.sum(), min=1.0)
        return weighted.sum() / d

    elif reduction == "mean":
        return weighted.mean()

    else:
        return weighted

def onset_weighted_squared_hinge_logits(
    logits,
    targets,
    margin=1.0,
    alpha0=1.0,
    alpha1=1.0,
    onset_boost=100.0,
    reduction="weightsum",
    element_weight=None,
):
    _assert_same_shape(logits, targets)

    """
    Onset Weighted Squared Hinge Loss on raw logits.

    Original:
        L = a_c * max(0, m - y * f)^2

    where:
        f = raw logits
        y in {-1, +1}

    Added:
        onset weighting for anomaly start positions
        (first 1 after a 0)

    Args:
        logits: raw model logits
        targets: binary labels
        margin: hinge margin
        alpha0: negative class weight
        alpha1: positive class weight
        onset_boost: extra weight for onset positions
        reduction: 'none' | 'mean' | 'sum' | 'weightsum'
        element_weight: optional per-element multiplier
    """

    # ---------------------------------------------------------
    # Labels -> {-1, +1}
    # ---------------------------------------------------------
    y_pm = torch.where(
        targets > 0,
        1.0,
        -1.0
    ).to(dtype=logits.dtype, device=logits.device)

    # ---------------------------------------------------------
    # Raw-logit squared hinge
    # ---------------------------------------------------------
    violation = margin - y_pm * logits
    losses = torch.clamp(
        violation,
        min=0.0
    ).pow(2)

    # ---------------------------------------------------------
    # Class weighting
    # ---------------------------------------------------------
    a0 = torch.as_tensor(
        alpha0,
        device=logits.device,
        dtype=losses.dtype
    )

    a1 = torch.as_tensor(
        alpha1,
        device=logits.device,
        dtype=losses.dtype
    )

    class_w = torch.where(
        targets > 0,
        a1,
        a0
    )

    # ---------------------------------------------------------
    # Onset detection
    # onset = current == 1 AND previous == 0
    # ---------------------------------------------------------
    prev = torch.zeros_like(targets)
    prev[:, 1:] = targets[:, :-1]

    onset_mask = (
        (targets > 0) &
        (prev == 0)
    ).float()

    onset_w = 1.0 + onset_boost * onset_mask

    # ---------------------------------------------------------
    # Total weight
    # ---------------------------------------------------------
    total_w = class_w * onset_w

    if element_weight is not None:
        ew = torch.as_tensor(
            element_weight,
            device=logits.device,
            dtype=losses.dtype
        )
        total_w = total_w * ew

    # ---------------------------------------------------------
    # Apply weighting
    # ---------------------------------------------------------
    weighted = losses * total_w

    # ---------------------------------------------------------
    # Reduction
    # ---------------------------------------------------------
    if reduction == "sum":
        return weighted.sum()

    elif reduction == "weightsum":
        d = torch.clamp(total_w.sum(), min=1.0)
        return weighted.sum() / d

    elif reduction == "mean":
        return weighted.mean()

    elif reduction == "none":
        return weighted

    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def dice_loss(logits, targets, reduction="mean", smooth=1.0, eps=1e-6):
    _assert_same_shape(logits, targets)

    """
    Global binary soft Dice loss with logits.

    Computes one Dice loss over the full batch/time tensor.

        p = sigmoid(logits)
        Dice = (2 * sum(p * y) + smooth) / (sum(p) + sum(y) + smooth + eps)
        Loss = 1 - Dice

    Args:
        logits: raw model outputs, same shape as targets
        targets: binary labels {0,1}, same shape as logits
        reduction: kept for API compatibility; all modes return the same scalar
        smooth: smoothing constant for empty/near-empty positive regions
        eps: small numerical constant to avoid division instability

    Returns:
        Scalar Dice loss.
    """

    targets = targets.float()
    p = torch.sigmoid(logits)

    intersection = (p * targets).sum()
    union = p.sum() + targets.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth + eps)
    loss = 1.0 - dice

    if reduction in ["mean", "weightsum", "sum", "none"]:
        return loss
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def tversky_loss(logits, targets, alpha=0.3, beta=0.7, reduction="mean", smooth=1.0, eps=1e-6):
    _assert_same_shape(logits, targets)

    """
    Global binary soft Tversky loss with logits.

        p = sigmoid(logits)
        TP = sum(p * y)
        FP = sum(p * (1 - y))
        FN = sum((1 - p) * y)

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth + eps)
        Loss = 1 - Tversky

    Args:
        logits: raw model outputs, same shape as targets
        targets: binary labels {0,1}, same shape as logits
        alpha: false-positive penalty
        beta: false-negative penalty; beta > alpha favours recall
        reduction: kept for API compatibility; all modes return the same scalar
        smooth: smoothing constant for empty/near-empty positive regions
        eps: small numerical constant to avoid division instability

    Returns:
        Scalar Tversky loss.
    """

    targets = targets.float()
    p = torch.sigmoid(logits)

    tp = (p * targets).sum()
    fp = (p * (1.0 - targets)).sum()
    fn = ((1.0 - p) * targets).sum()

    tversky_index = (tp + smooth) / (
        tp + float(alpha) * fp + float(beta) * fn + smooth + eps
    )

    loss = 1.0 - tversky_index

    if reduction in ["mean", "weightsum", "sum", "none"]:
        return loss
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def weighted_dice_loss(
    logits,
    targets,
    alpha0=1.0,
    alpha1=1.0,
    reduction="mean",
    smooth=1.0,
    eps=1e-6,
    element_weight=None,
):
    _assert_same_shape(logits, targets)

    """
    Class-weighted global binary soft Dice loss with logits.

    Weighting is applied inside the Dice overlap terms:

        Dice = (2 * sum(w_i * p_i * y_i) + smooth)
               /
               (sum(w_i * p_i) + sum(w_i * y_i) + smooth + eps)

    where:
        w_i = alpha1 if y_i = 1
              alpha0 if y_i = 0

    Args:
        logits: raw model outputs, same shape as targets
        targets: binary labels {0,1}, same shape as logits
        alpha0: negative/normal class weight
        alpha1: positive/anomaly class weight
        reduction: kept for API compatibility; all modes return the same scalar
        smooth: smoothing constant for empty/near-empty positive regions
        eps: small numerical constant
        element_weight: optional per-element multiplier

    Returns:
        Scalar weighted Dice loss.
    """

    targets = targets.float()
    p = torch.sigmoid(logits)

    a0 = torch.as_tensor(alpha0, device=logits.device, dtype=p.dtype)
    a1 = torch.as_tensor(alpha1, device=logits.device, dtype=p.dtype)

    class_w = torch.where(targets > 0, a1, a0)

    if element_weight is not None:
        ew = torch.as_tensor(element_weight, device=logits.device, dtype=p.dtype)
        total_w = class_w * ew
    else:
        total_w = class_w

    intersection = (total_w * p * targets).sum()
    union = (total_w * p).sum() + (total_w * targets).sum()

    dice = (2.0 * intersection + smooth) / (union + smooth + eps)
    loss = 1.0 - dice

    if reduction in ["mean", "weightsum", "sum", "none"]:
        return loss
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")