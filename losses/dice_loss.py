import torch
import torch.nn.functional as F


def dice_loss(
    logits,
    targets,
    smooth: float = 1e-5,
    reduction: str = "mean"
):
    """
    Dice Loss for 3D Multi-Organ and Tumor Segmentation
    (Paper-aligned implementation)

    This implementation follows the formulation used in the TRIPROMPT
    framework and standard medical imaging practice (FLARE, MSD, TMI).

    Key properties:
    ----------------
    - Operates on raw logits (numerically stable)
    - Uses sigmoid activation (multi-label formulation)
    - Supports multi-organ + tumor overlap
    - Robust to severe class imbalance
    - Applicable to all datasets used in the paper

    Args:
        logits (Tensor):
            Raw network outputs of shape (B, C, H, W, D),
            where C corresponds to organs and/or tumor classes.
        targets (Tensor):
            Ground-truth segmentation masks of shape (B, C, H, W, D),
            encoded as binary multi-label maps.
        smooth (float):
            Small constant to avoid division-by-zero.
        reduction (str):
            Reduction over batch and classes.
            Options: "mean", "sum", "none".

    Returns:
        Tensor:
            Scalar Dice loss (if reduction != "none"),
            otherwise per-class Dice loss.
    """

    # ------------------------------
    # 1. Multi-label probability
    # ------------------------------
    # Sigmoid is used instead of softmax because:
    # - Multi-organ + tumor segmentation allows overlapping labels
    # - Stabilizes training under long-tailed class distributions
    probs = torch.sigmoid(logits)

    # ------------------------------
    # 2. Flatten spatial dimensions
    # ------------------------------
    # Shape: (B, C, H*W*D)
    probs = probs.reshape(probs.shape[0], probs.shape[1], -1)
    targets = targets.reshape(targets.shape[0], targets.shape[1], -1)

    # ------------------------------
    # 3. Dice computation (per class)
    # ------------------------------
    intersection = torch.sum(probs * targets, dim=-1)
    cardinality = torch.sum(probs, dim=-1) + torch.sum(targets, dim=-1)

    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)

    # Dice loss
    dice_loss = 1.0 - dice_score  # shape: (B, C)

    # ------------------------------
    # 4. Reduction
    # ------------------------------
    if reduction == "mean":
        return dice_loss.mean()
    elif reduction == "sum":
        return dice_loss.sum()
    elif reduction == "none":
        return dice_loss
    else:
        raise ValueError(
            f"Invalid reduction mode '{reduction}'. "
            f"Expected one of ['mean', 'sum', 'none']."
        )
