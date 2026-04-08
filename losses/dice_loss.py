"""
losses/dice_loss.py
Dice Loss for TRIPROMPT 3D Multi-Organ and Tumor Segmentation.

Paper-aligned implementation following formulation in FLARE, MSD, TMI.

Key properties:
    - Operates on raw logits (numerically stable)
    - Uses sigmoid activation (multi-label formulation — NOT softmax)
    - Supports multi-organ + tumor overlap
    - Robust to severe class imbalance
    - Valid class masking: absent classes ignored during multi-dataset training
      to avoid false negatives across heterogeneous label spaces
"""

import torch
import torch.nn.functional as F


def dice_loss(
    logits,
    targets,
    smooth: float = 1e-5,
    reduction: str = "mean",
    valid_classes=None,
    valid_mask=None,
):
    """
    Dice Loss for 3D multi-organ and tumor segmentation.

    Args:
        logits (Tensor):
            Raw network outputs (B, C, H, W, D).
        targets (Tensor):
            Binary multi-label ground truth (B, C, H, W, D).
        smooth (float):
            Small constant to avoid division-by-zero (default=1e-5).
        reduction (str):
            'mean', 'sum', or 'none'.
        valid_classes (list of int, optional):
            List of class indices annotated in this dataset.
            If provided, loss is computed only over these classes.
            Other classes are masked out (no false negatives in
            multi-dataset joint training).
            Example: FLARE22 → [0,1,2,...,12], Pancreas-CT → [3]
        valid_mask (Tensor, optional):
            (B, C) or (C,) bool tensor — True where class is annotated.
            Takes precedence over valid_classes if both are provided.
            Returned by CTDataset per sample for full flexibility.

    Returns:
        Tensor: scalar Dice loss (or per-class (B, C) if reduction='none')
    """

    # ── 1. Multi-label probabilities (sigmoid, not softmax) ───────────────
    # Each class is treated independently — supports overlapping labels
    probs = torch.sigmoid(logits)              # (B, C, H, W, D)

    # ── 2. Flatten spatial dimensions ────────────────────────────────────
    B, C = probs.shape[0], probs.shape[1]
    probs   = probs.reshape(B, C, -1)          # (B, C, HWD)
    targets = targets.reshape(B, C, -1)        # (B, C, HWD)

    # ── 3. Dice per class ─────────────────────────────────────────────────
    intersection = (probs * targets).sum(dim=-1)           # (B, C)
    cardinality  = probs.sum(dim=-1) + targets.sum(dim=-1) # (B, C)
    dice_score   = (2.0 * intersection + smooth) / (cardinality + smooth)
    dice         = 1.0 - dice_score                        # (B, C)

    # ── 4. Valid class masking ────────────────────────────────────────────
    # Absent classes are zeroed out so they contribute nothing to the loss.
    # This prevents false negatives when training across datasets with
    # heterogeneous label spaces (e.g., KiTS19 has kidney but not pancreas).
    if valid_mask is not None:
        # valid_mask: (B, C) or (C,) bool tensor from CTDataset
        vm = valid_mask.to(logits.device)
        if vm.dim() == 1:
            vm = vm.unsqueeze(0).expand(B, -1)   # (B, C)
        vm = vm.bool().float()                    # 1 where valid, 0 where absent
        dice = dice * vm                          # zero out absent classes

        if reduction == "mean":
            # Average only over valid (annotated) classes
            n_valid = vm.sum().clamp(min=1)
            return dice.sum() / n_valid
        elif reduction == "sum":
            return dice.sum()
        elif reduction == "none":
            return dice
        else:
            raise ValueError(f"Invalid reduction '{reduction}'.")

    elif valid_classes is not None:
        # valid_classes: list of int indices annotated in this dataset
        mask = torch.zeros(C, device=logits.device)
        mask[valid_classes] = 1.0
        mask = mask.unsqueeze(0).expand(B, -1)   # (B, C)
        dice = dice * mask

        if reduction == "mean":
            n_valid = mask.sum().clamp(min=1)
            return dice.sum() / n_valid
        elif reduction == "sum":
            return dice.sum()
        elif reduction == "none":
            return dice
        else:
            raise ValueError(f"Invalid reduction '{reduction}'.")

    # ── 5. Standard reduction (no masking) ───────────────────────────────
    if reduction == "mean":
        return dice.mean()
    elif reduction == "sum":
        return dice.sum()
    elif reduction == "none":
        return dice
    else:
        raise ValueError(
            f"Invalid reduction mode '{reduction}'. "
            f"Expected one of ['mean', 'sum', 'none']."
        )
