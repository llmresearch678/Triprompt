import torch
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1e-5):
    """
    Dice loss for 3D medical image segmentation.

    Args:
        pred (Tensor): Raw model outputs (logits) of shape
                       (B, C, H, W, D)
        target (Tensor): Ground truth masks of shape
                         (B, C, H, W, D)
        smooth (float): Smoothing factor to avoid division by zero

    Returns:
        Tensor: Scalar Dice loss
    """

    # Apply sigmoid for multi-label segmentation
    pred = torch.sigmoid(pred)

    # Flatten spatial dimensions
    pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
    target = target.contiguous().view(target.shape[0], target.shape[1], -1)

    # Compute Dice score per class
    intersection = (pred * target).sum(dim=2)
    union = pred.sum(dim=2) + target.sum(dim=2)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Dice loss
    loss = 1.0 - dice

    # Average over classes and batch
    return loss.mean()
