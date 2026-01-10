import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for full reproducibility.

    This function enforces deterministic behavior across:
    - Python random
    - NumPy
    - PyTorch (CPU & CUDA)

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (important for reviewers)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    save_dir: str,
    filename: str = None
):
    """
    Save training checkpoint.

    Stores:
    - Model weights
    - Optimizer state
    - Epoch index

    Args:
        model (nn.Module): TRIPROMPT model
        optimizer (Optimizer): Optimizer (e.g., AdamW)
        epoch (int): Current epoch index
        save_dir (str): Directory to save checkpoints
        filename (str, optional): Custom checkpoint name
    """
    os.makedirs(save_dir, exist_ok=True)

    if filename is None:
        filename = f"epoch_{epoch + 1}.pth"

    checkpoint_path = os.path.join(save_dir, filename)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    print(f"[INFO] Checkpoint saved at: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    device: torch.device = torch.device("cpu"),
    strict: bool = True,
):
    """
    Load a checkpoint for inference or training resumption.

    Args:
        checkpoint_path (str): Path to checkpoint (.pth)
        model (nn.Module): Initialized TRIPROMPT model
        optimizer (Optimizer, optional): Optimizer to restore state
        device (torch.device): CPU or CUDA device
        strict (bool): Strict state_dict loading

    Returns:
        int: Epoch to resume from
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict
    )
    model.to(device)

    start_epoch = checkpoint.get("epoch", -1) + 1

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(
        f"[INFO] Loaded checkpoint '{checkpoint_path}' "
        f"(resuming from epoch {start_epoch})"
    )

    return start_epoch
