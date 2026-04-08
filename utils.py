"""
utils.py
Utility functions for TRIPROMPT.

Includes:
- Deterministic seed setting
- Checkpoint save/load
- PDP mask retrieval (cosine nearest-neighbor, top-k=5)
- Gradient-norm balancing for lambda_1, lambda_2
- Multi-dataset sampling strategy
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed=42):
    """
    Set deterministic random seeds across Python, NumPy, and PyTorch.
    Enables full reproducibility across 3 seeds: 42, 123, 2024.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Utils] Random seed set to {seed}")


# ─────────────────────────────────────────────
# Checkpoint Save / Load
# ─────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, loss, save_path, extra=None):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, save_path)
    print(f"[Checkpoint] Saved epoch {epoch} → {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Load model checkpoint. Allows training to resume exactly from saved epoch,
    including optimizer state.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt.get("epoch", 0)
    loss = ckpt.get("loss", None)
    print(f"[Checkpoint] Loaded from {checkpoint_path} (epoch {epoch}, loss {loss})")
    return epoch, loss


# ─────────────────────────────────────────────
# PDP Mask Retrieval
# ─────────────────────────────────────────────

def retrieve_pdp_masks(query_features, mask_bank_embeddings, class_id, k=5,
                       current_subject_idx=None):
    """
    Retrieve top-k PDP masks for a given class using class-conditional
    cosine nearest-neighbor search.

    Retrieval policy (per Sec. 2.1 of the paper):
    - Mask bank built exclusively from training subjects (frozen before evaluation)
    - Retrieval: cosine similarity between query backbone features and bank embeddings
    - Top-k = 5 masks retrieved per class (stable for k in {1, 3, 5, 10})
    - Cross-subject constraint enforced: current subject excluded (s != i)
    - No ground-truth masks from validation or test images are ever accessed

    Args:
        query_features:         (embed_dim,) or (1, embed_dim) query embedding
        mask_bank_embeddings:   dict {class_id: (N, embed_dim)} or (N, embed_dim) tensor
        class_id:               target anatomical class index
        k:                      number of masks to retrieve (default=5)
        current_subject_idx:    index of current subject to exclude (s != i)

    Returns:
        top_k_embeddings: (k, embed_dim) retrieved mask latent embeddings
        top_k_indices:    list of k indices in the bank
    """
    if isinstance(mask_bank_embeddings, dict):
        bank = mask_bank_embeddings[class_id]
    else:
        bank = mask_bank_embeddings  # (N, embed_dim)

    if bank.shape[0] == 0:
        return torch.zeros(k, query_features.shape[-1]), []

    device = query_features.device
    bank = bank.to(device)

    # Enforce cross-subject constraint (s != i)
    if current_subject_idx is not None:
        valid = [i for i in range(bank.shape[0]) if i != current_subject_idx]
        if len(valid) == 0:
            valid = list(range(bank.shape[0]))
        bank_filtered = bank[valid]
        index_map = valid
    else:
        bank_filtered = bank
        index_map = list(range(bank.shape[0]))

    # Cosine similarity
    q = F.normalize(query_features.view(1, -1), dim=-1)    # (1, D)
    b = F.normalize(bank_filtered, dim=-1)                  # (N', D)
    sims = (q @ b.T).squeeze(0)                             # (N',)

    k_actual = min(k, sims.shape[0])
    top_k_local = torch.topk(sims, k_actual).indices        # (k,)
    top_k_global = [index_map[i.item()] for i in top_k_local]
    top_k_embs = bank[top_k_global]                         # (k, D)

    # Pad to k if fewer than k available
    if k_actual < k:
        pad = top_k_embs[-1:].expand(k - k_actual, -1)
        top_k_embs = torch.cat([top_k_embs, pad], dim=0)

    return top_k_embs, top_k_global


# ─────────────────────────────────────────────
# Gradient-Norm Balancing
# ─────────────────────────────────────────────

def update_loss_weights(lambda_vals, grad_norms_seg, grad_norms_aux):
    """
    Gradient-norm balancing for lambda_1 and lambda_2 (Eq. 10).
    Updated PER TRAINING STEP over shared decoder and query parameters.

    lambda_i <- lambda_i * ||grad(L_seg)|| / ||grad(L_i)||

    Args:
        lambda_vals:     list of current [lambda_1, lambda_2]
        grad_norms_seg:  gradient norm of L_seg w.r.t. shared params
        grad_norms_aux:  list of gradient norms [||grad(L_1)||, ||grad(L_2)||]

    Returns:
        updated lambda_vals: list of updated [lambda_1, lambda_2]
    """
    updated = []
    for i, (lam, gnorm_aux) in enumerate(zip(lambda_vals, grad_norms_aux)):
        if gnorm_aux > 1e-8:
            new_lam = lam * (grad_norms_seg / gnorm_aux)
        else:
            new_lam = lam
        updated.append(new_lam)
    return updated


def compute_grad_norm(loss, parameters):
    """
    Compute gradient norm of a loss w.r.t. given parameters.
    Used for gradient-norm balancing.

    Args:
        loss:       scalar loss tensor
        parameters: iterable of model parameters (shared decoder + query modules)

    Returns:
        grad_norm: float
    """
    grads = torch.autograd.grad(loss, parameters, retain_graph=True,
                                create_graph=False, allow_unused=True)
    grad_norm = 0.0
    for g in grads:
        if g is not None:
            grad_norm += g.norm(2).item() ** 2
    return grad_norm ** 0.5


# ─────────────────────────────────────────────
# Multi-Dataset Sampling
# ─────────────────────────────────────────────

def build_dataset_sampler(dataset_sizes, mode="proportional"):
    """
    Build sampling weights for multi-dataset joint training.
    Mode 'proportional': sample proportional to dataset size.

    Args:
        dataset_sizes: dict {dataset_name: num_samples}
        mode:          'proportional' or 'uniform'

    Returns:
        weights: dict {dataset_name: sampling_weight}
    """
    total = sum(dataset_sizes.values())
    if mode == "proportional":
        weights = {k: v / total for k, v in dataset_sizes.items()}
    elif mode == "uniform":
        n = len(dataset_sizes)
        weights = {k: 1.0 / n for k in dataset_sizes}
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")

    print("[Sampler] Dataset sampling weights:")
    for k, w in weights.items():
        print(f"  {k}: {w:.4f}")
    return weights


# ─────────────────────────────────────────────
# Label Space Harmonization
# ─────────────────────────────────────────────

# Unified 13-class anatomical ontology used across all 11 datasets
UNIFIED_LABEL_MAP = {
    0:  "liver",
    1:  "right_kidney",
    2:  "spleen",
    3:  "pancreas",
    4:  "aorta",
    5:  "inferior_vena_cava",
    6:  "right_adrenal_gland",
    7:  "left_adrenal_gland",
    8:  "gallbladder",
    9:  "esophagus",
    10: "stomach",
    11: "duodenum",
    12: "left_kidney",
}


def get_valid_classes_for_dataset(dataset_name):
    """
    Returns the list of class indices annotated in a given dataset.
    Classes absent in a dataset are masked out during loss computation
    (ignored in Dice and CE losses) to avoid false negatives.

    Args:
        dataset_name: name of the dataset
    Returns:
        valid_classes: list of int class indices present in this dataset
    """
    dataset_class_map = {
        "FLARE22":      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "MSD_liver":    [0],
        "MSD_pancreas": [3],
        "MSD_spleen":   [2],
        "MSD_colon":    [],    # tumor-only, handled separately
        "LiTS":         [0],
        "KiTS19":       [1, 12],
        "AMOS":         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "WORD":         [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12],
        "CT_ORG":       [0, 1, 2, 12],
        "Pancreas_CT":  [3],
        "AbdomenCT1K":  [0, 1, 2, 3],
    }
    return dataset_class_map.get(dataset_name, list(range(13)))
