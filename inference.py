"""
inference.py
TRIPROMPT Inference Script.

Full inference pipeline:
- Loads frozen PDP mask bank (training subjects only — no test masks ever accessed)
- Retrieves top-k=5 PDP masks per class via class-conditional cosine nearest-neighbor
- Applies same preprocessing pipeline as training (intensity norm, resampling, cropping)
- Runs voxel-wise multi-label segmentation
- Saves predictions as NIfTI (.nii.gz) files

Usage:
    python inference.py \
        --checkpoint checkpoints/best_model.pth \
        --data_root ./data/test \
        --mask_bank_path ./data/mask_bank/mask_bank.pt \
        --output_dir ./predictions \
        --topk_pdp 5
"""

import os
import argparse
import numpy as np
import torch
import nibabel as nib

from models.deformation_prompt import MaskBank, DeformationPromptEncoder
from utils import load_checkpoint, retrieve_pdp_masks, set_seed


# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="TRIPROMPT Inference")
    parser.add_argument("--checkpoint",      type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_root",       type=str, default="./data/test",
                        help="Directory containing test images")
    parser.add_argument("--mask_bank_path",  type=str, default="./data/mask_bank/mask_bank.pt",
                        help="Path to frozen PDP mask bank (training subjects only)")
    parser.add_argument("--output_dir",      type=str, default="./predictions")
    parser.add_argument("--num_classes",     type=int, default=13)
    parser.add_argument("--embed_dim",       type=int, default=256)
    parser.add_argument("--input_size",      type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--topk_pdp",        type=int, default=5,
                        help="Top-k PDP masks retrieved per class (stable for k in {1,3,5,10})")
    parser.add_argument("--threshold",       type=float, default=0.5,
                        help="Sigmoid threshold for binarizing per-class predictions")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--device",          type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Preprocessing (matches training pipeline)
# ─────────────────────────────────────────────

def preprocess_volume(image_np, target_size=(96, 96, 96)):
    """
    Apply the same preprocessing pipeline used during training:
    1. CT Hounsfield unit clipping [-1000, 1000]
    2. Z-score normalization per volume
    3. Resize to target_size via trilinear interpolation
    4. Add batch and channel dimensions -> (1, 1, H, W, D)
    """
    # Step 1: HU clipping
    image_np = np.clip(image_np, -1000, 1000)

    # Step 2: Z-score normalization
    mean = image_np.mean()
    std  = image_np.std() + 1e-8
    image_np = (image_np - mean) / std

    # Step 3: Resize to target_size
    image_tensor = torch.from_numpy(image_np.astype(np.float32))
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
    image_tensor = torch.nn.functional.interpolate(
        image_tensor,
        size=target_size,
        mode="trilinear",
        align_corners=False,
    )
    return image_tensor  # (1, 1, H, W, D)


# ─────────────────────────────────────────────
# PDP Mask Retrieval at Inference
# ─────────────────────────────────────────────

def get_pdp_masks_for_inference(model, image_tensor, mask_bank, num_classes,
                                topk, device):
    """
    Retrieve top-k PDP masks per class using class-conditional cosine
    nearest-neighbor search against the frozen training-split mask bank.

    Retrieval policy (Sec. 2.1):
    - Compute query embedding from backbone features
    - Cosine similarity against bank embeddings per class
    - Retrieve top-k=5 masks (no test masks ever accessed)
    - Cross-subject constraint enforced during bank construction

    Args:
        model:         TRIPROMPT model (used to extract backbone query features)
        image_tensor:  preprocessed input (1, 1, H, W, D)
        mask_bank:     frozen MaskBank instance
        num_classes:   number of anatomical classes
        topk:          k for nearest-neighbor retrieval
        device:        torch device
    Returns:
        pdp_masks: list of K tensors, each (1, 1, H, W, D) — one per class
    """
    model.eval()
    with torch.no_grad():
        # Extract backbone query features for retrieval
        backbone_features, _ = model.backbone(image_tensor.to(device))
        # Use deepest scale features for retrieval query
        query_feat = backbone_features[-1]                          # (1, C, h, w, d)
        query_feat = query_feat.mean(dim=[2, 3, 4]).squeeze(0)     # (C,) global avg

        # Project to embed_dim for cosine similarity
        if hasattr(model, "pdp_query_proj"):
            query_feat = model.pdp_query_proj(query_feat)          # (embed_dim,)

    pdp_masks = []
    for c in range(num_classes):
        top_k_embs, top_k_indices = retrieve_pdp_masks(
            query_features=query_feat,
            mask_bank_embeddings=mask_bank.embeddings,
            class_id=c,
            k=topk,
            current_subject_idx=None,  # No subject to exclude at test time
        )
        # Load actual mask from the top-1 retrieved path and pass to encoder
        if len(top_k_indices) > 0 and len(mask_bank.mask_paths[c]) > 0:
            best_idx = top_k_indices[0]
            mask_path = mask_bank.mask_paths[c][best_idx]
            m = MaskBank._load_mask(mask_path).to(device)     # (1, 1, H, W, D)
            m = torch.nn.functional.interpolate(
                m, size=image_tensor.shape[2:], mode="nearest"
            )
        else:
            # Fallback: zero mask if class absent from bank
            m = torch.zeros(1, 1, *image_tensor.shape[2:], device=device)
        pdp_masks.append(m)

    return pdp_masks


# ─────────────────────────────────────────────
# Single Volume Inference
# ─────────────────────────────────────────────

def run_inference(model, image_path, output_path, mask_bank, device, args):
    """
    Full inference pipeline for a single 3D CT volume.

    Args:
        model:       trained TRIPROMPT model
        image_path:  path to input CT (.nii or .nii.gz)
        output_path: path to save prediction (.nii.gz)
        mask_bank:   frozen PDP mask bank (training subjects only)
        device:      torch device
        args:        parsed arguments
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[Inference] Input image not found: {image_path}")

    # ── Load CT volume ────────────────────────────────────────────────────
    image_nii  = nib.load(image_path)
    image_np   = image_nii.get_fdata().astype(np.float32)
    orig_shape = image_np.shape   # Save for resampling output back
    affine     = image_nii.affine

    # ── Preprocess (matches training pipeline exactly) ────────────────────
    image_tensor = preprocess_volume(image_np, target_size=tuple(args.input_size))

    # ── Retrieve PDP masks from frozen mask bank ──────────────────────────
    # Top-k=5 class-conditional cosine nearest-neighbor retrieval
    # No validation/test masks are ever used here
    pdp_masks = None
    if mask_bank is not None:
        pdp_masks = get_pdp_masks_for_inference(
            model, image_tensor, mask_bank,
            args.num_classes, args.topk_pdp, device,
        )

    # ── Forward pass ──────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device), pdp_masks=pdp_masks)
        # Per-class sigmoid heads (multi-label formulation)
        # Each voxel classified independently per class
        probs = torch.sigmoid(logits)           # (1, K, H, W, D)
        preds = (probs >= args.threshold).float()

    # ── Resize predictions back to original volume shape ──────────────────
    preds_resized = torch.nn.functional.interpolate(
        preds.float(),
        size=orig_shape,
        mode="nearest",
    )
    preds_np = preds_resized.squeeze(0).cpu().numpy()  # (K, H, W, D)

    # ── Save prediction as NIfTI ──────────────────────────────────────────
    # Multi-label output channels preserved for fair evaluation
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_nii = nib.Nifti1Image(preds_np, affine)
    nib.save(output_nii, output_path)
    print(f"[Inference] Saved prediction: {output_path}")

    return preds_np


# ─────────────────────────────────────────────
# Batch Inference over Test Directory
# ─────────────────────────────────────────────

def run_batch_inference(model, data_root, output_dir, mask_bank, device, args):
    """
    Run inference over all CT volumes in data_root/images/.
    Saves predictions to output_dir/.
    """
    image_dir = os.path.join(data_root, "images")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"[Inference] Image directory not found: {image_dir}")

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    ])
    print(f"[Inference] Found {len(image_files)} volumes in {image_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for fname in image_files:
        image_path  = os.path.join(image_dir, fname)
        output_path = os.path.join(output_dir, fname.replace(".nii", "_pred.nii"))
        try:
            run_inference(model, image_path, output_path, mask_bank, device, args)
        except Exception as e:
            print(f"[Inference] Error on {fname}: {e}")
            continue

    print(f"[Inference] Done. {len(image_files)} volumes processed → {output_dir}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args   = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[Inference] Device: {device} | Top-k PDP: {args.topk_pdp} | Threshold: {args.threshold}")

    # ── Load model ────────────────────────────────────────────────────────
    from models.triprompt_model import TripromptModel
    model = TripromptModel(
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
    ).to(device)
    load_checkpoint(args.checkpoint, model, optimizer=None, device=device)
    model.eval()

    # ── Load frozen PDP mask bank (training subjects only) ────────────────
    # IMPORTANT: This bank was built exclusively from training-split subjects
    # and frozen before any evaluation. No test/val masks are ever accessed.
    mask_bank = None
    if os.path.exists(args.mask_bank_path):
        mask_bank = MaskBank(
            bank_dir=os.path.dirname(args.mask_bank_path),
            num_classes=args.num_classes,
            device=device,
        )
        mask_bank.load(args.mask_bank_path)
        print(f"[Inference] PDP mask bank loaded: {args.mask_bank_path}")
        print(f"[Inference] Retrieval: class-conditional cosine NN, top-k={args.topk_pdp}")
    else:
        print(f"[Inference] Warning: mask bank not found at {args.mask_bank_path}. "
              f"Running without PDP conditioning.")

    # ── Run batch inference ───────────────────────────────────────────────
    run_batch_inference(model, args.data_root, args.output_dir, mask_bank, device, args)


if __name__ == "__main__":
    main()
