"""
build_mask_bank.py
Builds the PDP mask bank exclusively from training subjects.
Must be run ONCE before any evaluation begins. The bank is then frozen.

Usage:
    python build_mask_bank.py \
        --train_mask_dir ./data/train/masks \
        --bank_save_path ./data/mask_bank/mask_bank.pt \
        --checkpoint checkpoints/best_model.pth \
        --num_classes 13

IMPORTANT:
    - Only training-split masks are processed here
    - No validation or test masks are ever included
    - This enforces the cross-subject constraint (s != i) during inference
"""

import os
import argparse
import torch
from models.deformation_prompt import DeformationPromptEncoder, MaskBank


def get_class_mask_paths(mask_dir, num_classes):
    """
    Scan mask_dir for per-class binary mask files.
    Expected structure:
        mask_dir/
            class_00/
                subject_001.nii.gz
                subject_002.nii.gz
                ...
            class_01/
                ...
    Returns:
        dict {class_id: [path1, path2, ...]}
    """
    paths = {}
    for c in range(num_classes):
        class_dir = os.path.join(mask_dir, f"class_{c:02d}")
        if os.path.isdir(class_dir):
            files = sorted([
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if f.endswith(".nii.gz") or f.endswith(".nii")
            ])
            paths[c] = files
        else:
            paths[c] = []
            print(f"[Warning] No directory found for class {c}: {class_dir}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Build PDP Mask Bank from training subjects")
    parser.add_argument("--train_mask_dir", type=str, required=True,
                        help="Directory containing per-class training masks")
    parser.add_argument("--bank_save_path", type=str, default="./data/mask_bank/mask_bank.pt",
                        help="Path to save the built mask bank")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint to load PDP encoder weights")
    parser.add_argument("--num_classes", type=int, default=13)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[MaskBank] Building mask bank from: {args.train_mask_dir}")
    print(f"[MaskBank] Device: {args.device}")

    # Initialize PDP encoder
    encoder = DeformationPromptEncoder(
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
    ).to(args.device)

    # Load trained encoder weights if checkpoint provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state = ckpt.get("model_state_dict", ckpt)
        # Load only deformation encoder weights
        deform_state = {
            k.replace("deformation_encoder.", ""): v
            for k, v in state.items()
            if k.startswith("deformation_encoder.")
        }
        if deform_state:
            encoder.load_state_dict(deform_state, strict=False)
            print(f"[MaskBank] Loaded PDP encoder weights from {args.checkpoint}")
        else:
            print(f"[MaskBank] No deformation encoder weights found in checkpoint, using random init")

    # Collect training mask paths per class
    mask_paths = get_class_mask_paths(args.train_mask_dir, args.num_classes)
    total = sum(len(v) for v in mask_paths.values())
    print(f"[MaskBank] Found {total} masks across {args.num_classes} classes")

    # Build the mask bank
    bank = MaskBank(
        bank_dir=args.train_mask_dir,
        num_classes=args.num_classes,
        device=args.device,
    )
    bank.build(mask_paths, encoder, verbose=True)

    # Save frozen bank
    os.makedirs(os.path.dirname(args.bank_save_path), exist_ok=True)
    bank.save(args.bank_save_path)
    print(f"\n[MaskBank] Done. Mask bank saved to: {args.bank_save_path}")
    print("[MaskBank] This bank is now frozen and ready for inference.")


if __name__ == "__main__":
    main()
