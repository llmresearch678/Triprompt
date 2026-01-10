import os
import torch
import nibabel as nib
import numpy as np
from utils import load_checkpoint


def run_inference(
    model,
    image_path,
    output_path,
    device,
    threshold: float = 0.5
):
    """
    Inference pipeline for TRIPROMPT-style 3D segmentation.

    This function performs voxel-wise multi-label segmentation
    on a single 3D CT volume and saves the prediction as a
    NIfTI file for downstream evaluation or visualization.

    Args:
        model (nn.Module):
            Trained TRIPROMPT model.
        image_path (str):
            Path to input CT volume (.nii or .nii.gz).
        output_path (str):
            Path to save predicted segmentation (.nii.gz).
        device (torch.device):
            CUDA or CPU device.
        threshold (float):
            Sigmoid threshold for binarizing predictions.
            Default = 0.5 (as used in evaluation).
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    model.eval()

    # ---------------------------------------------------
    # 1. Load CT volume
    # ---------------------------------------------------
    # Shape after loading: (H, W, D)
    image_nii = nib.load(image_path)
    image = image_nii.get_fdata().astype(np.float32)

    # ---------------------------------------------------
    # 2. Add batch and channel dimensions
    # ---------------------------------------------------
    # Final shape: (1, 1, H, W, D)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)

    image = torch.from_numpy(image).to(device)

    # ---------------------------------------------------
    # 3. Forward pass
    # ---------------------------------------------------
    with torch.no_grad():
        logits = model(image)

        # Multi-label probability map
        probs = torch.sigmoid(logits)

        # Binarize predictions
        preds = (probs >= threshold).float()

    # ---------------------------------------------------
    # 4. Prepare output for saving
    # ---------------------------------------------------
    # Remove batch dimension: (C, H, W, D)
    preds = preds.squeeze(0).cpu().numpy()

    # IMPORTANT:
    # - If C = 1 → binary segmentation
    # - If C > 1 → multi-organ / tumor labels stored as channels
    #
    # We preserve channel dimension for multi-label compatibility
    # and evaluation consistency.
    affine = image_nii.affine

    output_nii = nib.Nifti1Image(preds, affine)
    nib.save(output_nii, output_path)

    print(f"[INFO] Segmentation saved to: {output_path}")


if __name__ == "__main__":
    """
    Example usage for standalone inference.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model checkpoint
    model = load_checkpoint(
        checkpoint_path="checkpoints/best_model.pth",
        device=device
    )

    run_inference(
        model=model,
        image_path="data/test/images/case_001.nii.gz",
        output_path="output/case_001_pred.nii.gz",
        device=device,
        threshold=0.5
    )
