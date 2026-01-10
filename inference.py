import torch
import nibabel as nib
import numpy as np
from utils import load_checkpoint


def inference(model, image_path, output_path, device):
    model.eval()

    image = nib.load(image_path).get_fdata().astype(np.float32)
    image = np.expand_dims(image, axis=(0, 1))  # (1, 1, H, W, D)
    image = torch.from_numpy(image).to(device)

    with torch.no_grad():
        logits = model(image)
        pred = torch.sigmoid(logits)
        pred = (pred > 0.5).float()

    pred = pred.cpu().numpy()[0, 0]
    nib.save(nib.Nifti1Image(pred, np.eye(4)), output_path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint("checkpoints/best_model.pth", device)

    inference(
        model,
        image_path="data/test/images/case_001.nii.gz",
        output_path="output/case_001_pred.nii.gz",
        device=device,
    )
