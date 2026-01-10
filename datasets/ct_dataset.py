import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib


class CTDataset(Dataset):
    """
    3D CT dataset for multi-organ / tumor segmentation.

    Expected directory structure:
        root/
          ├── images/
          │     └── case_xxx.nii.gz
          └── masks/
                └── case_xxx.nii.gz
    """

    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")

        self.cases = sorted(os.listdir(self.image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.cases)

    def load_nifti(self, path):
        return nib.load(path).get_fdata().astype(np.float32)

    def __getitem__(self, idx):
        case_id = self.cases[idx]

        img_path = os.path.join(self.image_dir, case_id)
        mask_path = os.path.join(self.mask_dir, case_id)

        image = self.load_nifti(img_path)
        mask = self.load_nifti(mask_path)

        # Add channel dimension: (C, H, W, D)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        if self.transform:
            sample = self.transform({"image": image, "mask": mask})
            image, mask = sample["image"], sample["mask"]

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask
