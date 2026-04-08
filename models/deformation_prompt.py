"""
deformation_prompt.py
Population-Level Deformation Prompt Encoder (E_DEF) for TRIPROMPT.

Key design principles:
- Operates exclusively on binary shape masks (no appearance cues)
- Sampled under strict cross-subject constraint (s != i)
- Mask bank built exclusively from training subjects, frozen before evaluation
- Retrieval: class-conditional cosine nearest-neighbor (top-k=5)
- Compact latent bottleneck prevents identity memorization
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DeformationPromptEncoder(nn.Module):
    """
    E_DEF: Lightweight 3D shape encoder that maps binary segmentation masks
    into compact deformation latent representations Q_d.

    Unlike SSMs, PDP is:
    - Jointly optimized with the segmentation objective
    - Integrated via differentiable hard-routed query refinement
    - Adaptive: suppresses influence when deformation is minimal
    """

    def __init__(self, embed_dim=256, num_classes=13):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Lightweight 3D conv encoder for binary masks
        # Input: binary mask (B, 1, H, W, D)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),   # Global pooling: eliminates spatial identity
        )

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        # Learnable deformation projection W_d
        self.W_d = nn.Linear(num_classes * embed_dim, num_classes * embed_dim, bias=False)

    def encode_mask(self, mask):
        """
        Encode a single binary mask into a deformation latent vector.
        Args:
            mask: binary mask (B, 1, H, W, D)
        Returns:
            latent: (B, embed_dim)
        """
        feat = self.encoder(mask).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, 64)
        return self.projection(feat)                                     # (B, embed_dim)

    def forward(self, masks_per_class):
        """
        Args:
            masks_per_class: list of K tensors, each (B, 1, H, W, D)
                             one binary mask per class, sampled from a different subject (s != i)
        Returns:
            Q_d: deformation prompt tokens (B, num_classes, embed_dim)
        """
        B = masks_per_class[0].shape[0]
        class_embeddings = []

        for mask_c in masks_per_class:
            emb = self.encode_mask(mask_c)         # (B, embed_dim)
            class_embeddings.append(emb.unsqueeze(1))

        # Stack: (B, K, embed_dim)
        E_d = torch.cat(class_embeddings, dim=1)

        # Apply learnable projection W_d
        B, K, C = E_d.shape
        Q_d = self.W_d(E_d.view(B, K * C)).view(B, K, C)
        return Q_d


class MaskBank:
    """
    Fixed mask bank constructed exclusively from training subjects.
    Used at validation/test time for PDP retrieval.
    No ground-truth masks from validation or test images are ever accessed.

    Retrieval policy: class-conditional cosine nearest-neighbor (top-k=5)
    """

    def __init__(self, bank_dir, num_classes=13, device="cpu"):
        """
        Args:
            bank_dir:    path to mask bank directory (built from training subjects only)
            num_classes: number of anatomical classes
            device:      torch device
        """
        self.bank_dir = bank_dir
        self.num_classes = num_classes
        self.device = device
        self.embeddings = {}   # class_id -> (N, embed_dim) tensor of latent embeddings
        self.mask_paths = {}   # class_id -> list of N mask file paths

    def build(self, train_mask_paths_per_class, encoder, verbose=True):
        """
        Build the mask bank by encoding all training masks.
        Must be called BEFORE any evaluation begins and then frozen.

        Args:
            train_mask_paths_per_class: dict {class_id: [path1, path2, ...]}
            encoder: DeformationPromptEncoder instance
        """
        encoder.eval()
        with torch.no_grad():
            for class_id in range(self.num_classes):
                paths = train_mask_paths_per_class.get(class_id, [])
                if len(paths) == 0:
                    if verbose:
                        print(f"[MaskBank] Warning: no masks for class {class_id}")
                    self.embeddings[class_id] = torch.zeros(0, encoder.embed_dim)
                    self.mask_paths[class_id] = []
                    continue

                embs = []
                for path in paths:
                    mask = self._load_mask(path).to(self.device)
                    emb = encoder.encode_mask(mask)   # (1, embed_dim)
                    embs.append(emb.squeeze(0).cpu())

                self.embeddings[class_id] = torch.stack(embs, dim=0)  # (N, embed_dim)
                self.mask_paths[class_id] = paths
                if verbose:
                    print(f"[MaskBank] Class {class_id}: {len(paths)} masks encoded.")

    def save(self, save_path):
        """Save the built mask bank to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "embeddings": self.embeddings,
            "mask_paths": self.mask_paths,
            "num_classes": self.num_classes,
        }, save_path)
        print(f"[MaskBank] Saved to {save_path}")

    def load(self, load_path):
        """Load a pre-built mask bank from disk."""
        data = torch.load(load_path, map_location=self.device)
        self.embeddings = data["embeddings"]
        self.mask_paths = data["mask_paths"]
        self.num_classes = data["num_classes"]
        print(f"[MaskBank] Loaded from {load_path}")

    def retrieve(self, query_features, class_id, k=5, current_subject_idx=None):
        """
        Retrieve top-k masks for a given class using cosine nearest-neighbor search.
        Enforces cross-subject constraint: current subject is excluded (s != i).

        Args:
            query_features:       (embed_dim,) query embedding for class_id
            class_id:             target anatomical class
            k:                    number of masks to retrieve (default=5)
            current_subject_idx:  index of current subject to exclude (s != i)
        Returns:
            top_k_embeddings: (k, embed_dim) retrieved mask embeddings
            top_k_indices:    indices of retrieved masks in the bank
        """
        bank_embs = self.embeddings[class_id].to(self.device)  # (N, embed_dim)

        if bank_embs.shape[0] == 0:
            # Fallback: return zero embedding if no masks available for this class
            return torch.zeros(k, query_features.shape[-1], device=self.device), []

        # Exclude current subject to enforce cross-subject constraint (s != i)
        if current_subject_idx is not None:
            valid_indices = [i for i in range(bank_embs.shape[0]) if i != current_subject_idx]
            if len(valid_indices) == 0:
                valid_indices = list(range(bank_embs.shape[0]))
            bank_embs_filtered = bank_embs[valid_indices]
            index_map = valid_indices
        else:
            bank_embs_filtered = bank_embs
            index_map = list(range(bank_embs.shape[0]))

        # Cosine similarity
        q = F.normalize(query_features.unsqueeze(0), dim=-1)        # (1, embed_dim)
        b = F.normalize(bank_embs_filtered, dim=-1)                  # (N', embed_dim)
        similarities = (q @ b.T).squeeze(0)                          # (N',)

        k_actual = min(k, similarities.shape[0])
        top_k_local = torch.topk(similarities, k_actual).indices     # (k,)
        top_k_global = [index_map[i.item()] for i in top_k_local]

        top_k_embeddings = bank_embs[top_k_global]                   # (k, embed_dim)

        # Pad to k if fewer than k available
        if k_actual < k:
            pad = top_k_embeddings[-1:].expand(k - k_actual, -1)
            top_k_embeddings = torch.cat([top_k_embeddings, pad], dim=0)

        return top_k_embeddings, top_k_global

    def retrieve_all_classes(self, query_features_per_class, k=5, current_subject_idx=None):
        """
        Retrieve top-k masks for all classes.
        Args:
            query_features_per_class: (num_classes, embed_dim)
            k:                        number of masks per class
            current_subject_idx:      subject index to exclude
        Returns:
            retrieved: list of K tensors, each (k, embed_dim)
        """
        retrieved = []
        for c in range(self.num_classes):
            embs, _ = self.retrieve(
                query_features_per_class[c],
                class_id=c,
                k=k,
                current_subject_idx=current_subject_idx,
            )
            retrieved.append(embs)
        return retrieved

    @staticmethod
    def _load_mask(path):
        """Load a binary mask from a NIfTI file and return as (1, 1, H, W, D) tensor."""
        import nibabel as nib
        img = nib.load(path)
        mask = torch.from_numpy(img.get_fdata()).float()
        mask = (mask > 0.5).float()
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
