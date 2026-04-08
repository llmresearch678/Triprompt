"""
structural_prompt.py
Structural Prompt Encoder (E_ANA) for TRIPROMPT.

Automatically generates class-specific structural prompts Q_a from
localized 3D sub-volumes predicted by a co-trained region-proposal head.
No human input required at test time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionProposalHead(nn.Module):
    """
    Lightweight 3D region-proposal head co-trained with the segmentation network.
    Predicts class-specific bounding proposals from backbone feature maps.
    Fully automatic — zero human input at test time.
    """

    def __init__(self, in_channels=48, num_classes=13):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
        )
        # Predict (cx, cy, cz, w, h, d) per class — 6 values
        self.head = nn.Conv3d(32, num_classes * 6, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, features):
        """
        Args:
            features: backbone feature map (B, C, H, W, D)
        Returns:
            proposals: (B, num_classes, 6) — normalized bbox proposals per class
        """
        x = self.conv(features)
        x = self.head(x)
        # Global average pool to get one proposal per class
        x = F.adaptive_avg_pool3d(x, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        proposals = x.view(-1, self.num_classes, 6)
        return torch.sigmoid(proposals)  # Normalize to [0, 1]


class StructuralPromptEncoder(nn.Module):
    """
    E_ANA: Maps localized 3D sub-volumes into compact structural embeddings Q_a.
    Sub-volumes are automatically cropped using proposals from RegionProposalHead.
    """

    def __init__(self, in_channels=1, embed_dim=256, num_classes=13, crop_size=(32, 32, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.crop_size = crop_size

        # Lightweight 3D conv encoder for sub-volumes
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.projection = nn.Linear(128, embed_dim)
        self.proposal_head = RegionProposalHead(in_channels=48, num_classes=num_classes)

    def crop_sub_volume(self, volume, proposal, crop_size):
        """
        Crop a localized sub-volume from the input volume using a proposal bbox.
        Args:
            volume:   (B, C, H, W, D)
            proposal: (B, 6) — (cx, cy, cz, w, h, d) normalized
            crop_size: target crop size
        Returns:
            cropped: (B, C, crop_H, crop_W, crop_D)
        """
        B, C, H, W, D = volume.shape
        cx = (proposal[:, 0] * H).long().clamp(crop_size[0] // 2, H - crop_size[0] // 2)
        cy = (proposal[:, 1] * W).long().clamp(crop_size[1] // 2, W - crop_size[1] // 2)
        cz = (proposal[:, 2] * D).long().clamp(crop_size[2] // 2, D - crop_size[2] // 2)

        crops = []
        for b in range(B):
            x1, x2 = cx[b] - crop_size[0] // 2, cx[b] + crop_size[0] // 2
            y1, y2 = cy[b] - crop_size[1] // 2, cy[b] + crop_size[1] // 2
            z1, z2 = cz[b] - crop_size[2] // 2, cz[b] + crop_size[2] // 2
            crop = volume[b:b+1, :, x1:x2, y1:y2, z1:z2]
            crop = F.interpolate(crop, size=crop_size, mode='trilinear', align_corners=False)
            crops.append(crop)
        return torch.cat(crops, dim=0)

    def forward(self, volume, backbone_features):
        """
        Args:
            volume:            input 3D volume (B, C, H, W, D)
            backbone_features: feature map from backbone for proposal head (B, C', H', W', D')
        Returns:
            Q_a: structural prompt tokens (B, num_classes, embed_dim)
        """
        B = volume.shape[0]
        proposals = self.proposal_head(backbone_features)  # (B, num_classes, 6)

        class_embeddings = []
        for c in range(self.num_classes):
            prop_c = proposals[:, c, :]                    # (B, 6)
            sub_vol = self.crop_sub_volume(volume, prop_c, self.crop_size)
            feat = self.encoder(sub_vol).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, 128)
            emb = self.projection(feat)                    # (B, embed_dim)
            class_embeddings.append(emb.unsqueeze(1))

        Q_a = torch.cat(class_embeddings, dim=1)           # (B, num_classes, embed_dim)
        return Q_a
