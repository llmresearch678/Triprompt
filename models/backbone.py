import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class Backbone(nn.Module):
    """
    Shared 3D Encoder Backbone for TRIPROMPT.

    This module implements a Swin-UNETR backbone used as a
    common volumetric feature extractor for all prompt modalities
    (structural, textual, and deformation-aware) and the segmentation head.

    IMPORTANT DESIGN CHOICES (Paper-aligned):
    ----------------------------------------
    1. The backbone architecture is used WITHOUT modification.
    2. No prompt logic or conditioning is embedded here.
    3. The backbone serves purely as a feature encoder.
    4. Architectural novelty lies outside the backbone (in prompting).

    This design ensures reproducibility, clarity of contribution,
    and compliance with IJCAI / IEEE TMI review standards.
    """

    def __init__(
        self,
        img_size,
        in_channels: int = 1,
        feature_size: int = 48,
        use_checkpoint: bool = False
    ):
        """
        Args:
            img_size (tuple or list):
                Spatial size of the input volume (H, W, D).
            in_channels (int):
                Number of input channels.
                Default = 1 for CT imaging.
            feature_size (int):
                Base feature size for Swin-UNETR.
            use_checkpoint (bool):
                Whether to use gradient checkpointing
                (disabled by default for clarity and stability).
        """
        super().__init__()

        self.encoder = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=feature_size,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.

        Args:
            x (Tensor):
                Input 3D volume of shape (B, C, H, W, D).

        Returns:
            Tensor:
                Dense volumetric feature representation produced
                by Swin-UNETR, used for segmentation and prompt conditioning.
        """
        return self.encoder(x)
