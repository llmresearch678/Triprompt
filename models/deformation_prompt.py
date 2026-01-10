import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformationPromptEncoder(nn.Module):
    """
    Population-level Deformation Prompt (PDP) Encoder.

    This module encodes non-rigid anatomical deformation patterns
    from binary segmentation masks, capturing population-level
    shape variability across subjects.

    IMPORTANT (Paper-aligned design):
    --------------------------------
    1. Input is a *binary shape mask only* (no image intensity).
    2. No appearance information is used (prevents leakage).
    3. Encodes deformation statistics, not instance identity.
    4. Output is a compact global embedding used as a prompt.

    This implementation supports all datasets used in the paper
    (FLARE22, MSD, LiTS, KiTS, AMOS, etc.) once harmonized.
    """

    def __init__(
        self,
        embed_dim: int,
        in_channels: int = 1
    ):
        """
        Args:
            embed_dim (int):
                Dimensionality of the deformation prompt embedding.
            in_channels (int):
                Number of input channels.
                Default = 1 for binary segmentation masks.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            # ---------------------------------------
            # Local shape encoding
            # ---------------------------------------
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),

            # ---------------------------------------
            # Projection to embedding space
            # ---------------------------------------
            nn.Conv3d(64, embed_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(embed_dim),
            nn.ReLU(inplace=True),

            # ---------------------------------------
            # Global deformation pooling
            # ---------------------------------------
            nn.AdaptiveAvgPool3d(output_size=1)
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for deformation prompt encoding.

        Args:
            mask (Tensor):
                Binary segmentation mask of shape (B, 1, H, W, D),
                sampled from a *different subject* than the input image.

        Returns:
            Tensor:
                Deformation prompt embedding of shape (B, embed_dim).
        """

        if mask.dim() != 5:
            raise ValueError(
                "DeformationPromptEncoder expects input of shape "
                "(B, 1, H, W, D)."
            )

        # Ensure mask is binary (robust to interpolation artifacts)
        mask = (mask > 0).float()

        # Encode deformation statistics
        z = self.encoder(mask)

        # Flatten global embedding
        z = z.view(z.size(0), -1)

        return z
