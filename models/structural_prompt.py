import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralPromptEncoder(nn.Module):
    """
    Structural Prompt Encoder (Qa).

    This module encodes localized anatomical appearance and
    spatial structure from backbone feature maps. It provides
    geometry-aware cues that ground segmentation queries in
    image-derived structure.

    IMPORTANT (Paper-aligned design):
    --------------------------------
    1. Operates on backbone feature maps, not raw images.
    2. Encodes *local appearance + spatial structure* only.
    3. Does NOT encode semantic labels or deformation priors.
    4. Produces a compact global prompt embedding via pooling.

    This design ensures clear separation between:
    - Structural cues (Qa)
    - Semantic cues (Qt)
    - Deformation cues (Qd)
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int
    ):
        """
        Args:
            in_channels (int):
                Number of input channels from the backbone.
            embed_dim (int):
                Dimensionality of the structural prompt embedding.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            # ---------------------------------------
            # Local structural feature extraction
            # ---------------------------------------
            nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(embed_dim),
            nn.ReLU(inplace=True),

            # ---------------------------------------
            # Global structural summarization
            # ---------------------------------------
            nn.AdaptiveAvgPool3d(output_size=1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for structural prompt encoding.

        Args:
            features (Tensor):
                Backbone feature maps of shape (B, C, H, W, D).

        Returns:
            Tensor:
                Structural prompt embedding of shape (B, embed_dim).
        """

        if features.dim() != 5:
            raise ValueError(
                "StructuralPromptEncoder expects input of shape "
                "(B, C, H, W, D)."
            )

        # Encode local structural cues
        q = self.encoder(features)

        # Flatten global embedding
        q = q.view(q.size(0), -1)

        return q
