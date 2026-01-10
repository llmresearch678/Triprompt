import torch
import torch.nn as nn

from .backbone import Backbone
from .triprompt_aligner import TriPromptAligner


class TriPromptModel(nn.Module):
    """
    TRIPROMPT: Deformation-Aware Multimodal Prompted Segmentation Model.

    This module integrates:
        - A shared 3D backbone (Swin-UNETR)
        - Structural, textual, and deformation prompts
        - Query-centric multimodal alignment
        - A lightweight segmentation head

    The backbone is used without modification.
    All architectural novelty resides in the prompt encoders
    and the query–prompt alignment mechanism.
    """

    def __init__(
        self,
        backbone: Backbone,
        aligner: TriPromptAligner,
        num_classes: int
    ):
        """
        Args:
            backbone (Backbone):
                Shared 3D encoder backbone (e.g., Swin-UNETR).
            aligner (TriPromptAligner):
                Query-centric multimodal prompt aligner.
            num_classes (int):
                Number of segmentation output channels
                (organs + tumors, multi-label).
        """
        super().__init__()

        self.backbone = backbone
        self.aligner = aligner

        # ---------------------------------------------------
        # Segmentation head
        # ---------------------------------------------------
        # 1×1×1 convolution maps backbone features to
        # multi-label segmentation logits.
        self.segmentation_head = nn.Conv3d(
            in_channels=self.backbone.encoder.out_channels,
            out_channels=num_classes,
            kernel_size=1,
            bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        qs: torch.Tensor,
        qa: torch.Tensor,
        qt: torch.Tensor,
        qd: torch.Tensor
    ):
        """
        Forward pass of TRIPROMPT.

        Args:
            x (Tensor):
                Input 3D volume of shape (B, 1, H, W, D).
            qs (Tensor):
                Initial segmentation query embeddings of shape (B, C).
            qa (Tensor):
                Structural prompt embeddings of shape (B, C).
            qt (Tensor):
                Text prompt embeddings of shape (B, C).
            qd (Tensor):
                Deformation prompt embeddings of shape (B, C).

        Returns:
            logits (Tensor):
                Segmentation logits of shape (B, num_classes, H, W, D).
            qs_aligned (Tensor):
                Prompt-aligned segmentation query embeddings of shape (B, C).
        """

        # ---------------------------------------------------
        # 1. Backbone feature extraction
        # ---------------------------------------------------
        # Output shape: (B, F, H, W, D)
        features = self.backbone(x)

        # ---------------------------------------------------
        # 2. Query-centric multimodal alignment
        # ---------------------------------------------------
        # Refines segmentation queries using Qa, Qt, Qd
        qs_aligned = self.aligner(
            qs=qs,
            qa=qa,
            qt=qt,
            qd=qd
        )

        # ---------------------------------------------------
        # 3. Segmentation prediction
        # ---------------------------------------------------
        logits = self.segmentation_head(features)

        return logits, qs_aligned
