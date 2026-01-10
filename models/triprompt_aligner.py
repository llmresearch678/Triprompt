import torch
import torch.nn as nn
import torch.nn.functional as F


class TriPromptAligner(nn.Module):
    """
    TriPrompt Aligner (Query-Centric Multimodal Alignment).

    This module aligns segmentation queries (Qs) with three prompt
    modalities:
        - Qa: Structural prompt
        - Qt: Text prompt
        - Qd: Deformation prompt (PDP)

    DESIGN (Paper-aligned):
    ----------------------
    1. Query-centric: Qs attends to prompts, not vice versa.
    2. Modality-aware: each prompt is normalized and gated.
    3. Stable: residual connection + LayerNorm.
    4. Lightweight: no architectural novelty beyond alignment.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim (int): Shared embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout for attention output.
        """
        super().__init__()

        # Cross-attention: queries = Qs, keys/values = {Qa, Qt, Qd}
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # (B, N, C)
        )

        # Modality gates (learnable scalars)
        self.gate_a = nn.Parameter(torch.tensor(1.0))
        self.gate_t = nn.Parameter(torch.tensor(1.0))
        self.gate_d = nn.Parameter(torch.tensor(1.0))

        # Normalization and residual projection
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        qs: torch.Tensor,
        qa: torch.Tensor,
        qt: torch.Tensor,
        qd: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for tri-prompt alignment.

        Args:
            qs (Tensor): Segmentation queries of shape (B, C)
            qa (Tensor): Structural prompt embeddings of shape (B, C)
            qt (Tensor): Text prompt embeddings of shape (B, C)
            qd (Tensor): Deformation prompt embeddings of shape (B, C)

        Returns:
            Tensor:
                Aligned query embeddings of shape (B, C)
        """

        # ------------------------------
        # Shape & sanity checks
        # ------------------------------
        if not (qs.dim() == qa.dim() == qt.dim() == qd.dim() == 2):
            raise ValueError(
                "All inputs must have shape (B, C)."
            )

        # ------------------------------
        # Normalize each modality
        # ------------------------------
        qs_n = self.norm_q(qs)
        qa_n = F.normalize(qa, dim=1)
        qt_n = F.normalize(qt, dim=1)
        qd_n = F.normalize(qd, dim=1)

        # ------------------------------
        # Modality gating (learned balance)
        # ------------------------------
        qa_n = self.gate_a * qa_n
        qt_n = self.gate_t * qt_n
        qd_n = self.gate_d * qd_n

        # ------------------------------
        # Stack prompts as keys/values
        # Shape: (B, 3, C)
        # ------------------------------
        prompts = torch.stack([qa_n, qt_n, qd_n], dim=1)

        # ------------------------------
        # Query-centric cross-attention
        # Query: (B, 1, C)
        # Key/Value: (B, 3, C)
        # ------------------------------
        attn_out, _ = self.cross_attn(
            query=qs_n.unsqueeze(1),
            key=prompts,
            value=prompts,
            need_weights=False
        )

        # ------------------------------
        # Residual + projection
        # ------------------------------
        attn_out = attn_out.squeeze(1)          # (B, C)
        attn_out = self.proj(attn_out)
        attn_out = self.dropout(attn_out)

        # Final residual connection
        aligned_qs = self.norm_out(qs + attn_out)

        return aligned_qs
