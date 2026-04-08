"""
triprompt_aligner.py
TRIPROMPT ALIGNER: Query-centric multimodal alignment module.

Components:
- TriQueryIntegrator (TQ): hard routing for Q_a, Q_d; soft attention for Q_s, Q_t
- Gumbel-Softmax + Straight-Through Estimator (STE) for differentiable hard spatial assignment
- PromptContextAligner: masked cross-attention for context-aware query refinement
- Temperature: fixed tau=0.5 (no annealing schedule)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Gumbel-Softmax Hard Spatial Assignment
# ─────────────────────────────────────────────

def gumbel_softmax_hard(logits, tau=0.5, dim=-1):
    """
    Differentiable hard spatial assignment via Gumbel-Softmax + STE.
    tau=0.5 fixed throughout training (no annealing).

    Forward:  one-hot hard selection
    Backward: soft Gumbel-Softmax gradients (straight-through estimator)
    """
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    soft = F.softmax((logits + gumbels) / tau, dim=dim)

    # Hard one-hot selection
    index = soft.max(dim=dim, keepdim=True)[1]
    hard = torch.zeros_like(soft).scatter_(dim, index, 1.0)

    # STE: forward=hard, backward=soft
    return hard - soft.detach() + soft


# ─────────────────────────────────────────────
# Multi-Head Cross-Attention (soft and hard)
# ─────────────────────────────────────────────

class SoftCrossAttention(nn.Module):
    """Standard scaled dot-product cross-attention (for Q_s and Q_t)."""

    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        """
        Args:
            query:     (B, K, C)
            key_value: (B, V, C)
        Returns:
            (B, K, C)
        """
        out, _ = self.attn(query, key_value, key_value)
        return self.norm(query + out)


class HardSpatialAttention(nn.Module):
    """
    Hard spatial attention for Q_a and Q_d.
    Uses Gumbel-Softmax + STE to prevent cross-modal blending.
    Structure and deformation prompts are spatially localized and
    benefit from hard routing to prevent prompt interference.
    """

    def __init__(self, embed_dim, num_heads=8, tau=0.5):
        super().__init__()
        self.tau = tau
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self, query, features):
        """
        Args:
            query:    (B, K, C) — Q_a or Q_d
            features: (B, V, C) — spatial feature tokens at scale l
        Returns:
            (B, K, C) — hard-routed refined query
        """
        B, K, C = query.shape
        V = features.shape[1]

        Q = self.q_proj(query)     # (B, K, C)
        K_ = self.k_proj(features) # (B, V, C)
        V_ = self.v_proj(features) # (B, V, C)

        # Affinity scores S^(l) = Q * X^T / sqrt(C)
        scores = torch.bmm(Q, K_.transpose(1, 2)) / (C ** 0.5)  # (B, K, V)

        # Differentiable hard assignment via Gumbel-Softmax + STE
        hard_mask = gumbel_softmax_hard(scores, tau=self.tau, dim=-1)  # (B, K, V)

        # Aggregate values using hard mask
        out = torch.bmm(hard_mask, V_)  # (B, K, C)
        out = self.out_proj(out)
        return self.norm(query + out)


# ─────────────────────────────────────────────
# TriQueryIntegrator
# ─────────────────────────────────────────────

class TriQueryIntegrator(nn.Module):
    """
    Multi-scale query-feature interaction:
    - Soft cross-attention for [Q_s, Q_t] (global semantic cues)
    - Hard spatial attention for [Q_a, Q_d] (spatially localized cues)
    Applied across L feature scales.
    """

    def __init__(self, embed_dim=256, num_heads=8, num_scales=5, tau=0.5):
        super().__init__()
        self.num_scales = num_scales

        # Soft attention layers for segmentation and text queries
        self.soft_attn = nn.ModuleList([
            SoftCrossAttention(embed_dim, num_heads) for _ in range(num_scales)
        ])

        # Hard attention layers for structural and deformation queries
        self.hard_attn_struct = nn.ModuleList([
            HardSpatialAttention(embed_dim, num_heads, tau) for _ in range(num_scales)
        ])
        self.hard_attn_deform = nn.ModuleList([
            HardSpatialAttention(embed_dim, num_heads, tau) for _ in range(num_scales)
        ])

        # Feature projection to embed_dim
        self.feat_projs = nn.ModuleList([
            nn.LazyLinear(embed_dim) for _ in range(num_scales)
        ])

    def forward(self, Q_s, Q_t, Q_a, Q_d, multi_scale_features):
        """
        Args:
            Q_s: segmentation queries   (B, K, C)
            Q_t: text prompt tokens     (B, K, C)
            Q_a: structural tokens      (B, K, C)
            Q_d: deformation tokens     (B, K, C)
            multi_scale_features: list of L feature tensors (B, C_l, H_l, W_l, D_l)
        Returns:
            Q_s_hat, Q_t_hat, Q_a_hat, Q_d_hat: refined tokens (B, K, C) each
        """
        Q_s_hat, Q_t_hat, Q_a_hat, Q_d_hat = Q_s, Q_t, Q_a, Q_d

        for l, feat in enumerate(multi_scale_features):
            B, C_l, H_l, W_l, D_l = feat.shape
            # Flatten spatial dims: (B, V_l, C_l) then project to embed_dim
            feat_flat = feat.view(B, C_l, -1).transpose(1, 2)        # (B, V_l, C_l)
            feat_proj = self.feat_projs[l](feat_flat)                 # (B, V_l, C)

            # Soft attention: update Q_s and Q_t
            Q_s_hat = self.soft_attn[l](Q_s_hat, feat_proj)
            Q_t_hat = self.soft_attn[l](Q_t_hat, feat_proj)

            # Hard attention: update Q_a and Q_d
            Q_a_hat = self.hard_attn_struct[l](Q_a_hat, feat_proj)
            Q_d_hat = self.hard_attn_deform[l](Q_d_hat, feat_proj)

        return Q_s_hat, Q_t_hat, Q_a_hat, Q_d_hat


# ─────────────────────────────────────────────
# PromptContextAligner
# ─────────────────────────────────────────────

class PromptContextAligner(nn.Module):
    """
    Masked cross-attention module for context-aware segmentation query refinement.
    Segmentation queries attend to refined structural, textual, and deformation prompts.
    Query-centric: avoids unrestricted symmetric mixing of all modalities.
    """

    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, Q_s_hat, Q_a_hat, Q_t_hat, Q_d_hat):
        """
        Args:
            Q_s_hat: refined segmentation queries (B, K, C)
            Q_a_hat: refined structural tokens    (B, K, C)
            Q_t_hat: refined text tokens          (B, K, C)
            Q_d_hat: refined deformation tokens   (B, K, C)
        Returns:
            O_s: context-aligned segmentation query features (B, K, C)
        """
        # Concatenate all prompt tokens as key-value context
        prompt_context = torch.cat([Q_a_hat, Q_t_hat, Q_d_hat], dim=1)  # (B, 3K, C)

        # Segmentation queries attend to the full prompt context
        attn_out, _ = self.cross_attn(Q_s_hat, prompt_context, prompt_context)
        O_s = self.norm1(Q_s_hat + attn_out)
        O_s = self.norm2(O_s + self.ffn(O_s))
        return O_s


# ─────────────────────────────────────────────
# Full TriPrompt Aligner
# ─────────────────────────────────────────────

class TriPromptAligner(nn.Module):
    """
    Full TRIPROMPT ALIGNER combining:
    1. TriQueryIntegrator: multi-scale hard/soft query-feature interaction
    2. PromptContextAligner: context-aware segmentation query refinement
    """

    def __init__(self, embed_dim=256, num_heads=8, num_scales=5, tau=0.5):
        super().__init__()
        self.query_integrator = TriQueryIntegrator(embed_dim, num_heads, num_scales, tau)
        self.context_aligner = PromptContextAligner(embed_dim, num_heads)

    def forward(self, Q_s, Q_t, Q_a, Q_d, multi_scale_features):
        """
        Args:
            Q_s: segmentation queries (B, K, C)
            Q_t: text tokens          (B, K, C)
            Q_a: structural tokens    (B, K, C)
            Q_d: deformation tokens   (B, K, C)
            multi_scale_features: list of L feature maps
        Returns:
            O_s: final aligned segmentation queries (B, K, C)
        """
        Q_s_hat, Q_t_hat, Q_a_hat, Q_d_hat = self.query_integrator(
            Q_s, Q_t, Q_a, Q_d, multi_scale_features
        )
        O_s = self.context_aligner(Q_s_hat, Q_a_hat, Q_t_hat, Q_d_hat)
        return O_s
