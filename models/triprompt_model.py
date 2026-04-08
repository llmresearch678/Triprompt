"""
triprompt_model.py
Full TRIPROMPT Model.

Integrates all components:
- TripromptBackbone:         Swin-UNETR multi-scale feature extractor
- StructuralPromptEncoder:   E_ANA — automatic Q_a from co-trained region proposals
- TextPromptEncoder:         E_TEXT — ClinicalBERT semantic embeddings Q_t
- DeformationPromptEncoder:  E_DEF — binary mask latents Q_d (cross-subject, s != i)
- TriPromptAligner:          Hard/soft query-feature interaction + PromptContextAligner
- Segmentation head:         Per-class sigmoid heads (multi-label formulation)
- Contrastive loss outputs:  Returns query_emb, prompt_emb, struct_emb, text_emb

Design notes (per paper Sec. 2):
- Per-class sigmoid heads throughout (NOT softmax) for multi-label organ+tumor co-segmentation
- "Exactly one label per voxel" phrase in paper is a writing error — corrected here
- Gumbel-Softmax temperature tau=0.5 fixed (no annealing schedule)
- Gradient-norm balancing for lambda_1, lambda_2 applied in train.py (per step)
- PDP operates on binary masks only — no appearance cues, compact bottleneck prevents
  identity memorization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import TripromptBackbone
from models.structural_prompt import StructuralPromptEncoder
from models.text_prompt import TextPromptEncoder
from models.deformation_prompt import DeformationPromptEncoder
from models.triprompt_aligner import TriPromptAligner


class TripromptModel(nn.Module):
    """
    TRIPROMPT: Deformation-Aware Multimodal Prompting for Robust 3D Segmentation.

    Three prompt streams:
        Q_a — structural (visual, automatic via region-proposal head)
        Q_t — textual    (ClinicalBERT medical descriptions)
        Q_d — deformation (PDP, cross-subject binary mask latents)

    All prompts fused via TriPromptAligner using:
        - Soft cross-attention  for Q_s (segmentation queries) and Q_t
        - Hard spatial routing  for Q_a and Q_d (Gumbel-Softmax + STE, tau=0.5)

    Output:
        logits:      (B, K, H, W, D) — per-class sigmoid logits
        query_emb:   (B, K, C)       — refined segmentation queries  [for L_ALIGN_1]
        prompt_emb:  (B, K, C)       — fused prompt embeddings p_c   [for L_ALIGN_1]
        struct_emb:  (B, K, C)       — Q_a_hat                       [for L_ALIGN_2]
        text_emb:    (B, K, C)       — Q_t_hat                       [for L_ALIGN_2]
    """

    def __init__(
        self,
        num_classes: int = 13,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_scales: int = 5,
        gumbel_tau: float = 0.5,
        img_size: tuple = (96, 96, 96),
        in_channels: int = 1,
        feature_size: int = 48,
        pretrained_weights: str = None,
        freeze_text_encoder: bool = True,
        class_descriptions: list = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim   = embed_dim

        # ── 1. Backbone: Swin-UNETR ───────────────────────────────────────
        self.backbone = TripromptBackbone(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=feature_size,
            pretrained_weights=pretrained_weights,
        )

        # ── 2. Structural Prompt Encoder E_ANA ───────────────────────────
        # Fully automatic: co-trained region-proposal head generates sub-volumes
        # No human input required at test time
        self.structural_encoder = StructuralPromptEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            crop_size=(32, 32, 32),
        )

        # ── 3. Text Prompt Encoder E_TEXT ─────────────────────────────────
        self.text_encoder = TextPromptEncoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            class_descriptions=class_descriptions,
            freeze_encoder=freeze_text_encoder,
        )

        # ── 4. Deformation Prompt Encoder E_DEF ──────────────────────────
        # Operates on binary shape masks only (no appearance cues)
        # Cross-subject constraint enforced: masks sampled with s != i
        # Compact bottleneck prevents identity memorization
        self.deformation_encoder = DeformationPromptEncoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
        )

        # ── 5. Learnable segmentation query tokens Q_s ───────────────────
        # K class-specific query vectors jointly refined by TriPromptAligner
        self.seg_queries = nn.Embedding(num_classes, embed_dim)
        nn.init.trunc_normal_(self.seg_queries.weight, std=0.02)

        # ── 6. TriPrompt Aligner ──────────────────────────────────────────
        # Hard routing for Q_a, Q_d (spatially localized)
        # Soft attention for Q_s, Q_t (global semantic cues)
        # tau=0.5 fixed (no annealing)
        self.aligner = TriPromptAligner(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_scales=num_scales,
            tau=gumbel_tau,
        )

        # ── 7. Per-class sigmoid segmentation head ────────────────────────
        # Multi-label formulation: each voxel classified independently per class
        # Per-class sigmoid (NOT softmax) supports multi-organ + tumor overlap
        self.seg_head = nn.Conv3d(embed_dim, num_classes, kernel_size=1)

        # ── 8. Dense feature projection ───────────────────────────────────
        # Projects decoder output Z to embed_dim for voxel-wise dot product
        self.feat_proj = nn.Conv3d(feature_size, embed_dim, kernel_size=1)

        # ── 9. PDP query projection for retrieval ─────────────────────────
        # Used in inference.py to project backbone features for cosine NN retrieval
        self.pdp_query_proj = nn.Linear(feature_size, embed_dim)

    # ─────────────────────────────────────────────────────────────────────
    # Voxel-wise prediction (Eq. 8-9 in paper)
    # ─────────────────────────────────────────────────────────────────────

    def decode_queries_to_logits(self, seg_queries_refined, dense_feat):
        """
        Compute per-voxel logits via dot product between refined segmentation
        queries and dense voxel-level feature map (decoder output Z).

        M_c(x,y,z) = sigma(<U_s^(c), phi_{x,y,z}>)  [Eq. 8]

        Per-class sigmoid heads (multi-label formulation):
        Each voxel can belong to multiple classes simultaneously,
        supporting multi-organ + tumor co-segmentation.
        NOTE: "exactly one label per voxel" in paper is a writing error.
              This implementation uses per-class sigmoid throughout.

        Args:
            seg_queries_refined: (B, K, C) — refined class-specific queries U_s
            dense_feat:          (B, C, H, W, D) — projected decoder feature map phi
        Returns:
            logits: (B, K, H, W, D)
        """
        B, K, C = seg_queries_refined.shape
        _, _, H, W, D = dense_feat.shape

        # Normalize queries and features for stable dot product
        q = F.normalize(seg_queries_refined, dim=-1)      # (B, K, C)
        f = F.normalize(dense_feat, dim=1)                 # (B, C, H, W, D)

        # Reshape for batch matrix multiply
        f_flat = f.view(B, C, -1)                          # (B, C, HWD)
        logits_flat = torch.bmm(q, f_flat)                 # (B, K, HWD)
        logits = logits_flat.view(B, K, H, W, D)           # (B, K, H, W, D)
        return logits

    # ─────────────────────────────────────────────────────────────────────
    # Forward Pass
    # ─────────────────────────────────────────────────────────────────────

    def forward(self, x, pdp_masks=None):
        """
        Full TRIPROMPT forward pass.

        Args:
            x:          Input CT volume (B, 1, H, W, D)
            pdp_masks:  List of K binary masks (B, 1, H, W, D), one per class,
                        each sampled from a DIFFERENT subject (s != i).
                        Retrieved from frozen training-split mask bank at inference.
                        If None, zero deformation conditioning is used (fallback).

        Returns:
            logits:      (B, K, H, W, D) — per-class sigmoid logits
            query_emb:   (B, K, C)       — L2-normalized refined seg queries [L_ALIGN_1]
            prompt_emb:  (B, K, C)       — L2-normalized fused prompt p_c    [L_ALIGN_1]
            struct_emb:  (B, K, C)       — L2-normalized Q_a_hat             [L_ALIGN_2]
            text_emb:    (B, K, C)       — L2-normalized Q_t_hat             [L_ALIGN_2]
        """
        B = x.shape[0]
        device = x.device

        # ── Step 1: Backbone — multi-scale features + dense map Z ─────────
        multi_scale_features, dense_decoder_out = self.backbone(x)
        # dense_decoder_out: (B, feature_size, H, W, D)

        # Project dense decoder output to embed_dim
        dense_feat = self.feat_proj(dense_decoder_out)    # (B, embed_dim, H, W, D)

        # ── Step 2: Structural Prompt Q_a ─────────────────────────────────
        # Generated fully automatically by co-trained region-proposal head
        # No human input required at test time
        backbone_feat_for_proposal = multi_scale_features[0]  # shallowest scale
        Q_a = self.structural_encoder(x, backbone_feat_for_proposal)  # (B, K, C)

        # ── Step 3: Text Prompt Q_t ───────────────────────────────────────
        # Class-level ClinicalBERT embeddings (same for all samples in batch)
        Q_t = self.text_encoder.forward_batch(B, device=device)       # (B, K, C)

        # ── Step 4: Deformation Prompt Q_d ───────────────────────────────
        # PDP: binary masks from OTHER training subjects (s != i)
        # Compact bottleneck ensures only population-level geometry is encoded
        if pdp_masks is not None:
            assert len(pdp_masks) == self.num_classes, (
                f"Expected {self.num_classes} PDP masks, got {len(pdp_masks)}"
            )
            Q_d = self.deformation_encoder(pdp_masks)                 # (B, K, C)
        else:
            # Fallback: zero deformation conditioning
            # Model relies on Q_a and Q_t only (as per paper Sec. 2.1 Note)
            Q_d = torch.zeros(B, self.num_classes, self.embed_dim, device=device)

        # ── Step 5: Segmentation Queries Q_s ─────────────────────────────
        query_idx = torch.arange(self.num_classes, device=device)
        Q_s = self.seg_queries(query_idx)                              # (K, C)
        Q_s = Q_s.unsqueeze(0).expand(B, -1, -1)                      # (B, K, C)

        # ── Step 6: TriPrompt Aligner ─────────────────────────────────────
        # Hard routing for Q_a, Q_d (spatially localized, benefit from discrete assignment)
        # Soft attention for Q_s, Q_t (global semantic cues)
        # Returns O_s: context-aligned segmentation queries
        O_s = self.aligner(Q_s, Q_t, Q_a, Q_d, multi_scale_features)  # (B, K, C)

        # Also retrieve intermediate refined tokens for alignment losses
        # (We re-run integrator to get Q_a_hat and Q_t_hat for L_ALIGN_2)
        Q_s_hat, Q_t_hat, Q_a_hat, Q_d_hat = self.aligner.query_integrator(
            Q_s, Q_t, Q_a, Q_d, multi_scale_features
        )

        # ── Step 7: Voxel-wise segmentation (Eq. 8) ──────────────────────
        # Per-class sigmoid: each voxel independently classified per class
        logits = self.decode_queries_to_logits(O_s, dense_feat)        # (B, K, H, W, D)

        # ── Step 8: Prepare contrastive alignment embeddings ─────────────
        # L_ALIGN_1: seg query <-> fused prompt p_c = Q_a_hat + Q_t_hat + Q_d_hat
        p_c         = Q_a_hat + Q_t_hat + Q_d_hat                      # (B, K, C)
        query_emb   = F.normalize(O_s,  dim=-1)                        # (B, K, C)
        prompt_emb  = F.normalize(p_c,  dim=-1)                        # (B, K, C)

        # L_ALIGN_2: structural <-> text prompt consistency
        struct_emb  = F.normalize(Q_a_hat, dim=-1)                     # (B, K, C)
        text_emb    = F.normalize(Q_t_hat, dim=-1)                     # (B, K, C)

        return logits, query_emb, prompt_emb, struct_emb, text_emb

    # ─────────────────────────────────────────────────────────────────────
    # Inference-only forward (returns logits only)
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, x, pdp_masks=None, threshold=0.5):
        """
        Convenience method for inference: returns binary segmentation mask.

        Args:
            x:          Input CT volume (B, 1, H, W, D)
            pdp_masks:  List of K binary PDP masks (optional)
            threshold:  Sigmoid binarization threshold (default=0.5)
        Returns:
            preds: (B, K, H, W, D) binary predictions
        """
        logits, _, _, _, _ = self.forward(x, pdp_masks=pdp_masks)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()

    # ─────────────────────────────────────────────────────────────────────
    # Parameter groups for optimizer (backbone vs. prompt modules)
    # ─────────────────────────────────────────────────────────────────────

    def get_param_groups(self, backbone_lr_scale=0.1):
        """
        Return parameter groups with separate LR for backbone and prompt modules.
        Backbone uses a lower LR (pretrained weights) while prompt modules use full LR.

        Args:
            backbone_lr_scale: LR multiplier for backbone (default=0.1)
        Returns:
            list of parameter group dicts for AdamW
        """
        backbone_params = list(self.backbone.parameters())
        backbone_ids    = set(id(p) for p in backbone_params)

        prompt_params = [
            p for p in self.parameters()
            if id(p) not in backbone_ids and p.requires_grad
        ]

        return [
            {"params": backbone_params, "lr_scale": backbone_lr_scale},
            {"params": prompt_params,   "lr_scale": 1.0},
        ]

    def count_parameters(self):
        """Print parameter counts per module."""
        modules = {
            "Backbone":              self.backbone,
            "StructuralEncoder":     self.structural_encoder,
            "TextEncoder":           self.text_encoder,
            "DeformationEncoder":    self.deformation_encoder,
            "TriPromptAligner":      self.aligner,
            "SegHead+Queries":       nn.ModuleList([self.seg_head,
                                                    self.seg_queries,
                                                    self.feat_proj]),
        }
        total = 0
        print("\n[TRIPROMPT] Parameter counts:")
        print(f"{'Module':<25} {'Params':>12}")
        print("-" * 40)
        for name, mod in modules.items():
            n = sum(p.numel() for p in mod.parameters())
            total += n
            print(f"{name:<25} {n:>12,}")
        print("-" * 40)
        print(f"{'Total':<25} {total:>12,}\n")
        return total
