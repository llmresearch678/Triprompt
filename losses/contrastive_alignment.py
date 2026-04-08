"""
losses/contrastive_alignment.py
Contrastive Alignment Losses for TRIPROMPT.

Implements both alignment objectives from Eq. (11) and Eq. (12):

L_ALIGN_1 (segmentation-to-prompt, Eq. 11):
    Anchors each segmentation query U_s^(c) to its matched fused
    prompt representation p_c = Q_a_hat^(c) + Q_t_hat^(c) + Q_d_hat^(c).
    Enforces semantic and physiological grounding.

L_ALIGN_2 (prompt-to-prompt, Eq. 12):
    Encourages consistency between structural prompt Q_a_hat^(c)
    and textual prompt Q_t_hat^(c) across classes.
    Promotes cross-modal semantic alignment.

Both losses use:
    - L2 normalization (cosine similarity)
    - Temperature-scaled cross-entropy (InfoNCE / NT-Xent style)
    - Batch-averaged formulation: (B, K, C) inputs
    - Valid class masking: absent classes ignored in loss computation
      (supports multi-dataset training with heterogeneous label spaces)
"""

import torch
import torch.nn.functional as F


def contrastive_alignment_loss(
    query_emb,
    prompt_emb,
    temperature: float = 0.07,
    reduction: str = "mean",
    valid_mask=None,
):
    """
    Contrastive Alignment Loss (L_ALIGN_1 or L_ALIGN_2).

    Supports both (K, C) and (B, K, C) input shapes.
    When inputs are (B, K, C), loss is computed per sample and averaged over batch.

    For L_ALIGN_1: query_emb = U_s (seg queries), prompt_emb = p_c (fused prompts)
    For L_ALIGN_2: query_emb = Q_a_hat (structural), prompt_emb = Q_t_hat (textual)

    Positive pairs: diagonal entries query[c] <-> prompt[c]
    Negative pairs: all off-diagonal entries (other classes)

    Args:
        query_emb  (Tensor): (K, C) or (B, K, C) — L2-normalized query embeddings
        prompt_emb (Tensor): (K, C) or (B, K, C) — L2-normalized prompt embeddings
        temperature (float): sharpness of similarity distribution (default=0.07)
        reduction  (str):    'mean', 'sum', or 'none' — reduction over classes
        valid_mask (Tensor): (K,) or (B, K) bool — True if class is annotated.
                             Absent classes are masked out to avoid false negatives
                             during multi-dataset joint training.

    Returns:
        Scalar loss tensor (or per-class if reduction='none')
    """
    # ── Handle both (K, C) and (B, K, C) inputs ──────────────────────────
    if query_emb.dim() == 2:
        # Single sample: add batch dim -> (1, K, C)
        query_emb  = query_emb.unsqueeze(0)
        prompt_emb = prompt_emb.unsqueeze(0)
        if valid_mask is not None and valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    B, K, C = query_emb.shape

    # ── L2 normalization (cosine similarity) ─────────────────────────────
    # Critical for:
    # - Preventing embedding magnitude collapse
    # - Stable cosine-based contrastive learning
    # - Multi-prompt alignment as per Eq. (11)-(12)
    query_emb  = F.normalize(query_emb,  dim=-1)   # (B, K, C)
    prompt_emb = F.normalize(prompt_emb, dim=-1)   # (B, K, C)

    # ── Compute loss per sample in batch ─────────────────────────────────
    batch_losses = []
    for b in range(B):
        q = query_emb[b]   # (K, C)
        p = prompt_emb[b]  # (K, C)

        # ── Valid class masking ───────────────────────────────────────────
        # Classes absent in this dataset are excluded from loss computation
        # to avoid false negatives during multi-dataset joint training
        if valid_mask is not None:
            vm = valid_mask[b] if valid_mask.dim() == 2 else valid_mask  # (K,)
            vm = vm.bool()
            if vm.sum() == 0:
                # No valid classes for this sample — skip
                batch_losses.append(torch.tensor(0.0, device=q.device))
                continue
            q = q[vm]  # (K', C)
            p = p[vm]  # (K', C)

        K_valid = q.shape[0]

        # ── Similarity matrix (K' x K') ───────────────────────────────────
        # query[c] should align with its corresponding prompt[c] (diagonal)
        # and be contrasted against all other class prompts (off-diagonal)
        logits = torch.matmul(q, p.T) / temperature  # (K', K')

        # ── Ground-truth: positive pairs on diagonal ──────────────────────
        labels = torch.arange(K_valid, device=logits.device)

        # ── Cross-entropy contrastive loss ────────────────────────────────
        loss_b = F.cross_entropy(logits, labels, reduction=reduction)
        batch_losses.append(loss_b)

    # ── Average over batch ────────────────────────────────────────────────
    if reduction == "none":
        # Stack per-class losses: (B, K')
        loss = torch.stack(batch_losses, dim=0)
    else:
        loss = torch.stack(batch_losses).mean()

    if squeeze_batch and reduction != "none":
        pass  # Already scalar

    return loss


def segmentation_prompt_alignment_loss(
    seg_queries,
    struct_emb,
    text_emb,
    deform_emb,
    temperature: float = 0.07,
    valid_mask=None,
):
    """
    L_ALIGN_1 (Eq. 11): Segmentation-to-Prompt Contrastive Loss.

    Anchors each refined segmentation query U_s^(c) to its matched
    fused prompt p_c = Q_a_hat^(c) + Q_t_hat^(c) + Q_d_hat^(c).

    Args:
        seg_queries (Tensor): (B, K, C) — refined segmentation queries U_s
        struct_emb  (Tensor): (B, K, C) — Q_a_hat
        text_emb    (Tensor): (B, K, C) — Q_t_hat
        deform_emb  (Tensor): (B, K, C) — Q_d_hat
        temperature (float):  contrastive temperature
        valid_mask  (Tensor): (B, K) or (K,) — valid class mask

    Returns:
        Scalar L_ALIGN_1 loss
    """
    # Fused prompt: p_c = Q_a_hat + Q_t_hat + Q_d_hat (Eq. 11)
    fused_prompt = struct_emb + text_emb + deform_emb   # (B, K, C)

    return contrastive_alignment_loss(
        query_emb=seg_queries,
        prompt_emb=fused_prompt,
        temperature=temperature,
        valid_mask=valid_mask,
    )


def prompt_prompt_alignment_loss(
    struct_emb,
    text_emb,
    temperature: float = 0.07,
    valid_mask=None,
):
    """
    L_ALIGN_2 (Eq. 12): Prompt-to-Prompt Contrastive Loss.

    Encourages cross-modal semantic consistency between structural
    prompt Q_a_hat^(c) and textual prompt Q_t_hat^(c) across classes.

    L_ALIGN_2 = -1/K sum_c log [
        exp(Q_a_hat^(c) . Q_t_hat^(c) / tau) /
        sum_j exp(Q_a_hat^(c) . Q_t_hat^(j) / tau)
    ]

    Args:
        struct_emb (Tensor): (B, K, C) — Q_a_hat structural embeddings
        text_emb   (Tensor): (B, K, C) — Q_t_hat text embeddings
        temperature (float): contrastive temperature
        valid_mask  (Tensor): (B, K) or (K,) — valid class mask

    Returns:
        Scalar L_ALIGN_2 loss
    """
    return contrastive_alignment_loss(
        query_emb=struct_emb,
        prompt_emb=text_emb,
        temperature=temperature,
        valid_mask=valid_mask,
    )


def total_alignment_loss(
    seg_queries,
    struct_emb,
    text_emb,
    deform_emb,
    lambda1=0.1,
    lambda2=0.1,
    temperature=0.07,
    valid_mask=None,
):
    """
    Combined contrastive alignment loss: lambda1 * L_ALIGN_1 + lambda2 * L_ALIGN_2.

    NOTE: lambda1 and lambda2 are updated per training step via gradient-norm
    balancing in train.py (Eq. 10). This function accepts current values.

    Args:
        seg_queries (Tensor): (B, K, C)
        struct_emb  (Tensor): (B, K, C)
        text_emb    (Tensor): (B, K, C)
        deform_emb  (Tensor): (B, K, C)
        lambda1     (float):  current weight for L_ALIGN_1
        lambda2     (float):  current weight for L_ALIGN_2
        temperature (float):  shared contrastive temperature
        valid_mask  (Tensor): (B, K) or (K,)

    Returns:
        total_loss, l_align1, l_align2
    """
    l_align1 = segmentation_prompt_alignment_loss(
        seg_queries, struct_emb, text_emb, deform_emb,
        temperature=temperature, valid_mask=valid_mask,
    )
    l_align2 = prompt_prompt_alignment_loss(
        struct_emb, text_emb,
        temperature=temperature, valid_mask=valid_mask,
    )
    total = lambda1 * l_align1 + lambda2 * l_align2
    return total, l_align1, l_align2
