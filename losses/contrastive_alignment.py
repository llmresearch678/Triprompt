import torch
import torch.nn.functional as F


def contrastive_alignment_loss(
    query_emb,
    prompt_emb,
    temperature: float = 0.07,
    reduction: str = "mean"
):
    """
    Contrastive Alignment Loss for TRIPROMPT

    This loss aligns segmentation queries with their corresponding
    prompt representations (structural + textual + deformation),
    enforcing semantic and physiological consistency as described
    in the TRIPROMPT framework.

    This implementation follows a normalized temperature-scaled
    cross-entropy (NT-Xent / InfoNCE-style) formulation.

    Args:
        query_emb (Tensor):
            Segmentation query embeddings of shape (K, C),
            where K is the number of classes (organs / tumors)
            and C is the embedding dimension.

        prompt_emb (Tensor):
            Corresponding prompt embeddings of shape (K, C),
            typically formed by combining Qa + Qt + Qd after
            alignment and projection.

        temperature (float):
            Temperature scaling parameter controlling the
            sharpness of the similarity distribution.

        reduction (str):
            Reduction mode over classes.
            Options: "mean", "sum", "none".

    Returns:
        Tensor:
            Scalar contrastive loss (or per-class loss if reduction="none").
    """

    # ---------------------------------------------------
    # 1. L2 normalization (cosine similarity)
    # ---------------------------------------------------
    # Normalization is critical to:
    # - Prevent embedding magnitude collapse
    # - Ensure cosine-similarity-based alignment
    # - Stabilize multi-prompt contrastive learning
    query_emb = F.normalize(query_emb, dim=1)
    prompt_emb = F.normalize(prompt_emb, dim=1)

    # ---------------------------------------------------
    # 2. Similarity matrix (K x K)
    # ---------------------------------------------------
    # Each query should align with its corresponding prompt
    # (diagonal entries), while being contrasted against
    # other class prompts (off-diagonal entries).
    logits = torch.matmul(query_emb, prompt_emb.T) / temperature

    # ---------------------------------------------------
    # 3. Ground-truth alignment labels
    # ---------------------------------------------------
    # Positive pairs lie on the diagonal:
    # query[c] ↔ prompt[c]
    labels = torch.arange(logits.size(0), device=logits.device)

    # ---------------------------------------------------
    # 4. Cross-entropy contrastive loss
    # ---------------------------------------------------
    loss = F.cross_entropy(logits, labels, reduction=reduction)

    return loss
