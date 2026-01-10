import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TextPromptEncoder(nn.Module):
    """
    Text Prompt Encoder (Qt).

    This module encodes medical semantic information using a
    pretrained clinical language model and projects it into
    a shared embedding space for multimodal alignment with
    segmentation queries and other prompts.

    IMPORTANT (Paper-aligned design):
    --------------------------------
    1. Uses a pretrained clinical language model (no architectural changes).
    2. Encodes *semantic priors only* (organ names, anatomy descriptions).
    3. Does NOT access image data or segmentation masks.
    4. Outputs a fixed-dimensional prompt embedding per class.

    This design ensures semantic conditioning without data leakage
    or task-specific bias.
    """

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 768,
        freeze_text_encoder: bool = True
    ):
        """
        Args:
            model_name (str):
                Name or path of the pretrained clinical language model.
            embed_dim (int):
                Dimensionality of the shared embedding space.
            freeze_text_encoder (bool):
                Whether to freeze the language model weights.
                Default=True for training stability and reproducibility.
        """
        super().__init__()

        # ---------------------------------------------------
        # 1. Pretrained medical language encoder
        # ---------------------------------------------------
        self.text_encoder = AutoModel.from_pretrained(model_name)

        # Optionally freeze the language model
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # ---------------------------------------------------
        # 2. Projection head
        # ---------------------------------------------------
        # Projects language embeddings into the shared
        # prompt embedding space used by TRIPROMPT
        self.proj = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for text prompt encoding.

        Args:
            input_ids (Tensor):
                Tokenized text input IDs of shape (B, L),
                where L is the token sequence length.
            attention_mask (Tensor):
                Attention mask of shape (B, L).

        Returns:
            Tensor:
                Text prompt embeddings of shape (B, embed_dim).
        """

        # ---------------------------------------------------
        # 1. Encode text semantics
        # ---------------------------------------------------
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # ---------------------------------------------------
        # 2. CLS token pooling
        # ---------------------------------------------------
        # The CLS token is used as a global semantic summary
        # of the medical text description.
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # ---------------------------------------------------
        # 3. Projection to shared embedding space
        # ---------------------------------------------------
        text_prompt = self.proj(cls_embedding)

        return text_prompt
