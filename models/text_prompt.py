"""
text_prompt.py
Medical Text Prompt Encoder (E_TEXT) for TRIPROMPT.
Uses pretrained ClinicalBERT to encode class-specific medical descriptions into Q_t.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# Default class-specific medical text descriptions
DEFAULT_CLASS_DESCRIPTIONS = [
    "liver: large solid abdominal organ responsible for metabolism and detoxification",
    "right kidney: bean-shaped retroperitoneal organ for urine filtration",
    "spleen: lymphatic organ in the upper left abdomen for blood filtration",
    "pancreas: elongated glandular organ producing digestive enzymes and insulin",
    "aorta: main arterial trunk descending through the thorax and abdomen",
    "inferior vena cava: large vein returning deoxygenated blood to the heart",
    "right adrenal gland: small triangular endocrine gland above the right kidney",
    "left adrenal gland: small triangular endocrine gland above the left kidney",
    "gallbladder: pear-shaped sac beneath the liver storing bile",
    "esophagus: muscular tube connecting the pharynx to the stomach",
    "stomach: J-shaped digestive organ between the esophagus and small intestine",
    "duodenum: first segment of the small intestine receiving chyme from the stomach",
    "left kidney: bean-shaped retroperitoneal organ for urine filtration",
]


class TextPromptEncoder(nn.Module):
    """
    E_TEXT: Maps class-specific medical text descriptions into semantic embeddings Q_t.
    Uses ClinicalBERT as the pretrained language encoder.
    Text prompts are class-level (not patient-specific) and fixed at inference.
    """

    def __init__(
        self,
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        embed_dim=256,
        num_classes=13,
        class_descriptions=None,
        freeze_encoder=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.class_descriptions = class_descriptions or DEFAULT_CLASS_DESCRIPTIONS
        assert len(self.class_descriptions) == num_classes, \
            f"Expected {num_classes} descriptions, got {len(self.class_descriptions)}"

        # Load ClinicalBERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Projection head: BERT hidden size (768) -> embed_dim
        self.projection = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        # Learnable textual prompt projection W_t
        self.W_t = nn.Linear(embed_dim, embed_dim, bias=False)

    def encode_texts(self, descriptions, device):
        """
        Tokenize and encode a list of text descriptions using ClinicalBERT.
        Returns CLS token embeddings of shape (len(descriptions), 768).
        """
        tokens = self.tokenizer(
            descriptions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.bert(**tokens)

        # Use CLS token as sentence representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (N, 768)
        return cls_embeddings

    def forward(self, device=None):
        """
        Encode all class descriptions and return text prompt tokens Q_t.
        Returns:
            Q_t: text prompt tokens (num_classes, embed_dim)
        """
        if device is None:
            device = next(self.parameters()).device

        # Encode all class descriptions
        cls_embeddings = self.encode_texts(self.class_descriptions, device)  # (K, 768)
        projected = self.projection(cls_embeddings)                           # (K, embed_dim)
        Q_t = self.W_t(projected)                                             # (K, embed_dim)
        return Q_t

    def forward_batch(self, batch_size, device=None):
        """
        Returns Q_t expanded for a batch.
        Returns:
            Q_t: (B, num_classes, embed_dim)
        """
        Q_t = self.forward(device)                             # (K, embed_dim)
        return Q_t.unsqueeze(0).expand(batch_size, -1, -1)    # (B, K, embed_dim)
