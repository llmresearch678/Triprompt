import torch
import torch.nn as nn
from transformers import AutoModel

class TextPromptEncoder(nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", embed_dim=768):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.proj(cls)
