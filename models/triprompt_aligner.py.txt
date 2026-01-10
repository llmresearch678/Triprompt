import torch
import torch.nn as nn

class TriPromptAligner(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8)

    def forward(self, qs, qa, qt, qd):
        prompts = torch.stack([qa, qt, qd], dim=0)
        out, _ = self.attn(qs.unsqueeze(0), prompts, prompts)
        return out.squeeze(0)
