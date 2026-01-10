import torch
import torch.nn as nn

class StructuralPromptEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

    def forward(self, x):
        q = self.encoder(x)
        return q.flatten(1)
