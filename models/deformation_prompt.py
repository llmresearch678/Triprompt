import torch
import torch.nn as nn

class DeformationPromptEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, embed_dim, 3, padding=1),
            nn.AdaptiveAvgPool3d(1)
        )

    def forward(self, mask):
        z = self.encoder(mask)
        return z.flatten(1)
