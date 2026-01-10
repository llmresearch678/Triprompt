import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class Backbone(nn.Module):
    def __init__(self, img_size, in_channels, feature_size):
        super().__init__()
        self.encoder = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=feature_size,
            feature_size=feature_size,
            use_checkpoint=False
        )

    def forward(self, x):
        return self.encoder(x)
