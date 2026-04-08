"""
backbone.py
Swin-UNETR backbone wrapper for TRIPROMPT.
Loads pretrained Swin-UNETR and exposes multi-scale feature maps.
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class TripromptBackbone(nn.Module):
    """
    Swin-UNETR backbone for volumetric 3D feature extraction.
    Returns multi-scale feature maps F = {f_l}_{l=1}^{L}.
    """

    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=13,
        feature_size=48,
        pretrained_weights=None,
    ):
        super().__init__()
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True,
        )
        if pretrained_weights is not None:
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            # Load only encoder weights if full checkpoint
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.swin_unetr.load_state_dict(state_dict, strict=False)
            print(f"[Backbone] Loaded pretrained weights from {pretrained_weights}")

    def forward(self, x):
        """
        Args:
            x: Input volume (B, C, H, W, D)
        Returns:
            hidden_states: list of multi-scale feature tensors
            decoder_out:   dense voxel-level embedding Z (B, H, W, D, C)
        """
        hidden_states_out = self.swin_unetr.swinViT(x, self.swin_unetr.normalize)
        # Decoder forward to get dense feature map Z
        enc0 = self.swin_unetr.encoder1(x)
        enc1 = self.swin_unetr.encoder2(hidden_states_out[0])
        enc2 = self.swin_unetr.encoder3(hidden_states_out[1])
        enc3 = self.swin_unetr.encoder4(hidden_states_out[2])
        enc4 = self.swin_unetr.encoder10(hidden_states_out[4])

        dec4 = self.swin_unetr.decoder5(enc4, hidden_states_out[3])
        dec3 = self.swin_unetr.decoder4(dec4, enc3)
        dec2 = self.swin_unetr.decoder3(dec3, enc2)
        dec1 = self.swin_unetr.decoder2(dec2, enc1)
        dec0 = self.swin_unetr.decoder1(dec1, enc0)

        multi_scale_features = [enc0, enc1, enc2, enc3, enc4]
        return multi_scale_features, dec0
