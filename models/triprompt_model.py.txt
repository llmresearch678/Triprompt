import torch
import torch.nn as nn
from .backbone import Backbone
from .triprompt_aligner import TriPromptAligner

class TriPromptModel(nn.Module):
    def __init__(self, backbone, aligner, num_classes):
        super().__init__()
        self.backbone = backbone
        self.aligner = aligner
        self.head = nn.Conv3d(backbone.encoder.out_channels, num_classes, 1)

    def forward(self, x, qs, qa, qt, qd):
        feats = self.backbone(x)
        qs_refined = self.aligner(qs, qa, qt, qd)
        logits = self.head(feats)
        return logits, qs_refined
