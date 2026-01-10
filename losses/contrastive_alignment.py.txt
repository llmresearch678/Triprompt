import torch
import torch.nn.functional as F

def contrastive_loss(qs, qp, temperature=0.07):
    qs = F.normalize(qs, dim=1)
    qp = F.normalize(qp, dim=1)
    logits = qs @ qp.T / temperature
    labels = torch.arange(qs.size(0)).to(qs.device)
    return F.cross_entropy(logits, labels)
