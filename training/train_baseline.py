"""Training pipeline for baseline neural schedulability models."""
import torch
import torch.nn as nn

class AsymmetricNormalizedMSELoss(nn.Module):
    """Asymmetric loss function from Baruah et al. (2025).

    Penalizes underestimations (R' < R) by factor w to avoid optimistic certificates.
    """
    def __init__(self, weight: float = 100.0, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = pred - target
        norm_factor = target + self.eps
        normalized_error = error / norm_factor

        weights = torch.where(error < 0, torch.full_like(error, self.weight), torch.ones_like(error))
        loss = weights * (normalized_error ** 2)
        return loss.mean()
