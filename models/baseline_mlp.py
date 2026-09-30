"""PyTorch neural network architectures for Response-Time regression.

Updated with 128 -> 64 -> 64 hidden layers and asymmetric loss default w=25.
"""
import torch
import torch.nn as nn
from typing import Tuple


class AsymmetricNormalizedMSELoss(nn.Module):
    """Normalized MSE with asymmetric weighting for negative errors (w=25)."""

    def __init__(self, weight: float = 25.0, eps: float = 1e-6):
        super().__init__()
        self.weight = float(weight)
        self.eps = float(eps)

    def forward(self, r_pred: torch.Tensor, r_true: torch.Tensor) -> torch.Tensor:
        r_true_safe = torch.clamp(r_true, min=self.eps)
        norm_error = (r_pred - r_true) / r_true_safe
        weights = torch.where(norm_error < 0.0, self.weight, 1.0)
        return torch.mean((weights * norm_error) ** 2)


class ResponseTimeMLP(nn.Module):
    """Regression MLP predicting R'_2 .. R'_n with 128 -> 64 -> 64 hidden layers."""

    def __init__(self, n: int):
        super().__init__()
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        self.n = n

        self.net = nn.Sequential(
            nn.Linear(3 * n, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n - 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointSchedulabilityMLP(nn.Module):
    """Unified network with 128 -> 64 -> 64 hidden layers."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

        # Regression branch (C, T, 1/T)
        self.reg_net = nn.Sequential(
            nn.Linear(3 * n, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n - 1),
            nn.Softplus(),
        )

        # Classification branch (C, T, 1/T, D)
        self.clf_net = nn.Sequential(
            nn.Linear(4 * n, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_reg: torch.Tensor, x_clf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r_pred = self.reg_net(x_reg)
        p_sched = self.clf_net(x_clf).squeeze(-1)
        return r_pred, p_sched