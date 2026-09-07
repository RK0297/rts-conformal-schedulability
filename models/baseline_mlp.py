"""Baseline MLP Architectures for Response Time Prediction (Baruah et al. 2025)."""
import torch
import torch.nn as nn
from typing import List, Tuple

class ResponseTimeMLP(nn.Module):
    def __init__(self, n: int, hidden_dims: List[int] = [30, 30, 30, 30]):
        super().__init__()
        self.n = n
        input_dim = 3 * n
        output_dim = n - 1

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class JointSchedulabilityMLP(nn.Module):
    def __init__(self, n: int, hidden_dims: List[int] = [30, 30, 30, 30]):
        super().__init__()
        self.n = n
        self.reg_head = ResponseTimeMLP(n=n, hidden_dims=hidden_dims)
        self.clf_head = nn.Sequential(
            nn.Linear(4 * n, 15),
            nn.ReLU(),
            nn.Linear(15, 15),
            nn.ReLU(),
            nn.Linear(15, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_reg: torch.Tensor, x_clf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r_pred = self.reg_head(x_reg)
        p_sched = self.clf_head(x_clf).squeeze(-1)
        return r_pred, p_sched
