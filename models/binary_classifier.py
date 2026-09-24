"""Binary classification baseline from Baruah et al. (2025) Section 4.2 (Fig. 6).

- Inputs: 4n features [C_1, T_1, 1/T_1, D_1, ..., C_n, T_n, 1/T_n, D_n] in DM order.
- Architecture: 2 hidden layers of 15 neurons each with ReLU.
- Output: 1 sigmoid node representing p_hat(Schedulable).
"""
import torch
import torch.nn as nn


class BinarySchedulabilityMLP(nn.Module):
    """2-layer binary classifier predicting schedulability probability."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

        self.net = nn.Sequential(
            nn.Linear(4 * n, 15),
            nn.ReLU(),
            nn.Linear(15, 15),
            nn.ReLU(),
            nn.Linear(15, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting p_hat(Schedulable)."""
        return self.net(x).squeeze(-1)
