"""Training pipeline for baseline neural schedulability models."""
import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models.baseline_mlp import ResponseTimeMLP, JointSchedulabilityMLP

class AsymmetricNormalizedMSELoss(nn.Module):
    def __init__(self, weight: float = 100.0, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = pred - target
        norm_factor = target + self.eps
        normalized_error = error / norm_factor
        weights = torch.where(error < 0, torch.full_like(error, self.weight), torch.ones_like(error))
        return (weights * (normalized_error ** 2)).mean()

def train_joint(
    data_npz: str,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    w: float = 100.0,
    save_path: Optional[str] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(data_npz)
    n = int(data["n"])
    model = JointSchedulabilityMLP(n=n).to(device)
    print(f"Training JointSchedulabilityMLP on {device}...")
    return model
