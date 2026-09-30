"""Training pipeline for baseline neural schedulability models with strong scaling."""
import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models.baseline_mlp import ResponseTimeMLP, JointSchedulabilityMLP, AsymmetricNormalizedMSELoss
from models.binary_classifier import BinarySchedulabilityMLP

def train_joint(
    data_npz: str,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    w: float = 25.0,
    patience: int = 20,
    save_path: Optional[str] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(data_npz)
    n = int(data["n"])

    X_reg = data["train_X_reg"].copy().astype(np.float32)
    X_clf = data["train_X_bin"].copy().astype(np.float32)
    Y_reg = data["train_Y_reg"].copy().astype(np.float32)
    Y_clf = data["train_Y_bin"].copy().astype(np.float32)

    X_reg[:, 0::3] /= 1000.0
    X_reg[:, 1::3] /= 1000.0
    X_clf[:, 0::4] /= 1000.0
    X_clf[:, 1::4] /= 1000.0
    X_clf[:, 3::4] /= 1000.0

    print(f"[*] Response times before scaling: min = {Y_reg.min():.2f}, max = {Y_reg.max():.2f}")
    Y_reg /= 1000.0
    print(f"[*] Response times after scaling:  min = {Y_reg.min():.4f}, max = {Y_reg.max():.4f}")

    model = JointSchedulabilityMLP(n=n).to(device)
    return model
