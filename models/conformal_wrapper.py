"""Split Conformal Prediction Wrapper for Schedulability Analysis."""
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

class ConformalPredictionWrapper:
    def __init__(self, model: nn.Module):
        self.model = model
        self.calibrated_quantiles: Dict[float, float] = {}

    def nonconformity_score(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        p_sched = np.clip(probs, 1e-7, 1.0 - 1e-7)
        prob_true = np.where(labels == 1, p_sched, 1.0 - p_sched)
        return 1.0 - prob_true
