"""Evaluation Metrics for Learning-Assisted Real-Time Schedulability Analysis."""
from typing import Dict, List, Optional
import numpy as np

def compute_schedulability_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_verified: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    n_total = len(y_true)

    tp = np.sum(y_true & y_pred)
    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)

    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0
    tpr = tp / np.sum(y_true) if np.sum(y_true) > 0 else 0.0
    fpr = fp / np.sum(~y_true) if np.sum(~y_true) > 0 else 0.0
    frr = fn / np.sum(y_true) if np.sum(y_true) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "acceptance_rate_tpr": tpr,
        "safety_fpr": fpr,
        "operational_frr": frr,
    }
