"""Evaluation metrics for Real-Time Schedulability Analysis.

Distinguishes between:
1. Safety False Positive Rate (FPR): unschedulable systems certified as schedulable (MUST BE 0.000%)
2. Operational False Rejection Rate (FRR): schedulable systems rejected by conservative certificate failure
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class SchedulabilityMetrics:
    total_samples: int
    true_schedulable: int
    true_unschedulable: int
    tp: int
    fp: int  # Safety false positives
    tn: int
    fn: int  # Operational false rejections

    predictive_accuracy: float
    acceptance_rate: float  # Sensitivity / TPR: TP / (TP + FN)
    safety_fpr: float       # FP / (FP + TN)
    operational_frr: float  # FN / (TP + FN)

    conformal_coverage: Optional[float] = None
    uncertainty_rate: Optional[float] = None


def compute_metrics(
    y_true: np.ndarray,
    is_certified: np.ndarray,
    prediction_sets: Optional[List[set]] = None,
) -> SchedulabilityMetrics:
    """Computes safety and performance metrics."""
    N = len(y_true)
    y_t = np.array(y_true, dtype=int)
    cert = np.array(is_certified, dtype=bool)

    n_sched = int(np.sum(y_t == 1))
    n_unsched = int(np.sum(y_t == 0))

    tp = int(np.sum((y_t == 1) & cert))
    fp = int(np.sum((y_t == 0) & cert))
    tn = int(np.sum((y_t == 0) & (~cert)))
    fn = int(np.sum((y_t == 1) & (~cert)))

    accuracy = (tp + tn) / max(1, N)
    acceptance = tp / max(1, n_sched) if n_sched > 0 else 0.0
    safety_fpr = fp / max(1, n_unsched) if n_unsched > 0 else 0.0
    operational_frr = fn / max(1, n_sched) if n_sched > 0 else 0.0

    conformal_coverage = None
    uncertainty_rate = None

    if prediction_sets is not None:
        covered = 0
        uncertain = 0
        for idx, p_set in enumerate(prediction_sets):
            label = "Schedulable" if y_t[idx] == 1 else "Unschedulable"
            if label in p_set:
                covered += 1
            if len(p_set) > 1:
                uncertain += 1

        conformal_coverage = covered / max(1, N)
        uncertainty_rate = uncertain / max(1, N)

    return SchedulabilityMetrics(
        total_samples=N,
        true_schedulable=n_sched,
        true_unschedulable=n_unsched,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        predictive_accuracy=accuracy,
        acceptance_rate=acceptance,
        safety_fpr=safety_fpr,
        operational_frr=operational_frr,
        conformal_coverage=conformal_coverage,
        uncertainty_rate=uncertainty_rate,
    )
