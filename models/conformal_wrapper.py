"""Split Conformal Prediction Wrapper for Schedulability Analysis.

Provides statistical control over classification uncertainty:
- Primary non-conformity score: S = 1 - p_hat(y | x)
- Calibrated empirical quantile: q_hat_alpha
- Bounded prediction sets: C(x) in {{Sched}, {Unsched}, {Sched, Unsched}}
- Safety guarantee: Classical RTA certificate verifier is ALWAYS run
  on any "Schedulable" claim before admitting a task set.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch
import torch.nn as nn

try:
    from verification.rta_verifier import verify_single_taskset
    from verification.certificate import CertificateResult
except ImportError:
    pass


@dataclass
class TriagedOutput:
    """Outcome of Conformal Prediction + Classical Verification."""
    prediction_set: Set[str]
    is_uncertain: bool
    status: str
    is_safe: bool
    certificate_result: Optional[object] = None


class ConformalSchedulabilityWrapper:
    """Wraps a schedulability model with Split Conformal Prediction."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.cal_scores: Optional[np.ndarray] = None
        self.m: int = 0
        self.quantiles: Dict[float, float] = {}

    def compute_nonconformity(self, p_sched: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Computes score S = 1 - p_hat(y | x)."""
        p_true = np.where(y_true == 1, p_sched, 1.0 - p_sched)
        return 1.0 - p_true

    def calibrate(self, p_sched_cal: np.ndarray, y_true_cal: np.ndarray):
        """Calibrates non-conformity scores on hold-out calibration split."""
        self.cal_scores = self.compute_nonconformity(p_sched_cal, y_true_cal)
        self.m = len(self.cal_scores)
        self.quantiles.clear()

    def get_quantile(self, alpha: float) -> float:
        """Computes empirical quantile with finite-sample correction."""
        if self.cal_scores is None:
            raise RuntimeError("Wrapper is not calibrated. Call calibrate() first.")
        if alpha not in self.quantiles:
            level = min(1.0, np.ceil((self.m + 1) * (1.0 - alpha)) / self.m)
            self.quantiles[alpha] = float(np.quantile(self.cal_scores, level, method="higher"))
        return self.quantiles[alpha]

    def predict_set(self, p_sched: float, alpha: float) -> Set[str]:
        """Constructs prediction set for an individual instance."""
        q_hat = self.get_quantile(alpha)
        threshold = 1.0 - q_hat

        p_set = set()
        if p_sched >= threshold:
            p_set.add("Schedulable")
        if p_sched <= q_hat:
            p_set.add("Unschedulable")
        if len(p_set) == 0:
            p_set.add("Schedulable" if p_sched >= 0.5 else "Unschedulable")
        return p_set

    def predict_batch_sets(self, p_sched: np.ndarray, alpha: float) -> List[Set[str]]:
        """Constructs prediction sets for a batch of predictions."""
        q_hat = self.get_quantile(alpha)
        threshold = 1.0 - q_hat

        sets = []
        for p in p_sched:
            s = set()
            if p >= threshold:
                s.add("Schedulable")
            if p <= q_hat:
                s.add("Unschedulable")
            if len(s) == 0:
                s.add("Schedulable" if p >= 0.5 else "Unschedulable")
            sets.append(s)
        return sets

    def triage_taskset(
        self,
        C: np.ndarray,
        D: np.ndarray,
        T: np.ndarray,
        p_sched: float,
        R_pred: np.ndarray,
        alpha: float,
    ) -> TriagedOutput:
        """Full triage pipeline combining CP confidence with RTA certificate verification."""
        from verification.rta_verifier import verify_single_taskset

        p_set = self.predict_set(p_sched, alpha=alpha)
        is_uncertain = len(p_set) > 1

        cert_res = None
        # Strict Safety: Only accept Schedulable if the RTA certificate passes
        if "Schedulable" in p_set:
            cert_res = verify_single_taskset(C, D, T, R_pred)
            if cert_res.is_valid:
                status = "VERIFIED_SCHEDULABLE"
            else:
                status = "UNCERTAIN_CERT_FAILED" if is_uncertain else "FALLBACK_RTA_CERT_FAILED"
        else:
            status = "FAST_REJECT"

        return TriagedOutput(
            prediction_set=p_set,
            is_uncertain=is_uncertain,
            status=status,
            is_safe=True,  # Verifier guarantees zero unsafe admissions
            certificate_result=cert_res,
        )
