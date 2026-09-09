"""Classical Polynomial-Time Verifier for Predicted Response Times (Baruah et al. 2025)."""
import time
from typing import Dict, List, Tuple
import numpy as np
import torch

def verify_certificate(
    c_vals: np.ndarray,
    d_vals: np.ndarray,
    periods: np.ndarray,
    r_pred: np.ndarray,
) -> Tuple[bool, Dict[str, any]]:
    n = len(c_vals)
    r_full = np.zeros(n, dtype=np.int64)
    r_full[0] = int(round(c_vals[0]))
    for i in range(1, n):
        r_full[i] = int(round(r_pred[i - 1]))

    for i in range(n):
        if r_full[i] > d_vals[i]:
            return False, {"failed_task": i, "reason": "deadline_miss"}

        interference = 0
        for j in range(i):
            num_releases = (r_full[i] + periods[j] - 1) // periods[j]
            interference += num_releases * c_vals[j]
        f_r = c_vals[i] + interference
        if r_full[i] < f_r:
            return False, {"failed_task": i, "reason": "recurrence_violation"}

    return True, {"verified": True}
