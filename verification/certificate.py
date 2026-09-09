"""Data structures for Schedulability Certificates."""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class CertificateResult:
    """Outcome of polynomial-time RTA certificate verification."""
    is_valid: bool
    deadline_passed: bool
    recurrence_passed: bool
    predicted_R: np.ndarray
    deadline_violations: List[int]
    recurrence_violations: List[int]
    verification_time_us: float

    def __str__(self) -> str:
        if self.is_valid:
            return "CERTIFICATE VALID: System is provably schedulable."
        reasons = []
        if not self.deadline_passed:
            reasons.append(f"Deadline exceeded at tasks {self.deadline_violations}")
        if not self.recurrence_passed:
            reasons.append(f"Recurrence violated at tasks {self.recurrence_violations}")
        return f"CERTIFICATE INVALID: {', '.join(reasons)}"
