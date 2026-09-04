"""Exact Response-Time Analysis (RTA) for Fixed-Priority Uniprocessor Scheduling.

Implements the classical Joseph & Pandya (1986) recurrence:
    R_i = C_i + \sum_{j \in hp(i)} \lceil R_i / T_j \rceil C_j
under Deadline-Monotonic (DM) priority assignment.
"""
from typing import Tuple
import numpy as np

def exact_rta(
    C: np.ndarray,
    D: np.ndarray,
    T: np.ndarray,
    max_iterations: int = 1000,
    max_bound: int = 100000,
) -> Tuple[bool, np.ndarray]:
    n = len(C)
    R = np.zeros(n, dtype=np.int64)
    C_int = np.round(C).astype(np.int64)
    D_int = np.round(D).astype(np.int64)
    T_int = np.round(T).astype(np.int64)

    R[0] = C_int[0]
    for i in range(1, n):
        r_curr = C_int[i]
        converged = False
        for _ in range(max_iterations):
            interference = 0
            for j in range(i):
                num_releases = (r_curr + T_int[j] - 1) // T_int[j]
                interference += num_releases * C_int[j]
            r_next = C_int[i] + interference
            if r_next == r_curr:
                R[i] = r_next
                converged = True
                break
            if r_next > max_bound:
                R[i] = r_next
                converged = True
                break
            r_curr = r_next
        if not converged:
            R[i] = r_curr

    is_schedulable = bool(np.all(R <= D_int))
    return is_schedulable, R
