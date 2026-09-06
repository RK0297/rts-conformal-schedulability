"""Synthetic task set generator for Fixed-Priority constrained-deadline systems.

Implements the workload synthesis protocol from Baruah et al. (2025):
- UUniSort utilization distribution (Bini & Buttazzo 2005)
- Uniform period distribution in [1, 1000] (or log-uniform)
- Execution times C_i = round(U_i * T_i) with C_i >= 1
- Deadlines D_i uniformly sampled in [C_i, T_i]
- Deadline-Monotonic (DM) priority sorting (ascending D_i)
- Ground-truth RTA labeling via exact Joseph & Pandya recurrence
- Partitions data into 70% Train, 15% Calibration, 15% Test
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    from data.exact_rta import exact_rta
except ImportError:
    from exact_rta import exact_rta

def uunisort(n: int, target_u: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    points = np.sort(rng.uniform(0.0, target_u, size=n - 1))
    utilizations = np.zeros(n, dtype=np.float64)
    utilizations[0] = points[0]
    for i in range(1, n - 1):
        utilizations[i] = points[i] - points[i - 1]
    utilizations[n - 1] = target_u - points[-1]
    utilizations = np.maximum(utilizations, 1e-5)
    return utilizations * (target_u / np.sum(utilizations))

def generate_single_taskset(
    n: int,
    target_u: float,
    period_min: int = 1,
    period_max: int = 1000,
    log_uniform: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    u_tasks = uunisort(n, target_u, rng=rng)
    if log_uniform:
        log_min = np.log(max(1.0, float(period_min)))
        log_max = np.log(float(period_max))
        periods = np.round(np.exp(rng.uniform(log_min, log_max, size=n))).astype(np.int64)
        periods = np.clip(periods, period_min, period_max)
    else:
        periods = rng.integers(period_min, period_max + 1, size=n, dtype=np.int64)

    c_vals = np.round(u_tasks * periods).astype(np.int64)
    c_vals = np.clip(c_vals, 1, periods)

    d_vals = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if c_vals[i] == periods[i]:
            d_vals[i] = periods[i]
        else:
            d_vals[i] = rng.integers(int(c_vals[i]), int(periods[i]) + 1)

    # DM Sorting
    dm_order = np.argsort(d_vals)
    c_vals = c_vals[dm_order]
    d_vals = d_vals[dm_order]
    periods = periods[dm_order]

    is_sched, r_true = exact_rta(c_vals, d_vals, periods)

    feats_reg = []
    feats_bin = []
    for i in range(n):
        feats_reg.extend([c_vals[i], periods[i], 1.0 / periods[i]])
        feats_bin.extend([c_vals[i], periods[i], 1.0 / periods[i], d_vals[i]])

    return {
        "x_reg": np.array(feats_reg, dtype=np.float32),
        "x_bin": np.array(feats_bin, dtype=np.float32),
        "y_reg": np.array(r_true[1:], dtype=np.float32),
        "y_bin": 1.0 if is_sched else 0.0,
        "C": c_vals,
        "D": d_vals,
        "T": periods,
        "R": r_true,
    }
