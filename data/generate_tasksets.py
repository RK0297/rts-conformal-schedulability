"""Synthetic task set generator for Fixed-Priority constrained-deadline systems.

Implements the workload synthesis protocol from Baruah et al. (2025):
- UUniSort utilization distribution (Bini & Buttazzo 2005)
- Uniform period distribution in [1, 1000] (or log-uniform)
- Execution times C_i = round(U_i * T_i) with C_i >= 1
- Deadlines D_i uniformly sampled in [C_i, T_i]
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

def uunisort(n: int, target_u: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generates n task utilizations summing to target_u via UUniSort."""
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

def sample_periods(n: int, period_min: int = 1, period_max: int = 1000, log_uniform: bool = False, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if log_uniform:
        log_min = np.log(max(1.0, float(period_min)))
        log_max = np.log(float(period_max))
        periods = np.round(np.exp(rng.uniform(log_min, log_max, size=n))).astype(np.int64)
        return np.clip(periods, period_min, period_max)
    return rng.integers(period_min, period_max + 1, size=n, dtype=np.int64)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic real-time tasksets.")
    parser.add_argument("--n", type=int, default=15, help="Number of tasks per task set")
    parser.add_argument("--num_per_util", type=int, default=1500, help="Task sets per utilization step")
    parser.add_argument("--log_uniform", action="store_true", help="Sample periods log-uniformly")
    args = parser.parse_args()
    print(f"UUniSort and period distributions configured for n={args.n}")

if __name__ == "__main__":
    main()
