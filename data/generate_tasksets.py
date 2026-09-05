"""Synthetic task set generator for Fixed-Priority constrained-deadline systems.

Implements the workload synthesis protocol from Baruah et al. (2025):
- UUniSort utilization distribution (Bini & Buttazzo 2005)
- Uniform period distribution in [1, 1000] (or log-uniform)
- Execution times C_i = round(U_i * T_i) with C_i >= 1
- Deadlines D_i uniformly sampled in [C_i, T_i]
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

def partition_dataset(all_data: Dict[str, np.ndarray], train_ratio: float = 0.70, calib_ratio: float = 0.15):
    total = len(all_data["x_reg"])
    indices = np.arange(total)
    np.random.default_rng(42).shuffle(indices)

    n_train = int(total * train_ratio)
    n_calib = int(total * calib_ratio)

    idx_train = indices[:n_train]
    idx_calib = indices[n_train:n_train + n_calib]
    idx_test = indices[n_train + n_calib:]

    return {
        "train_X_reg": all_data["x_reg"][idx_train],
        "train_X_bin": all_data["x_bin"][idx_train],
        "train_Y_reg": all_data["y_reg"][idx_train],
        "train_Y_bin": all_data["y_bin"][idx_train],
        "calib_X_reg": all_data["x_reg"][idx_calib],
        "calib_X_bin": all_data["x_bin"][idx_calib],
        "calib_Y_reg": all_data["y_reg"][idx_calib],
        "calib_Y_bin": all_data["y_bin"][idx_calib],
        "test_X_reg": all_data["x_reg"][idx_test],
        "test_X_bin": all_data["x_bin"][idx_test],
        "test_Y_reg": all_data["y_reg"][idx_test],
        "test_Y_bin": all_data["y_bin"][idx_test],
    }
