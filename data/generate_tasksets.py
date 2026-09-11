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

# Support direct execution and module execution
try:
    from data.exact_rta import exact_rta
except ImportError:
    from exact_rta import exact_rta


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


def generate_single_taskset(
    n: int,
    target_u: float,
    period_min: int = 1,
    period_max: int = 1000,
    log_uniform: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """Generates a single valid constrained-deadline taskset in DM order."""
    if rng is None:
        rng = np.random.default_rng()

    # 1. Utilizations
    u_tasks = uunisort(n, target_u, rng=rng)

    # 2. Periods
    if log_uniform:
        log_min = np.log(max(1.0, float(period_min)))
        log_max = np.log(float(period_max))
        periods = np.round(np.exp(rng.uniform(log_min, log_max, size=n))).astype(np.int64)
        periods = np.clip(periods, period_min, period_max)
    else:
        periods = rng.integers(period_min, period_max + 1, size=n, dtype=np.int64)

    # 3. Execution times C_i = round(U_i * T_i), with C_i in [1, T_i]
    c_vals = np.round(u_tasks * periods).astype(np.int64)
    c_vals = np.clip(c_vals, 1, periods)

    # 4. Deadlines D_i uniform in [C_i, T_i]
    d_vals = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if c_vals[i] == periods[i]:
            d_vals[i] = periods[i]
        else:
            d_vals[i] = rng.integers(int(c_vals[i]), int(periods[i]) + 1)

    # 5. DM Sorting: ascending deadlines
    dm_order = np.argsort(d_vals)
    c_vals = c_vals[dm_order]
    d_vals = d_vals[dm_order]
    periods = periods[dm_order]

    # 6. Exact RTA Ground Truth
    is_sched, r_true = exact_rta(c_vals, d_vals, periods)

    # 7. Features:
    # Regression features: [C_i, T_i, 1/T_i] (dimension 3n) - Deadlines excluded!
    feats_reg = []
    # Binary features: [C_i, T_i, 1/T_i, D_i] (dimension 4n)
    feats_bin = []
    for i in range(n):
        feats_reg.extend([c_vals[i], periods[i], 1.0 / periods[i]])
        feats_bin.extend([c_vals[i], periods[i], 1.0 / periods[i], d_vals[i]])

    return {
        "x_reg": np.array(feats_reg, dtype=np.float32),
        "x_bin": np.array(feats_bin, dtype=np.float32),
        "y_reg": np.array(r_true[1:], dtype=np.float32),  # Target: R'_2 .. R'_n
        "y_bin": 1.0 if is_sched else 0.0,
        "C": c_vals,
        "D": d_vals,
        "T": periods,
        "R": r_true,
        "U": target_u,
    }


def generate_taskset_collection(
    n: int,
    num_per_util: int = 1000,
    util_levels: Optional[List[float]] = None,
    log_uniform: bool = False,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generates tasksets across utilizations and packages them into NumPy arrays."""
    if util_levels is None:
        util_levels = [round(u, 2) for u in np.arange(0.1, 1.05, 0.1)]

    rng = np.random.default_rng(seed)
    total_samples = len(util_levels) * num_per_util

    x_reg_arr = np.zeros((total_samples, 3 * n), dtype=np.float32)
    x_bin_arr = np.zeros((total_samples, 4 * n), dtype=np.float32)
    y_reg_arr = np.zeros((total_samples, n - 1), dtype=np.float32)
    y_bin_arr = np.zeros(total_samples, dtype=np.float32)
    c_arr = np.zeros((total_samples, n), dtype=np.int64)
    d_arr = np.zeros((total_samples, n), dtype=np.int64)
    t_arr = np.zeros((total_samples, n), dtype=np.int64)
    r_arr = np.zeros((total_samples, n), dtype=np.int64)
    u_arr = np.zeros(total_samples, dtype=np.float32)

    idx = 0
    for u in util_levels:
        for _ in range(num_per_util):
            sample = generate_single_taskset(
                n=n, target_u=u, log_uniform=log_uniform, rng=rng
            )
            x_reg_arr[idx] = sample["x_reg"]
            x_bin_arr[idx] = sample["x_bin"]
            y_reg_arr[idx] = sample["y_reg"]
            y_bin_arr[idx] = sample["y_bin"]
            c_arr[idx] = sample["C"]
            d_arr[idx] = sample["D"]
            t_arr[idx] = sample["T"]
            r_arr[idx] = sample["R"]
            u_arr[idx] = sample["U"]
            idx += 1

    return {
        "X_reg": x_reg_arr,
        "X_bin": x_bin_arr,
        "Y_reg": y_reg_arr,
        "Y_bin": y_bin_arr,
        "C": c_arr,
        "D": d_arr,
        "T": t_arr,
        "R": r_arr,
        "U": u_arr,
        "n": np.array(n, dtype=np.int32),
    }


def split_and_save(
    data: Dict[str, np.ndarray],
    n: int,
    output_dir: Path,
    tag: str = "uniform",
    seed: int = 12345,
):
    """Splits into 70% Train, 15% Calibration, 15% Test and exports compressed npz."""
    n_samples = len(data["Y_bin"])
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    n_train = int(0.70 * n_samples)
    n_calib = int(0.15 * n_samples)

    idx_train = indices[:n_train]
    idx_calib = indices[n_train : n_train + n_calib]
    idx_test = indices[n_train + n_calib :]

    save_dict = {
        "n": np.array(n, dtype=np.int32),
        "train_X_reg": data["X_reg"][idx_train],
        "train_X_bin": data["X_bin"][idx_train],
        "train_Y_reg": data["Y_reg"][idx_train],
        "train_Y_bin": data["Y_bin"][idx_train],
        "train_C": data["C"][idx_train],
        "train_D": data["D"][idx_train],
        "train_T": data["T"][idx_train],
        "train_R": data["R"][idx_train],
        "train_U": data["U"][idx_train],
        "calib_X_reg": data["X_reg"][idx_calib],
        "calib_X_bin": data["X_bin"][idx_calib],
        "calib_Y_reg": data["Y_reg"][idx_calib],
        "calib_Y_bin": data["Y_bin"][idx_calib],
        "calib_C": data["C"][idx_calib],
        "calib_D": data["D"][idx_calib],
        "calib_T": data["T"][idx_calib],
        "calib_R": data["R"][idx_calib],
        "calib_U": data["U"][idx_calib],
        "test_X_reg": data["X_reg"][idx_test],
        "test_X_bin": data["X_bin"][idx_test],
        "test_Y_reg": data["Y_reg"][idx_test],
        "test_Y_bin": data["Y_bin"][idx_test],
        "test_C": data["C"][idx_test],
        "test_D": data["D"][idx_test],
        "test_T": data["T"][idx_test],
        "test_R": data["R"][idx_test],
        "test_U": data["U"][idx_test],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"tasksets_n{n}_{tag}.npz"
    np.savez_compressed(out_file, **save_dict)
    print(f"[+] Saved n={n} ({tag}) to {out_file} (Train: {len(idx_train)}, Calib: {len(idx_calib)}, Test: {len(idx_test)})")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic real-time tasksets.")
    parser.add_argument("--n", type=int, default=4, help="System size (number of tasks)")
    parser.add_argument("--num_per_util", type=int, default=1000, help="Task sets per utilization level (10 levels)")
    parser.add_argument("--log_uniform", action="store_true", help="Sample periods log-uniformly")
    parser.add_argument("--all_n", action="store_true", help="Generate for n in [2, 3, 4, 6, 8, 10, 15, 20]")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    target_ns = [2, 3, 4, 6, 8, 10, 15, 20] if args.all_n else [args.n]
    tag = "log_uniform" if args.log_uniform else "uniform"

    for n_val in target_ns:
        print(f"Generating tasksets for n={n_val} ({tag}), {args.num_per_util} per utilization level...")
        data = generate_taskset_collection(
            n=n_val,
            num_per_util=args.num_per_util,
            log_uniform=args.log_uniform,
            seed=args.seed + n_val,
        )
        split_and_save(data, n=n_val, output_dir=out_dir, tag=tag)


if __name__ == "__main__":
    main()
