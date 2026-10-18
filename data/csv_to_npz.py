"""Convert a Bosch-style task-set CSV into the .npz layout used by train_baseline.py.

Usage:
  python data/csv_to_npz.py --csv data/raw/bosch_tasksets_n15.csv \
         --out data/processed/bosch_n15.npz --label rm_dm_label

Keys written, for prefix p in {train, cal, test} (Train / Calibration / Test partitions, SHUFFLED):
  n
  p_X_reg  (m, 3n)    per task [C, T, 1/T]        units: milliseconds
  p_X_bin  (m, 4n)    per task [C, T, 1/T, D]     units: milliseconds
  p_Y_reg  (m, n-1)   R_2..R_n                    units: milliseconds
  p_Y_bin  (m,)       1 = schedulable (DM ground truth)
  p_Y_rm   (m,)       1 = schedulable (RM ground truth)
  p_Y_edf  (m,)       1 = schedulable (EDF ground truth)
  p_C_us, p_D_us, p_T_us  (m, n) int64            exact integers for the verifier
  p_util   (m,)       utilization level the set was generated at
"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification.schedulability import edf_demand_bound_analysis, rate_monotonic_rta


def recompute_R(C, T, cap_mult=10):
    """Smallest fixed point of the RTA recurrence WITHOUT stopping at the deadline.

    Needed because the CSV stores R = -1 for tasks that miss their deadline, and a
    regression loss divides by R. If the iteration passes cap_mult * T the value is
    capped (the task is hopelessly late, only the fact that it is late matters).
    """
    n = len(C)
    out = []
    for i in range(n):
        cap = cap_mult * T[i]
        R = C[i]
        while True:
            Rn = C[i] + sum(-(-R // T[j]) * C[j] for j in range(i))
            if Rn == R:
                break
            if Rn > cap:
                R = cap
                break
            R = Rn
        out.append(R)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to input CSV (e.g. data/raw/bosch_tasksets_n15.csv)")
    ap.add_argument("--out", required=True, help="Output .npz path (e.g. data/processed/bosch_n15.npz)")
    ap.add_argument("--label", choices=["rm_dm_label", "edf_label"], default="rm_dm_label")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    csv_path = Path(a.csv)
    if not csv_path.exists():
        fallback = Path("data/raw") / a.csv
        if fallback.exists():
            csv_path = fallback
        else:
            raise FileNotFoundError(f"Could not find CSV file at {a.csv} or {fallback}")

    rows = list(csv.DictReader(open(csv_path)))
    m = len(rows)
    n = int(rows[0]["n"])
    C = np.array([[int(r[f"C{i}"]) for i in range(1, n + 1)] for r in rows], dtype=np.int64)
    D = np.array([[int(r[f"D{i}"]) for i in range(1, n + 1)] for r in rows], dtype=np.int64)
    T = np.array([[int(r[f"T{i}"]) for i in range(1, n + 1)] for r in rows], dtype=np.int64)
    R = np.array([[int(r[f"R{i}"]) for i in range(1, n + 1)] for r in rows], dtype=np.int64)
    y = np.array([int(r[a.label]) for r in rows], dtype=np.float32)
    y_rm = np.array(
        [rate_monotonic_rta(c, d, t)[0] for c, d, t in zip(C, D, T)],
        dtype=np.float32,
    )
    y_edf = np.array(
        [edf_demand_bound_analysis(c, d, t)[0] for c, d, t in zip(C, D, T)],
        dtype=np.float32,
    )
    util = np.array([float(r["util_target"]) for r in rows], dtype=np.float32)

    # replace R = -1 (deadline missed) by the uncut response time
    fixed = 0
    for k in range(m):
        if (R[k] < 0).any():
            R[k] = np.array(recompute_R(C[k].tolist(), T[k].tolist()), dtype=np.int64)
            fixed += 1

    # features in ms, task-major interleaved: [C1,T1,1/T1, C2,T2,1/T2, ...]
    Cm, Tm, Dm = C / 1000.0, T / 1000.0, D / 1000.0
    inv = 1.0 / Tm
    X_reg = np.stack([Cm, Tm, inv], axis=2).reshape(m, 3 * n).astype(np.float32)
    X_bin = np.stack([Cm, Tm, inv, Dm], axis=2).reshape(m, 4 * n).astype(np.float32)
    Y_reg = (R[:, 1:] / 1000.0).astype(np.float32)

    # SHUFFLE: the CSV is ordered by utilization, train_baseline.py validates on the
    # last 20% of the training array. Unshuffled, validation would be only U=0.9/1.0.
    perm = np.random.default_rng(a.seed).permutation(m)
    n_tr, n_cal = int(0.70 * m), int(0.15 * m)
    parts = {
        "train": perm[:n_tr],
        "cal": perm[n_tr:n_tr + n_cal],
        "test": perm[n_tr + n_cal:],
    }

    arrays = {"n": np.array(n)}
    for p, idx in parts.items():
        arrays[f"{p}_X_reg"] = X_reg[idx]
        arrays[f"{p}_X_bin"] = X_bin[idx]
        arrays[f"{p}_Y_reg"] = Y_reg[idx]
        arrays[f"{p}_Y_bin"] = y[idx]
        arrays[f"{p}_Y_rm"] = y_rm[idx]
        arrays[f"{p}_Y_edf"] = y_edf[idx]
        arrays[f"{p}_C"] = C[idx]
        arrays[f"{p}_D"] = D[idx]
        arrays[f"{p}_T"] = T[idx]
        arrays[f"{p}_C_us"] = C[idx]
        arrays[f"{p}_D_us"] = D[idx]
        arrays[f"{p}_T_us"] = T[idx]
        arrays[f"{p}_U"] = util[idx]
        arrays[f"{p}_util"] = util[idx]

    # Provide calib_* aliases for cal_* so any script expecting either works seamlessly
    for k in list(arrays.keys()):
        if k.startswith("cal_"):
            arrays[k.replace("cal_", "calib_")] = arrays[k]

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)

    print(f"[+] {m} sets, n={n}, label={a.label}; recomputed R for {fixed} sets with a missed deadline")
    for p, idx in parts.items():
        print(f"    {p:5s}: {len(idx):5d} sets, schedulable = {100 * y[idx].mean():.1f}%, "
              f"mean util level = {util[idx].mean():.2f}")
    print(f"[+] wrote {out_path}")


if __name__ == "__main__":
    main()
