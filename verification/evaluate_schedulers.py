"""Evaluate RM, DM, or EDF against stored task-set arrays.

Examples:
    python verification/evaluate_schedulers.py --data data/processed/tasksets_n10_uniform.npz --analysis rm
    python verification/evaluate_schedulers.py --data data/processed/tasksets_n10_uniform.npz --analysis dm
    python verification/evaluate_schedulers.py --data data/processed/tasksets_n10_uniform.npz --analysis edf
    python verification/evaluate_schedulers.py --data data/processed/tasksets_n10_uniform.npz --analysis all
"""
import argparse
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

try:
    from verification.schedulability import (
        deadline_monotonic_rta,
        edf_demand_bound_analysis,
        rate_monotonic_rta,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from verification.schedulability import (
        deadline_monotonic_rta,
        edf_demand_bound_analysis,
        rate_monotonic_rta,
    )


def evaluate_dataset(
    data_path: str,
    analysis_type: str = "all",
    split: str = "test",
) -> Dict[str, float]:
    """Evaluate schedulability analyses on an NPZ dataset."""
    data = np.load(data_path)
    c_key = f"{split}_C" if f"{split}_C" in data else f"{split}_C_us"
    d_key = f"{split}_D" if f"{split}_D" in data else f"{split}_D_us"
    t_key = f"{split}_T" if f"{split}_T" in data else f"{split}_T_us"

    if c_key not in data:
        raise KeyError(f"Could not find key '{c_key}' in {data_path}. Available keys: {list(data.keys())}")

    C, D, T = data[c_key], data[d_key], data[t_key]
    n_samples = len(C)

    analyses = ["rm", "dm", "edf"] if analysis_type == "all" else [analysis_type]
    results: Dict[str, List[bool]] = {a: [] for a in analyses}

    for c, d, t in zip(C, D, T):
        if "rm" in results:
            rm_ok, _ = rate_monotonic_rta(c, d, t)
            results["rm"].append(rm_ok)
        if "dm" in results:
            dm_ok, _ = deadline_monotonic_rta(c, d, t)
            results["dm"].append(dm_ok)
        if "edf" in results:
            edf_ok, _ = edf_demand_bound_analysis(c, d, t)
            results["edf"].append(edf_ok)

    rates = {}
    print("\n" + "=" * 72)
    print(f"SCHEDULABILITY EVALUATION: {Path(data_path).name} ({split} split, N={len(C[0])} tasks)")
    print("=" * 72)
    print(f"Total task sets evaluated: {n_samples}\n")
    print(f"{'Algorithm':<28} | {'Schedulable':<12} | {'Acceptance Rate':<15}")
    print("-" * 72)

    for a in analyses:
        arr = np.array(results[a], dtype=bool)
        n_sched = int(np.sum(arr))
        pct = 100.0 * n_sched / n_samples
        rates[a] = pct
        full_name = {
            "rm": "Rate-Monotonic (RM - Static T)",
            "dm": "Deadline-Monotonic (DM - Static D)",
            "edf": "Earliest-Deadline First (EDF - Dynamic)",
        }[a]
        print(f"{full_name:<28} | {n_sched:<12} | {pct:>6.2f}%")

    if analysis_type == "all":
        rm_arr = np.array(results["rm"])
        dm_arr = np.array(results["dm"])
        edf_arr = np.array(results["edf"])

        # Check dominance: RM <= DM <= EDF
        rm_subset_dm = np.all(~rm_arr | dm_arr)
        dm_subset_edf = np.all(~dm_arr | edf_arr)
        print("-" * 72)
        print("Dominance Hierarchy Invariants (Sched(RM) <= Sched(DM) <= Sched(EDF)):")
        print(f"  * RM <= DM satisfied for all sets:   {'PASSED (100%)' if rm_subset_dm else 'FAILED'}")
        print(f"  * DM <= EDF satisfied for all sets:  {'PASSED (100%)' if dm_subset_edf else 'FAILED'}")
        print(f"  * Extra sets schedulable by DM over RM:  {int(np.sum(dm_arr & ~rm_arr))}")
        print(f"  * Extra sets schedulable by EDF over DM: {int(np.sum(edf_arr & ~dm_arr))}")

    print("=" * 72 + "\n")
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classical schedulability analyses.")
    parser.add_argument("--data", required=True, help="NPZ dataset containing test_C/test_D/test_T")
    parser.add_argument("--analysis", choices=("rm", "dm", "edf", "all"), default="all")
    parser.add_argument("--split", choices=("train", "calib", "test"), default="test")
    args = parser.parse_args()

    evaluate_dataset(args.data, analysis_type=args.analysis, split=args.split)


if __name__ == "__main__":
    main()
