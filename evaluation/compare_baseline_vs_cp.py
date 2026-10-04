"""Comparative evaluation: Unverified DL vs Verified Baseline vs Conformal Prediction.

Evaluates test performance, computes safety FPR and operational FRR,
and exports results to experiments/results/.
"""
import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch

try:
    from models.baseline_mlp import JointSchedulabilityMLP
    from models.conformal_wrapper import ConformalSchedulabilityWrapper
    from verification.rta_verifier import verify_batch_tasksets
    from evaluation.metrics import compute_metrics
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from models.baseline_mlp import JointSchedulabilityMLP
    from models.conformal_wrapper import ConformalSchedulabilityWrapper
    from verification.rta_verifier import verify_batch_tasksets
    from evaluation.metrics import compute_metrics


def run_comparison(
    data_npz: str,
    checkpoint: str,
    alphas: List[float] = [0.01, 0.05, 0.10, 0.20],
    results_dir: str = "experiments/results",
) -> pd.DataFrame:
    """Compares baseline methods and conformal prediction on test data."""
    data = np.load(data_npz)
    n = int(data["n"])

    X_reg_test = data["test_X_reg"]
    X_clf_test = data["test_X_bin"]
    y_test = data["test_Y_bin"]
    C_test = data["test_C"]
    D_test = data["test_D"]
    T_test = data["test_T"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointSchedulabilityMLP(n=n).to(device)

    # Critical fix: load checkpoint with weights_only=False
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Scale test inputs by 1000.0
    X_reg_test_scaled = X_reg_test.copy().astype(np.float32)
    X_reg_test_scaled[:, 0::3] /= 1000.0
    X_reg_test_scaled[:, 1::3] /= 1000.0

    X_clf_test_scaled = X_clf_test.copy().astype(np.float32)
    X_clf_test_scaled[:, 0::4] /= 1000.0
    X_clf_test_scaled[:, 1::4] /= 1000.0
    X_clf_test_scaled[:, 3::4] /= 1000.0

    # Test inference
    with torch.no_grad():
        tx_reg = torch.tensor(X_reg_test_scaled, dtype=torch.float32, device=device)
        tx_clf = torch.tensor(X_clf_test_scaled, dtype=torch.float32, device=device)
        r_pred_t, p_sched_t = model(tx_reg, tx_clf)
        # Critical fix: apply inverse scaling on predicted R'
        r_pred = r_pred_t.cpu().numpy() * 1000.0
        p_sched = p_sched_t.cpu().numpy()

    # 1. Unverified Baseline (DL Sigmoid > 0.5)
    unverified_sched = p_sched > 0.5
    m_unverified = compute_metrics(y_test, unverified_sched)

    # 2. Baruah et al. (2025) Verified Baseline (Certificate Check)
    is_valid, _, _ = verify_batch_tasksets(C_test, D_test, T_test, r_pred)
    m_verified = compute_metrics(y_test, is_valid)

    # 3. Conformal Prediction Layer
    cp = ConformalSchedulabilityWrapper(model=model)

    # Scale calib inputs
    X_reg_cal_scaled = data["calib_X_reg"].copy().astype(np.float32)
    X_reg_cal_scaled[:, 0::3] /= 1000.0
    X_reg_cal_scaled[:, 1::3] /= 1000.0

    X_clf_cal_scaled = data["calib_X_bin"].copy().astype(np.float32)
    X_clf_cal_scaled[:, 0::4] /= 1000.0
    X_clf_cal_scaled[:, 1::4] /= 1000.0
    X_clf_cal_scaled[:, 3::4] /= 1000.0

    with torch.no_grad():
        tx_reg_cal = torch.tensor(X_reg_cal_scaled, dtype=torch.float32, device=device)
        tx_clf_cal = torch.tensor(X_clf_cal_scaled, dtype=torch.float32, device=device)
        _, p_cal_t = model(tx_reg_cal, tx_clf_cal)
        cp.calibrate(p_sched_cal=p_cal_t.cpu().numpy(), y_true_cal=data["calib_Y_bin"])

    rows = []
    rows.append({
        "Method": "Unverified Baseline (DL Sigmoid)",
        "alpha": "N/A",
        "Accuracy (%)": m_unverified.predictive_accuracy * 100,
        "Acceptance Rate (%)": m_unverified.acceptance_rate * 100,
        "Safety FPR (%)": m_unverified.safety_fpr * 100,
        "Operational FRR (%)": m_unverified.operational_frr * 100,
        "Uncertainty Rate (%)": 0.0,
        "Coverage (%)": "N/A",
    })

    rows.append({
        "Method": "Baruah et al. (2025) Verified",
        "alpha": "N/A",
        "Accuracy (%)": m_verified.predictive_accuracy * 100,
        "Acceptance Rate (%)": m_verified.acceptance_rate * 100,
        "Safety FPR (%)": m_verified.safety_fpr * 100,
        "Operational FRR (%)": m_verified.operational_frr * 100,
        "Uncertainty Rate (%)": 0.0,
        "Coverage (%)": "N/A",
    })

    for a in alphas:
        pred_sets = cp.predict_batch_sets(p_sched, alpha=a)
        # Schedulable only if both claimed and certificate is valid
        claimed_sched = np.array(["Schedulable" in s for s in pred_sets])
        accepted_and_valid = claimed_sched & is_valid

        m_cp = compute_metrics(y_test, accepted_and_valid, prediction_sets=pred_sets)
        rows.append({
            "Method": f"CP Augmented (alpha={a:.2f})",
            "alpha": a,
            "Accuracy (%)": m_cp.predictive_accuracy * 100,
            "Acceptance Rate (%)": m_cp.acceptance_rate * 100,
            "Safety FPR (%)": m_cp.safety_fpr * 100,
            "Operational FRR (%)": m_cp.operational_frr * 100,
            "Uncertainty Rate (%)": (m_cp.uncertainty_rate or 0.0) * 100,
            "Coverage (%)": (m_cp.conformal_coverage or 0.0) * 100,
        })

    df = pd.DataFrame(rows)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"comparison_n{n}.csv"
    md_path = out_dir / f"comparison_n{n}.md"

    df.to_csv(csv_path, index=False)
    md_table = df.to_markdown(index=False)
    with open(md_path, "w") as f:
        f.write(f"# Schedulability Evaluation (n={n})\n\n{md_table}\n")

    # Generate and save summary plots
    try:
        from evaluation.plots import plot_conformal_tradeoff
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from evaluation.plots import plot_conformal_tradeoff

    # Extract CP rows for trade-off curve
    cp_rows = [r for r in rows if r["alpha"] != "N/A"]
    if cp_rows:
        alpha_vals = np.array([r["alpha"] for r in cp_rows])
        cov_vals = np.array([r["Coverage (%)"] / 100.0 for r in cp_rows])
        unc_vals = np.array([r["Uncertainty Rate (%)"] / 100.0 for r in cp_rows])
        frr_vals = np.array([r["Operational FRR (%)"] / 100.0 for r in cp_rows])

        plot_path = out_dir / f"conformal_tradeoff_n{n}.png"
        plot_conformal_tradeoff(
            alphas=alpha_vals,
            coverage=cov_vals,
            uncertainty=unc_vals,
            frr=frr_vals,
            save_path=str(plot_path),
        )
        print(f"[+] Saved Conformal Trade-off plot to {plot_path}")

    # Create Bar Comparison plot (Baseline vs Verified vs CP)
    try:
        import matplotlib.pyplot as plt
        methods = [r["Method"] for r in rows]
        accs = [r["Accuracy (%)"] for r in rows]
        accepts = [r["Acceptance Rate (%)"] for r in rows]
        fprs = [r["Safety FPR (%)"] for r in rows]

        fig, ax = plt.subplots(figsize=(10, 5))
        x_indices = np.arange(len(methods))
        width = 0.25

        ax.bar(x_indices - width, accs, width, label="Accuracy (%)", color="#1f77b4")
        ax.bar(x_indices, accepts, width, label="Acceptance Rate (%)", color="#2ca02c")
        ax.bar(x_indices + width, fprs, width, label="Safety FPR (%)", color="#d62728")

        ax.set_ylabel("Percentage (%)", fontweight="bold")
        ax.set_title(f"Method Comparison for n={n} (Zero False Positives Guaranteed by Certificate)", fontweight="bold")
        ax.set_xticks(x_indices)
        ax.set_xticklabels([m.split("(")[0].strip() + ("\n(" + m.split("(")[1] if "(" in m else "") for m in methods], fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right")
        plt.tight_layout()

        bar_plot_path = out_dir / f"method_comparison_n{n}.png"
        plt.savefig(bar_plot_path, dpi=300)
        plt.close()
        print(f"[+] Saved Method Comparison bar plot to {bar_plot_path}")
    except Exception as e:
        print(f"[!] Plotting warning: {e}")

    print(f"\n[+] Evaluation Summary for n={n}:\n")
    print(md_table)
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs CP Schedulability.")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--results_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    run_comparison(args.data, args.checkpoint, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
