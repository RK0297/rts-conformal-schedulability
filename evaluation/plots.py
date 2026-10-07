"""Scientific visualization routines for Learning-Assisted RTS Schedulability Analysis.

Replicates core figures from Baruah et al. (2025) and generates Conformal Prediction
calibration diagnostics for the 70% presentation milestone.
"""
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np

# Set clean, publication-ready styling
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.2,
    "figure.autolayout": True,
})


def plot_schedulability_acceptance(
    utilizations: np.ndarray,
    exact_rta: np.ndarray,
    unverified_dl: np.ndarray,
    certified: np.ndarray,
    n: int,
    save_path: Optional[str] = None,
):
    """Replicates Fig 4 / Core RTS curve: Acceptance Ratio vs Task Set Utilization.
    
    Shows Ground Truth Exact RTA, Unverified DL (with unsafe over-admission),
    and Verified Certificate (guaranteed zero false positives).
    """
    plt.figure(figsize=(8, 5.2))
    plt.plot(utilizations, exact_rta * 100, "k--o", label="Exact RTA (Ground Truth)", alpha=0.85)
    plt.plot(utilizations, unverified_dl * 100, "r:^", label="Unverified DL (Sigmoid > 0.5)", alpha=0.85)
    plt.plot(utilizations, certified * 100, "b-s", label="Baruah et al. Certified (Zero FPR)")

    plt.fill_between(
        utilizations,
        certified * 100,
        unverified_dl * 100,
        where=(unverified_dl > certified),
        color="salmon",
        alpha=0.18,
        label="Unverified Unsafe Risk / FRR Zone",
    )

    plt.xlabel("Task Set Utilization (U)", fontweight="bold")
    plt.ylabel("Schedulability Acceptance Rate (%)", fontweight="bold")
    plt.title(f"Schedulability Acceptance vs. Utilization (n = {n})", fontweight="bold")
    plt.ylim(-2, 105)
    plt.xlim(0.05, 1.05)
    plt.xticks(utilizations)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", frameon=True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[+] Saved Schedulability Acceptance plot to {save_path}")
    else:
        plt.show()


def plot_classifier_metrics(
    utilizations: np.ndarray,
    accuracy: np.ndarray,
    tpr: np.ndarray,
    tnr: np.ndarray,
    fpr: np.ndarray,
    n: int,
    save_path: Optional[str] = None,
):
    """Replicates Baruah et al. Fig 2: Binary Classifier Performance vs Utilization."""
    fig, ax1 = plt.subplots(figsize=(8, 5.2))

    ax1.plot(utilizations, accuracy * 100, "b-o", label="Overall Accuracy")
    ax1.plot(utilizations, tpr * 100, "g--s", label="True Positive Rate (TPR)")
    ax1.plot(utilizations, tnr * 100, color="darkorange", linestyle="-.", marker="d", label="True Negative Rate (TNR)")
    ax1.set_xlabel("Task Set Utilization (U)", fontweight="bold")
    ax1.set_ylabel("Rate (%)", fontweight="bold")
    ax1.set_ylim(-5, 105)
    ax1.set_xlim(0.05, 1.05)
    ax1.set_xticks(utilizations)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Secondary axis for False Positive Rate
    ax2 = ax1.twinx()
    ax2.plot(utilizations, fpr * 100, "r-^", linewidth=2.5, label="Safety False Positive Rate (FPR)")
    ax2.fill_between(utilizations, 0, fpr * 100, color="red", alpha=0.15)
    ax2.set_ylabel("Safety FPR (%) [Deadline Miss Hazard]", color="red", fontweight="bold")
    ax2.set_ylim(-1, max(30.0, np.max(fpr * 100) * 1.5))
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", frameon=True)

    plt.title(f"Unverified Neural Classifier Metrics vs. Utilization (n = {n})", fontweight="bold")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[+] Saved Classifier Metrics plot to {save_path}")
    else:
        plt.show()


def plot_conformal_calibration(
    alphas: np.ndarray,
    empirical_coverage: np.ndarray,
    p_single_sched: np.ndarray,
    p_single_unsched: np.ndarray,
    p_uncertain: np.ndarray,
    n: int,
    save_path: Optional[str] = None,
):
    """Conformal Prediction Calibration diagnostics: Coverage and Set Size Distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) Statistical Coverage vs Guaranteed Bound (1 - alpha)
    target_coverage = (1.0 - alphas) * 100.0
    ax1.plot(alphas, empirical_coverage * 100.0, "b-o", linewidth=2.5, label="Empirical Coverage")
    ax1.plot(alphas, target_coverage, "k--", linewidth=2.0, label="Theoretical Bound (1 - α)")
    ax1.fill_between(alphas, target_coverage, empirical_coverage * 100.0, color="blue", alpha=0.12, label="Coverage Margin")
    ax1.set_xlabel("Significance Level (α)", fontweight="bold")
    ax1.set_ylabel("Coverage (%)", fontweight="bold")
    ax1.set_title("(a) Marginal Coverage Verification", fontweight="bold")
    ax1.set_ylim(75, 102)
    ax1.set_xticks(alphas)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", frameon=True)

    # (b) Prediction Set Breakdown Across Alphas
    x = np.arange(len(alphas))
    width = 0.25

    ax2.bar(x - width, p_single_sched * 100, width, label="{Schedulable}", color="#2ca02c")
    ax2.bar(x, p_single_unsched * 100, width, label="{Unschedulable}", color="#d62728")
    ax2.bar(x + width, p_uncertain * 100, width, label="{Uncertain / Fallback}", color="#ff7f0e")

    ax2.set_xlabel("Significance Level (α)", fontweight="bold")
    ax2.set_ylabel("Proportion of Test Set (%)", fontweight="bold")
    ax2.set_title("(b) Prediction Set Distribution Across α", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"α={a:.2f}" for a in alphas])
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    plt.suptitle(f"Split Conformal Prediction Diagnostics (n = {n})", fontweight="bold")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[+] Saved Conformal Calibration plot to {save_path}")
    else:
        plt.show()


def plot_method_comparison_bar(
    methods: List[str],
    accuracies: List[float],
    acceptance_rates: List[float],
    safety_fprs: List[float],
    n: int,
    save_path: Optional[str] = None,
):
    """Bar comparison contrasting Unverified DL vs Certified Baseline vs CP-Triaged."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x_indices = np.arange(len(methods))
    width = 0.26

    ax.bar(x_indices - width, accuracies, width, label="Overall Accuracy (%)", color="#1f77b4")
    ax.bar(x_indices, acceptance_rates, width, label="Acceptance Rate (%)", color="#2ca02c")
    ax.bar(x_indices + width, safety_fprs, width, label="Safety FPR (%) [Hazard]", color="#d62728")

    # Add text labels on the Safety FPR bars to emphasize 0.000%
    for i, fpr in enumerate(safety_fprs):
        if fpr == 0.0:
            ax.text(x_indices[i] + width, 2.5, "0.000%\n(SAFE)", ha="center", va="bottom", fontsize=8.5, color="green", fontweight="bold")
        else:
            ax.text(x_indices[i] + width, fpr + 1.5, f"{fpr:.1f}%\n(UNSAFE)", ha="center", va="bottom", fontsize=8.5, color="red", fontweight="bold")

    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title(f"Method Comparison (n = {n}): Absolute Safety Guarantee of Certificates", fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(methods, fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper right", frameon=True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[+] Saved Method Comparison bar plot to {save_path}")
    else:
        plt.show()


def plot_loss_weight_sensitivity(
    weights: np.ndarray,
    certified_rates: np.ndarray,
    save_path: Optional[str] = None,
):
    """Replicates Baruah et al. Fig 9: Sensitivity to Asymmetric Undershoot Penalty w."""
    plt.figure(figsize=(7.5, 4.8))
    plt.semilogx(weights, certified_rates * 100, "b-o", linewidth=2.5, markersize=7)
    plt.axvline(x=25.0, color="red", linestyle="--", label="Implemented Setting (w = 25)")
    plt.axvline(x=100.0, color="gray", linestyle=":", label="Paper Theoretical Value (w = 100)")

    plt.xlabel("Asymmetric Penalty Weight w (Log Scale)", fontweight="bold")
    plt.ylabel("Certificate Acceptance Rate (%)", fontweight="bold")
    plt.title("Impact of Asymmetric Undershoot Penalty Weight w", fontweight="bold")
    plt.ylim(0, 105)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="lower right")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[+] Saved Loss Weight Sensitivity plot to {save_path}")
    else:
        plt.show()
