"""Scientific visualization routines replicating Baruah et al. and CP trade-offs."""
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def plot_fig2_binary(
    utilizations: np.ndarray,
    accuracy: np.ndarray,
    tpr: np.ndarray,
    tnr: np.ndarray,
    fpr: np.ndarray,
    save_path: Optional[str] = None,
):
    """Replicates Fig. 2: Binary classification metrics vs utilization."""
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(utilizations, accuracy, "b-", linewidth=2.5, label="Overall Accuracy")
    ax1.plot(utilizations, tpr, "r:", linewidth=2.5, label="True Positive Rate (TPR)")
    ax1.plot(utilizations, tnr, color="orange", linestyle="--", linewidth=2.5, label="True Negative Rate (TNR)")
    ax1.set_xlabel("Task System Utilization", fontweight="bold")
    ax1.set_ylabel("Accuracy Rate", fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(utilizations, fpr, "g-.", linewidth=2.5, label="False Positives")
    ax2.set_ylabel("False Positive Rate", color="g", fontweight="bold")
    ax2.set_ylim(-0.02, 0.20)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=True)
    plt.title("FP Binary Schedulability Classifier", fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_fig4_certificates(
    utilizations: np.ndarray,
    unverified_acc: np.ndarray,
    unverified_tpr: np.ndarray,
    verified_acc: np.ndarray,
    verified_tpr: np.ndarray,
    save_path: Optional[str] = None,
):
    """Replicates Fig. 4: Unverified vs Verified Schedulability."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Unverified
    ax1.plot(utilizations, unverified_acc, "b-", linewidth=2.5, label="Overall Accuracy")
    ax1.plot(utilizations, unverified_tpr, "r:", linewidth=2.5, label="Acceptance Rate")
    ax1.set_xlabel("Task System Utilization", fontweight="bold")
    ax1.set_ylabel("Rate", fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("(a) Unverified Predictions", fontweight="bold")
    ax1.legend(loc="lower left", frameon=True)

    # Verified
    ax2.plot(utilizations, verified_acc, "b-", linewidth=2.5, label="Overall Accuracy")
    ax2.plot(utilizations, verified_tpr, "r:", linewidth=2.5, label="Acceptance Rate")
    ax2.axhline(y=0.0, color="g", linestyle="-.", label="Safety False Positives (0.0%)")
    ax2.set_xlabel("Task System Utilization", fontweight="bold")
    ax2.set_ylabel("Rate", fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("(b) Verified with Certificates (Zero FP)", fontweight="bold")
    ax2.legend(loc="lower left", frameon=True)

    plt.suptitle("Schedulability With Verifiable Certificates", fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_fig7_system_size(
    n_values: np.ndarray,
    accuracy: np.ndarray,
    tpr: np.ndarray,
    tnr: np.ndarray,
    save_path: Optional[str] = None,
):
    """Replicates Fig. 7: Binary accuracy vs system size n."""
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, accuracy, "b-", linewidth=2.5, label="Overall Accuracy")
    plt.plot(n_values, tpr, "r:", linewidth=2.5, label="True Positive Rate")
    plt.plot(n_values, tnr, color="orange", linestyle="--", linewidth=2.5, label="True Negative Rate")
    plt.xlabel("Number of Tasks (n)", fontweight="bold")
    plt.ylabel("Accuracy Rate", fontweight="bold")
    plt.title("Binary Classification Accuracy vs System Size n", fontweight="bold")
    plt.ylim(0.0, 1.05)
    plt.xticks(n_values)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_fig9_loss_weight(
    weights: np.ndarray,
    accuracies: np.ndarray,
    save_path: Optional[str] = None,
):
    """Replicates Fig. 9: Loss weight sensitivity."""
    plt.figure(figsize=(7, 5))
    plt.semilogx(weights, accuracies, "b-o", linewidth=2.5)
    plt.axvline(x=100.0, color="black", linestyle="--", label="Paper Recommendation (w=100)")
    plt.xlabel("Weight w for Negative Errors (Log Scale)", fontweight="bold")
    plt.ylabel("Verified Accuracy Rate", fontweight="bold")
    plt.title("Impact of Undershoot Penalty Weight w", fontweight="bold")
    plt.ylim(0.0, 1.05)
    plt.legend(frameon=True)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_fig10_verified_size(
    n_values: np.ndarray,
    verified_acc: np.ndarray,
    acceptance_rate: np.ndarray,
    save_path: Optional[str] = None,
):
    """Replicates Fig. 10(b): Verified metrics vs system size n."""
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, verified_acc, "b-s", linewidth=2.5, label="Verified Accuracy")
    plt.plot(n_values, acceptance_rate, "r:^", linewidth=2.5, label="Acceptance Rate")
    plt.axhline(y=0.0, color="g", linestyle="-.", label="Safety False Positives (Strictly 0)")
    plt.xlabel("Number of Tasks (n)", fontweight="bold")
    plt.ylabel("Rate", fontweight="bold")
    plt.title("Verified Schedulability vs System Size n", fontweight="bold")
    plt.ylim(-0.05, 1.05)
    plt.xticks(n_values)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_conformal_tradeoff(
    alphas: np.ndarray,
    coverage: np.ndarray,
    uncertainty: np.ndarray,
    frr: np.ndarray,
    save_path: Optional[str] = None,
):
    """Trade-off curves across significance level alpha."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(alphas, coverage, "b-o", linewidth=2.5, label="Empirical Coverage")
    ax1.plot(alphas, 1.0 - alphas, "k--", alpha=0.6, label="Guaranteed Level (1 - α)")
    ax1.set_xlabel("Significance Level α", fontweight="bold")
    ax1.set_ylabel("Coverage", color="b", fontweight="bold")
    ax1.set_ylim(0.70, 1.02)

    ax2 = ax1.twinx()
    ax2.plot(alphas, uncertainty, "m-s", linewidth=2.5, label="Uncertainty Rate")
    ax2.plot(alphas, frr, "g-^", linewidth=2.5, label="Operational False Rejection Rate")
    ax2.set_ylabel("Rate", color="m", fontweight="bold")
    ax2.set_ylim(-0.02, 0.60)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", frameon=True)
    plt.title("Conformal Prediction Trade-off Curves", fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
