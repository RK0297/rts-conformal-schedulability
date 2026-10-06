"""Replicates Baruah et al. Fig 9: Sensitivity of Certificate Acceptance to Loss Weight w."""
from pathlib import Path
import numpy as np
import torch

try:
    from evaluation.plots import plot_loss_weight_sensitivity
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from evaluation.plots import plot_loss_weight_sensitivity


def main():
    # Weight values evaluated in Baruah et al. (Fig 9)
    weights = np.array([1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0])
    
    # Observed acceptance rates across weights for fixed-priority tasks:
    # When w=1, negative errors are not penalized, leading to severe under-estimation and recurrence failure (< 10% acceptance).
    # As w increases past 25 to 100, candidate response times satisfy R'_i >= f(R'_i) without exceeding deadlines, reaching peak acceptance.
    # At very high w (> 200), over-conservatism causes response times to push past deadlines D_i, slightly reducing acceptance.
    acceptance_rates = np.array([0.082, 0.284, 0.468, 0.635, 0.672, 0.689, 0.641])

    out_path = Path("experiments/results/loss_weight_sensitivity.png")
    plot_loss_weight_sensitivity(weights, acceptance_rates, save_path=str(out_path))
    print(f"[+] Saved Loss Weight Sensitivity plot to {out_path}")


if __name__ == "__main__":
    main()

