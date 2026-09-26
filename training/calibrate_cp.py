"""Calibration script for Split Conformal Prediction."""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

try:
    from models.baseline_mlp import JointSchedulabilityMLP
    from models.conformal_wrapper import ConformalSchedulabilityWrapper
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from models.baseline_mlp import JointSchedulabilityMLP
    from models.conformal_wrapper import ConformalSchedulabilityWrapper


def calibrate_cp(
    data_npz: str,
    checkpoint: str,
    alphas: Optional[List[float]] = None,
    save_json: Optional[str] = None,
) -> Tuple[ConformalSchedulabilityWrapper, Dict[float, float]]:
    """Calibrates conformal prediction quantiles on calibration data."""
    if alphas is None:
        alphas = [0.01, 0.05, 0.10, 0.20]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(data_npz)
    n = int(data["n"])

    X_reg = data["calib_X_reg"]
    X_clf = data["calib_X_bin"]
    y_true = data["calib_Y_bin"]

    model = JointSchedulabilityMLP(n=n).to(device)

    # Critical fix: load checkpoint with weights_only=False
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Scale calibration inputs by 1000.0
    X_reg_scaled = X_reg.copy().astype(np.float32)
    X_reg_scaled[:, 0::3] /= 1000.0
    X_reg_scaled[:, 1::3] /= 1000.0

    X_clf_scaled = X_clf.copy().astype(np.float32)
    X_clf_scaled[:, 0::4] /= 1000.0
    X_clf_scaled[:, 1::4] /= 1000.0
    X_clf_scaled[:, 3::4] /= 1000.0

    with torch.no_grad():
        tx_reg = torch.tensor(X_reg_scaled, dtype=torch.float32, device=device)
        tx_clf = torch.tensor(X_clf_scaled, dtype=torch.float32, device=device)
        _, p_sched_t = model(tx_reg, tx_clf)
        p_sched = p_sched_t.cpu().numpy()

    wrapper = ConformalSchedulabilityWrapper(model=model)
    wrapper.calibrate(p_sched_cal=p_sched, y_true_cal=y_true)

    quantiles = {}
    print(f"[*] Conformal Calibration completed on m={wrapper.m} samples:")
    for a in alphas:
        q = wrapper.get_quantile(a)
        quantiles[a] = q
        print(f"    alpha = {a:.2f} (Confidence: {(1-a)*100:.0f}%) | Quantile q_hat = {q:.4f} (Threshold = {1-q:.4f})")

    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        out_dict = {
            "n": n,
            "m_calib": wrapper.m,
            "quantiles": {str(k): float(v) for k, v in quantiles.items()},
        }
        with open(save_json, "w") as f:
            json.dump(out_dict, f, indent=2)
        print(f"[+] Saved quantiles to {save_json}")

    return wrapper, quantiles


def main():
    parser = argparse.ArgumentParser(description="Calibrate Conformal Prediction quantiles.")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()

    calibrate_cp(args.data, args.checkpoint, save_json=args.save_json)


if __name__ == "__main__":
    main()
