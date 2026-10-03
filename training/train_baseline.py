"""Training pipeline for baseline neural schedulability models with strong scaling."""
import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    from models.baseline_mlp import ResponseTimeMLP, JointSchedulabilityMLP, AsymmetricNormalizedMSELoss
    from models.binary_classifier import BinarySchedulabilityMLP
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from models.baseline_mlp import ResponseTimeMLP, JointSchedulabilityMLP, AsymmetricNormalizedMSELoss
    from models.binary_classifier import BinarySchedulabilityMLP


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_joint(
    data_npz: str,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    w: float = 25.0,
    patience: int = 20,
    save_path: Optional[str] = None,
) -> Tuple[JointSchedulabilityMLP, dict]:
    device = get_device()
    data = np.load(data_npz)
    n = int(data["n"])

    # Load raw data
    X_reg = data["train_X_reg"].copy().astype(np.float32)
    X_clf = data["train_X_bin"].copy().astype(np.float32)
    Y_reg = data["train_Y_reg"].copy().astype(np.float32)
    Y_clf = data["train_Y_bin"].copy().astype(np.float32)

    # 1. Simple strong scaling: divide C, T, D by 1000.0
    X_reg[:, 0::3] /= 1000.0  # C
    X_reg[:, 1::3] /= 1000.0  # T
    # Note: column 2::3 is 1/T, which is already in [0.001, 1.0]

    X_clf[:, 0::4] /= 1000.0  # C
    X_clf[:, 1::4] /= 1000.0  # T
    X_clf[:, 3::4] /= 1000.0  # D

    # 2. Print min/max of response times before and after scaling
    print(f"[*] Response times before scaling: min = {Y_reg.min():.2f}, max = {Y_reg.max():.2f}")
    Y_reg /= 1000.0
    print(f"[*] Response times after scaling:  min = {Y_reg.min():.4f}, max = {Y_reg.max():.4f}")

    n_samples = len(X_reg)
    n_val = int(0.20 * n_samples)
    n_tr = n_samples - n_val

    train_ds = TensorDataset(
        torch.from_numpy(X_reg[:n_tr]),
        torch.from_numpy(X_clf[:n_tr]),
        torch.from_numpy(Y_reg[:n_tr]),
        torch.from_numpy(Y_clf[:n_tr]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_reg[n_tr:]),
        torch.from_numpy(X_clf[n_tr:]),
        torch.from_numpy(Y_reg[n_tr:]),
        torch.from_numpy(Y_clf[n_tr:]),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = JointSchedulabilityMLP(n=n).to(device)
    reg_loss_fn = AsymmetricNormalizedMSELoss(weight=w)
    clf_loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_loss = float("inf")
    best_weights = None
    no_improve = 0

    print(f"[*] Training JointSchedulabilityMLP (n={n}) on {device} (Epochs: {epochs}, Batch size: {batch_size}, lr: {lr})...")
    for epoch in range(1, epochs + 1):
        model.train()
        for bx_reg, bx_clf, by_reg, by_clf in train_loader:
            bx_reg, bx_clf = bx_reg.to(device), bx_clf.to(device)
            by_reg, by_clf = by_reg.to(device), by_clf.to(device)

            optimizer.zero_grad()
            r_pred, p_sched = model(bx_reg, bx_clf)
            loss = reg_loss_fn(r_pred, by_reg) + clf_loss_fn(p_sched, by_clf)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx_reg, bx_clf, by_reg, by_clf in val_loader:
                bx_reg, bx_clf = bx_reg.to(device), bx_clf.to(device)
                by_reg, by_clf = by_reg.to(device), by_clf.to(device)
                r_pred, p_sched = model(bx_reg, bx_clf)
                val_loss += (reg_loss_fn(r_pred, by_reg) + clf_loss_fn(p_sched, by_clf)).item() * len(bx_reg)
        val_loss /= n_val

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or no_improve == 0 or epoch == epochs:
            print(f"    Epoch {epoch:03d}/{epochs} | Val Loss: {val_loss:.4f} (Best: {best_loss:.4f})")

        if no_improve >= patience:
            print(f"[!] Early stopping at epoch {epoch}. Best Val Loss: {best_loss:.4f}")
            break

    if best_weights:
        model.load_state_dict(best_weights)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"[+] Saved model checkpoint to {save_path}")

    return model, {"best_val_loss": best_loss}


def main():
    parser = argparse.ArgumentParser(description="Train baseline schedulability model.")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--model_type", type=str, choices=["reg", "joint"], default="joint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--w", type=float, default=25.0)
    parser.add_argument("--save_path", type=str, required=True)

    args = parser.parse_args()
    train_joint(
        data_npz=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        w=args.w,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()