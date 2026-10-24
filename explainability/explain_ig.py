#!/usr/bin/env python3
"""Integrated Gradients (IG) explainability + certificate report for the joint
schedulability MLP (outputs R'_i and p_hat).

Place this file in your repo (e.g. explain/) so ../models/baseline_mlp.py is importable,
or pass --repo_root.

  1) Validate the script itself first (needs only torch + captum, no data/model):
       python 2026-10-26-explain_ig.py --selftest

  2) Explain your trained model on the uniform test split AND the Bosch test split:
       python 2026-10-26-explain_ig.py --ckpt experiments/results/joint_n15.pt \
           --data data/processed/tasksets_n15_uniform.npz data/processed/bosch_n15.npz \
           --tags uniform bosch --out_dir experiments/results/ig_n15

Requires: pip install torch captum numpy

What it reports (all printed, and saved as CSV + one markdown file):
  Table 1  certificate result + WHY schedulable sets were rejected
  Table 2  physics check: share of IG attribution on the task itself + higher-priority
           tasks, versus the chance level
  Table 3  which input types (C, T, 1/T, D) the network leans on
  Table 4  p_hat attribution by task rank
IG attributions are approximations of "who pushed the output"; the verifier, not IG,
is what guarantees safety.

ASSUMPTIONS (I have not seen models/baseline_mlp.py):
  * JointSchedulabilityMLP(n=n) has forward(x_reg, x_clf) -> (r_pred, p_sched)
  * r_pred has n-1 columns (R_2..R_n) or n columns (R_1..R_n); both handled
  * input scaling is the one in train_baseline.py (C, T, D divided by 1000)
  * task order in the arrays is deadline-monotonic (task 1 = highest priority)
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("PyTorch is required: pip install torch")

PREFIX = "2026-10-26-ig-"


# ----------------------------------------------------------------------------
# data / model helpers
# ----------------------------------------------------------------------------
def scale_inputs(X_reg, X_clf):
    """Same scaling as train_baseline.py."""
    X_reg = X_reg.astype(np.float32).copy()
    X_clf = X_clf.astype(np.float32).copy()
    X_reg[:, 0::3] /= 1000.0
    X_reg[:, 1::3] /= 1000.0
    X_clf[:, 0::4] /= 1000.0
    X_clf[:, 1::4] /= 1000.0
    X_clf[:, 3::4] /= 1000.0
    return X_reg, X_clf


def need_keys(d, keys, name):
    miss = [k for k in keys if k not in d.files]
    if miss:
        sys.exit(f"{name}: missing keys {miss}. Keys present: {d.files}")


def load_model(ckpt, n, repo_root):
    sys.path.insert(0, str(repo_root))
    try:
        from models.baseline_mlp import JointSchedulabilityMLP
    except ImportError as e:
        sys.exit(f"Cannot import models.baseline_mlp from {repo_root}: {e}\n"
                 f"Use --repo_root to point at the repo root.")
    model = JointSchedulabilityMLP(n=n)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


class OutputWrapper(nn.Module):
    """Joint model -> one tensor (batch, k+1): R' columns first, p_hat in the last column."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_reg, x_clf):
        r, p = self.model(x_reg, x_clf)
        if r.dim() == 1:
            r = r.unsqueeze(1)
        p = p.reshape(p.shape[0], -1)[:, :1]
        return torch.cat([r, p], dim=1)


# ----------------------------------------------------------------------------
# Integrated Gradients
# ----------------------------------------------------------------------------
def compute_ig(wrapped, Xr, Xc, base_r, base_c, targets, n_steps, batch=256):
    """Returns {target: (attr_reg (m,3n), attr_clf (m,4n), completeness_err_pct)}."""
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        sys.exit("captum is required: pip install captum")
    ig = IntegratedGradients(wrapped)
    with torch.no_grad():
        fx = wrapped(Xr, Xc)
        fb = wrapped(base_r.expand(len(Xr), -1).contiguous(),
                     base_c.expand(len(Xc), -1).contiguous())
    out = {}
    for t in targets:
        a_r, a_c, deltas = [], [], []
        for s in range(0, len(Xr), batch):
            xr, xc = Xr[s:s + batch], Xc[s:s + batch]
            br = base_r.expand(len(xr), -1).contiguous()
            bc = base_c.expand(len(xc), -1).contiguous()
            (ar, ac), delta = ig.attribute(
                (xr, xc), baselines=(br, bc), target=t,
                n_steps=n_steps, return_convergence_delta=True)
            a_r.append(ar.detach().numpy())
            a_c.append(ac.detach().numpy())
            deltas.append(delta.detach().numpy())
        denom = (fx[:, t] - fb[:, t]).abs().mean().item() + 1e-12
        err_pct = float(np.abs(np.concatenate(deltas)).mean() / denom * 100.0)
        out[t] = (np.concatenate(a_r), np.concatenate(a_c), err_pct)
    return out


def aggregate(a_r, a_c, n):
    """|attribution| per task (m, n) and % share per input type."""
    mr = np.abs(a_r).reshape(-1, n, 3)    # C, T, 1/T
    mc = np.abs(a_c).reshape(-1, n, 4)    # C, T, 1/T, D
    mass = mr.sum(2) + mc.sum(2)
    feat = {
        "C": mr[:, :, 0].sum() + mc[:, :, 0].sum(),
        "T": mr[:, :, 1].sum() + mc[:, :, 1].sum(),
        "1/T": mr[:, :, 2].sum() + mc[:, :, 2].sum(),
        "D": mc[:, :, 3].sum(),
    }
    tot = sum(feat.values()) + 1e-30
    return mass, {k: 100.0 * float(v) / tot for k, v in feat.items()}


def hp_mass(mass, task_idx):
    """Mean share of |attribution| on tasks 0..task_idx (task itself + higher priority)."""
    tot = mass.sum(1)
    ok = tot > 1e-12
    if not ok.any():
        return float("nan")
    return float((mass[ok, :task_idx + 1].sum(1) / tot[ok]).mean())


# ----------------------------------------------------------------------------
# certificate (integer arithmetic, exactly the Baruah et al. check)
# ----------------------------------------------------------------------------
def verify(Rp, C, D, T):
    """Rp, C, D, T: int64 (m, n). Returns (deadline_ok, recurrence_ok, shortfall)."""
    m, n = Rp.shape
    ok_dl = Rp <= D
    ok_rec = np.zeros((m, n), dtype=bool)
    short = np.zeros((m, n), dtype=np.int64)
    for i in range(n):
        f = C[:, i].copy()
        for j in range(i):
            f = f + (-(-Rp[:, i] // T[:, j])) * C[:, j]
        ok_rec[:, i] = Rp[:, i] >= f
        short[:, i] = np.maximum(f - Rp[:, i], 0)
    return ok_dl, ok_rec, short


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


# ----------------------------------------------------------------------------
# output formatting
# ----------------------------------------------------------------------------
def fmt(v):
    if isinstance(v, float):
        return "n/a" if math.isnan(v) else f"{v:.2f}"
    return str(v)


def md_table(rows, cols):
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])


# ----------------------------------------------------------------------------
# self-test: checks the script's own logic, no data / trained model needed
# ----------------------------------------------------------------------------
class ToyModel(nn.Module):
    """Linear toy. causal: R'_i depends on C of tasks 0..i. anti-causal: only on the last task."""

    def __init__(self, n, causal):
        super().__init__()
        self.n, self.causal = n, causal
        self.register_buffer("w", torch.linspace(1.0, 2.0, n))

    def forward(self, xr, xc):
        C = xr[:, 0::3]
        cols = []
        for i in range(1, self.n):
            if self.causal:
                cols.append((C[:, :i + 1] * self.w[:i + 1]).sum(1))
            else:
                cols.append(C[:, self.n - 1] * self.w[self.n - 1])
        r = torch.stack(cols, 1)
        p = torch.sigmoid((xc[:, 0::4] * self.w).sum(1) / self.n)
        return r, p


def selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name + (f"   [{extra}]" if extra else ""))
        ok = ok and bool(cond)

    # --- verifier on the worked example (C,D,T) = (1,4,4), (2,6,6), (3,12,12); exact R = 1,3,10
    C = np.array([[1, 2, 3]], dtype=np.int64)
    D = np.array([[4, 6, 12]], dtype=np.int64)
    T = np.array([[4, 6, 12]], dtype=np.int64)

    def cert(R, Dm=D):
        dl, rec, sh = verify(np.array([R], dtype=np.int64), C, Dm, T)
        return bool((dl & rec).all()), int(sh.max())

    check("exact R=[1,3,10] certifies", cert([1, 3, 10])[0])
    c, sh = cert([1, 3, 9])
    check("under-estimate R3'=9 rejected, short by exactly 1", (not c) and sh == 1, f"short={sh}")
    check("over-estimate R3'=11 still certifies", cert([1, 3, 11])[0])
    check("R3'=13 rejected (deadline 12)", not cert([1, 3, 13])[0])
    D_bad = np.array([[4, 6, 9]], dtype=np.int64)     # unschedulable: true R3 = 10 > 9
    check("unschedulable set (D3=9): no R3' in 1..199 certifies",
          not any(cert([1, 3, r3], D_bad)[0] for r3 in range(1, 200)))

    # --- IG pipeline on toy linear models (IG is exact for linear models, zero baseline)
    try:
        import captum  # noqa: F401
    except ImportError:
        print("SKIP  IG checks (captum not installed: pip install captum)")
        return ok
    n = 6
    g = torch.Generator().manual_seed(0)
    Xr = torch.rand(64, 3 * n, generator=g) + 0.1
    Xc = torch.rand(64, 4 * n, generator=g) + 0.1
    z_r, z_c = torch.zeros(1, 3 * n), torch.zeros(1, 4 * n)
    for causal in (True, False):
        name = "causal" if causal else "anti-causal"
        wrapped = OutputWrapper(ToyModel(n, causal)).eval()
        res = compute_ig(wrapped, Xr, Xc, z_r, z_c, list(range(n - 1)), n_steps=64)
        hps = []
        for t in range(n - 1):                       # output column t is task t+1 (0-based)
            mass, _ = aggregate(res[t][0], res[t][1], n)
            hps.append(hp_mass(mass, t + 1))
        errs = [res[t][2] for t in range(n - 1)]
        if causal:
            check("causal toy: all attribution on own + higher-priority tasks",
                  min(hps) > 0.999, f"min hp_mass={min(hps):.4f}")
        else:
            check("anti-causal toy: ~no attribution on higher-priority tasks",
                  max(hps[:-1]) < 0.001, f"max hp_mass={max(hps[:-1]):.4f}")
        check(f"IG completeness error small ({name})", max(errs) < 0.5, f"max err={max(errs):.4f}%")

    print("\nALL SELF-TESTS PASSED" if ok else "\nSOME SELF-TESTS FAILED")
    return ok


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Integrated Gradients + certificate report")
    ap.add_argument("--ckpt", help="trained JointSchedulabilityMLP state_dict (.pt)")
    ap.add_argument("--data", nargs="+", help="one or more .npz datasets")
    ap.add_argument("--tags", nargs="+", help="short names for the datasets (default: file names)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_explain", type=int, default=500, help="sets used for IG per dataset")
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--baseline", choices=["mean", "zeros"], default="mean",
                    help="mean = average training input of the FIRST dataset")
    ap.add_argument("--out_dir", default="ig_results")
    ap.add_argument("--repo_root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.ckpt or not a.data:
        ap.error("--ckpt and --data are required (or use --selftest)")
    tags = a.tags or [Path(p).stem for p in a.data]
    if len(tags) != len(a.data):
        ap.error("--tags must have the same count as --data")

    rng = np.random.default_rng(a.seed)
    d0 = np.load(a.data[0])
    n = int(d0["n"])
    model = load_model(a.ckpt, n, a.repo_root)
    wrapped = OutputWrapper(model).eval()

    if a.baseline == "zeros":
        base_r, base_c = torch.zeros(1, 3 * n), torch.zeros(1, 4 * n)
    else:
        need_keys(d0, ["train_X_reg", "train_X_bin"], a.data[0])
        tr_r, tr_c = scale_inputs(d0["train_X_reg"], d0["train_X_bin"])
        base_r = torch.from_numpy(tr_r.mean(0, keepdims=True))
        base_c = torch.from_numpy(tr_c.mean(0, keepdims=True))

    summary, physics, feats, phat_rows = [], [], [], []

    for tag, path in zip(tags, a.data):
        d = np.load(path)
        if int(d["n"]) != n:
            sys.exit(f"{path}: n={int(d['n'])} but the model was built for n={n}")
        s = a.split
        need_keys(d, [f"{s}_X_reg", f"{s}_X_bin", f"{s}_Y_bin"], path)
        Xr_raw, Xc_raw = d[f"{s}_X_reg"], d[f"{s}_X_bin"]
        y = d[f"{s}_Y_bin"].astype(bool)
        m = len(y)
        Xr_np, Xc_np = scale_inputs(Xr_raw, Xc_raw)
        Xr, Xc = torch.from_numpy(Xr_np), torch.from_numpy(Xc_np)

        # ---- predictions on the whole split
        chunks = []
        with torch.no_grad():
            for st in range(0, m, 4096):
                chunks.append(wrapped(Xr[st:st + 4096], Xc[st:st + 4096]).numpy())
        out_all = np.concatenate(chunks)
        r_dim = out_all.shape[1] - 1
        off = n - r_dim
        if off not in (0, 1):
            sys.exit(f"model returns {r_dim} response times for n={n}; expected n-1 or n")
        r_scaled = out_all[:, :r_dim].astype(np.float64)
        p_hat = out_all[:, r_dim]

        # ---- exact integers for the verifier
        if f"{s}_C_us" in d.files:
            C = d[f"{s}_C_us"].astype(np.int64)
            D = d[f"{s}_D_us"].astype(np.int64)
            T = d[f"{s}_T_us"].astype(np.int64)
            factor, unit = 1e6, "us"          # scaled output -> ms -> us
        else:
            C = np.rint(Xc_raw[:, 0::4]).astype(np.int64)
            T = np.maximum(np.rint(Xc_raw[:, 1::4]).astype(np.int64), 1)
            D = np.rint(Xc_raw[:, 3::4]).astype(np.int64)
            factor, unit = 1000.0, "raw units"  # assumes integer C, T, D in the npz

        # ceil() = round UP: an over-estimate can never create a false positive
        Rp = np.ones((m, n), dtype=np.int64)
        Rp[:, off:] = np.clip(np.ceil(r_scaled * factor), 1, 10 ** 12).astype(np.int64)
        if off == 1:
            Rp[:, 0] = C[:, 0]

        ok_dl, ok_rec, short = verify(Rp, C, D, T)
        cert = (ok_dl & ok_rec).all(1)
        tp, fp = int((cert & y).sum()), int((cert & ~y).sum())
        n_s = int(y.sum())
        rej = y & ~cert
        rec_fail, dl_fail = (~ok_rec).any(1), (~ok_dl).any(1)
        fail_tasks = (~(ok_dl & ok_rec)).sum(1)[rej]
        max_short = short.max(1)[rej]
        rel_short = (short / np.maximum(Rp + short, 1)).max(1)[rej] * 100.0
        pred = p_hat > 0.5

        # ---- Integrated Gradients on a random subset
        idx = np.sort(rng.choice(m, size=min(a.n_explain, m), replace=False))
        targets = list(range(r_dim)) + [r_dim]          # last = p_hat
        print(f"[*] {tag}: IG on {len(idx)} sets x {len(targets)} outputs, {a.n_steps} steps ...")
        res = compute_ig(wrapped, Xr[idx], Xc[idx], base_r, base_c, targets, a.n_steps)

        hp_all, lift_all, errs = [], [], []
        feat_acc = []
        for t in range(r_dim):
            task = t + off                                   # 0-based task index of this output
            mass, fshare = aggregate(res[t][0], res[t][1], n)
            hp = hp_mass(mass, task)
            chance = (task + 1) / n
            physics.append({"tag": tag, "output": f"R'_{task + 1}", "hp_mass": hp,
                            "chance": chance, "lift": hp - chance})
            hp_all.append(hp)
            if task < n - 1:
                lift_all.append(hp - chance)
            feat_acc.append(fshare)
            errs.append(res[t][2])
        feats.append({"tag": tag, "output": "R'_i (mean)",
                      **{k: float(np.mean([f[k] for f in feat_acc])) for k in ["C", "T", "1/T", "D"]}})

        mass_p, fshare_p = aggregate(res[r_dim][0], res[r_dim][1], n)
        errs.append(res[r_dim][2])
        feats.append({"tag": tag, "output": "p_hat", **fshare_p})
        share = mass_p.mean(0) / (mass_p.mean(0).sum() + 1e-30) * 100.0
        for j in range(n):
            phat_rows.append({"tag": tag, "task_rank": j + 1, "share_pct": float(share[j])})

        summary.append({
            "tag": tag, "n": n, "sets": m, "schedulable": n_s,
            "phat_acc_pct": 100.0 * float((pred == y).mean()),
            "phat_fpr_pct": 100.0 * float((pred & ~y).sum() / max((~y).sum(), 1)),
            "cert_accept_pct": 100.0 * tp / max(n_s, 1),
            "cert_FP": fp,
            "rej_recurrence_only": int((rej & rec_fail & ~dl_fail).sum()),
            "rej_deadline_only": int((rej & dl_fail & ~rec_fail).sum()),
            "rej_both": int((rej & rec_fail & dl_fail).sum()),
            "median_failing_tasks": med(fail_tasks),
            f"median_shortfall_{unit.replace(' ', '_')}": med(max_short),
            "median_rel_shortfall_pct": med(rel_short),
            "ig_mean_hp_mass": float(np.nanmean(hp_all)),
            "ig_mean_lift": float(np.nanmean(lift_all)) if lift_all else float("nan"),
            "ig_completeness_err_pct": float(max(errs)),
        })

    # ---- write + print
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c_sum = list(summary[0].keys())
    c_phy = ["tag", "output", "hp_mass", "chance", "lift"]
    c_feat = ["tag", "output", "C", "T", "1/T", "D"]
    c_ph = ["tag", "task_rank", "share_pct"]
    write_csv(out / f"{PREFIX}summary.csv", summary, c_sum)
    write_csv(out / f"{PREFIX}physics.csv", physics, c_phy)
    write_csv(out / f"{PREFIX}feature-share.csv", feats, c_feat)
    write_csv(out / f"{PREFIX}phat-by-task.csv", phat_rows, c_ph)

    md = [
        "# Integrated Gradients report",
        f"- checkpoint: `{a.ckpt}`; split: `{a.split}`; IG steps: {a.n_steps}; "
        f"baseline: {a.baseline}; sets explained per dataset: {a.n_explain}",
        "- cert_FP must be 0. If it is not, the verifier or the unit conversion is wrong.",
        "- hp_mass = share of |attribution| on the task itself + higher-priority tasks; "
        "chance = (rank/n); lift = hp_mass - chance. Physics says R'_i depends only on tasks <= i.",
        "",
        "## Table 1 - certificate result and why schedulable sets were rejected",
        md_table(summary, c_sum), "",
        "## Table 2 - physics check, by output",
        md_table(physics, c_phy), "",
        "## Table 3 - share (%) of |attribution| by input type",
        md_table(feats, c_feat), "",
        "## Table 4 - p_hat attribution (%) by task rank (1 = highest priority)",
        md_table(phat_rows, c_ph), "",
        "Attributions are approximations. IG with a baseline depends on the baseline chosen.",
    ]
    (out / f"{PREFIX}report.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n[+] saved to {out}/")


if __name__ == "__main__":
    main()