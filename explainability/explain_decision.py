#!/usr/bin/env python3
"""Show HOW the joint network reaches SCHEDULABLE / UNSCHEDULABLE / UNCERTAIN.

Run:
  python 2026-10-26-explain_decision.py --selftest
  python 2026-10-26-explain_decision.py --ckpt experiments/results/joint_bosch_n15.pt \
         --data data/processed/bosch_n15.npz --out_dir experiments/results/decision_bosch_n15

What it prints (and saves as markdown + CSV):
  Table A  the network's layers: sizes, number of weights
  Table B  the decision rule: which p_hat ranges mean schedulable / unschedulable / uncertain
  Table C  results over the whole test split (decision vs truth, certificate)
  Cards    for a few example task sets, the full story:
             inputs -> layer values (z = W x + b) -> final logit = w . h + b
             -> p_hat = sigmoid(logit) -> decision -> certificate
             plus which tasks pushed the logit up/down (Integrated Gradients)

IMPORTANT HONESTY NOTES
  * "UNCERTAIN" here is computed with STANDARD split conformal prediction on p_hat
    (score = 1 - p_hat of the true class, calibration split = cal_*). This is MY
    implementation of the textbook method. Check it matches your repo's conformal
    code; if your repo differs, replace conformal_thresholds()/decide().
  * I have not seen models/baseline_mlp.py. Layers are found automatically with hooks
    (nn.Linear and nn.Sigmoid modules). If your model applies sigmoid inside forward()
    with no nn.Sigmoid module, the script falls back to logit = log(p/(1-p)).
  * Requires: pip install torch captum numpy
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("PyTorch is required: pip install torch")

PREFIX = "2026-10-26-decision-"
SCHED, UNSCHED, UNCERT = "SCHEDULABLE", "UNSCHEDULABLE", "UNCERTAIN"


# ----------------------------------------------------------------------------
# data / model helpers (same conventions as train_baseline.py)
# ----------------------------------------------------------------------------
def scale_inputs(X_reg, X_clf):
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


# ----------------------------------------------------------------------------
# tracing: record every Linear / Sigmoid call during one forward pass
# ----------------------------------------------------------------------------
def trace(model, xr, xc):
    records, handles = [], []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            def hk(m, inp, out, name=name):
                records.append(("linear", name, inp[0].detach(), out.detach()))
            handles.append(mod.register_forward_hook(hk))
        elif isinstance(mod, nn.Sigmoid):
            def hk(m, inp, out, name=name):
                records.append(("sigmoid", name, inp[0].detach(), out.detach()))
            handles.append(mod.register_forward_hook(hk))
    try:
        with torch.no_grad():
            r, p = model(xr, xc)
    finally:
        for h in handles:
            h.remove()
    if r.dim() == 1:
        r = r.unsqueeze(1)
    return r.detach(), p.detach().reshape(-1), records


def find_logit_and_last_linear(p, records):
    """Returns (logit tensor (m,), index of the last Linear record, note)."""
    pf = p.reshape(-1)
    # 1) exact: an nn.Sigmoid module whose output equals p
    for rec in reversed(records):
        if rec[0] == "sigmoid" and rec[3].numel() == pf.numel() \
                and torch.allclose(rec[3].reshape(-1), pf, atol=1e-5):
            logit = rec[2].reshape(-1)
            break
    else:
        logit = torch.logit(pf.clamp(1e-7, 1 - 1e-7))
    # 2) the Linear with ONE output whose value equals the logit
    ok = (pf > 1e-3) & (pf < 1 - 1e-3)
    for k in range(len(records) - 1, -1, -1):
        kind, _, _, out = records[k]
        if kind == "linear" and out.shape[-1] == 1 and out.numel() == pf.numel():
            o = out.reshape(-1)
            if ok.any() and torch.allclose(o[ok], logit[ok], atol=1e-2, rtol=1e-2):
                return logit, k, ""
    return logit, None, ("could not match a final Linear(.,1) to the logit; "
                         "the Wx+b decomposition is skipped")


# ----------------------------------------------------------------------------
# conformal prediction (standard split conformal, binary, class-conditional)
# ----------------------------------------------------------------------------
def _q(scores, alpha):
    m = len(scores)
    if m == 0:
        return np.inf
    k = int(np.ceil((m + 1) * (1 - alpha)))
    if k > m:
        return np.inf
    return float(np.sort(scores)[k - 1])


def conformal_thresholds(p_cal, y_cal, alpha, marginal):
    # nonconformity score = 1 - p(true class)
    s = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
    if marginal:
        q = _q(s, alpha)
        return q, q
    return _q(s[y_cal == 0], alpha), _q(s[y_cal == 1], alpha)


def decide(p, q0, q1):
    """Prediction set -> label. class 1 in set iff 1-p<=q1 ; class 0 in set iff p<=q0."""
    in1 = (1.0 - p) <= q1
    in0 = p <= q0
    lab = np.full(p.shape, UNCERT, dtype=object)
    lab[in1 & ~in0] = SCHED
    lab[in0 & ~in1] = UNSCHED
    why = np.full(p.shape, "", dtype=object)
    why[in1 & in0] = "both labels plausible"
    why[~in1 & ~in0] = "neither label plausible"
    return lab, why


def region_text(q0, q1):
    hi = 1.0 - q1          # class 1 allowed when p >= hi
    lo = q0                # class 0 allowed when p <= lo
    rows = []
    if hi <= lo:
        rows.append((f"p_hat < {hi:.4f}", UNSCHED, "only 'unschedulable' is plausible"))
        rows.append((f"{hi:.4f} <= p_hat <= {lo:.4f}", UNCERT, "both labels plausible"))
        rows.append((f"p_hat > {lo:.4f}", SCHED, "only 'schedulable' is plausible"))
    else:
        rows.append((f"p_hat <= {lo:.4f}", UNSCHED, "only 'unschedulable' is plausible"))
        rows.append((f"{lo:.4f} < p_hat < {hi:.4f}", UNCERT, "neither label plausible"))
        rows.append((f"p_hat >= {hi:.4f}", SCHED, "only 'schedulable' is plausible"))
    return rows


# ----------------------------------------------------------------------------
# certificate (integer arithmetic, same check as the Baruah et al. verifier)
# ----------------------------------------------------------------------------
def verify(Rp, C, D, T):
    m, n = Rp.shape
    ok_dl = Rp <= D
    ok_rec = np.zeros((m, n), dtype=bool)
    for i in range(n):
        f = C[:, i].copy()
        for j in range(i):
            f = f + (-(-Rp[:, i] // T[:, j])) * C[:, j]
        ok_rec[:, i] = Rp[:, i] >= f
    return (ok_dl & ok_rec).all(1)


# ----------------------------------------------------------------------------
# Integrated Gradients on the LOGIT
# ----------------------------------------------------------------------------
class LogitWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, xr, xc):
        _, p = self.model(xr, xc)
        out = torch.logit(p.reshape(-1).clamp(1e-6, 1 - 1e-6))
        # p_hat may depend on x_clf only; this zero term keeps x_reg in the graph
        # so Captum does not raise "tensor not used in the graph" (its IG stays 0).
        return out + 0.0 * xr.sum(dim=1)


def ig_per_task(model, xr, xc, base_r, base_c, n, n_steps):
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        sys.exit("captum is required: pip install captum")
    w = LogitWrapper(model).eval()
    ig = IntegratedGradients(w)
    br = base_r.expand(len(xr), -1).contiguous()
    bc = base_c.expand(len(xc), -1).contiguous()
    (ar, ac), delta = ig.attribute((xr, xc), baselines=(br, bc), n_steps=n_steps,
                                   return_convergence_delta=True)
    ar = ar.detach().numpy().reshape(-1, n, 3).sum(-1)
    ac = ac.detach().numpy().reshape(-1, n, 4).sum(-1)
    with torch.no_grad():
        base_logit = w(br, bc).numpy()
    return ar + ac, base_logit, np.abs(delta.detach().numpy())


# ----------------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------------
def f(v, nd=3):
    if isinstance(v, (float, np.floating)):
        if np.isinf(v):
            return "inf"
        return f"{v:.{nd}f}"
    return str(v)


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(f(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])


def anatomy(model):
    rows = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            rows.append({"layer": name, "type": "Linear  z = W x + b",
                         "in": mod.in_features, "out": mod.out_features,
                         "weights": mod.in_features * mod.out_features,
                         "biases": mod.out_features})
        elif isinstance(mod, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.GELU)):
            rows.append({"layer": name, "type": type(mod).__name__ + " (squashing step)",
                         "in": "", "out": "", "weights": 0, "biases": 0})
    return rows


# ----------------------------------------------------------------------------
# one example card
# ----------------------------------------------------------------------------
def card(idx, title, ctx):
    L = []
    p = float(ctx["p"][idx])
    logit = float(ctx["logit"][idx])
    L.append(f"### {title}  (test set #{idx})")
    L.append(f"- truth: **{'schedulable' if ctx['y'][idx] else 'UNschedulable'}**")
    L.append(f"- network says p_hat = **{p:.4f}**  ->  rule says **{ctx['lab'][idx]}**"
             + (f" ({ctx['why'][idx]})" if ctx['why'][idx] else ""))
    cert = bool(ctx["cert"][idx])
    L.append(f"- certificate (the safety check): **{'PASS' if cert else 'FAIL'}**"
             + ("" if ctx["lab"][idx] == SCHED else "  (only matters if the label is SCHEDULABLE)"))
    L.append("")
    L.append("**Step 1 - what goes in:** 15 tasks, each with run time C, period T, deadline D (ms)")
    L.append("")

    # layer-by-layer values
    L.append("**Step 2 - layers (each Linear computes z = W x + b):**")
    rows = []
    for kind, name, inp, out in ctx["records"]:
        if kind != "linear":
            continue
        z = out[idx].reshape(-1).numpy()
        rows.append({"layer": name, "inputs": int(inp[idx].numel()), "outputs": int(z.size),
                     "z_min": float(z.min()), "z_mean": float(z.mean()), "z_max": float(z.max()),
                     "z>0 count": int((z > 0).sum())})
    L.append(md_table(rows, ["layer", "inputs", "outputs", "z_min", "z_mean", "z_max", "z>0 count"]))
    L.append("")

    # last-layer decomposition
    k = ctx["last_k"]
    if k is not None:
        _, lname, inp, _ = ctx["records"][k]
        mod = ctx["modules"][lname]
        h = inp[idx].reshape(-1).numpy()
        w = mod.weight.detach().reshape(-1).numpy()
        b = float(mod.bias.detach().reshape(-1)[0]) if mod.bias is not None else 0.0
        contrib = w * h
        order = np.argsort(contrib)
        L.append(f"**Step 3 - the final decision number (layer `{lname}`):**")
        L.append(f"`logit = w . h + b`   where h = {h.size} hidden neuron values, "
                 f"w = {h.size} learned weights, b = {b:.4f}")
        crow = []
        top_pos = [j for j in order[::-1][:3] if contrib[j] > 0]
        top_neg = [j for j in order[:3] if contrib[j] < 0]
        shown = set(top_pos + top_neg)
        for j in top_pos + top_neg:
            crow.append({"part": f"neuron #{j}", "h (value)": float(h[j]),
                         "w (weight)": float(w[j]), "w x h": float(contrib[j])})
        rest = sum(contrib[j] for j in range(h.size) if j not in shown)
        crow.append({"part": f"all other {h.size - len(shown)} neurons", "h (value)": "",
                     "w (weight)": "", "w x h": float(rest)})
        crow.append({"part": "bias b", "h (value)": "", "w (weight)": "", "w x h": b})
        crow.append({"part": "**TOTAL = logit**", "h (value)": "", "w (weight)": "",
                     "w x h": float(contrib.sum() + b)})
        L.append(md_table(crow, ["part", "h (value)", "w (weight)", "w x h"]))
        err = abs(float(contrib.sum() + b) - logit)
        L.append(f"check: sum = {contrib.sum() + b:.4f}, network logit = {logit:.4f}, "
                 f"difference = {err:.5f}")
    else:
        L.append(f"**Step 3:** {ctx['note']}")
    L.append("")
    L.append(f"**Step 4 - squash to a probability:** p_hat = sigmoid({logit:.4f}) = "
             f"1 / (1 + e^-{logit:.4f}) = **{p:.4f}**")
    L.append("")
    L.append(f"**Step 5 - decision rule** (from Table B): p_hat = {p:.4f} -> **{ctx['lab'][idx]}**")
    L.append("")

    # per-task push
    if ctx["ig"] is not None and idx in ctx["ig_index"]:
        r = ctx["ig_index"][idx]
        push = ctx["ig"][r]
        order = np.argsort(-np.abs(push))[:6]
        trow = []
        for j in sorted(order):
            trow.append({"task rank": int(j + 1),
                         "C (ms)": float(ctx["Cms"][idx, j]), "T (ms)": float(ctx["Tms"][idx, j]),
                         "D (ms)": float(ctx["Dms"][idx, j]), "push on logit": float(push[j])})
        L.append("**Step 6 - which tasks pushed the number** (Integrated Gradients on the logit; "
                 "positive = toward SCHEDULABLE, negative = toward UNSCHEDULABLE; top 6 shown)")
        L.append(md_table(trow, ["task rank", "C (ms)", "T (ms)", "D (ms)", "push on logit"]))
        L.append(f"starting point (average task set) logit = {float(ctx['base_logit'][r]):.4f}; "
                 f"sum of all 15 pushes = {push.sum():.4f}; "
                 f"start + pushes = {float(ctx['base_logit'][r]) + push.sum():.4f} "
                 f"(network logit {logit:.4f})")
    L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# self-test (no data, no trained model needed)
# ----------------------------------------------------------------------------
class ToyJoint(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.reg = nn.Linear(3 * n, n - 1)
        self.fc1 = nn.Linear(4 * n, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sig = nn.Sigmoid()

    def forward(self, xr, xc):
        return self.reg(xr), self.sig(self.fc2(self.act(self.fc1(xc)))).squeeze(-1)


def selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name + (f"   [{extra}]" if extra else ""))
        ok = ok and bool(cond)

    n = 5
    torch.manual_seed(0)
    m = ToyJoint(n).eval()
    xr, xc = torch.rand(40, 3 * n), torch.rand(40, 4 * n)
    r, p, rec = trace(m, xr, xc)
    logit, k, note = find_logit_and_last_linear(p, rec)
    check("final Linear(.,1) found by hooks", k is not None and rec[k][1] == "fc2", note)
    if k is not None:
        mod = dict(m.named_modules())[rec[k][1]]
        h = rec[k][2]
        recon = ((h * mod.weight).sum(1) + mod.bias).detach()
        check("w.h + b reproduces the logit", torch.allclose(recon, logit, atol=1e-5),
              f"max diff {float((recon - logit).abs().max()):.2e}")
    check("sigmoid(logit) == p_hat", torch.allclose(torch.sigmoid(logit), p, atol=1e-5))

    # conformal rule vs brute-force prediction sets
    rng = np.random.default_rng(0)
    pc = rng.uniform(0, 1, 400)
    yc = (rng.uniform(0, 1, 400) < pc).astype(int)
    for marginal in (False, True):
        q0, q1 = conformal_thresholds(pc, yc, 0.1, marginal)
        grid = np.linspace(0, 1, 2001)
        lab, _ = decide(grid, q0, q1)
        good = True
        for pv, lv in zip(grid, lab):
            s1, s0 = (1 - pv) <= q1, pv <= q0
            exp = SCHED if (s1 and not s0) else UNSCHED if (s0 and not s1) else UNCERT
            good = good and (exp == lv)
        rows = region_text(q0, q1)
        check(f"decision rule matches prediction sets (marginal={marginal})", good,
              f"q0={q0:.3f} q1={q1:.3f}; {rows[0][0]} -> {rows[0][1]}")
    # coverage sanity: true label inside the set for >= ~(1-alpha) of fresh points
    q0, q1 = conformal_thresholds(pc, yc, 0.1, True)
    pt = rng.uniform(0, 1, 4000)
    yt = (rng.uniform(0, 1, 4000) < pt).astype(int)
    cover = np.mean(np.where(yt == 1, (1 - pt) <= q1, pt <= q0))
    check("marginal conformal coverage ~ >= 90%", cover >= 0.87, f"coverage={cover:.3f}")

    try:
        import captum  # noqa: F401
        base_r, base_c = torch.zeros(1, 3 * n), torch.zeros(1, 4 * n)
        att, bl, delta = ig_per_task(m, xr[:4], xc[:4], base_r, base_c, n, 100)
        with torch.no_grad():
            lg = LogitWrapper(m)(xr[:4], xc[:4]).numpy()
        gap = np.abs(att.sum(1) + bl - lg).max()
        check("IG: baseline logit + task pushes = logit", gap < 0.05, f"max gap {gap:.4f}")
    except ImportError:
        print("SKIP  IG check (captum not installed)")

    print("\nALL SELF-TESTS PASSED" if ok else "\nSOME SELF-TESTS FAILED")
    return ok


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def run_in_chunks(model, X_r, X_c, bs=2048):
    rs, ps, recs = [], [], []
    for s in range(0, len(X_r), bs):
        r, p, rec = trace(model, X_r[s:s + bs], X_c[s:s + bs])
        rs.append(r)
        ps.append(p)
        recs.append(rec)
    merged = []
    for j in range(len(recs[0])):
        kind, name = recs[0][j][0], recs[0][j][1]
        merged.append((kind, name,
                       torch.cat([rc[j][2] for rc in recs]), torch.cat([rc[j][3] for rc in recs])))
    return torch.cat(rs), torch.cat(ps), merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--data")
    ap.add_argument("--alpha", type=float, default=0.1, help="conformal error level")
    ap.add_argument("--marginal", action="store_true",
                    help="one pooled threshold instead of one per class")
    ap.add_argument("--n_per_group", type=int, default=1)
    ap.add_argument("--n_steps", type=int, default=200)
    ap.add_argument("--out_dir", default="decision_results")
    ap.add_argument("--repo_root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.ckpt or not a.data:
        ap.error("--ckpt and --data are required (or use --selftest)")

    d = np.load(a.data)
    n = int(d["n"])
    need_keys(d, ["train_X_reg", "train_X_bin", "cal_X_reg", "cal_X_bin", "cal_Y_bin",
                  "test_X_reg", "test_X_bin", "test_Y_bin"], a.data)
    model = load_model(a.ckpt, n, a.repo_root)
    modules = dict(model.named_modules())

    # ---- calibration -> thresholds
    cr, cc = scale_inputs(d["cal_X_reg"], d["cal_X_bin"])
    _, p_cal, _ = run_in_chunks(model, torch.from_numpy(cr), torch.from_numpy(cc))
    y_cal = d["cal_Y_bin"].astype(int)
    q0, q1 = conformal_thresholds(p_cal.numpy(), y_cal, a.alpha, a.marginal)

    # ---- test split
    tr_np, tc_np = scale_inputs(d["test_X_reg"], d["test_X_bin"])
    Xr, Xc = torch.from_numpy(tr_np), torch.from_numpy(tc_np)
    y = d["test_Y_bin"].astype(int)
    r, p, records = run_in_chunks(model, Xr, Xc)
    logit, last_k, note = find_logit_and_last_linear(p, records)
    p_np = p.numpy().astype(np.float64)
    lab, why = decide(p_np, q0, q1)

    # ---- certificate
    m = len(y)
    r_dim = r.shape[1]
    off = n - r_dim
    if off not in (0, 1):
        sys.exit(f"model returns {r_dim} response times for n={n}; expected n-1 or n")
    if "test_C_us" in d.files:
        C, D, T = (d["test_C_us"].astype(np.int64), d["test_D_us"].astype(np.int64),
                   d["test_T_us"].astype(np.int64))
        factor = 1e6
    else:
        C = np.rint(d["test_X_bin"][:, 0::4]).astype(np.int64)
        T = np.maximum(np.rint(d["test_X_bin"][:, 1::4]).astype(np.int64), 1)
        D = np.rint(d["test_X_bin"][:, 3::4]).astype(np.int64)
        factor = 1000.0
    Rp = np.ones((m, n), dtype=np.int64)
    Rp[:, off:] = np.clip(np.ceil(r.numpy().astype(np.float64) * factor), 1, 10 ** 12).astype(np.int64)
    if off == 1:
        Rp[:, 0] = C[:, 0]
    cert = verify(Rp, C, D, T)

    # ---- tables
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    A = anatomy(model)
    cA = ["layer", "type", "in", "out", "weights", "biases"]
    B = [{"p_hat range": rg, "decision": lb, "meaning": mn} for rg, lb, mn in region_text(q0, q1)]
    cB = ["p_hat range", "decision", "meaning"]

    Crows = []
    for name, dec in (("truly schedulable", y == 1), ("truly UNschedulable", y == 0)):
        Crows.append({"truth": name, "count": int(dec.sum()),
                      SCHED: int(((lab == SCHED) & dec).sum()),
                      UNSCHED: int(((lab == UNSCHED) & dec).sum()),
                      UNCERT: int(((lab == UNCERT) & dec).sum())})
    cC = ["truth", "count", SCHED, UNSCHED, UNCERT]
    accepted = (lab == SCHED) & cert
    S = [{
        "alpha": a.alpha, "mode": "marginal" if a.marginal else "per-class",
        "q0": q0, "q1": q1,
        "uncertain_pct": 100.0 * float((lab == UNCERT).mean()),
        "SCHEDULABLE_but_truly_unschedulable": int(((lab == SCHED) & (y == 0)).sum()),
        "SCHEDULABLE_and_certificate_PASS": int(accepted.sum()),
        "accepted_but_truly_unschedulable": int((accepted & (y == 0)).sum()),
        "truly_schedulable_accepted_pct": 100.0 * float((accepted & (y == 1)).sum() / max((y == 1).sum(), 1)),
    }]
    cS = list(S[0].keys())

    write_csv(out / f"{PREFIX}anatomy.csv", A, cA)
    write_csv(out / f"{PREFIX}rule.csv", B, cB)
    write_csv(out / f"{PREFIX}confusion.csv", Crows, cC)
    write_csv(out / f"{PREFIX}summary.csv", S, cS)

    # ---- pick examples
    rng = np.random.default_rng(a.seed)
    groups = [
        ("clearly SCHEDULABLE, truly schedulable", (lab == SCHED) & (y == 1)),
        ("clearly UNSCHEDULABLE, truly unschedulable", (lab == UNSCHED) & (y == 0)),
        ("UNCERTAIN", lab == UNCERT),
        ("DANGEROUS: said SCHEDULABLE but truly unschedulable", (lab == SCHED) & (y == 0)),
        ("MISSED: said UNSCHEDULABLE but truly schedulable", (lab == UNSCHED) & (y == 1)),
    ]
    chosen = []
    for title, mask in groups:
        ids = np.where(mask)[0]
        if len(ids) == 0:
            continue
        pick = rng.choice(ids, size=min(a.n_per_group, len(ids)), replace=False)
        chosen += [(title, int(i)) for i in pick]

    # ---- IG for the chosen sets
    tr_all, tc_all = scale_inputs(d["train_X_reg"], d["train_X_bin"])
    base_r = torch.from_numpy(tr_all.mean(0, keepdims=True))
    base_c = torch.from_numpy(tc_all.mean(0, keepdims=True))
    ids = [i for _, i in chosen]
    ig_arr, base_logit, delta = (None, None, None)
    ig_index = {}
    if ids:
        ig_arr, base_logit, delta = ig_per_task(model, Xr[ids], Xc[ids], base_r, base_c, n, a.n_steps)
        ig_index = {i: k for k, i in enumerate(ids)}

    ctx = dict(p=p_np, logit=logit.numpy(), y=y, lab=lab, why=why, cert=cert, records=records,
               last_k=last_k, note=note, modules=modules, ig=ig_arr, ig_index=ig_index,
               base_logit=base_logit,
               Cms=d["test_X_bin"][:, 0::4], Tms=d["test_X_bin"][:, 1::4], Dms=d["test_X_bin"][:, 3::4])

    md = ["# How the network decides: SCHEDULABLE / UNSCHEDULABLE / UNCERTAIN", "",
          f"- checkpoint `{a.ckpt}`, data `{a.data}`, test sets: {m}",
          f"- uncertain-bucket method: split conformal prediction, alpha = {a.alpha}, "
          f"{'one pooled threshold' if a.marginal else 'one threshold per class'}, "
          f"calibration sets = {len(y_cal)}",
          "- the CERTIFICATE (not p_hat) is what finally guarantees safety", "",
          "## Table A - the network's layers", md_table(A, cA), "",
          "Each Linear layer computes **z = W x + b**: every output is a weighted sum of all its "
          "inputs plus a bias. ReLU keeps positive z and sets negative z to 0. The last layer has "
          "one output (the logit); sigmoid turns it into a probability p_hat.", "",
          "## Table B - the decision rule", md_table(B, cB), "",
          "q0 and q1 come from the calibration split: they say how far p_hat may be from the "
          "truth on 'typical' sets. Close to 0 or 1 = confident; in the middle = uncertain.", "",
          "## Table C - what happened on the test split", md_table(Crows, cC), "",
          md_table(S, cS), "",
          "'DANGEROUS' = the label says SCHEDULABLE but the set is truly unschedulable. "
          "The certificate must catch every one of these (accepted_but_truly_unschedulable = 0).", "",
          "## Worked examples"]
    for title, i in chosen:
        md.append(card(i, title, ctx))
    if ids:
        md.append(f"IG convergence gap (max): {float(delta.max()):.4f}")
    text = "\n".join(md)
    (out / f"{PREFIX}report.md").write_text(text)
    print(text)
    print(f"\n[+] saved to {out}/")


if __name__ == "__main__":
    main()