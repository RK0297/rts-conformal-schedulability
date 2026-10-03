# Methodology: Learning-Assisted Schedulability Analysis with Conformal Prediction

This document describes the 4-phase methodology combining deep learning response-time regression, classical verifiable certificates, and split conformal prediction for real-time task systems.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Sporadic Task Set Input   |
                      |   Tau_i = (C_i, D_i, T_i)   |
                      |   DM order: D_1 <= ... <= D_n|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Learning-Enabled Component |
                      |  - Regresses R'_2 ... R'_n  |
                      |  - Predicts p_hat(Sched)    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Conformal Prediction Layer  |
                      | Non-conformity: S = 1-p_hat |
                      | Quantile q_hat at error a   |
                      +--------------+--------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
    Set: {Unschedulable}                   Set: {Sched} or {Sched, Unsched}
     [ FAST REJECT ]                                       |
                                                           v
                                            +-----------------------------+
                                            | Polynomial RTA Verifier     |
                                            | - R'_i >= C_i + sum ceil... |
                                            | - R'_i <= D_i               |
                                            +--------------+--------------+
                                                           |
                                           Passed? --------+-------- Failed?
                                              |                         |
                                              v                         v
                                      VERIFIED SCHEDULABLE     ESCALATE TO EXACT RTA
                                      (Zero False Positives)   (Avoids False Rejection)
```

---

## Phase 1: Workload Generation & Ground Truth
1. **Task Model:** Independent constrained-deadline sporadic tasks $\tau_i = (C_i, D_i, T_i)$ with $1 \le C_i \le D_i \le T_i$.
2. **Utilization Assignment:** Total utilization $U \in [0.1, 1.0]$ in steps of $0.1$. Individual utilizations $U_i$ generated via the UUniSort algorithm (Bini & Buttazzo 2005).
3. **Periods:** Sampled from $\mathcal{U}[1, 1000]$ (uniform) or log-uniform $\ln(T_i) \sim \mathcal{U}[\ln 1, \ln 1000]$.
4. **Execution Times:** $C_i = \max(1, \text{round}(U_i \cdot T_i))$.
5. **Deadlines:** Sampled uniformly $D_i \sim \mathcal{U}[C_i, T_i]$.
6. **Deadline-Monotonic (DM) Sorting:** Sorted in ascending order of deadlines ($D_1 \le D_2 \le \dots \le D_n$).
7. **Exact RTA Ground Truth:** Computed via classical Joseph & Pandya (1986) fix-point iteration until convergence or deadline violation.

---

## Phase 2: Baseline Model & Certificate Verification
1. **Model Architecture:**
   - Input: $3n$ features ($C_i, T_i, 1/T_i$) for each task. Deadlines are excluded from regression.
   - Hidden: 4 fully-connected layers $\times$ 30 neurons (ReLU).
   - Output: $n-1$ neurons predicting candidate response times $R_2', \dots, R_n'$ ($R_1' = C_1$).
2. **Asymmetric Loss Function ($w = 100$):**
   $$\mathcal{L} = \begin{cases}
   \left( \frac{R_i' - R_i}{R_i} \right)^2 & \text{if } R_i' \ge R_i \\
   \left( w \cdot \frac{R_i' - R_i}{R_i} \right)^2 & \text{if } R_i' < R_i
   \end{cases}$$
   Penalizes under-estimations by $w^2 = 10,000\times$, ensuring predicted values satisfy the recurrence.
3. **Classical Certificate Verifier:**
   Checks in $O(n^2)$ integer operations:
   - $R_i' \le D_i \quad \forall i$
   - $R_i' \ge C_i + \sum_{j < i} \lceil R_i'/T_j \rceil C_j \quad \forall i$
   Passing certificate guarantees schedulability with zero false positives.

---

## Phase 3: Conformal Prediction Layer
1. **Split-Conformal Calibration:**
   Data is split into Train (70%), Calibration (15%), and Test (15%).
2. **Non-Conformity Scoring:**
   Primary classification score:
   $$S_k = 1 - \hat{p}(y_k \mid x_k)$$
3. **Calibrated Quantile:**
   For user-selected error rate $\alpha \in (0, 1)$:
   $$\hat{q}_\alpha = \text{Quantile}\left(\{S_k\}_{k=1}^m, \; \frac{\lceil (m+1)(1-\alpha) \rceil}{m}\right)$$
4. **Prediction Sets:**
   $$\mathcal{C}(x) = \{ y : \hat{p}(y \mid x) \ge 1 - \hat{q}_\alpha \}$$
   - $\{\text{Schedulable}\}$: High confidence schedulable.
   - $\{\text{Unschedulable}\}$: High confidence unschedulable.
   - $\{\text{Schedulable, Unschedulable}\}$: **Uncertain** boundary case.
5. **Safety Invariance:** Schedulability is **never** certified on CP probability alone; the polynomial RTA verifier must always validate the candidate certificate $R_i'$.

---

## Phase 4: Evaluation & Triaged Actions
- **Safety False Positive Rate (FPR):** Must remain strictly $0.000\%$.
- **Operational False Rejection Rate (FRR):** Percentage of schedulable systems unnecessarily rejected. CP quantifies and reduces this by escalating ambiguous cases to fallback exact RTA.
- **Uncertainty Rate:** Percentage of task sets designated as uncertain.
- **Acceptance Rate:** Fraction of true schedulable sets verifiably admitted.
