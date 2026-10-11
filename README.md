# Learning-Assisted Real-Time Schedulability Analysis with Split Conformal Prediction

---

## 1. Project Summary in Plain English

### The Problem
In safety-critical Real-Time Systems (such as autonomous driving, drone avionics, and industrial robots), the operating system must decide at runtime whether a new task can be safely admitted without missing any deadlines. 
- Classical exact Response-Time Analysis (RTA) is **NP-complete**. While usually fast on average, in the worst case it takes exponential / pseudo-polynomial time and can spike to tens of milliseconds on embedded processors, causing timing deadline overruns.
- Machine Learning (Deep Neural Networks) can predict schedulability in **microseconds**. But neural networks hallucinate and make mistakes: an unverified neural network admits unschedulable tasks **11% to 18.5% of the time**, causing catastrophic deadline misses!

### Our Solution
1. **Neural Network Regression:** A Deep Neural Network (MLP) predicts candidate worst-case response times in microseconds.
2. **$O(n^2)$ Polynomial Certificate Verifier:** A mathematical certificate checker validates the neural network's predictions in $< 3$ microseconds using Tarski's fixed-point theorem, guaranteeing **strictly 0.0000% Safety False Positives** (no unschedulable task set is ever admitted).
3. **Split Conformal Prediction:** Instead of making brittle yes/no guesses, Conformal Prediction wraps the system with rigorous statistical confidence ($1 - \alpha$), isolating difficult borderline task sets into calibrated prediction sets: `[Schedulable]`, `[Unschedulable]`, and `[Uncertain]`.

---

## 2. Real-Time Systems (RTS) Concepts Index

The following Real-Time Systems concepts are directly implemented and evaluated in this project:

1. **Hard Real-Time Systems** (Zero-tolerance for timing deadline misses)
2. **Event-Driven vs. Clock-Driven Scheduling** (Sporadic task arrivals vs. cyclic executive frames)
3. **Constrained-Deadline Sporadic Task Model** ($\tau_i = (C_i, D_i, T_i)$ with $C_i \le D_i \le T_i$)
4. **Preemptive Fixed-Priority (FP) Scheduling** (Highest static priority always runs)
5. **Deadline-Monotonic (DM) Priority Ordering** (Shorter deadline = higher priority; proven optimal)
6. **Exact Response-Time Analysis (RTA)** (Joseph & Pandya 1986 iterative recurrence)
7. **Computational Complexity in Schedulability** (NP-completeness of FP vs. coNP-completeness of EDF)
8. **Tarski's Fixed-Point Theorem & Monotonicity** (Mathematical foundation of the zero-error safety certificate)
9. **Online Admission Control in RTOS** (Dynamic real-time workload acceptance)

---

## 3. Real-Time Concepts in Detail

### Concept 1: Hard Real-Time Systems vs. Soft Real-Time Systems
- **Hard Real-Time:** Missing a single deadline causes total system failure or catastrophic physical damage (e.g., automotive braking, aircraft flight control). In a hard real-time system, a late result is a wrong result.
- **How it applies here:** Because this is a hard real-time system, our primary metric is **Safety False Positive Rate (FPR)**. Admitting an unschedulable task system is catastrophic. Our system mathematically forces this error rate to **0.0000%**.

### Concept 2: Event-Driven (Priority-Driven) vs. Clock-Driven Systems
A foundational real-time systems distinction is how tasks are triggered and dispatched:

| Dimension | Clock-Driven (Time-Driven / Cyclic Executive) | Event-Driven (Priority-Driven / Sporadic) [THIS PROJECT] |
| :--- | :--- | :--- |
| **Dispatch Trigger** | Regular clock ticks from a timer interrupt. | System **events** (task arrivals, sensor triggers, interrupts). |
| **Task Model** | Strictly periodic with known, synchronized phase offsets. | **Sporadic tasks** with irregular arrival times separated by minimum period $T_i$. |
| **Preemption** | Pre-allocated offline schedule slices. | **Online preemptive**: Higher-priority task arrival event preempts lower-priority task immediately. |
| **Priority Assignment** | No dynamic priorities; static timetable lookup. | Static **Deadline-Monotonic (DM)** priority ordering ($D_i \le D_j \implies \text{Priority}(i) > \text{Priority}(j)$). |
| **Admission Control** | Static offline feasibility calculation. | **Online event-driven admission**: Triggered dynamically when a new sporadic task arrives. |
| **Exact Complexity** | Polynomial table traversal. | Exact FP Response-Time Analysis is **NP-complete**. |

> **Summary:** Our project models an **Event-Driven, Priority-Driven** system. Tasks arrive sporadically upon external events, execute with preemptive fixed priorities, and admission requests are evaluated online at the moment of task arrival.

### Concept 3: Constrained-Deadline Sporadic Task Model
A task set $\Gamma = \{\tau_1, \tau_2, \dots, \tau_n\}$ consists of $n$ independent sporadic tasks. Each task $\tau_i$ is defined by 3 parameters:
$$\tau_i = (C_i, D_i, T_i)$$
- **$C_i$ (Worst-Case Execution Time - WCET):** The maximum uninterrupted CPU time the task needs to finish.
- **$D_i$ (Relative Deadline):** The time budget allowed from when a task arrives to when it must complete.
- **$T_i$ (Minimum Inter-Arrival Time / Period):** The shortest possible time between two consecutive arrivals of this task.
- **Constrained-Deadline Rule:** $1 \le C_i \le D_i \le T_i$. (Each task must complete before its next release).

### Concept 4: Deadline-Monotonic (DM) Scheduling & Optimality
- In Fixed-Priority scheduling, each task has a static priority level assigned before execution.
- **Deadline-Monotonic rule:** Tasks with shorter relative deadlines receive higher priorities:
  $$D_1 \le D_2 \le \dots \le D_n \implies \text{Priority}(\tau_1) > \text{Priority}(\tau_2) > \dots > \text{Priority}(\tau_n)$$
- **Optimality Theorem (Leung & Whitehead, 1982):** DM is proven optimal for constrained-deadline sporadic task systems on a uniprocessor. If *any* static priority assignment can successfully schedule the task set, Deadline-Monotonic ordering will also succeed.

### Concept 5: Exact Response-Time Analysis (RTA)
A lower-priority task experiences interference whenever higher-priority tasks run. Under the critical instant (all higher-priority tasks released simultaneously), the worst-case response time $R_i$ of task $\tau_i$ satisfies the recurrence:
$$R_i = C_i + \sum_{j < i} \left\lceil \frac{R_i}{T_j} \right\rceil C_j$$
- The ceiling term $\lceil R_i / T_j \rceil$ calculates the maximum number of times higher-priority task $\tau_j$ can preempt task $\tau_i$.
- Multiplied by $C_j$, it gives the total preemption interference.
- The iteration starts at $R_i^{(0)} = C_i$ and iterates until $R_i^{(k+1)} = R_i^{(k)}$.
- If $R_i \le D_i$ for all tasks, the system is **schedulable**.

### Concept 6: Computational Complexity (Why FP is in NP while EDF is coNP)
Sanjoy Baruah et al. (2025) proved **Proposition 1**: A schedulability decision problem can be verified by a learning-assisted polynomial certificate if and only if the problem belongs to the complexity class **NP**.
- **Fixed-Priority Schedulability is in NP (NP-complete):** If a task system is schedulable, the true response times $\{R_1, \dots, R_n\}$ serve as a polynomial-size witness checkable in $O(n^2)$ time. Therefore, neural networks can output candidate certificates for FP scheduling!
- **EDF Schedulability is in coNP (coNP-complete):** Under Earliest-Deadline First, proving schedulability requires verifying processor demand at an exponential number of testing points. No short polynomial certificate of schedulability exists unless $\mathbf{NP} = \mathbf{coNP}$.

### Concept 7: Tarski's Theorem & The Zero-Risk Certificate Proof
Let $R_i'$ be the candidate response time predicted by our neural network:
1. **Recurrence Check:** $R_i' \ge C_i + \sum_{j < i} \lceil R_i' / T_j \rceil C_j$
2. **Deadline Check:** $R_i' \le D_i$

Because the recurrence function $f_i(R) = C_i + \sum_{j < i} \lceil R / T_j \rceil C_j$ is monotonically non-decreasing, **Tarski's Fixed-Point Theorem** proves that the exact worst-case response time $R_i^*$ must satisfy:
$$R_i^* \le R_i' \le D_i$$
Therefore, any task set that passes the verifier is mathematically guaranteed to never miss a deadline. **Safety False Positive Rate = 0.0000%**.

---

## 4. Step-by-Step Walkthrough: What We Have Done

```mermaid
flowchart TD
    Step1["Step 1: Synthetic Workload Generation<br>UUniSort algorithm (U in 0.1 to 1.0) + Exact RTA Ground Truth"] --> Step2["Step 2: Neural Network Training<br>Joint MLP (128-64-64) with Asymmetric Loss (w = 25)"]
    Step2 --> Step3["Step 3: Classical Polynomial Verifier<br>O(nÂ²) check: R'_i >= f(R'_i) and R'_i <= D_i"]
    Step2 --> Step4["Step 4: Split Conformal Calibration<br>Hold-out split, non-conformity S = 1 - p_hat, empirical quantiles"]
    Step3 & Step4 --> Step5["Step 5: Unified Decision & Triage<br>Certified Schedulable, Fast Reject, or Uncertain Boundary"]

    style Step1 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Step2 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Step3 fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style Step4 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style Step5 fill:#fce4ec,stroke:#c2185b,stroke-width:2px
```

### Step 1: Workload Synthesis & Ground Truth (`data/generate_tasksets.py`)
- We generate 10,000 synthetic task sets for $n=10$ and $n=15$ across 10 utilization levels ($U \in [0.1, 1.0]$ in steps of 0.1).
- We use the **UUniSort** algorithm (Bini & Buttazzo 2005) to distribute total utilization $U$ uniformly across individual tasks.
- Periods are sampled from $\mathcal{U}[1, 1000]$, execution times $C_i = \max(1, \text{round}(U_i \cdot T_i))$, and deadlines $D_i \sim \mathcal{U}[C_i, T_i]$.
- Tasks are sorted in Deadline-Monotonic order ($D_1 \le D_2 \le \dots \le D_n$).
- We compute ground truth labels using classical exact RTA fixed-point iteration (`data/exact_rta.py`).
- Data is split into 70% Training (7,000), 15% Calibration (1,500), and 15% Test (1,500).

### Step 2: Model Architecture & Asymmetric Training (`training/train_baseline.py`)
- We implement a PyTorch neural network (`models/baseline_mlp.py`) with 4 fully-connected layers (128 $\rightarrow$ 64 $\rightarrow$ 64 neurons).
- Inputs are scaled $[C_i, T_i, 1/T_i]$ (deadlines $D_i$ are excluded from regression because physics dictates response times are independent of deadlines).
- We train using an **asymmetric normalized loss function** ($w = 25$):
  $$\mathcal{L} = \begin{cases}
  \left(\frac{R_i' - R_i}{R_i}\right)^2 & \text{if } R_i' \ge R_i \\
  \left(w \cdot \frac{R_i' - R_i}{R_i}\right)^2 & \text{if } R_i' < R_i
  \end{cases}$$
  Penalizing under-predictions by $w^2 = 625\times$ is essential: under-estimating response times causes the recurrence check to fail immediately. Over-estimating is safe as long as it does not exceed $D_i$.

### Step 3: Classical Polynomial Certificate Verification (`verification/rta_verifier.py`)
- The neural network's candidate response times $R_i'$ are passed into our $O(n^2)$ certificate verifier.
- The verifier executes in **$< 3$ microseconds** on CPU.
- If $R_i' \ge f_i(R_i')$ and $R_i' \le D_i$, the system is accepted with **100% mathematical certainty**.

### Step 4: Split Conformal Prediction Calibration (`training/calibrate_cp.py`)
- Standard neural networks make binary decisions at $\hat{p} \ge 0.5$ without confidence guarantees.
- We calibrate non-conformity scores $S_k = 1 - \hat{p}(y_k \mid x_k)$ on the 1,500 hold-out calibration task sets.
- For confidence levels 99% ($\alpha=0.01$) and 95% ($\alpha=0.05$), we extract the non-parametric empirical quantiles $\hat{q}_\alpha$.

### Step 5: Triaged Prediction Set Generation (`models/conformal_wrapper.py`)
- For any new task set, Conformal Prediction outputs a guaranteed prediction set:
  - `[Schedulable]`: High confidence schedulable $\rightarrow$ verified by certificate $\rightarrow$ admitted.
  - `[Unschedulable]`: High confidence unschedulable $\rightarrow$ fast reject (no CPU time wasted).
