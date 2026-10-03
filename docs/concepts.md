# Real-Time Systems Concept Mapping: Unit-I & Unit-II

This guide maps each technical component of the repository to standard Real-Time Systems course concepts.

---

## Unit-I: Real-Time Systems Fundamentals & Timing Constraints

### 1. Hard Real-Time Systems
- **Concept:** Systems where missing a single timing constraint (deadline) can lead to catastrophic consequences.
- **Repository Implementation:** Enforced via the zero-false-positive certificate verifier (`verification/rta_verifier.py`). An unschedulable task system is never admitted.

### 2. Constrained-Deadline Sporadic Task Model
- **Concept:** A set of recurrent tasks $\Gamma = \{\tau_1, \dots, \tau_n\}$ where each task $\tau_i = (C_i, D_i, T_i)$ has:
  - Worst-Case Execution Time: $C_i$
  - Relative Deadline: $D_i$
  - Minimum Inter-Arrival Period: $T_i$
  - Constraint: $D_i \le T_i$.
- **Repository Implementation:** Implemented in `data/generate_tasksets.py`.

### 3. Deadline-Monotonic (DM) Scheduling
- **Concept:** Fixed-priority scheduling where tasks with shorter relative deadlines receive higher priorities ($D_i \le D_j \implies \text{Priority}(i) > \text{Priority}(j)$). Proven optimal for constrained-deadline sporadic tasks on uniprocessors (Leung & Whitehead, 1982).
- **Repository Implementation:** Tasks are sorted in ascending order of deadlines prior to feature extraction and analysis.

### 4. Response-Time Analysis (RTA)
- **Concept:** Exact necessary and sufficient test for preemptive fixed-priority schedulability:
  $$R_i = C_i + \sum_{j \in hp(i)} \left\lceil \frac{R_i}{T_j} \right\rceil C_j$$
  Schedulable iff $R_i \le D_i$ for all tasks.
- **Repository Implementation:** Implemented in `data/exact_rta.py` using integer fix-point iteration.

### 5. Task & System Utilization
- **Concept:** $U_i = C_i / T_i$, total utilization $U = \sum_{i=1}^n U_i$.
- **Repository Implementation:** UUniSort algorithm distributes total utilization $U \in [0.1, 1.0]$ across $n$ tasks.

---

## Unit-II: RTOS, Priority Scheduling & Computational Complexity

### 1. Preemptive Fixed-Priority Scheduling
- **Concept:** The processor always executes the active task with the highest static priority. Preemptions occur whenever a higher-priority task arrives.
- **Repository Implementation:** Evaluated in both exact RTA and the polynomial certificate verifier.

### 2. Computational Complexity: NP-Completeness vs coNP-Completeness
- **Fixed-Priority Schedulability is in NP:** If a system is schedulable, candidate response times $R_i'$ serve as a polynomial-time witness checkable in $O(n^2)$ time.
- **EDF Schedulability is coNP-complete:** Proving schedulability under EDF requires checking all points in an exponential testing set $\mathcal{T}(\Gamma)$. No short polynomial certificate of schedulability exists unless NP = coNP.
- **Baruah et al. Proposition 1:** The verifiable certificate approach is applicable if and only if the schedulability decision problem belongs to NP.

### 3. Admission Control in RTOS
- **Concept:** Deciding dynamically whether to accept a newly submitted real-time task without violating existing deadlines.
- **Repository Implementation:** Neural inference (< 1 ms) combined with rapid certificate verification provides predictable, bounded execution times suitable for online admission control paths.
