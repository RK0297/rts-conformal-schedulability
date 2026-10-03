# Paper Summary: Baruah et al. (2025)

**Title:** Learning-assisted schedulability analysis: opportunities and limitations  
**Citation:** Real-Time Systems (2025) 61:332–358  

---

## 1. Problem & Motivation
- **Runtime Admission Control:** In dynamic real-time cyber-physical systems (CPS), tasks arrive or modify operational modes dynamically, requiring rapid online schedulability checks.
- **Limitation of Classical Exact RTA:** Determining whether a constrained-deadline sporadic task system is Fixed-Priority (FP) schedulable is **NP-complete**. While average-case response-time iterations are fast, the worst-case running time is pseudo-polynomial and can spike significantly (e.g., up to 70 ms on embedded platforms), jeopardizing runtime guarantees.
- **Role of Machine Learning:** Deep neural networks (MLPs) offer fast, deterministic execution times (< 1 ms on embedded CPUs, < 4 µs on FPGA).
- **The Core Challenge:** Neural networks occasionally make classification errors:
  - *False negative:* Declaring a schedulable system unschedulable (conservative, hurts resource utilization).
  - *False positive:* Declaring an unschedulable system schedulable (**unsafe**, safety hazard).
- **Goal:** Design a learning-enabled framework that guarantees **zero false positives** (never certifiably admits an unschedulable workload).

---

## 2. Theoretical Foundation: Proposition 1
Baruah et al. relate the applicability of learning-assisted schedulability verification directly to computational complexity:

> **Proposition 1:** Restricting the verification algorithm to run in at most polynomial time, it is **necessary and sufficient** for a schedulability condition to belong to the complexity class **NP** in order to be checkable via a learning-generated certificate.

### Fixed-Priority (FP) Scheduling
- Preemptive uniprocessor FP schedulability is **NP-complete**.
- A candidate certificate consists of candidate response times $R_i'$ for each task $\tau_i$.
- Checking whether $R_i' \le D_i$ and verifying the RTA recurrence takes $O(n^2)$ arithmetic steps, which is polynomial time. Hence, FP fits the framework.

### Earliest-Deadline First (EDF) Scheduling
- Constrained-deadline EDF schedulability is **coNP-complete**.
- Exact verification requires checking processor demand $\sum \text{dbf}_i(t) \le t$ over an exponential testing set $\mathcal{T}(\Gamma)$.
- A short certificate exists for *unschedulability* (a single violating point $t$), but not for *schedulability* (unless NP = coNP). Hence, EDF cannot directly use this framework.

---

## 3. The Certificate Mechanism

Under Deadline-Monotonic (DM) priority ordering ($D_1 \le D_2 \le \dots \le D_n$), the classical RTA recurrence is:
$$R_i \ge C_i + \sum_{j \in hp(i)} \left\lceil \frac{R_i}{T_j} \right\rceil C_j$$

### Verification Rule:
Given neural predictions $R_2', R_3', \dots, R_n'$ (with $R_1' = C_1$ exact):
1. **Recurrence Check:** Check that for each task $\tau_i$:
   $$R_i' \ge C_i + \sum_{j < i} \left\lceil \frac{R_i'}{T_j} \right\rceil C_j$$
2. **Deadline Check:** Check that for each task $\tau_i$:
   $$R_i' \le D_i$$

### Why Over-approximating Certificates are Safe:
The recurrence function $f_i(R) = C_i + \sum_{j < i} \lceil R / T_j \rceil C_j$ is monotonically non-decreasing. If candidate $R_i'$ satisfies $R_i' \ge f_i(R_i')$ and $R_i' \le D_i$, then by Tarski's fixed point theorem, the minimal positive fixed point $R_i^*$ (exact worst-case response time) must satisfy:
$$R_i^* \le R_i' \le D_i$$
Therefore, any task set that passes certificate verification is provably schedulable with **strictly zero false positives**.

---

## 4. Model Architecture & Loss Function
- **Input Features ($3n$):** $[C_i, T_i, 1/T_i]$ for all tasks in DM order. Deadlines $D_i$ are excluded because response times are independent of deadlines.
- **Topology:** 4 hidden fully-connected layers of 30 neurons each with ReLU activation functions. Output layer has $n-1$ neurons predicting $R_2', \dots, R_n'$.
- **Asymmetric Loss Function:**
  $$\mathcal{L} = \begin{cases}
  \left( \frac{R_i' - R_i}{R_i} \right)^2 & \text{if } R_i' \ge R_i \\
  \left( w \cdot \frac{R_i' - R_i}{R_i} \right)^2 & \text{if } R_i' < R_i
  \end{cases}$$
  With penalty weight $w = 100$, severely penalizing under-predictions that would cause certificate rejection.
