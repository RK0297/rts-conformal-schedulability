# Replication Understanding Guide (for Your Own Preparation)

This is **not a slide deck**.  
This is a personal preparation document so you can confidently explain:
- real-time systems concepts used in this repo,
- what the original paper does,
- what we replicated,
- what results we got,
- and why our numbers differ.

---

## 1) One-Line Story of This Project

We use deep learning to speed up schedulability checks for fixed-priority real-time tasks, but we keep safety by verifying every accepted result with a classical mathematical certificate; then we add conformal prediction to mark uncertain cases instead of making risky binary decisions.

---

## 2) Core Real-Time Systems Concepts (Simple Language)

## 2.1 Hard Real-Time System
- In hard real-time, a missed deadline is a system failure.
- So a method is acceptable only if it never labels an unsafe task set as safe.

**In this repo:** safety is protected by the verifier (`verification/rta_verifier.py`), not by neural confidence.

## 2.2 Sporadic Constrained-Deadline Task Model
Each task is:
- `C` = worst-case execution time
- `D` = relative deadline
- `T` = minimum period (minimum gap between two releases)

Constraint: `C <= D <= T`.

**In this repo:** task sets are generated in `data/generate_tasksets.py`.

## 2.3 Deadline-Monotonic (DM) Scheduling
- Fixed-priority scheduling.
- Smaller deadline => higher priority.
- Tasks are sorted by deadline before analysis.

## 2.4 Response-Time Analysis (RTA)
- Exact schedulability check under fixed priority.
- For each task, response time includes its own execution + interference from higher-priority tasks.
- If response time of every task is within its deadline, the task set is schedulable.

**In this repo:** exact RTA logic supports ground truth and verification pipeline.

## 2.5 Safety Metrics
- **False Positive (unsafe):** unschedulable predicted as schedulable (catastrophic).
- **False Rejection (safe but rejected):** schedulable predicted/rejected as unschedulable (conservative but costly).

Goal in this project: keep false positives at zero.

---

## 3) Original Paper Idea (Baruah et al., 2025)

Problem:
- Exact checks can be costly in worst case for online admission decisions.
- Pure ML is fast but unsafe.

Key idea:
1. ML predicts candidate response times.
2. A polynomial-time verifier checks certificate inequalities.
3. Accept only if verifier passes.

Important theorem-level point:
- This certificate trick is practical for NP-style schedulability settings (like fixed-priority case used here).

Resulting guarantee:
- Verified acceptance has zero unsafe admissions (zero safety false positives).

---

## 4) What **We** Added in Replication

We replicated the verified-learning setup and added a conformal prediction (CP) layer.

Why CP?
- Baseline verified method can reject too many truly schedulable sets when ML prediction is slightly off.
- CP gives calibrated uncertainty sets instead of only hard yes/no.

CP output types:
- `{Schedulable}` (confident schedulable class),
- `{Unschedulable}` (confident unschedulable class),
- `{Schedulable, Unschedulable}` (uncertain/boundary case).

Operational policy in this repo:
- Never certify schedulability from CP alone.
- Certification still requires verifier passing.
- Uncertain cases are separated and can be escalated to exact analysis.

---

## 5) Pipeline in This Repository (Explain This Clearly)

1. **Generate data** across utilization levels (`U = 0.1 to 1.0`).
2. **Compute exact labels/response-time references** using classical analysis.
3. **Train neural model** (joint regression/classification setup in project flow).
4. **Run verifier** on model candidate response times.
5. **Calibrate CP** on calibration split for chosen `alpha`.
6. **Compare baseline vs verified vs CP-augmented behavior** using evaluation scripts and markdown summaries.

---

## 6) How to Explain Our Results (n=15 and n=20)

From `experiments/results/comparison_n15.md` and `comparison_n20.md`:

1. **Unverified baseline**
   - High apparent acceptance/accuracy,
   - but non-zero safety FPR (unsafe in hard real-time context).

2. **Verified approach**
   - Safety FPR becomes exactly `0`,
   - but acceptance drops (higher operational false rejection).

3. **CP-augmented reporting**
   - Keeps verified safety behavior,
   - quantifies uncertainty,
   - coverage tracks confidence target (especially at lower alpha values),
   - gives a controlled uncertainty signal for escalation strategy.

Main message:
- In safety-critical systems, verified safety is non-negotiable.
- CP is useful for transparency/calibration around ambiguous decisions.

---

## 7) Why Our Acceptance May Be Lower Than Paper

Use this if asked during discussion:

1. Smaller training scale vs large-scale original experiments.
2. Harder interference structure at larger task counts (`n=15`, `n=20`).
3. Sensitivity of recurrence inequalities to under/over prediction.
4. Practical architecture/training-budget differences in replication.

This is normal in replication studies: safety guarantee can still hold while acceptance differs.

---

## 8) Part-1 and Part-2 Preparation Split

As requested, do not present full project in one go.

## Part 1 (First Presentation, ~70% scope: what we have done)
Focus on:
1. Real-time basics (task model, DM, RTA, hard real-time safety).
2. Original paper problem and certificate idea.
3. Our replication pipeline and implementation flow.
4. Core result: verified safety FPR = 0, plus key n=15/n=20 table interpretation.
5. Why CP layer was added and what “uncertain” means.

End Part 1 with:
- “We have replicated safety-preserving verified learning behavior and added calibrated uncertainty reporting.”

## Part 2 (Later Presentation: remaining depth)
Focus on:
1. Deeper theoretical interpretation (NP vs coNP insight and limits).
2. Detailed alpha-wise CP behavior and coverage interpretation.
3. Full limitations and future work directions.
4. Replication gap analysis (training scale/model capacity/data diversity).
5. Suggested improvements for next milestone.

---

## 9) Quick Self-Check Before You Present Part 1

You are ready if you can answer:
1. Why pure ML is unsafe in hard real-time.
2. Why verifier gives zero false positives.
3. Difference between safety FPR and operational FRR.
4. What CP adds without breaking safety policy.
5. Why replication numbers can differ from original paper.

If you can explain these five without notes, your Part 1 is strong.

