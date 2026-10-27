"""Reference schedulability analyses for RM (Rate-Monotonic), DM (Deadline-Monotonic), and EDF (Earliest-Deadline First).

Theoretical Foundations:
1. Rate-Monotonic (RM):
   - Static priority: shorter period T_i means higher priority.
   - Proven optimal among static-priority uniprocessor schedulers for implicit-deadline tasks (Liu & Layland 1973).
   - Fast sufficient checks: Liu-Layland utilization bound U <= n(2^(1/n) - 1) and Bini et al. Hyperbolic Bound.
   - Exact analysis: Response-Time Analysis (RTA) evaluated under RM priority ordering.

2. Deadline-Monotonic (DM):
   - Static priority: shorter relative deadline D_i means higher priority.
   - Proven optimal among static-priority uniprocessor schedulers for constrained-deadline tasks (Leung & Whitehead 1982).
   - Exact analysis: Joseph & Pandya (1986) iterative fixed-point recurrence R_i = C_i + sum_{j<i} ceil(R_i / T_j) * C_j.

3. Earliest-Deadline First (EDF):
   - Dynamic priority: at each instant, the job with the earliest absolute deadline executes.
   - Proven optimal among all uniprocessor scheduling algorithms (Dertouzos 1973).
   - For implicit deadlines (D_i = T_i): schedulable iff sum(C_i / T_i) <= 1.0.
   - For constrained deadlines (D_i <= T_i): exact analysis via Processor Demand Analysis (Baruah et al. 1990).
     Verifies dbf(t) <= t for all deadline points t <= L*, where L* is bounded by the synchronous
     busy period and the Baruah/Spuri testing bound.

Dominance Hierarchy for Constrained Deadlines:
   Schedulable(RM) <= Schedulable(DM) <= Schedulable(EDF)
"""
from typing import Dict, List, Optional, Tuple
import numpy as np


# =============================================================================
# 1. FIXED-PRIORITY RESPONSE-TIME ANALYSIS (RTA)
# =============================================================================

def fixed_priority_rta(
    C: np.ndarray,
    D: np.ndarray,
    T: np.ndarray,
    priority_order: np.ndarray,
    max_iterations: int = 1000,
    max_bound: Optional[int] = None,
) -> Tuple[bool, np.ndarray]:
    """Run exact response-time analysis for a specified static priority order.

    Args:
        C: Worst-case execution times (WCET) of tasks.
        D: Relative deadlines of tasks.
        T: Minimum inter-arrival times (periods) of tasks.
        priority_order: Permutation array of task indices from highest to lowest priority.
        max_iterations: Maximum recurrence iterations per task.
        max_bound: Cap on response-time iterations (defaults to 10 * max(T)).

    Returns:
        (is_schedulable, response_times):
            is_schedulable: True if R_i <= D_i for all tasks i.
            response_times: Array of computed worst-case response times indexed by original task id.
    """
    C = np.asarray(C, dtype=np.int64)
    D = np.asarray(D, dtype=np.int64)
    T = np.asarray(T, dtype=np.int64)
    order = np.asarray(priority_order, dtype=np.int64)
    n = len(C)

    if C.shape != D.shape or C.shape != T.shape or order.shape != (n,):
        raise ValueError("C, D, T, and priority_order must describe the same tasks")
    if np.any(C <= 0) or np.any(T <= 0) or np.any(D < C):
        raise ValueError("Tasks must satisfy C > 0, T > 0, and D >= C")
    if not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError("priority_order must be a valid permutation of task indices")

    if max_bound is None:
        max_bound = max(100000, int(np.max(T)) * 10)

    response = np.zeros(n, dtype=np.int64)
    for position, task in enumerate(order):
        r = int(C[task])
        converged = False
        higher = order[:position]
        for _ in range(max_iterations):
            # ceil(r / T[higher]) = (r + T[higher] - 1) // T[higher]
            releases = (r + T[higher] - 1) // T[higher]
            next_r = int(C[task] + np.sum(releases * C[higher]))
            if next_r == r:
                converged = True
                break
            if next_r > max_bound:
                r = next_r
                converged = True
                break
            r = next_r
        if not converged:
            r = max_bound
        response[task] = r

    return bool(np.all(response <= D)), response


def rate_monotonic_rta(
    C: np.ndarray, D: np.ndarray, T: np.ndarray, **kwargs
) -> Tuple[bool, np.ndarray]:
    """Exact Rate-Monotonic Analysis (RMA).

    Assigns static priorities based on period: shorter period T_i => higher priority.
    Uses exact Response-Time Analysis recurrence.
    """
    order = np.argsort(np.asarray(T), kind="stable")
    return fixed_priority_rta(C, D, T, order, **kwargs)


def deadline_monotonic_rta(
    C: np.ndarray, D: np.ndarray, T: np.ndarray, **kwargs
) -> Tuple[bool, np.ndarray]:
    """Exact Deadline-Monotonic Analysis (DMA).

    Assigns static priorities based on relative deadline: shorter deadline D_i => higher priority.
    Proven optimal for constrained-deadline uniprocessor systems (Leung & Whitehead 1982).
    """
    order = np.argsort(np.asarray(D), kind="stable")
    return fixed_priority_rta(C, D, T, order, **kwargs)


# =============================================================================
# 2. SUFFICIENT UTILIZATION BOUNDS FOR RATE-MONOTONIC SCHEDULING
# =============================================================================

def liu_layland_bound(n: int) -> float:
    """Compute the classical Liu & Layland (1973) utilization bound: U_LL = n * (2^(1/n) - 1)."""
    if n <= 0:
        return 0.0
    return float(n * (2.0 ** (1.0 / n) - 1.0))


def rate_monotonic_sufficient_test(C: np.ndarray, T: np.ndarray) -> bool:
    """Sufficient (O(n)) Liu & Layland schedulability test for implicit deadlines (D_i = T_i).

    If sum(C_i / T_i) <= n(2^(1/n) - 1), the system is guaranteed schedulable.
    """
    C = np.asarray(C, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    u_total = float(np.sum(C / T))
    return u_total <= liu_layland_bound(len(C))


def hyperbolic_bound_test(C: np.ndarray, T: np.ndarray) -> bool:
    """Sufficient (O(n)) Hyperbolic Bound test (Bini, Buttazzo, Buttazzo 2001).

    If prod(U_i + 1) <= 2, the system is guaranteed schedulable under Rate-Monotonic.
    Strictly dominates the Liu & Layland bound.
    """
    C = np.asarray(C, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    u_i = C / T
    return bool(np.prod(u_i + 1.0) <= 2.0)


# =============================================================================
# 3. EARLIEST-DEADLINE FIRST (EDF) DEMAND-BOUND ANALYSIS
# =============================================================================

def demand_bound(C: np.ndarray, D: np.ndarray, T: np.ndarray, t: int) -> int:
    """Compute EDF Processor Demand Bound Function dbf(t) for interval length t.

    dbf(t) = sum_{i=1}^n max(0, floor((t - D_i) / T_i) + 1) * C_i
    """
    if t <= 0:
        return 0
    jobs = np.maximum(0, (t - D) // T + 1)
    return int(np.sum(jobs * C))


def synchronous_busy_period(
    C: np.ndarray,
    T: np.ndarray,
    cap: Optional[int] = None,
    max_iterations: int = 2000,
    max_bound: int = 1000000,
) -> Optional[int]:
    """Find the length of the synchronous busy period L on a preemptive uniprocessor.

    L is the smallest positive fixed point of:
        w^(0) = sum(C_i)
        w^(k+1) = sum(ceil(w^(k) / T_i) * C_i)

    Returns:
        The integer busy period length, or None if utilization >= 1.0 (unbounded).
    """
    C = np.asarray(C, dtype=np.int64)
    T = np.asarray(T, dtype=np.int64)
    u = float(np.sum(C / T))
    if u >= 1.0:
        return None

    limit = min(max_bound, cap) if cap is not None else max_bound
    w = int(np.sum(C))
    for _ in range(max_iterations):
        if w >= limit:
            return limit
        # ceil(w / T) = (w + T - 1) // T
        releases = (w + T - 1) // T
        next_w = int(np.sum(releases * C))
        if next_w == w:
            return w
        w = next_w

    return w


def edf_demand_bound_analysis(
    C: np.ndarray,
    D: np.ndarray,
    T: np.ndarray,
    max_iterations: int = 2000,
    max_points: int = 10000,
) -> Tuple[bool, int]:
    """Exact EDF schedulability test for constrained-deadline sporadic tasks (C_i <= D_i <= T_i).

    Uses Processor Demand Analysis (Demand Bound Function - DBF).
    Evaluates dbf(t) <= t for all absolute deadline points t <= L*, where L* is bounded
    by the synchronous busy period and the Baruah/Spuri testing bound.

    Returns:
        (is_schedulable, failing_point_or_horizon):
            - If schedulable: (True, horizon)
            - If unschedulable: (False, first_failing_time_point)
    """
    C = np.asarray(C, dtype=np.int64)
    D = np.asarray(D, dtype=np.int64)
    T = np.asarray(T, dtype=np.int64)

    if C.shape != D.shape or C.shape != T.shape:
        raise ValueError("C, D, and T must have the same shape")
    if np.any(C <= 0) or np.any(T <= 0) or np.any(D < C) or np.any(D > T):
        raise ValueError("Tasks must satisfy 0 < C <= D <= T")

    u = float(np.sum(C / T))
    if u > 1.0:
        return False, int(np.min(D))

    # Fast path: For implicit deadlines (D == T), U <= 1.0 is exact necessary and sufficient
    if np.array_equal(D, T):
        return True, int(np.max(T))

    # If U == 1.0 but some deadlines are strictly constrained (D < T), demand will exceed capacity
    if u >= 1.0:
        return False, int(np.min(D))

    # Baruah/Spuri formula bound: sum((T_i - D_i) * U_i) / (1 - U)
    u_i = C / T
    slack = max(1e-6, 1.0 - u)
    baruah_bound = int(np.floor(np.sum((T - D) * u_i) / slack))
    max_d = int(np.max(D))
    formula_bound = max(max_d, baruah_bound)

    # Compute testing horizon L* capped at formula_bound
    busy_period = synchronous_busy_period(C, T, cap=formula_bound, max_iterations=max_iterations)

    if busy_period is not None:
        horizon = min(busy_period, formula_bound)
    else:
        horizon = formula_bound

    # Collect all testing points: absolute deadlines d_{i,k} = k * T_i + D_i <= horizon
    points = set()
    for d, t_period in zip(D, T):
        p = int(d)
        t_val = int(t_period)
        while p <= horizon:
            points.add(p)
            p += t_val
            if len(points) > max_points:
                break

    sorted_points = sorted(points)
    for point in sorted_points:
        # Compute dbf(point)
        demand = demand_bound(C, D, T, point)
        if demand > point:
            return False, point

    return True, horizon


# =============================================================================
# 4. UNIFIED COMPARATIVE SCHEDULABILITY CHECK
# =============================================================================

def evaluate_all_schedulers(
    C: np.ndarray,
    D: np.ndarray,
    T: np.ndarray,
) -> Dict[str, bool]:
    """Run RM, DM, and EDF schedulability analyses on a single task set.

    Validates that the theoretical dominance hierarchy holds:
        Schedulable(RM) <= Schedulable(DM) <= Schedulable(EDF)
    """
    rm_sched, _ = rate_monotonic_rta(C, D, T)
    dm_sched, _ = deadline_monotonic_rta(C, D, T)
    edf_sched, _ = edf_demand_bound_analysis(C, D, T)

    # Verify theoretical dominance invariants
    if rm_sched and not dm_sched:
        raise AssertionError("Theoretical violation: RM schedulable but DM unschedulable!")
    if dm_sched and not edf_sched:
        raise AssertionError("Theoretical violation: DM schedulable but EDF unschedulable!")

    return {
        "RM": rm_sched,
        "DM": dm_sched,
        "EDF": edf_sched,
    }
