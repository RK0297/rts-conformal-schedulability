"""Comprehensive unit tests for RM (Rate-Monotonic), DM (Deadline-Monotonic), and EDF schedulability analyses."""
import numpy as np
import pytest

from verification.schedulability import (
    deadline_monotonic_rta,
    demand_bound,
    edf_demand_bound_analysis,
    evaluate_all_schedulers,
    hyperbolic_bound_test,
    liu_layland_bound,
    rate_monotonic_rta,
    rate_monotonic_sufficient_test,
    synchronous_busy_period,
)


def test_liu_layland_and_hyperbolic_bounds():
    """Verify classical Liu-Layland and Hyperbolic bounds."""
    # n=1: U = 1.0
    assert abs(liu_layland_bound(1) - 1.0) < 1e-6
    # n=2: 2 * (sqrt(2) - 1) approx 0.8284
    assert abs(liu_layland_bound(2) - 0.828427) < 1e-4
    # n -> inf: ln(2) approx 0.6931
    assert abs(liu_layland_bound(100) - 0.6955) < 1e-3

    # Task set with U = 0.5 <= 0.828
    c = np.array([1, 1])
    t = np.array([4, 4])
    assert rate_monotonic_sufficient_test(c, t)
    assert hyperbolic_bound_test(c, t)


def test_rm_and_dm_priority_order_and_response_times():
    """Verify RM sorts by T and DM sorts by D."""
    c = np.array([1, 1])
    d = np.array([2, 4])
    t = np.array([5, 3])
    # Under RM: task 1 has T=3 (higher priority, R=1 <= 4), task 0 has T=5 (lower priority, R=2 <= 2)
    rm_ok, rm_r = rate_monotonic_rta(c, d, t)
    # Under DM: task 0 has D=2 (higher priority, R=1 <= 2), task 1 has D=4 (lower priority, R=2 <= 4)
    dm_ok, dm_r = deadline_monotonic_rta(c, d, t)
    assert rm_ok and dm_ok
    assert np.array_equal(rm_r, np.array([2, 1]))
    assert np.array_equal(dm_r, np.array([1, 2]))


def test_dm_strictly_dominates_rm_for_constrained_deadlines():
    """A classic task set where DM succeeds but RM fails.

    Example:
      tau_1: C=2, D=3, T=10  (short deadline, long period)
      tau_2: C=3, D=7, T=4   (longer deadline, short period)

    Under RM:
      tau_2 has priority (T_2=4 < T_1=10) -> R_2 = C_2 = 3 <= 7.
      tau_1 has lower priority -> R_1:
        R_1^(0) = 2
        R_1^(1) = 2 + ceil(2/4)*3 = 5
        R_1^(2) = 2 + ceil(5/4)*3 = 2 + 2*3 = 8 > D_1=3! Deadline miss!

    Under DM:
      tau_1 has priority (D_1=3 < D_2=7) -> R_1 = C_1 = 2 <= 3 (OK).
      tau_2 has lower priority -> R_2:
        R_2^(0) = 3
        R_2^(1) = 3 + ceil(3/10)*2 = 5
        R_2^(2) = 3 + ceil(5/10)*2 = 5 <= D_2=7 (OK!).
    """
    c = np.array([2, 3])
    d = np.array([3, 7])
    t = np.array([10, 4])

    rm_ok, rm_r = rate_monotonic_rta(c, d, t)
    dm_ok, dm_r = deadline_monotonic_rta(c, d, t)

    assert not rm_ok, "RM should fail this task set"
    assert dm_ok, "DM should succeed on this task set"
    assert rm_r[0] > d[0]
    assert dm_r[0] <= d[0] and dm_r[1] <= d[1]


def test_edf_strictly_dominates_dm():
    """A task set where EDF succeeds but fixed-priority DM fails.

    Example:
      tau_1: C=2, D=4, T=5
      tau_2: C=3, D=5, T=6
    Total utilization U = 2/5 + 3/6 = 0.4 + 0.5 = 0.90 <= 1.0.

    Under DM:
      tau_1 has D=4 (high prio): R_1 = 2 <= 4.
      tau_2 has D=5 (low prio): R_2 = 3 + ceil(3/5)*2 = 5.
      Next: R_2 = 3 + ceil(5/5)*2 = 5 <= 5.
      Wait, let's pick one where DM fails:
      tau_1: C=3, D=5, T=6  (U1 = 0.5)
      tau_2: C=4, D=8, T=9  (U2 = 0.444) Total U = 0.944
      Under DM: tau_1 has D=5, R_1 = 3 <= 5.
      tau_2 has D=8: R_2^(0)=4, R_2^(1)=4 + ceil(4/6)*3 = 7.
      R_2^(2) = 4 + ceil(7/6)*3 = 4 + 6 = 10 > 8! Fails DM!

    Under EDF:
      dbf(t) <= t holds for all t <= L*. Schedulable!
    """
    c = np.array([3, 4])
    d = np.array([5, 8])
    t = np.array([6, 9])

    dm_ok, dm_r = deadline_monotonic_rta(c, d, t)
    edf_ok, horizon = edf_demand_bound_analysis(c, d, t)

    assert not dm_ok, "DM must fail this task set due to interference"
    assert edf_ok, "EDF must successfully schedule this feasible task set"
    assert horizon > 0


def test_edf_catches_deadline_overflow_at_horizon():
    """Verify that the busy-period bound does not terminate prematurely.

    Task set:
      tau_1: C=8, D=9, T=10
      tau_2: C=4, D=14, T=25
    Total utilization: U = 8/10 + 4/25 = 0.96.
    At t = 19:
      tau_1 has 2 jobs (deadlines 9, 19) -> demand = 16
      tau_2 has 1 job (deadline 14) -> demand = 4
      Total demand = 20 > 19!
    Must be detected as unschedulable at t=19!
    """
    c = np.array([8, 4])
    d = np.array([9, 14])
    t = np.array([10, 25])

    ok, point = edf_demand_bound_analysis(c, d, t)
    assert not ok, "EDF must reject this task set because dbf(19) = 20 > 19"
    assert point == 19, f"Expected first failing point to be 19, got {point}"


def test_edf_implicit_deadlines_up_to_full_utilization():
    """Under EDF with D == T, schedulability is exact iff U <= 1.0."""
    # U = 1.0 exact
    c1 = np.array([1, 2, 2])
    d1 = np.array([5, 5, 5])
    t1 = np.array([5, 5, 5])
    ok1, _ = edf_demand_bound_analysis(c1, d1, t1)
    assert ok1, "EDF should accept 100% utilization when D == T"

    # U = 1.2 overload
    c2 = np.array([2, 2, 2])
    d2 = np.array([5, 5, 5])
    t2 = np.array([5, 5, 5])
    ok2, _ = edf_demand_bound_analysis(c2, d2, t2)
    assert not ok2, "EDF must reject systems with U > 1.0"


def test_synchronous_busy_period():
    """Verify synchronous busy period computation."""
    c = np.array([1, 2])
    t = np.array([4, 6])
    # w^(0) = 3
    # w^(1) = ceil(3/4)*1 + ceil(3/6)*2 = 1 + 2 = 3. Fixed point = 3.
    assert synchronous_busy_period(c, t) == 3

    # Overload case U >= 1.0 returns None
    c_over = np.array([3, 3])
    t_over = np.array([4, 4])
    assert synchronous_busy_period(c_over, t_over) is None


def test_evaluate_all_schedulers_dominance():
    """Verify RM <= DM <= EDF invariant across various task sets."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        n = rng.integers(2, 6)
        u_target = rng.uniform(0.1, 0.9)
        # Random periods in [10, 200]
        t = rng.integers(10, 200, size=n)
        c = np.maximum(1, (u_target / n * t).astype(np.int64))
        # Ensure C <= D <= T
        d = np.array([rng.integers(c[i], t[i] + 1) for i in range(n)])

        res = evaluate_all_schedulers(c, d, t)
        # Dominance hierarchy:
        if res["RM"]:
            assert res["DM"], "Violation: RM schedulable but DM unschedulable"
        if res["DM"]:
            assert res["EDF"], "Violation: DM schedulable but EDF unschedulable"


def test_input_validation():
    """Verify that invalid inputs raise descriptive ValueErrors."""
    c = np.array([5, 2])
    d = np.array([4, 5])  # D[0] < C[0] is invalid!
    t = np.array([10, 10])

    with pytest.raises(ValueError, match="satisfy C > 0, T > 0, and D >= C"):
        rate_monotonic_rta(c, d, t)

    with pytest.raises(ValueError, match="satisfy 0 < C <= D <= T"):
        edf_demand_bound_analysis(c, d, t)
