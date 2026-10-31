"""
DOIM-Traffic-RL: Objective Metric and Fairness Validation Tests
Verifies mathematical definitions, units, optimization directions, bounds,
and population semantics of the 5 traffic objectives.
"""

import pytest
import numpy as np

from objectives.objective_normalizer import ObjectiveNormalizer
from objectives.fairness import CorridorFairnessCalculator
from objectives.doim_reward import DOIMReward


def test_objective_normalizer_calibration_and_inversion():
    """
    Verifies that all 5 objectives are bounded in [0, 1] as urgency values,
    and that throughput is inverted so that higher throughput yields lower urgency.
    """
    normalizer = ObjectiveNormalizer()

    # Best-case traffic: 0 waiting, 0 queue, 0 emissions, maximum throughput (20), 0 disparity
    best_raw = np.array([0.0, 0.0, 0.0, 20.0, 0.0], dtype=np.float32)
    best_urgency = normalizer.normalize(best_raw)
    np.testing.assert_allclose(best_urgency, [0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-5)

    # Worst-case traffic: 600 waiting, 20 queue, 120000 emissions, 0 throughput, 1.0 disparity
    worst_raw = np.array([600.0, 20.0, 120000.0, 0.0, 1.0], dtype=np.float32)
    worst_urgency = normalizer.normalize(worst_raw)
    np.testing.assert_allclose(worst_urgency, [1.0, 1.0, 1.0, 1.0, 1.0], atol=1e-5)

    # Throughput monotonicity test: increasing throughput must decrease urgency
    low_tp = normalizer.normalize(np.array([100.0, 5.0, 10000.0, 2.0, 0.2]))
    high_tp = normalizer.normalize(np.array([100.0, 5.0, 10000.0, 15.0, 0.2]))
    assert high_tp[3] < low_tp[3], "Higher throughput must yield lower urgency penalty!"


def test_jain_fairness_index_properties():
    """
    Verifies Jain's Fairness Index mathematical properties:
      - J = 1.0 under identical delays
      - J = 1/K under single-vehicle monopoly
      - Disparity = 1 - J in [0, 1]
    """
    calc = CorridorFairnessCalculator()

    # Perfectly equitable delays across 5 vehicles
    equal_delays = [30.0, 30.0, 30.0, 30.0, 30.0]
    jain_equal = calc.compute_jain_index(equal_delays)
    assert jain_equal == pytest.approx(1.0, rel=1e-5)
    assert calc.compute_delay_disparity(equal_delays) == pytest.approx(0.0, rel=1e-5)

    # Extreme starvation: 1 vehicle starved with 100s, 4 vehicles with 0s
    extreme_delays = [100.0, 0.0, 0.0, 0.0, 0.0]
    jain_extreme = calc.compute_jain_index(extreme_delays)
    assert jain_extreme == pytest.approx(0.2, rel=1e-5) # 1 / 5 = 0.2
    assert calc.compute_delay_disparity(extreme_delays) == pytest.approx(0.8, rel=1e-5)

    # Empty list handling
    assert calc.compute_jain_index([]) == 1.0
    assert calc.compute_delay_disparity([]) == 0.0


def test_distinguish_vehicle_and_intersection_fairness():
    """
    Verifies that vehicle-level disparity and intersection-level disparity
    are conceptually and quantitatively distinct.
    """
    calc = CorridorFairnessCalculator()

    # Scenario: 2 intersections have identical average delay (50s each),
    # but intersection 1 has high internal disparity [10s, 90s] while intersection 2 has [50s, 50s].
    int_delays = [50.0, 50.0]
    int_disparity = calc.compute_delay_disparity(int_delays)
    assert int_disparity == pytest.approx(0.0, abs=1e-5) # Intersections appear perfectly fair!

    # But vehicle-level delays [10, 90, 50, 50] reveal unfair individual treatment:
    all_veh_delays = [10.0, 90.0, 50.0, 50.0]
    veh_disparity = calc.compute_vehicle_fairness_disparity(all_veh_delays)
    assert veh_disparity > 0.1, "Vehicle disparity must detect individual variance hidden by spatial aggregation!"


def test_reward_engine_sign_alignment():
    """
    Verifies that DOIM reward is strictly monotonic with respect to all 5 objectives:
    Decreasing waiting time, queue, emissions, or disparity increases reward.
    Increasing throughput increases reward.
    """
    reward_engine = DOIMReward()
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    good_urgency = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    bad_urgency = np.array([0.8, 0.8, 0.8, 0.8, 0.8], dtype=np.float32)

    r_good = reward_engine.compute(good_urgency, weights)
    r_bad = reward_engine.compute(bad_urgency, weights)

    assert r_good > r_bad, "Desirable traffic conditions must yield higher reward than congested conditions"
    assert r_good <= 0.0 and r_bad <= 0.0, "Urgency-weighted reward must remain non-positive"
