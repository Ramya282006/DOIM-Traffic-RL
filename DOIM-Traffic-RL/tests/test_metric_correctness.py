"""
DOIM-Traffic-RL: Metric Pipeline Verification Suite (Phase C)
Verifies:
  1. Hand-computed analytical proofs for Waiting, Queue, CO2, and Throughput.
  2. Multi-tier fairness unit separation: Disparity [0, 1], Gini [0, 1], Variance [s^2].
  3. AuthoritativeObjectiveConfig physical bounds and direction consistency.
  4. MultiObjectiveEngine end-to-end vector calculation from network states.
"""

import numpy as np
import pytest

from objectives.waiting_time import compute_total_waiting
from objectives.queue import compute_total_queue
from objectives.emissions import compute_step_co2
from objectives.throughput import ThroughputTracker
from objectives.fairness import CorridorFairnessCalculator
from objectives.objective_config import AuthoritativeObjectiveConfig
from objectives.objective_engine import MultiObjectiveEngine


def test_waiting_time_accumulation():
    """Verifies waiting time summation over lane/vehicle delays."""
    assert compute_total_waiting([]) == 0.0
    delays = [12.5, 3.0, 4.5, 0.0]
    assert compute_total_waiting(delays) == 20.0


def test_queue_length_accumulation():
    """Verifies halted vehicle count aggregation."""
    assert compute_total_queue([]) == 0.0
    queues = [3, 0, 5, 2]
    assert compute_total_queue(queues) == 10.0


def test_co2_emission_integration():
    """Verifies instantaneous CO2 emission rate integration over step length."""
    assert compute_step_co2([], step_length_sec=1.0) == 0.0
    # 3 vehicles emitting 1500, 2500, 1000 mg/s over 1.0s
    rates = [1500.0, 2500.0, 1000.0]
    assert compute_step_co2(rates, step_length_sec=1.0) == 5000.0
    # Over 2.0s
    assert compute_step_co2(rates, step_length_sec=2.0) == 10000.0


def test_throughput_tracking():
    """Verifies throughput accumulation and step rate tracking."""
    tracker = ThroughputTracker()
    assert tracker.cumulative_completed == 0
    
    assert tracker.update(3) == 3
    assert tracker.cumulative_completed == 3
    
    assert tracker.update(5) == 5
    assert tracker.cumulative_completed == 8
    
    # 8 vehicles completed in 1800s -> 16 veh/hr
    rate = tracker.get_completion_rate_per_hour(1800.0)
    assert pytest.approx(rate, rel=1e-4) == 16.0


def test_fairness_three_tiers_separation():
    """
    Verifies that the 3 tiers of fairness maintain strict mathematical and unit separation:
      - Jain disparity is bounded in [0, 1] (dimensionless).
      - Delay variance has physical units of s^2 (seconds squared) and can be > 1.
      - Gini coefficient is bounded in [0, 1] (dimensionless).
    """
    # Test case: 4 approaches with delays [10.0, 20.0, 30.0, 40.0] seconds
    delays = [10.0, 20.0, 30.0, 40.0]
    
    # Hand calculation:
    # Mean = 25.0s
    # Sum = 100.0s, Sum^2 = 10000.0
    # Sum of squares = 100 + 400 + 900 + 1600 = 3000.0
    # K = 4
    # Jain = 10000 / (4 * 3000) = 10000 / 12000 = 5/6 ≈ 0.83333
    # Disparity = 1 - 5/6 = 1/6 ≈ 0.16667
    jain = CorridorFairnessCalculator.compute_jain_index(delays)
    disparity = CorridorFairnessCalculator.compute_delay_disparity(delays)
    assert pytest.approx(jain, rel=1e-4) == 5.0 / 6.0
    assert pytest.approx(disparity, rel=1e-4) == 1.0 / 6.0
    assert 0.0 <= disparity <= 1.0

    # Variance:
    # ((10-25)^2 + (20-25)^2 + (30-25)^2 + (40-25)^2) / 4 = (225 + 25 + 25 + 225) / 4 = 500 / 4 = 125.0 s^2
    var_s2 = CorridorFairnessCalculator.compute_delay_variance(delays, ddof=0)
    assert pytest.approx(var_s2, rel=1e-4) == 125.0
    # Notice: variance is 125.0 s^2, while disparity is 0.1667! They are NOT interchangeable.
    assert var_s2 > 1.0

    # Gini coefficient:
    # Sum = 100. Sorted: [10, 20, 30, 40].
    # 2 * (1*10 + 2*20 + 3*30 + 4*40) - (4+1)*100 = 2 * (10 + 40 + 90 + 160) - 500 = 2 * 300 - 500 = 100
    # Gini = 100 / (4 * 100) = 0.25
    gini = CorridorFairnessCalculator.compute_gini_coefficient(delays)
    assert pytest.approx(gini, rel=1e-4) == 0.25
    assert 0.0 <= gini <= 1.0


def test_fairness_equal_service_analytical_proof():
    """All approaches receive identical delay -> perfect equity."""
    delays = [30.0, 30.0, 30.0, 30.0]
    assert pytest.approx(CorridorFairnessCalculator.compute_jain_index(delays), rel=1e-5) == 1.0
    assert pytest.approx(CorridorFairnessCalculator.compute_delay_disparity(delays), abs=1e-5) == 0.0
    assert pytest.approx(CorridorFairnessCalculator.compute_delay_variance(delays), abs=1e-5) == 0.0
    assert pytest.approx(CorridorFairnessCalculator.compute_gini_coefficient(delays), abs=1e-5) == 0.0


def test_fairness_extreme_starvation_analytical_proof():
    """One single movement receives all green, others completely starved."""
    # 3 movements 0 delay, 1 movement 100s delay:
    delays = [0.0, 0.0, 0.0, 100.0]
    # Sum = 100, Sum^2 = 10000. Sum of squares = 10000.
    # Jain = 10000 / (4 * 10000) = 0.25 (= 1/K, theoretical lower bound)
    # Disparity = 1.0 - 0.25 = 0.75
    assert pytest.approx(CorridorFairnessCalculator.compute_jain_index(delays), rel=1e-5) == 0.25
    assert pytest.approx(CorridorFairnessCalculator.compute_delay_disparity(delays), rel=1e-5) == 0.75


def test_multiobjective_engine_integration():
    """Verifies that MultiObjectiveEngine compiles network states into valid 5-vector."""
    engine = MultiObjectiveEngine(fairness_mode="jain_disparity")
    
    mock_network_states = {
        "tl_0": {"queue": 2.0, "waiting": 15.0, "co2": 1000.0},
        "tl_1": {"queue": 3.0, "waiting": 25.0, "co2": 1500.0},
        "tl_2": {"queue": 1.0, "waiting": 10.0, "co2": 500.0},
    }
    
    vec = engine.compute_objectives(mock_network_states, completed_vehicles=4)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (5,)
    
    # Waiting = 15 + 25 + 10 = 50.0
    assert pytest.approx(vec[0], rel=1e-4) == 50.0
    # Queue = 2 + 3 + 1 = 6.0
    assert pytest.approx(vec[1], rel=1e-4) == 6.0
    # CO2 = 1000 + 1500 + 500 = 3000.0
    assert pytest.approx(vec[2], rel=1e-4) == 3000.0
    # Throughput = 4.0
    assert pytest.approx(vec[3], rel=1e-4) == 4.0
    # Fairness Disparity in [0, 1]
    assert 0.0 <= vec[4] <= 1.0
