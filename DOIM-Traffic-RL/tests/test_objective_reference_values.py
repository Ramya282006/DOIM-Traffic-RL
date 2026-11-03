"""
DOIM-Traffic-RL: Independent Reference Validation Tests for Traffic Objectives and Fairness Metrics.
Validates hand-calculated analytical benchmarks for delay, queue, emissions, throughput,
and the three-tier equity metrics (approach, intersection, vehicle).
"""

import pytest
import numpy as np

from objectives.emissions import compute_step_co2
from objectives.throughput import ThroughputTracker
from objectives.waiting_time import compute_total_waiting
from objectives.queue import compute_total_queue
from objectives.fairness import CorridorFairnessCalculator


class TestObjectiveReferenceValues:
    """Hand-calculated analytical reference tests for primary traffic objectives."""

    def test_total_delay_reference(self):
        """Total waiting time matches exact sum of vehicle waiting times."""
        delays = [12.5, 3.5, 4.0, 0.0, 10.0]
        # Hand-calculated sum: 12.5 + 3.5 + 4.0 + 0.0 + 10.0 = 30.0
        assert compute_total_waiting(delays) == pytest.approx(30.0, abs=1e-6)

    def test_total_delay_empty(self):
        """Empty vehicle list yields exactly 0.0 total waiting time."""
        assert compute_total_waiting([]) == 0.0

    def test_queue_length_reference(self):
        """Queue length matches exact sum of halting counts."""
        halting_counts = [3, 5, 2, 0, 8]
        # Hand-calculated sum: 3 + 5 + 2 + 0 + 8 = 18
        assert compute_total_queue(halting_counts) == pytest.approx(18.0, abs=1e-6)

    def test_queue_empty(self):
        """Empty halting counts list yields exactly 0.0 queue."""
        assert compute_total_queue([]) == 0.0

    def test_co2_emission_reference_and_gram_conversion(self):
        """
        CO2 calculation matches analytical integral:
        Rate: [100.0, 200.0, 300.0] mg/s over dt=1.0s -> 600.0 mg -> 0.60 g.
        """
        rates = [100.0, 200.0, 300.0]
        mg_emitted = compute_step_co2(rates, step_length_sec=1.0)
        assert mg_emitted == pytest.approx(600.0, abs=1e-6)

        # Non-unit step length: dt=2.5s
        mg_emitted_2p5 = compute_step_co2(rates, step_length_sec=2.5)
        assert mg_emitted_2p5 == pytest.approx(600.0 * 2.5, abs=1e-6)

        # Gram conversion check
        grams_emitted = mg_emitted / 1000.0
        assert grams_emitted == pytest.approx(0.60, abs=1e-6)

    def test_co2_negative_rate_clamping(self):
        """Physically impossible negative emission rates are clipped to 0.0."""
        rates = [-50.0, 100.0, -20.0]
        assert compute_step_co2(rates, step_length_sec=1.0) == pytest.approx(100.0, abs=1e-6)

    def test_throughput_cumulative_and_hourly_rate(self):
        """
        Throughput tracker correctly accumulates completions and computes
        analytical vehicles/hour rate:
        e.g., 20 vehicles in 180 seconds = (20 / 180) * 3600 = 400.0 veh/hr.
        """
        tracker = ThroughputTracker()
        assert tracker.update(5) == 5
        assert tracker.update(10) == 10
        assert tracker.update(5) == 5
        assert tracker.cumulative_completed == 20

        # Hourly rate at t=180s
        rate = tracker.get_completion_rate_per_hour(180.0)
        assert rate == pytest.approx(400.0, abs=1e-6)

        # Zero simulation time guard
        assert tracker.get_completion_rate_per_hour(0.0) == 0.0


class TestFairnessReferenceValues:
    """Hand-calculated analytical tests for multi-tier fairness and equity metrics."""

    def test_jains_index_perfect_equality(self):
        """When all delays are identical, Jain's index must be exactly 1.0."""
        delays = [15.0, 15.0, 15.0, 15.0]
        jain = CorridorFairnessCalculator.compute_jain_index(delays)
        assert jain == pytest.approx(1.0, abs=1e-6)
        disparity = CorridorFairnessCalculator.compute_delay_disparity(delays)
        assert disparity == pytest.approx(0.0, abs=1e-6)

    def test_jains_index_extreme_inequality(self):
        """
        For K=4 where only 1 entity receives delay X and K-1 entities receive 0:
        J(x) = X^2 / (4 * X^2) = 1/4 = 0.25.
        Disparity = 1 - 0.25 = 0.75.
        """
        delays = [0.0, 0.0, 0.0, 10.0]
        jain = CorridorFairnessCalculator.compute_jain_index(delays)
        assert jain == pytest.approx(0.25, abs=1e-6)
        disparity = CorridorFairnessCalculator.compute_delay_disparity(delays)
        assert disparity == pytest.approx(0.75, abs=1e-6)

    def test_jains_index_hand_calculated_sequence(self):
        """
        Delays = [10, 20, 30, 40]
        sum(x) = 100 -> (sum(x))^2 = 10000
        sum(x^2) = 100 + 400 + 900 + 1600 = 3000
        K * sum(x^2) = 4 * 3000 = 12000
        J(x) = 10000 / 12000 = 5/6 = 0.833333...
        Disparity = 1 - 5/6 = 1/6 = 0.166667...
        """
        delays = [10.0, 20.0, 30.0, 40.0]
        jain = CorridorFairnessCalculator.compute_jain_index(delays)
        assert jain == pytest.approx(5.0 / 6.0, abs=1e-5)
        disparity = CorridorFairnessCalculator.compute_delay_disparity(delays)
        assert disparity == pytest.approx(1.0 / 6.0, abs=1e-5)

    def test_jains_index_scale_invariance(self):
        """J(c * x) == J(x) for any positive scalar c > 0."""
        delays = np.array([12.0, 4.0, 28.0, 19.0])
        j1 = CorridorFairnessCalculator.compute_jain_index(delays)
        j2 = CorridorFairnessCalculator.compute_jain_index(delays * 3.7)
        assert j1 == pytest.approx(j2, abs=1e-6)

    def test_gini_coefficient_reference_values(self):
        """
        Hand-calculated Gini coefficient for [10, 20, 30, 40]:
        n = 4, sum = 100
        sum(i * x_i) = 1(10) + 2(20) + 3(30) + 4(40) = 10 + 40 + 90 + 160 = 300
        2 * 300 - (4 + 1) * 100 = 600 - 500 = 100
        G = 100 / (4 * 100) = 0.25
        """
        delays = [10.0, 20.0, 30.0, 40.0]
        gini = CorridorFairnessCalculator.compute_gini_coefficient(delays)
        assert gini == pytest.approx(0.25, abs=1e-5)

    def test_gini_coefficient_perfect_equality_and_scale_invariance(self):
        """Gini is 0.0 for identical delays, and invariant under scalar multiplication."""
        equal_delays = [25.0, 25.0, 25.0, 25.0]
        assert CorridorFairnessCalculator.compute_gini_coefficient(equal_delays) == pytest.approx(0.0, abs=1e-6)

        varied = np.array([5.0, 15.0, 30.0, 60.0])
        g1 = CorridorFairnessCalculator.compute_gini_coefficient(varied)
        g2 = CorridorFairnessCalculator.compute_gini_coefficient(varied * 5.2)
        assert g1 == pytest.approx(g2, abs=1e-6)

    def test_worst_approach_delay_and_ratio(self):
        """Worst approach delay matches max, and ratio matches max / (min + eps)."""
        delays = [10.0, 50.0, 20.0, 5.0]
        worst = CorridorFairnessCalculator.compute_worst_approach_delay(delays)
        assert worst == pytest.approx(50.0, abs=1e-6)

        ratio = CorridorFairnessCalculator.compute_max_to_min_disparity_ratio(delays, eps=0.0)
        assert ratio == pytest.approx(50.0 / 5.0, abs=1e-6)

    def test_three_tier_fairness_structures(self):
        """Verify dictionary keys and metric completeness across all three tiers."""
        delays_4 = [10.0, 15.0, 20.0, 25.0]
        delays_9 = [5.0 * i for i in range(1, 10)]
        delays_veh = [0.0, 12.0, 45.0, 60.0, 120.0]

        app_metrics = CorridorFairnessCalculator.compute_approach_fairness(delays_4)
        assert app_metrics["tier"] == "approach_corridor"
        assert "jain_index" in app_metrics
        assert "delay_disparity" in app_metrics
        assert "worst_approach_delay_s" in app_metrics
        assert "gini" in app_metrics

        int_metrics = CorridorFairnessCalculator.compute_intersection_fairness(delays_9)
        assert int_metrics["tier"] == "intersection"
        assert "worst_intersection_delay_s" in int_metrics

        veh_metrics = CorridorFairnessCalculator.compute_vehicle_fairness(delays_veh)
        assert veh_metrics["tier"] == "vehicle"
        assert "p95_vehicle_delay_s" in veh_metrics
        assert "max_vehicle_delay_s" in veh_metrics

