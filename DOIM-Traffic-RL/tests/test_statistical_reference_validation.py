"""
DOIM-Traffic-RL: Independent Reference Validation Tests for Statistical Rigor and Effect Sizes.
Validates hand-calculated analytical benchmarks for Cliff's delta directions,
Holm-Bonferroni FWER step-down corrections, 2D hypervolume exact geometry,
and Student-t confidence intervals against closed-form solutions.
"""

import pytest
import numpy as np
from scipy import stats

from analysis.statistical_analyzer import StatisticalAnalyzer


class TestStatisticalReferenceValidation:
    """Analytical benchmark validation for the statistical methodology suite."""

    def test_cliffs_delta_direction_benefit_metric(self):
        """For benefit metrics (throughput, reward), x > y yields positive delta."""
        # Candidate treatment x is strictly higher than baseline y
        x = np.array([100.0, 110.0, 120.0])
        y = np.array([50.0, 60.0, 70.0])

        delta = StatisticalAnalyzer.cliffs_delta(x, y, higher_is_better=True)
        assert delta == pytest.approx(1.0, abs=1e-6)

        # Baseline y compared to x yields negative delta
        delta_rev = StatisticalAnalyzer.cliffs_delta(y, x, higher_is_better=True)
        assert delta_rev == pytest.approx(-1.0, abs=1e-6)

        # compute_metric_effect_size for throughput
        eff = StatisticalAnalyzer.compute_metric_effect_size(x, y, "throughput")
        assert eff == pytest.approx(1.0, abs=1e-6)

    def test_cliffs_delta_direction_cost_metric(self):
        """
        For cost metrics (delay, queue, emissions), lower values indicate superior performance.
        Candidate x with lower delay than baseline y must yield positive effect size.
        """
        # Candidate x achieves low delay: [20, 25, 30]
        # Baseline y achieves high delay: [50, 60, 70]
        x = np.array([20.0, 25.0, 30.0])
        y = np.array([50.0, 60.0, 70.0])

        # Raw Cliff's delta without adjustment is negative (-1.0) because x < y
        raw_delta = StatisticalAnalyzer.cliffs_delta(x, y, higher_is_better=True)
        assert raw_delta == pytest.approx(-1.0, abs=1e-6)

        # Adjusted Cliff's delta with higher_is_better=False is positive (+1.0)
        adj_delta = StatisticalAnalyzer.cliffs_delta(x, y, higher_is_better=False)
        assert adj_delta == pytest.approx(1.0, abs=1e-6)

        # Automated compute_metric_effect_size recognizes cost metrics
        for cost_metric in ["delay", "total_delay", "queue", "queue_length", "co2", "emissions", "fairness_disparity"]:
            eff = StatisticalAnalyzer.compute_metric_effect_size(x, y, cost_metric)
            assert eff == pytest.approx(1.0, abs=1e-6), f"Failed for cost metric {cost_metric}"

    def test_cliffs_delta_ties_and_identical_distributions(self):
        """Identical distributions yield exactly 0.0 effect size."""
        x = np.array([10.0, 20.0, 30.0, 40.0])
        assert StatisticalAnalyzer.cliffs_delta(x, x) == pytest.approx(0.0, abs=1e-6)

        # Partial overlap:
        # x = [1, 3], y = [2, 4]
        # comparisons: (1>2:0, 1<2:1), (1>4:0, 1<4:1), (3>2:1, 3<2:0), (3>4:0, 3<4:1)
        # greater = 1, less = 3 -> delta = (1 - 3)/4 = -0.5
        x2 = np.array([1.0, 3.0])
        y2 = np.array([2.0, 4.0])
        assert StatisticalAnalyzer.cliffs_delta(x2, y2) == pytest.approx(-0.5, abs=1e-6)

    def test_holm_bonferroni_step_down_hand_calculated(self):
        """
        Validates step-down threshold logic against hand-calculated analytical benchmark:
        m = 3 hypotheses with p = [0.01, 0.04, 0.03] at alpha = 0.05.
        Rank 1: p=0.01, threshold = 0.05 / 3 = 0.01667 -> 0.01 <= 0.01667 -> SIGNIFICANT.
        Rank 2: p=0.03, threshold = 0.05 / 2 = 0.02500 -> 0.03 > 0.02500 -> NOT SIGNIFICANT.
        Rank 3: p=0.04, threshold = 0.05 / 1 = 0.05000 -> Step-down terminates -> NOT SIGNIFICANT.
        """
        p_vals = [0.01, 0.04, 0.03]
        results = StatisticalAnalyzer.holm_bonferroni_correction(p_vals, alpha=0.05)

        assert len(results) == 3

        # Index 0 was 0.01 -> Rank 1 -> Significant
        assert results[0]["rank"] == 1
        assert results[0]["significant"] is True
        assert results[0]["threshold"] == pytest.approx(0.05 / 3.0, abs=1e-5)

        # Index 1 was 0.04 -> Rank 3 -> Not significant
        assert results[1]["rank"] == 3
        assert results[1]["significant"] is False

        # Index 2 was 0.03 -> Rank 2 -> Not significant
        assert results[2]["rank"] == 2
        assert results[2]["significant"] is False
        assert results[2]["threshold"] == pytest.approx(0.05 / 2.0, abs=1e-5)

    def test_holm_bonferroni_empty_and_all_rejected(self):
        """Edge cases: empty list, and all p-values highly significant."""
        assert StatisticalAnalyzer.holm_bonferroni_correction([]) == []

        p_strong = [0.0001, 0.0005, 0.0010]
        results = StatisticalAnalyzer.holm_bonferroni_correction(p_strong, alpha=0.05)
        for r in results:
            assert r["significant"] is True

    def test_student_t_confidence_interval_analytical(self):
        """
        Hand-calculated Student-t confidence interval:
        Data: [10.0, 12.0, 14.0], n=3, mean=12.0, std=2.0, se = 2 / sqrt(3) = 1.1547.
        df = 2. For 95% CI: t_crit = stats.t.ppf(0.975, df=2) ≈ 4.30265.
        Margin h = 1.1547 * 4.30265 ≈ 4.96827.
        Bounds = [12.0 - 4.96827, 12.0 + 4.96827] ≈ [7.03173, 16.96827].
        """
        data = np.array([10.0, 12.0, 14.0])
        mean, ci_low, ci_high = StatisticalAnalyzer.compute_confidence_interval(data, confidence=0.95)

        assert mean == pytest.approx(12.0, abs=1e-5)
        expected_t = stats.t.ppf(0.975, df=2)
        expected_h = (2.0 / np.sqrt(3)) * expected_t

        assert ci_low == pytest.approx(12.0 - expected_h, abs=1e-5)
        assert ci_high == pytest.approx(12.0 + expected_h, abs=1e-5)

    def test_hypervolume_2d_exact_geometry(self):
        """
        Exact geometry test:
        Reference point: (1.0, 1.0).
        Point 1: (0.2, 0.6)
        Point 2: (0.5, 0.3)
        Dominated union area:
          Area([0.2, 1.0] x [0.6, 1.0]) = 0.8 * 0.4 = 0.32
          Area([0.5, 1.0] x [0.3, 1.0]) = 0.5 * 0.7 = 0.35
          Overlap([0.5, 1.0] x [0.6, 1.0]) = 0.5 * 0.4 = 0.20
          Union Area = 0.32 + 0.35 - 0.20 = 0.47
        """
        points = np.array([
            [0.2, 0.6],
            [0.5, 0.3],
            [0.8, 0.8],  # Dominated by both, should not affect Pareto front
        ])
        hv = StatisticalAnalyzer.compute_hypervolume_2d(points, ref_point=(1.0, 1.0))
        assert hv == pytest.approx(0.47, abs=1e-5)

    def test_hypervolume_sensitivity_scaling(self):
        """Hypervolume increases monotonically as reference point scale expands."""
        points = np.array([[0.2, 0.4], [0.3, 0.3]])
        sens = StatisticalAnalyzer.compute_hypervolume_sensitivity(points, reference_scales=[1.0, 1.2, 1.5])

        assert sens[1.0] < sens[1.2] < sens[1.5]

