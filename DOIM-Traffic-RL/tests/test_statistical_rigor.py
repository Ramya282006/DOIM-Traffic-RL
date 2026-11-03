"""
DOIM-Traffic-RL: Statistical Rigor & Multi-Objective Hypervolume Suite (Phase I)
Verifies:
  1. Multi-objective non-dominated Pareto front computation.
  2. 2D Hypervolume calculation with exact hand-calculated analytical benchmark.
  3. Cliff's delta effect size calculation.
  4. Holm-Bonferroni step-down correction for multiple comparisons.
"""

import numpy as np
import pytest

from analysis.statistical_analyzer import StatisticalAnalyzer


def test_pareto_front_computation():
    """
    In minimization, point A dominates point B if A_i <= B_i for all i and A_j < B_j for some j.
    """
    # Points:
    # P1: (0.2, 0.8) - non-dominated
    # P2: (0.5, 0.5) - non-dominated
    # P3: (0.8, 0.2) - non-dominated
    # P4: (0.6, 0.7) - dominated by P2 (0.5 <= 0.6 and 0.5 <= 0.7)
    # P5: (0.9, 0.9) - dominated by all
    points = np.array([
        [0.2, 0.8],
        [0.5, 0.5],
        [0.8, 0.2],
        [0.6, 0.7],
        [0.9, 0.9],
    ])
    
    pareto = StatisticalAnalyzer.compute_pareto_front(points)
    assert len(pareto) == 3
    # P4 and P5 must be excluded
    for p in pareto:
        assert not (np.allclose(p, [0.6, 0.7]) or np.allclose(p, [0.9, 0.9]))


def test_hypervolume_2d_analytical_benchmark():
    """
    Hand-calculated analytical verification of 2D Hypervolume:
    Reference point R = (1.0, 1.0).
    Non-dominated points:
      P1 = (0.3, 0.6)
      P2 = (0.7, 0.2)
      
    Geometry of dominated region bounded by R=(1.0, 1.0):
      From x = 0.3 to 0.7: height = 1.0 - 0.6 = 0.4. Area = (0.7 - 0.3) * 0.4 = 0.4 * 0.4 = 0.16.
      From x = 0.7 to 1.0: height = 1.0 - 0.2 = 0.8. Area = (1.0 - 0.7) * 0.8 = 0.3 * 0.8 = 0.24.
      Total Hypervolume = 0.16 + 0.24 = 0.40.
    """
    points = np.array([
        [0.3, 0.6],
        [0.7, 0.2],
    ])
    hv = StatisticalAnalyzer.compute_hypervolume_2d(points, ref_point=(1.0, 1.0))
    assert pytest.approx(hv, rel=1e-5) == 0.40


def test_cliffs_delta_effect_size():
    """Verifies Cliff's delta: +1.0 for complete stochastic dominance, 0.0 for equal distributions."""
    x = np.array([10.0, 11.0, 12.0])
    y = np.array([1.0, 2.0, 3.0])
    # x strictly greater than y
    assert pytest.approx(StatisticalAnalyzer.cliffs_delta(x, y), abs=1e-5) == 1.0
    assert pytest.approx(StatisticalAnalyzer.cliffs_delta(y, x), abs=1e-5) == -1.0
    
    # Identical distributions
    assert pytest.approx(StatisticalAnalyzer.cliffs_delta(x, x), abs=1e-5) == 0.0


def test_holm_bonferroni_step_down():
    """Verifies Holm-Bonferroni correction with known p-values."""
    p_values = [0.005, 0.015, 0.040, 0.150]
    results = StatisticalAnalyzer.holm_bonferroni_correction(p_values, alpha=0.05)
    
    assert len(results) == 4
    # m = 4:
    # rank 1: p=0.005, threshold = 0.05 / 4 = 0.0125 -> Significant!
    assert results[0]["significant"] is True
    # rank 2: p=0.015, threshold = 0.05 / 3 = 0.01667 -> Significant!
    assert results[1]["significant"] is True
    # rank 3: p=0.040, threshold = 0.05 / 2 = 0.025 -> Not significant!
    assert results[2]["significant"] is False
    # rank 4: p=0.150, threshold = 0.05 / 1 = 0.050 -> Not significant!
    assert results[3]["significant"] is False
