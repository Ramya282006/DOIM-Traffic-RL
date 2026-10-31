"""
DOIM-Traffic-RL: Unit tests for Fairness Metric Mathematical Definitions (P12-07).
Verifies:
1. Primary Metric: Delay Disparity (1 - Jain's Index), dimensionless in [0, 1].
2. Secondary Metric: Delay Variance, in seconds squared (s^2).
3. Exact manual calculation matches.
4. Metric separation: StatisticalAnalyzer routes values <= 1.0 to disparity and > 1.0 to variance,
   preventing accidental averaging of heterogeneous units.
"""

import pytest
import numpy as np

from objectives.fairness import CorridorFairnessCalculator
from analysis.statistical_analyzer import StatisticalAnalyzer


def test_manually_calculable_fairness_examples():
    """
    Tests exact manual calculations of Jain's Index, Delay Disparity, and Delay Variance.
    
    Case A: Identical Delays [20, 20, 20, 20]
      - Sum = 80, Sum_Sq = 4 * 400 = 1600
      - Jain = 80^2 / (4 * 1600) = 6400 / 6400 = 1.0
      - Disparity = 1.0 - 1.0 = 0.0 (dimensionless)
      - Variance = 0.0 s^2
    """
    delays_a = [20.0, 20.0, 20.0, 20.0]
    calc = CorridorFairnessCalculator()

    assert calc.compute_jain_index(delays_a) == pytest.approx(1.0, rel=1e-6)
    assert calc.compute_delay_disparity(delays_a) == pytest.approx(0.0, rel=1e-6)
    assert calc.compute_delay_variance(delays_a) == pytest.approx(0.0, rel=1e-6)

    """
    Case B: Two Corridors [10, 30]
      - Sum = 40, Sum_Sq = 100 + 900 = 1000
      - K = 2
      - Jain = 40^2 / (2 * 1000) = 1600 / 2000 = 0.80
      - Disparity = 1.0 - 0.80 = 0.20 (dimensionless)
      - Mean = 20.0
      - Variance (ddof=0) = ((10-20)^2 + (30-20)^2)/2 = (100 + 100)/2 = 100.0 s^2
    """
    delays_b = [10.0, 30.0]
    assert calc.compute_jain_index(delays_b) == pytest.approx(0.80, rel=1e-6)
    assert calc.compute_delay_disparity(delays_b) == pytest.approx(0.20, rel=1e-6)
    assert calc.compute_delay_variance(delays_b, ddof=0) == pytest.approx(100.0, rel=1e-6)

    """
    Case C: Extreme Asymmetry [0, 0, 0, 100]
      - Sum = 100, Sum_Sq = 10000
      - K = 4
      - Jain = 100^2 / (4 * 10000) = 10000 / 40000 = 0.25 (1/K)
      - Disparity = 1.0 - 0.25 = 0.75 (dimensionless)
    """
    delays_c = [0.0, 0.0, 0.0, 100.0]
    assert calc.compute_jain_index(delays_c) == pytest.approx(0.25, rel=1e-6)
    assert calc.compute_delay_disparity(delays_c) == pytest.approx(0.75, rel=1e-6)


def test_metric_heterogeneity_isolation():
    """
    Verifies that StatisticalAnalyzer separates Delay Disparity (<= 1.0)
    from historical Delay Variance (> 1.0) without averaging them.
    """
    analyzer = StatisticalAnalyzer()
    # Mock row with modern disparity
    row_disparity = {"average_fairness": 0.25, "scenario": "test", "algorithm": "IA_MOPPO", "train_seed": 42}
    # Mock row with historical variance
    row_variance = {"average_fairness": 387.94, "scenario": "test", "algorithm": "Fixed_Time", "train_seed": 42}

    import pandas as pd
    df = pd.DataFrame([
        {
            "scenario": "test",
            "algorithm": "IA_MOPPO",
            "train_seed": 42,
            "episode_id": 0,
            "fairness_disparity": 0.25,
            "fairness_variance": np.nan,
            "average_waiting": 50.0,
        },
        {
            "scenario": "test",
            "algorithm": "Fixed_Time",
            "train_seed": 42,
            "episode_id": 0,
            "fairness_disparity": np.nan,
            "fairness_variance": 387.94,
            "average_waiting": 55.0,
        },
    ])

    agg = analyzer.aggregate_by_scenario_and_algorithm(df)
    ia_row = agg[agg["algorithm"] == "IA_MOPPO"].iloc[0]
    ft_row = agg[agg["algorithm"] == "Fixed_Time"].iloc[0]

    # IA_MOPPO has disparity 0.25, not variance
    assert ia_row["fairness_disparity_mean"] == pytest.approx(0.25)
    assert np.isnan(ia_row.get("fairness_variance_mean", np.nan))

    # Fixed_Time has variance 387.94, not disparity
    assert ft_row["fairness_variance_mean"] == pytest.approx(387.94)
    assert np.isnan(ft_row.get("fairness_disparity_mean", np.nan))
