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


def test_delay_variance_below_one_classified_as_variance(tmp_path):
    """
    Critical regression test: Verifies that delay variance values below 1.0
    (e.g., Var(d) = 0.01 s^2) are NEVER erroneously classified as Delay Disparity
    due to numerical thresholding (raw_fairness <= 1.0).
    Explicit metric identity must route it to fairness_variance.
    """
    run_dir = tmp_path / "variance_low_run"
    run_dir.mkdir()

    # Step CSV with 1 row
    (run_dir / "metrics_step.csv").write_text(
        "step,simulation_time,reward,raw_waiting,raw_queue,raw_co2,raw_throughput,raw_fairness\n"
        "0,1.0,-0.1,0.2,0,500,1,0.01\n",
        encoding="utf-8",
    )
    # Summary JSON with explicit delay_variance identity
    import json
    summary_data = {
        "algorithm": "FW_PPO",
        "scenario": "test_scen",
        "train_seed": 42,
        "eval_seed": 1042,
        "total_steps": 1,
        "average_fairness": 0.01,
        "fairness_metric_name": "delay_variance",
        "fairness_metric_unit": "s^2",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_data), encoding="utf-8")

    analyzer = StatisticalAnalyzer(runs_root=tmp_path)
    df = analyzer.load_completed_runs(reject_incomplete=False)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["fairness_metric_name"] == "delay_variance"
    assert row["fairness_metric_unit"] == "s^2"
    assert row["fairness_variance"] == pytest.approx(0.01)
    assert np.isnan(row["fairness_disparity"])
    assert bool(row["fairness_eligible"]) is True


def test_delay_disparity_near_upper_bound(tmp_path):
    """
    Verifies that high delay disparity near the theoretical upper bound (e.g., 0.95)
    is correctly classified as Delay Disparity and not confused with variance.
    """
    run_dir = tmp_path / "disparity_high_run"
    run_dir.mkdir()

    (run_dir / "metrics_step.csv").write_text(
        "step,simulation_time,reward,raw_waiting,raw_queue,raw_co2,raw_throughput,raw_fairness\n"
        "0,1.0,-0.5,100.0,5,10000,0,0.95\n",
        encoding="utf-8",
    )
    import json
    summary_data = {
        "algorithm": "IA_MOPPO",
        "scenario": "test_scen",
        "train_seed": 42,
        "eval_seed": 1042,
        "total_steps": 1,
        "average_fairness": 0.95,
        "fairness_metric_name": "delay_disparity",
        "fairness_metric_unit": "dimensionless",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_data), encoding="utf-8")

    analyzer = StatisticalAnalyzer(runs_root=tmp_path)
    df = analyzer.load_completed_runs(reject_incomplete=False)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["fairness_metric_name"] == "delay_disparity"
    assert row["fairness_metric_unit"] == "dimensionless"
    assert row["fairness_disparity"] == pytest.approx(0.95)
    assert np.isnan(row["fairness_variance"])
    assert bool(row["fairness_eligible"]) is True


def test_unidentifiable_metric_classified_unknown_ineligible(tmp_path):
    """
    Verifies that a run lacking explicit fairness metric identity metadata
    is classified as UNKNOWN / ineligible rather than guessing based on numerical values.
    """
    run_dir = tmp_path / "unknown_metric_run"
    run_dir.mkdir()

    (run_dir / "metrics_step.csv").write_text(
        "step,simulation_time,reward,raw_waiting,raw_queue,raw_co2,raw_throughput,raw_fairness\n"
        "0,1.0,-0.2,10.0,1,2000,1,0.5\n",
        encoding="utf-8",
    )
    import json
    # No fairness_metric_name or unit provided
    summary_data = {
        "algorithm": "Fixed_Time",
        "scenario": "test_scen",
        "train_seed": 42,
        "eval_seed": 1042,
        "total_steps": 1,
        "average_fairness": 0.5,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_data), encoding="utf-8")

    analyzer = StatisticalAnalyzer(runs_root=tmp_path)
    df = analyzer.load_completed_runs(reject_incomplete=False)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["fairness_metric_name"] == "UNKNOWN"
    assert row["fairness_metric_unit"] == "UNKNOWN"
    assert np.isnan(row["fairness_disparity"])
    assert np.isnan(row["fairness_variance"])
    assert bool(row["fairness_eligible"]) is False

