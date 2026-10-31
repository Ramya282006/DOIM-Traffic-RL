"""
Unit tests for Scenario-Specific Statistical Analysis (Phase 2).
Verifies:
1. Scenario separation (no cross-scenario mixing)
2. Repeated evaluation episode intra-model averaging (seed as statistical unit)
3. Duplicate run rejection
4. Paired vs Welch's hypothesis tests
5. Holm-Bonferroni multiple testing correction
6. Insufficient sample size reporting
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from analysis.statistical_analyzer import StatisticalAnalyzer


@pytest.fixture
def mock_runs_directory(tmp_path: Path):
    """
    Creates synthetic structured runs across multiple scenarios, seeds, and repeated episodes.
    """
    runs = [
        # Scenario 1: north_south_surge, seed 42, 2 evaluation episodes (should average into 1 seed unit)
        {
            "run_dir": "IA_MOPPO_north_south_surge_s42_ep0",
            "algorithm": "IA_MOPPO",
            "scenario": "north_south_surge",
            "train_seed": 42,
            "episode_id": 0,
            "average_waiting": 50.0,
            "average_queue": 2.0,
            "average_reward": -5.0,
            "start_time": "2026-10-31T10:00:00",
        },
        {
            "run_dir": "IA_MOPPO_north_south_surge_s42_ep1",
            "algorithm": "IA_MOPPO",
            "scenario": "north_south_surge",
            "train_seed": 42,
            "episode_id": 1,
            "average_waiting": 60.0,
            "average_queue": 2.4,
            "average_reward": -6.0,
            "start_time": "2026-10-31T10:05:00",
        },
        # Scenario 1: north_south_surge, seed 123
        {
            "run_dir": "IA_MOPPO_north_south_surge_s123_ep0",
            "algorithm": "IA_MOPPO",
            "scenario": "north_south_surge",
            "train_seed": 123,
            "episode_id": 0,
            "average_waiting": 45.0,
            "average_queue": 1.8,
            "average_reward": -4.5,
            "start_time": "2026-10-31T10:10:00",
        },
        # Scenario 1: north_south_surge, seed 456
        {
            "run_dir": "IA_MOPPO_north_south_surge_s456_ep0",
            "algorithm": "IA_MOPPO",
            "scenario": "north_south_surge",
            "train_seed": 456,
            "episode_id": 0,
            "average_waiting": 40.0,
            "average_queue": 1.5,
            "average_reward": -4.0,
            "start_time": "2026-10-31T10:15:00",
        },
        # Baseline FW_PPO in north_south_surge with matching seeds 42, 123, 456
        {
            "run_dir": "FW_PPO_north_south_surge_s42",
            "algorithm": "FW_PPO",
            "scenario": "north_south_surge",
            "train_seed": 42,
            "episode_id": 0,
            "average_waiting": 65.0,
            "average_queue": 2.5,
            "average_reward": -7.0,
            "start_time": "2026-10-31T10:00:00",
        },
        {
            "run_dir": "FW_PPO_north_south_surge_s123",
            "algorithm": "FW_PPO",
            "scenario": "north_south_surge",
            "train_seed": 123,
            "episode_id": 0,
            "average_waiting": 55.0,
            "average_queue": 2.2,
            "average_reward": -6.0,
            "start_time": "2026-10-31T10:05:00",
        },
        {
            "run_dir": "FW_PPO_north_south_surge_s456",
            "algorithm": "FW_PPO",
            "scenario": "north_south_surge",
            "train_seed": 456,
            "episode_id": 0,
            "average_waiting": 50.0,
            "average_queue": 2.0,
            "average_reward": -5.5,
            "start_time": "2026-10-31T10:10:00",
        },
        # Scenario 2: rush_hour - completely different scenario!
        {
            "run_dir": "IA_MOPPO_rush_hour_s42",
            "algorithm": "IA_MOPPO",
            "scenario": "rush_hour",
            "train_seed": 42,
            "episode_id": 0,
            "average_waiting": 150.0,
            "average_queue": 8.0,
            "average_reward": -25.0,
            "start_time": "2026-10-31T11:00:00",
        },
    ]

    for r in runs:
        r_dir = tmp_path / r["run_dir"]
        r_dir.mkdir(parents=True, exist_ok=True)
        # Create summary.json
        with open(r_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(r, f)
        # Create valid manifest.json
        manifest = {
            "experiment_id": r["run_dir"],
            "algorithm": r["algorithm"],
            "scenario": r["scenario"],
            "train_seed": r["train_seed"],
            "evaluation_seed": r["train_seed"],
            "training_timesteps": 1000,
            "episode_id": r.get("episode_id", 0),
            "evaluation_mode": "deterministic_eval",
            "config_hash": "cfg_test",
            "code_commit": "2c71f93",
            "start_time": r["start_time"],
            "completion_status": "COMPLETED",
        }
        with open(r_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        # Create dummy metrics_step.csv
        with open(r_dir / "metrics_step.csv", "w", encoding="utf-8") as f:
            f.write("step,reward,raw_waiting\n1,-1.0,50.0\n")

    return tmp_path


def test_scenario_isolation(mock_runs_directory: Path):
    """
    Verifies that north_south_surge and rush_hour are NOT aggregated together.
    """
    analyzer = StatisticalAnalyzer(runs_root=mock_runs_directory)
    df = analyzer.load_run_summaries()
    agg = analyzer.aggregate_by_scenario_and_algorithm(df)

    scenarios = set(agg["scenario"].unique())
    assert "north_south_surge" in scenarios
    assert "rush_hour" in scenarios
    assert len(agg) == 3 # (north_south_surge, FW_PPO), (north_south_surge, IA_MOPPO), (rush_hour, IA_MOPPO)

    # Check IA_MOPPO in north_south_surge does not include rush_hour waiting time (150s)
    ns_iamoppo = agg[(agg["scenario"] == "north_south_surge") & (agg["algorithm"] == "IA_MOPPO")].iloc[0]
    # For seed 42: mean of (50.0 + 60.0)/2 = 55.0
    # Seeds: 55.0, 45.0, 40.0 -> mean is 46.6667
    assert np.isclose(ns_iamoppo["average_waiting_mean"], (55.0 + 45.0 + 40.0) / 3.0, atol=1e-3)
    assert ns_iamoppo["n_independent_seeds"] == 3
    assert ns_iamoppo["n_evaluation_episodes"] == 4


def test_intra_model_averaging_preserves_seed_units(mock_runs_directory: Path):
    """
    Verifies that multiple evaluation episodes within the same model are averaged
    before statistical testing, preventing pseudo-replication.
    """
    analyzer = StatisticalAnalyzer(runs_root=mock_runs_directory)
    df = analyzer.load_run_summaries()
    agg = analyzer.aggregate_by_scenario_and_algorithm(df)

    ns_iamoppo = agg[(agg["scenario"] == "north_south_surge") & (agg["algorithm"] == "IA_MOPPO")].iloc[0]
    assert ns_iamoppo["n_independent_seeds"] == 3
    assert ns_iamoppo["n_evaluation_episodes"] == 4
    # Min is min across seed means: min(55, 45, 40) = 40.0
    assert ns_iamoppo["average_waiting_min"] == 40.0
    # Max is max across seed means: max(55, 45, 40) = 55.0
    assert ns_iamoppo["average_waiting_max"] == 55.0


def test_duplicate_run_rejection(tmp_path: Path):
    """
    Verifies that identical run records for the same algorithm, scenario, seed, and episode
    are deduplicated, keeping the latest timestamp.
    """
    r1 = tmp_path / "run_older"
    r2 = tmp_path / "run_newer"
    r1.mkdir()
    r2.mkdir()

    for r_dir, ts, wait in [(r1, "2026-10-31T08:00:00", 99.0), (r2, "2026-10-31T09:00:00", 35.0)]:
        with open(r_dir / "summary.json", "w") as f:
            json.dump({
                "algorithm": "IA_MOPPO",
                "scenario": "surge",
                "train_seed": 42,
                "episode_id": 0,
                "average_waiting": wait,
                "start_time": ts,
            }, f)
        with open(r_dir / "manifest.json", "w") as f:
            json.dump({
                "experiment_id": r_dir.name,
                "algorithm": "IA_MOPPO",
                "scenario": "surge",
                "train_seed": 42,
                "evaluation_seed": 42,
                "training_timesteps": 1000,
                "episode_id": 0,
                "evaluation_mode": "train_rollout",
                "config_hash": "cfg",
                "code_commit": "2c71f93",
                "start_time": ts,
                "completion_status": "COMPLETED",
            }, f)
        with open(r_dir / "metrics_step.csv", "w") as f:
            f.write("step,raw_waiting\n1,35.0\n")

    analyzer = StatisticalAnalyzer(runs_root=tmp_path)
    df = analyzer.load_run_summaries()
    assert len(df) == 1 # Deduplicated to 1
    assert df.iloc[0]["average_waiting"] == 35.0 # Kept the newer run


def test_paired_hypothesis_testing(mock_runs_directory: Path):
    """
    Verifies paired t-test execution when seeds match (seeds 42, 123, 456).
    """
    analyzer = StatisticalAnalyzer(runs_root=mock_runs_directory)
    test_res = analyzer.run_hypothesis_tests(
        scenario="north_south_surge",
        baseline_algo="FW_PPO",
        target_algo="IA_MOPPO",
        metric="average_waiting",
    )
    assert test_res["test_type"] == "Paired Student's t-test"
    assert test_res["n_baseline_seeds"] == 3
    assert test_res["n_target_seeds"] == 3
    # IA_MOPPO (46.67s) outperforms FW_PPO (56.67s)
    assert test_res["target_mean"] < test_res["baseline_mean"]
    assert test_res["cohen_d"] < 0 # Negative difference is improvement for waiting time


def test_insufficient_samples_reported(tmp_path: Path):
    """
    Verifies that when N < 2 independent seeds, statistical test explicitly reports insufficient samples.
    """
    r_dir = tmp_path / "single_run"
    r_dir.mkdir()
    with open(r_dir / "summary.json", "w") as f:
        json.dump({
            "algorithm": "IA_MOPPO",
            "scenario": "surge",
            "train_seed": 42,
            "average_waiting": 50.0,
        }, f)
    with open(r_dir / "manifest.json", "w") as f:
        json.dump({
            "experiment_id": "single",
            "algorithm": "IA_MOPPO",
            "scenario": "surge",
            "train_seed": 42,
            "evaluation_seed": 42,
            "training_timesteps": 1000,
            "episode_id": 0,
            "evaluation_mode": "train_rollout",
            "config_hash": "cfg",
            "code_commit": "2c71f93",
            "start_time": "2026-10-31T00:00:00",
            "completion_status": "COMPLETED",
        }, f)
    with open(r_dir / "metrics_step.csv", "w") as f:
        f.write("step,raw_waiting\n1,50.0\n")

    analyzer = StatisticalAnalyzer(runs_root=tmp_path)
    res = analyzer.run_hypothesis_tests(
        scenario="surge",
        baseline_algo="FW_PPO",
        target_algo="IA_MOPPO",
        metric="average_waiting",
    )
    assert res["status"] == "INSUFFICIENT_INDEPENDENT_SEEDS"


def test_holm_bonferroni_correction():
    """
    Verifies mathematical correctness of Holm-Bonferroni step-down thresholding.
    """
    p_vals = [0.005, 0.02, 0.04, 0.20]
    alpha = 0.05
    results = StatisticalAnalyzer.holm_bonferroni_correction(p_vals, alpha=alpha)

    # Hypothesis 1: p=0.005 <= 0.05/4 = 0.0125 -> Significant
    assert results[0]["significant"]
    assert results[0]["rank"] == 1
    assert results[0]["threshold"] == 0.05 / 4.0

    # Hypothesis 2: p=0.02 <= 0.05/3 = 0.01667 -> False (0.02 > 0.01667) -> Not significant
    assert not results[1]["significant"]
    assert results[1]["rank"] == 2

    # Subsequent hypotheses must not be rejected
    assert not results[2]["significant"]
    assert not results[3]["significant"]
