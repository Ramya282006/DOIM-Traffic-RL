"""
DOIM-Traffic-RL: Unit tests for Strict Scenario Isolation (P12-01).
Uses synthetic data deliberately containing multiple scenarios, disparate training seeds,
and repeated evaluation episodes to mathematically verify:
1. Every result strictly retains its scenario identity.
2. Scenarios are never combined or contaminated across tables.
3. Repeated evaluation episodes within the same trained policy are averaged first (intra-model).
4. Independent training seeds serve as the primary statistical unit.
5. Incomplete/rejected runs are excluded from comparison tables but audited in rejected_runs.json.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from analysis.statistical_analyzer import StatisticalAnalyzer
from common.experiment import ManifestStatus


@pytest.fixture
def multi_scenario_synthetic_dataset(tmp_path: Path):
    """
    Creates synthetic run directories across 2 distinct scenarios:
      - 'north_south_surge'
      - 'rush_hour'
    Including repeated evaluation episodes, multiple seeds, and an incomplete run.
    """
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    test_runs = [
        # Scenario 1: north_south_surge, seed 42, ep 0
        {
            "id": "IA_MOPPO_north_south_surge_s42_ep0",
            "algo": "IA_MOPPO",
            "scen": "north_south_surge",
            "train_seed": 42,
            "eval_seed": 1042,
            "ep_id": 0,
            "waiting": 50.0,
            "queue": 2.0,
            "status": "COMPLETED",
        },
        # Scenario 1: north_south_surge, seed 42, ep 1 (repeated eval episode)
        {
            "id": "IA_MOPPO_north_south_surge_s42_ep1",
            "algo": "IA_MOPPO",
            "scen": "north_south_surge",
            "train_seed": 42,
            "eval_seed": 1043,
            "ep_id": 1,
            "waiting": 70.0,  # Average with ep0 must be (50 + 70)/2 = 60.0
            "queue": 4.0,   # Average with ep0 must be (2 + 4)/2 = 3.0
            "status": "COMPLETED",
        },
        # Scenario 1: north_south_surge, seed 123, ep 0
        {
            "id": "IA_MOPPO_north_south_surge_s123_ep0",
            "algo": "IA_MOPPO",
            "scen": "north_south_surge",
            "train_seed": 123,
            "eval_seed": 1123,
            "ep_id": 0,
            "waiting": 40.0,
            "queue": 1.0,
            "status": "COMPLETED",
        },
        # Scenario 2: rush_hour, seed 42, ep 0 (DIFFERENT SCENARIO)
        {
            "id": "IA_MOPPO_rush_hour_s42_ep0",
            "algo": "IA_MOPPO",
            "scen": "rush_hour",
            "train_seed": 42,
            "eval_seed": 1042,
            "ep_id": 0,
            "waiting": 999.0, # High outlier in rush_hour: MUST NEVER contaminate north_south_surge
            "queue": 50.0,
            "status": "COMPLETED",
        },
        # Scenario 1: north_south_surge, seed 999 (INCOMPLETE RUN)
        {
            "id": "IA_MOPPO_north_south_surge_s999_aborted",
            "algo": "IA_MOPPO",
            "scen": "north_south_surge",
            "train_seed": 999,
            "eval_seed": 1999,
            "ep_id": 0,
            "waiting": 10.0,
            "queue": 0.5,
            "status": "INITIALIZED", # Should be rejected and not included
        },
    ]

    for r in test_runs:
        run_folder = runs_dir / r["id"]
        run_folder.mkdir(parents=True, exist_ok=True)

        # Write manifest
        manifest = {
            "experiment_id": r["id"],
            "algorithm": r["algo"],
            "scenario": r["scen"],
            "train_seed": r["train_seed"],
            "evaluation_seed": r["eval_seed"],
            "training_timesteps": 1000,
            "episode_id": r["ep_id"],
            "evaluation_mode": "deterministic_eval",
            "config_hash": "testhash",
            "code_commit": "testcommit",
            "start_time": "2026-10-31T10:00:00",
            "end_time": "2026-10-31T10:05:00" if r["status"] == "COMPLETED" else None,
            "completion_status": r["status"],
        }
        with open(run_folder / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        # Write summary
        summary = {
            "run_id": r["id"],
            "algorithm": r["algo"],
            "scenario": r["scen"],
            "train_seed": r["train_seed"],
            "eval_seed": r["eval_seed"],
            "episode_id": r["ep_id"],
            "average_waiting": r["waiting"],
            "average_queue": r["queue"],
            "average_fairness": 0.15,
            "total_steps": 100 if r["status"] == "COMPLETED" else 0,
        }
        with open(run_folder / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f)

        # Write dummy metrics CSV
        if r["status"] == "COMPLETED":
            with open(run_folder / "metrics_step.csv", "w", encoding="utf-8") as f:
                f.write("step,simulation_time,reward,raw_waiting,raw_queue,raw_co2,raw_throughput,raw_fairness\n")
                f.write(f"1,1.0,-0.5,{r['waiting']},{r['queue']},1000,1,0.15\n")

    return runs_dir


def test_scenario_isolation_and_intra_model_averaging(multi_scenario_synthetic_dataset: Path, tmp_path: Path):
    """
    Verifies that StatisticalAnalyzer:
    1. Rejects incomplete run s999.
    2. Groups strictly by (scenario, algorithm).
    3. Calculates seed-level summaries before cross-seed mean.
    4. Guarantees rush_hour values (waiting=999) NEVER bleed into north_south_surge.
    """
    analyzer = StatisticalAnalyzer(runs_root=multi_scenario_synthetic_dataset)
    df = analyzer.load_run_summaries()

    # Verify rejected incomplete run
    assert len(analyzer.rejected_runs) == 1
    assert analyzer.rejected_runs[0]["run_dir"] == "IA_MOPPO_north_south_surge_s999_aborted"
    assert "INITIALIZED" in analyzer.rejected_runs[0]["reason"]

    # Verify loaded completed records
    assert len(df) == 4

    # Perform aggregation
    agg = analyzer.aggregate_by_scenario_and_algorithm(df)
    assert len(agg) == 2  # Exactly 2 scenario rows: north_south_surge and rush_hour

    # Verify north_south_surge row
    nss_row = agg[agg["scenario"] == "north_south_surge"].iloc[0]
    assert nss_row["n_independent_seeds"] == 2  # Seed 42 and Seed 123
    assert nss_row["n_evaluation_episodes"] == 3  # Seed 42 has 2 episodes, Seed 123 has 1

    # Mathematical verification of intra-model averaging:
    # Seed 42 average waiting = (50.0 + 70.0) / 2 = 60.0
    # Seed 123 average waiting = 40.0
    # Cross-seed mean = (60.0 + 40.0) / 2 = 50.0
    assert nss_row["average_waiting_mean"] == pytest.approx(50.0, abs=1e-5)
    # If rush_hour (999.0) had contaminated this table, mean would be > 200.0!
    assert nss_row["average_waiting_mean"] < 100.0

    # Cross-seed std: ddof=1 sample std of [60.0, 40.0] = 14.1421356
    expected_std = float(np.std([60.0, 40.0], ddof=1))
    assert nss_row["average_waiting_std"] == pytest.approx(expected_std, rel=1e-4)

    # Verify rush_hour row
    rh_row = agg[agg["scenario"] == "rush_hour"].iloc[0]
    assert rh_row["n_independent_seeds"] == 1
    assert rh_row["average_waiting_mean"] == pytest.approx(999.0, abs=1e-5)

    # Test export tables isolation
    out_tables = tmp_path / "tables"
    analyzer.export_summary_tables(output_dir=out_tables)

    assert (out_tables / "benchmark_comparison.csv").exists()
    assert (out_tables / "rejected_runs.json").exists()
    assert (out_tables / "north_south_surge_benchmark_comparison.csv").exists()
    assert (out_tables / "rush_hour_benchmark_comparison.csv").exists()

    # Read exported north_south_surge table and verify ZERO rush_hour references
    nss_exported = pd.read_csv(out_tables / "north_south_surge_benchmark_comparison.csv")
    assert all(nss_exported["scenario"] == "north_south_surge")
    assert not any("rush_hour" in s for s in nss_exported["scenario"])
