"""
DOIM-Traffic-RL: Unit tests for Experiment Runner Interface and Constructor Aliasing (P12-04).
Verifies:
1. StatisticalAnalyzer(runs_dir=...) keyword alias compatibility.
2. StatisticalAnalyzer(runs_root=...) canonical interface.
3. Initialization, aggregation, and export of rejected runs audit.
"""

from pathlib import Path
import json
import pytest

from analysis.statistical_analyzer import StatisticalAnalyzer


def test_statistical_analyzer_constructor_alias(tmp_path: Path):
    """
    Verifies that StatisticalAnalyzer accepts both canonical 'runs_root'
    and legacy alias 'runs_dir' without raising TypeError.
    """
    # 1. Canonical interface
    analyzer_canonical = StatisticalAnalyzer(runs_root=tmp_path)
    assert analyzer_canonical.runs_root == tmp_path

    # 2. Legacy alias interface (P12-04 regression test)
    analyzer_alias = StatisticalAnalyzer(runs_dir=tmp_path)
    assert analyzer_alias.runs_root == tmp_path


def test_rejected_runs_audit_generation(tmp_path: Path):
    """
    Verifies that rejected runs are tracked with explicit reasons and exported to rejected_runs.json.
    """
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Valid run
    valid_dir = runs_dir / "valid_run"
    valid_dir.mkdir()
    with open(valid_dir / "manifest.json", "w") as f:
        json.dump({
            "experiment_id": "valid_001",
            "algorithm": "Fixed_Time",
            "scenario": "north_south_surge",
            "train_seed": 42,
            "evaluation_seed": 42,
            "training_timesteps": 128,
            "episode_id": 0,
            "evaluation_mode": "deterministic_eval",
            "config_hash": "h123",
            "code_commit": "c123",
            "start_time": "2026-10-31T00:00:00",
            "completion_status": "COMPLETED",
        }, f)
    with open(valid_dir / "summary.json", "w") as f:
        json.dump({"average_waiting": 50.0, "average_fairness": 0.2}, f)
    with open(valid_dir / "metrics_step.csv", "w") as f:
        f.write("step,time\n1,1.0\n")

    # Incomplete run
    bad_dir = runs_dir / "incomplete_run"
    bad_dir.mkdir()
    with open(bad_dir / "manifest.json", "w") as f:
        json.dump({
            "experiment_id": "bad_001",
            "algorithm": "Fixed_Time",
            "scenario": "north_south_surge",
            "train_seed": 42,
            "evaluation_seed": 42,
            "training_timesteps": 128,
            "episode_id": 0,
            "evaluation_mode": "deterministic_eval",
            "config_hash": "h123",
            "code_commit": "c123",
            "start_time": "2026-10-31T00:00:00",
            "completion_status": "INITIALIZED", # Not completed
        }, f)

    analyzer = StatisticalAnalyzer(runs_root=runs_dir)
    df = analyzer.load_run_summaries()

    assert len(df) == 1
    assert len(analyzer.rejected_runs) == 1
    assert analyzer.rejected_runs[0]["run_dir"] == "incomplete_run"
    assert "INITIALIZED" in analyzer.rejected_runs[0]["reason"]

    out_tables = tmp_path / "tables"
    analyzer.export_summary_tables(output_dir=out_tables)

    rejected_file = out_tables / "rejected_runs.json"
    assert rejected_file.exists()
    with open(rejected_file, "r") as f:
        audit = json.load(f)
    assert audit["rejected_count"] == 1
    assert audit["rejected_runs"][0]["run_dir"] == "incomplete_run"
