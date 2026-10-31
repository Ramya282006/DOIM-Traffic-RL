"""
Unit tests for strict Phase 1 experiment schema validation.
"""

import json
from pathlib import Path
import pytest
from common.experiment import validate_experiment_manifest, ExperimentRun


def test_valid_manifest_schema(tmp_path: Path):
    """
    Verifies that a completed experiment run generates a conforming manifest.
    """
    config = {"config_hash": "abc123hash"}
    run = ExperimentRun(
        experiment_name="schema_test",
        algorithm="IA_MOPPO",
        config=config,
        train_seed=42,
        eval_seed=1042,
        scenario="north_south_surge",
        output_root=tmp_path,
        training_timesteps=1000,
        episode_id=1,
        evaluation_mode="deterministic_eval",
        experiment_id="EXP_001",
    )
    run.log_step(
        step=1,
        sim_time=1.0,
        reward=-0.5,
        raw_objs=[10.0, 2.0, 50.0, 5.0, 0.1],
        norm_objs=[0.1, 0.1, 0.1, 0.1, 0.1],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        actions=[0, 0, 0, 0, 0],
    )
    run.close(status="COMPLETED")

    is_valid, msg = validate_experiment_manifest(run.manifest_path)
    assert is_valid, f"Validation failed on valid manifest: {msg}"
    assert "Valid" in msg


def test_incomplete_run_rejection(tmp_path: Path):
    """
    Verifies that incomplete, aborted, or initialized runs are rejected.
    """
    manifest = {
        "experiment_id": "EXP_INCOMPLETE",
        "algorithm": "IA_MOPPO",
        "scenario": "north_south_surge",
        "train_seed": 42,
        "evaluation_seed": 42,
        "training_timesteps": 5000,
        "episode_id": 0,
        "evaluation_mode": "train_rollout",
        "config_hash": "hash123",
        "code_commit": "2c71f93",
        "start_time": "2026-10-31T00:00:00Z",
        "completion_status": "INITIALIZED", # not completed
    }
    is_valid, msg = validate_experiment_manifest(manifest)
    assert not is_valid
    assert "Incomplete or unverified" in msg


def test_missing_fields_rejection():
    """
    Verifies that manifests missing required fields are rejected.
    """
    manifest = {
        "algorithm": "IA_MOPPO",
        "scenario": "north_south_surge",
        # missing experiment_id, train_seed, etc.
    }
    is_valid, msg = validate_experiment_manifest(manifest)
    assert not is_valid
    assert "Missing required schema field" in msg


def test_invalid_evaluation_mode():
    """
    Verifies rejection of arbitrary or undefined evaluation modes.
    """
    manifest = {
        "experiment_id": "EXP_TEST",
        "algorithm": "IA_MOPPO",
        "scenario": "north_south_surge",
        "train_seed": 42,
        "evaluation_seed": 42,
        "training_timesteps": 5000,
        "episode_id": 0,
        "evaluation_mode": "invalid_magic_mode",
        "config_hash": "hash123",
        "code_commit": "2c71f93",
        "start_time": "2026-10-31T00:00:00Z",
        "completion_status": "COMPLETED",
    }
    is_valid, msg = validate_experiment_manifest(manifest)
    assert not is_valid
    assert "Invalid evaluation_mode" in msg
