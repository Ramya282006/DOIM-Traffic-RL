"""
DOIM-Traffic-RL: Unit tests for Experiment Manifest Classification & Migration (P12-02).
Verifies:
1. Canonical VALID manifest recognition.
2. Historical Phase 10 manifest migration to LEGACY_VALID.
3. Incomplete run detection (status='INITIALIZED', 0 steps).
4. Corrupted / missing field rejection (INVALID).
5. Strict mode rejection when allow_legacy=False.
"""

from pathlib import Path
import json
import pytest

from common.experiment import (
    classify_experiment_manifest,
    validate_experiment_manifest,
    ManifestStatus,
)


def test_classify_canonical_valid():
    """Verifies that a full modern manifest is classified as VALID."""
    manifest = {
        "experiment_id": "EXP_VALID_001",
        "algorithm": "IA_MOPPO",
        "scenario": "north_south_surge",
        "train_seed": 42,
        "evaluation_seed": 1042,
        "training_timesteps": 1000,
        "episode_id": 0,
        "evaluation_mode": "deterministic_eval",
        "config_hash": "testhash123",
        "code_commit": "commit456",
        "start_time": "2026-10-31T10:00:00",
        "end_time": "2026-10-31T10:05:00",
        "completion_status": "COMPLETED",
    }
    status, reason, parsed = classify_experiment_manifest(manifest)
    assert status == ManifestStatus.VALID
    assert parsed["experiment_id"] == "EXP_VALID_001"
    assert parsed["evaluation_mode"] == "deterministic_eval"


def test_classify_historical_legacy_valid():
    """
    Verifies that a completed Phase 10 historical manifest is classified
    as LEGACY_VALID with all fields mapped to canonical schema.
    """
    historical_manifest = {
        "run_id": "Fixed_Time_north_south_surge_s42_20261031_151810",
        "experiment_name": "Fixed_Time_north_south_surge",
        "algorithm": "Fixed_Time",
        "scenario": "north_south_surge",
        "train_seed": 42,
        "eval_seed": 42,
        "start_time": "2026-10-31T09:48:10.717261+00:00",
        "end_time": "2026-10-31T09:48:18.805933+00:00",
        "status": "COMPLETED",
        "git_sha": "fd264346a9f3f5c7a6859c045561b9650b7dc922",
        "config_hash": "7888a45453a99435",
        "total_steps": 128,
        "summary_metrics": {"average_waiting": 54.73},
    }

    status, reason, migrated = classify_experiment_manifest(historical_manifest, allow_legacy=True)
    assert status == ManifestStatus.LEGACY_VALID
    assert migrated["experiment_id"] == "Fixed_Time_north_south_surge_s42_20261031_151810"
    assert migrated["evaluation_seed"] == 42
    assert migrated["training_timesteps"] == 128
    assert migrated["code_commit"] == "fd264346a9f3f5c7a6859c045561b9650b7dc922"
    assert migrated["completion_status"] == "COMPLETED"
    assert migrated["is_legacy_migrated"] is True

    # Rejection when allow_legacy=False
    status_strict, reason_strict, _ = classify_experiment_manifest(historical_manifest, allow_legacy=False)
    assert status_strict == ManifestStatus.INVALID


def test_classify_incomplete_initialized_run():
    """
    Verifies that an aborted or crashed run with status='INITIALIZED' is classified as INCOMPLETE.
    """
    aborted_manifest = {
        "run_id": "IA_MOPPO_north_south_surge_s42_20261031_150516",
        "experiment_name": "IA_MOPPO_north_south_surge",
        "algorithm": "IA_MOPPO",
        "scenario": "north_south_surge",
        "train_seed": 42,
        "eval_seed": 42,
        "start_time": "2026-10-31T09:35:16.783933+00:00",
        "end_time": None,
        "status": "INITIALIZED",
        "total_steps": 0,
    }
    status, reason, parsed = classify_experiment_manifest(aborted_manifest, allow_legacy=True)
    assert status == ManifestStatus.INCOMPLETE
    assert "status='INITIALIZED'" in reason
    assert parsed is None

    is_valid, msg = validate_experiment_manifest(aborted_manifest, allow_legacy=True)
    assert not is_valid
    assert "[INCOMPLETE]" in msg


def test_classify_corrupted_json(tmp_path: Path):
    """
    Verifies that a malformed JSON file is classified as INVALID with exact reason.
    """
    bad_file = tmp_path / "corrupt_manifest.json"
    with open(bad_file, "w") as f:
        f.write("{this is not valid json")

    status, reason, parsed = classify_experiment_manifest(bad_file)
    assert status == ManifestStatus.INVALID
    assert "Corrupted or unparseable" in reason


def test_evaluation_mode_alias_normalization():
    """
    Verifies that aliases such as 'deterministic_evaluation' are normalized to 'deterministic_eval'.
    """
    manifest = {
        "experiment_id": "EXP_ALIAS_001",
        "algorithm": "IA_MOPPO",
        "scenario": "north_south_surge",
        "train_seed": 42,
        "evaluation_seed": 42,
        "training_timesteps": 1000,
        "episode_id": 0,
        "evaluation_mode": "deterministic_evaluation", # alias
        "config_hash": "hash123",
        "code_commit": "commit456",
        "start_time": "2026-10-31T10:00:00",
        "completion_status": "COMPLETED",
    }
    status, reason, parsed = classify_experiment_manifest(manifest)
    assert status == ManifestStatus.VALID
    assert parsed["evaluation_mode"] == "deterministic_eval"
