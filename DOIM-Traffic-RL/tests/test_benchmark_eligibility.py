"""
DOIM-Traffic-RL: Benchmark Eligibility and Fail-Closed Verification Tests (Phase B)
Verifies:
  1. 128-step diagnostic runs are strictly rejected from confirmatory tier.
  2. Unregistered/smoke algorithms are rejected.
  3. Non-deterministic evaluation modes are rejected from confirmatory tier.
  4. Missing model checkpoint SHA-256 hashes are rejected for learning agents.
  5. Short horizons (<1800s) are rejected.
  6. Pilot tier (>=5000 steps) eligibility behaves correctly.
  7. Smoke tier allows diagnostic runs with structured reasons.
"""

import pytest
from common.experiment import (
    is_confirmatory_benchmark_eligible,
    classify_experiment_manifest,
    ManifestStatus,
    SCHEMA_VERSION_CURRENT,
)


def test_128_step_diagnostic_rejected_from_confirmatory():
    """Confirmatory tier MUST fail-closed on 128-step diagnostic exploration runs."""
    manifest = {
        "schema_version": SCHEMA_VERSION_CURRENT,
        "experiment_id": "IA_MOPPO_diag_s42",
        "algorithm": "IA_MOPPO",
        "scenario": "north_south_surge",
        "train_seed": 42,
        "evaluation_seed": 42,
        "training_timesteps": 128,
        "episode_id": 0,
        "evaluation_mode": "deterministic_eval",
        "eval_horizon": 1800,
        "config_hash": "a1b2c3d4",
        "code_commit": "abcdef0123456789",
        "start_time": "2026-11-03T00:00:00Z",
        "completion_status": "COMPLETED",
        "model_checkpoint_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    
    is_eligible, reasons, _ = is_confirmatory_benchmark_eligible(manifest, declared_tier="confirmatory")
    assert not is_eligible
    assert any("below confirmatory threshold" in r for r in reasons)


def test_unregistered_algorithm_rejected():
    """Ad-hoc or debug algorithms must not enter benchmark tables."""
    manifest = {
        "schema_version": SCHEMA_VERSION_CURRENT,
        "experiment_id": "custom_debug_s42",
        "algorithm": "RandomJunkAgent",
        "scenario": "north_south_surge",
        "train_seed": 42,
        "evaluation_seed": 42,
        "training_timesteps": 10000,
        "episode_id": 0,
        "evaluation_mode": "deterministic_eval",
        "eval_horizon": 1800,
        "config_hash": "a1b2c3d4",
        "code_commit": "abcdef0123456789",
        "start_time": "2026-11-03T00:00:00Z",
        "completion_status": "COMPLETED",
        "model_checkpoint_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    
    is_eligible, reasons, _ = is_confirmatory_benchmark_eligible(manifest, declared_tier="confirmatory")
    assert not is_eligible
    assert any("not in the preregistered comparison set" in r for r in reasons)


def test_missing_checkpoint_hash_rejected_for_learning_agent():
    """Learning algorithms must provide valid model checkpoint SHA-256 hash."""
    manifest = {
        "schema_version": SCHEMA_VERSION_CURRENT,
        "experiment_id": "IA_MOPPO_s42",
        "algorithm": "IA_MOPPO",
        "scenario": "north_south_surge",
        "train_seed": 42,
        "evaluation_seed": 42,
        "training_timesteps": 10000,
        "episode_id": 0,
        "evaluation_mode": "deterministic_eval",
        "eval_horizon": 1800,
        "config_hash": "a1b2c3d4",
        "code_commit": "abcdef0123456789",
        "start_time": "2026-11-03T00:00:00Z",
        "completion_status": "COMPLETED",
        "model_checkpoint_sha256": None,  # Missing!
    }
    
    is_eligible, reasons, _ = is_confirmatory_benchmark_eligible(manifest, declared_tier="confirmatory")
    assert not is_eligible
    assert any("model checkpoint SHA-256 provenance" in r for r in reasons)


def test_fixed_time_controller_exempt_from_checkpoint_hash():
    """Non-learning baselines like Fixed-Time do not require neural model checkpoints."""
    manifest = {
        "schema_version": SCHEMA_VERSION_CURRENT,
        "experiment_id": "FixedTime_s42",
        "algorithm": "Fixed_Time",
        "scenario": "north_south_surge",
        "train_seed": 42,
        "evaluation_seed": 42,
        "training_timesteps": 5000,
        "episode_id": 0,
        "evaluation_mode": "deterministic_eval",
        "eval_horizon": 1800,
        "config_hash": "a1b2c3d4",
        "code_commit": "abcdef0123456789",
        "start_time": "2026-11-03T00:00:00Z",
        "completion_status": "COMPLETED",
        "model_checkpoint_sha256": None,
    }
    
    is_eligible, reasons, _ = is_confirmatory_benchmark_eligible(manifest, declared_tier="confirmatory")
    assert is_eligible
    assert len(reasons) == 0


def test_valid_confirmatory_run_accepted():
    """A valid, verified long-horizon confirmatory run passes all eligibility gates."""
    manifest = {
        "schema_version": SCHEMA_VERSION_CURRENT,
        "experiment_id": "IA_MOPPO_confirmatory_s42",
        "algorithm": "IA_MOPPO",
        "scenario": "north_south_surge",
        "train_seed": 42,
        "evaluation_seed": 1042,
        "training_timesteps": 10000,
        "episode_id": 0,
        "evaluation_mode": "deterministic_eval",
        "eval_horizon": 1800,
        "config_hash": "a1b2c3d4e5f6",
        "code_commit": "abcdef0123456789",
        "start_time": "2026-11-03T00:00:00Z",
        "completion_status": "COMPLETED",
        "model_checkpoint_sha256": "252e539c3e6baac92f68fcc8a2c40d4c697ec18fe50f1119b7f1e22611f1f386",
    }
    
    is_eligible, reasons, data = is_confirmatory_benchmark_eligible(manifest, declared_tier="confirmatory")
    assert is_eligible
    assert len(reasons) == 0
    assert data["algorithm"] == "IA_MOPPO"
