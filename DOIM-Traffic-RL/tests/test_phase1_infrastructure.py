"""
DOIM-Traffic-RL: Phase 1 Infrastructure & Reproducibility Test Suite
Validates configuration management, seeding, experiment tracking,
and verification tooling.
"""

import os
from pathlib import Path
import tempfile
import numpy as np
import pytest

from common.config import (
    load_experiment_config,
    snapshot_config,
    compute_config_hash,
    ConfigDict,
)
from common.seeding import (
    seed_everything,
    generate_independent_seeds,
    format_sumo_seed,
)
from common.experiment import ExperimentRun
from common.verification import ExperimentVerifier
from environment.sumo_locator import find_sumo_binary


def test_config_loading_and_hash():
    """Verify that configuration loads default.yaml, supports attribute access, and computes SHA-256."""
    config = load_experiment_config(default_config_path="configs/default.yaml")
    assert isinstance(config, ConfigDict)
    assert config.experiment.project_name == "DOIM-Traffic-RL"
    assert config.objectives.num_objectives == 5
    assert len(config.policy.training_seeds if hasattr(config.policy, 'training_seeds') else config.reproducibility.training_seeds) >= 3
    assert len(config.config_hash) == 16


def test_config_snapshot():
    """Verify that configuration snapshots are created and readable."""
    config = load_experiment_config(default_config_path="configs/default.yaml")
    with tempfile.TemporaryDirectory() as tmp_dir:
        snap_path = snapshot_config(config, tmp_dir)
        assert snap_path.exists()
        assert snap_path.name == "config_snapshot.yaml"


def test_seed_everything_determinism():
    """Verify that seed_everything yields identical NumPy and Python random outputs."""
    seed_everything(42)
    a1 = np.random.rand(5)
    
    seed_everything(42)
    a2 = np.random.rand(5)
    
    np.testing.assert_allclose(a1, a2)
    
    seed_everything(999)
    a3 = np.random.rand(5)
    assert not np.allclose(a1, a3)


def test_independent_seed_generation():
    """Verify that derived child seeds are unique and reproducible."""
    seeds1 = generate_independent_seeds(master_seed=42, count=5)
    seeds2 = generate_independent_seeds(master_seed=42, count=5)
    assert seeds1 == seeds2
    assert len(set(seeds1)) == 5
    for s in seeds1:
        assert int(format_sumo_seed(s)) > 0


def test_experiment_run_lifecycle_and_verification():
    """Verify that ExperimentRun logs steps, generates manifest, and passes verification."""
    config = load_experiment_config(default_config_path="configs/default.yaml")
    with tempfile.TemporaryDirectory() as tmp_dir:
        run = ExperimentRun(
            experiment_name="unit_test_run",
            algorithm="test_algo",
            config=config,
            train_seed=42,
            output_root=tmp_dir,
        )
        assert run.manifest_path.exists()
        assert run.run_dir.exists()
        
        # Log 5 test steps
        for step in range(5):
            run.log_step(
                step=step,
                sim_time=float(step),
                reward=-0.5,
                raw_objs=[10.0, 2.0, 500.0, 1.0, 20.0],
                norm_objs=[0.2, 0.1, 0.3, 0.4, 0.2],
                weights=[0.2, 0.2, 0.2, 0.2, 0.2],
                actions=[0, 0, 0, 0, 0],
                shield_interventions=0,
            )
            
        summary = run.close(status="COMPLETED")
        assert summary["total_steps"] == 5
        assert summary["total_reward"] == -2.5
        assert run.summary_json_path.exists()
        
        # Verify finite metrics
        ok, msg = ExperimentVerifier.verify_finite_metrics(run.metrics_csv_path)
        assert ok, msg
        
        # Verify JSON/CSV consistency
        ok, msg = ExperimentVerifier.verify_json_csv_consistency(run.run_dir)
        assert ok, msg
        
        # Verify simplex condition
        ok, msg = ExperimentVerifier.verify_simplex_weights(summary["average_weights"])
        assert ok, msg


def test_sumo_locator():
    """Verify that find_sumo_binary locates a valid executable."""
    binary = find_sumo_binary(prefer_gui=False)
    assert binary is not None
    assert Path(binary).exists()
    assert "sumo" in binary.lower()
