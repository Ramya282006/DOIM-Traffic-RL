"""
DOIM-Traffic-RL: Single Documented End-to-End Smoke Test
Executes a rapid, verifiable end-to-end test of simulation stepping,
observation extraction, adaptive reward calculation, fixed baseline stepping,
and manifest verification.
"""

import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from common.config import load_experiment_config
from common.seeding import seed_everything
from common.experiment import ExperimentRun
from common.verification import ExperimentVerifier
from environment.sumo_env import DOIMTrafficEnv
from environment.fixed_weight_env import FixedWeightTrafficEnv


def run_smoke_test(num_steps: int = 30) -> bool:
    print("=" * 80)
    print("DOIM-TRAFFIC-RL END-TO-END VERIFICATION SMOKE TEST")
    print("=" * 80)

    # 1. Config & Seeding
    config = load_experiment_config("configs/default.yaml")
    seed_everything(42)
    print(f"[1/5] Configuration loaded. SHA256 Hash: {config.config_hash}")

    # 2. Experiment Run Harness
    run = ExperimentRun(
        experiment_name="smoke_test",
        algorithm="DOIM_SMOKE",
        config=config,
        train_seed=42,
        scenario="directional_shift",
        output_root="results/runs",
    )
    print(f"[2/5] Experiment harness initialized: {run.run_id}")

    # 3. Test DOIMTrafficEnv with Live SUMO
    print("[3/5] Launching DOIMTrafficEnv in live SUMO...")
    sumo_cfg = config.simulation.sumo_config
    env = DOIMTrafficEnv(sumo_cfg)
    obs, info = env.reset(seed=42)
    
    assert obs.shape == (69,), f"Expected shape (69,), got {obs.shape}"
    assert np.all(np.isfinite(obs)), "Initial observation has non-finite values!"
    print(f"      Initial observation shape: {obs.shape} (finite: True)")

    for step in range(num_steps):
        # Sample legal random multi-discrete action: 5 binary decisions
        action = np.random.randint(0, 2, size=5, dtype=np.int64)
        obs, reward, terminated, truncated, info = env.step(action)
        
        raw_objs = info["raw_objectives"]
        norm_objs = info["normalized_objectives"]
        weights = info["doim_weights"]
        
        run.log_step(
            step=step,
            sim_time=info["time"],
            reward=reward,
            raw_objs=raw_objs,
            norm_objs=norm_objs,
            weights=weights,
            actions=action.tolist(),
            shield_interventions=0,
        )

    env.close()
    summary = run.close(status="COMPLETED")
    print(f"      Stepped {num_steps} simulation seconds successfully.")
    print(f"      Total Reward: {summary['total_reward']:.4f}, Mean Waiting: {summary['average_waiting']:.2f}s")

    # 4. Test FixedWeightTrafficEnv with Live SUMO
    print("[4/5] Testing FixedWeightTrafficEnv baseline with live SUMO...")
    fixed_env = FixedWeightTrafficEnv(sumo_cfg)
    f_obs, f_info = fixed_env.reset(seed=42)
    for _ in range(num_steps):
        action = np.zeros(5, dtype=np.int64)
        f_obs, f_rew, f_term, f_trunc, f_info = fixed_env.step(action)
    fixed_env.close()
    print(f"      Fixed baseline stepped {num_steps} simulation seconds successfully.")

    # 5. Automated Verification Checks
    print("[5/5] Running automated verification checks on outputs...")
    ok, msg = ExperimentVerifier.verify_finite_metrics(run.metrics_csv_path)
    assert ok, f"Finite metrics check failed: {msg}"
    print(f"      [OK] Finite metrics: {msg}")

    ok, msg = ExperimentVerifier.verify_json_csv_consistency(run.run_dir)
    assert ok, f"Consistency check failed: {msg}"
    print(f"      [OK] CSV/JSON consistency: {msg}")

    ok, msg = ExperimentVerifier.verify_simplex_weights(summary["average_weights"])
    assert ok, f"Simplex check failed: {msg}"
    print(f"      [OK] Adaptive weights simplex: {msg}")

    print("=" * 80)
    print("ALL SMOKE TEST CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_smoke_test(num_steps=30)
    sys.exit(0 if success else 1)
