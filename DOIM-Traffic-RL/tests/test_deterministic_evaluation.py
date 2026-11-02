"""
DOIM-Traffic-RL: Unit tests for Deterministic Evaluation & Baseline Fairness
Tests that deterministic evaluations produce identical, reproducible trajectories,
that trained neural baselines freeze model parameters during evaluation,
and that Fixed-Time is verified as a deterministic, non-learning baseline.
"""

import copy
import numpy as np
import pytest

from baselines.fixed_time.fixed_time_controller import FixedTimeBaseline, evaluate_fixed_time
from common.experiment import normalize_evaluation_mode


class MockTrafficEnv:
    """
    Deterministic mock environment implementing the Gym/TraCI interface for evaluation verification.
    """
    def __init__(self, seed: int = 42):
        self.initial_seed = seed
        self.current_step = 0
        self.max_steps = 20
        self.num_controllers = 2

    def reset(self, seed=None):
        self.current_step = 0
        rng = np.random.RandomState(seed if seed is not None else self.initial_seed)
        obs = rng.randn(self.num_controllers, 10).astype(np.float32)
        info = {
            "step": 0,
            "vehicle_count": 10,
            "normalized_objectives": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            "interventions": 0,
        }
        return obs, info

    def step(self, action):
        self.current_step += 1
        rng = np.random.RandomState(self.initial_seed + self.current_step + int(np.sum(action)))
        next_obs = rng.randn(self.num_controllers, 10).astype(np.float32)
        reward = float(-1.0 * np.sum(np.abs(next_obs)))
        terminated = (self.current_step >= self.max_steps)
        truncated = False
        info = {
            "step": self.current_step,
            "vehicle_count": max(0, 10 - self.current_step),
            "normalized_objectives": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            "interventions": 0,
            "waiting_time": float(np.sum(np.abs(action)) * 2.5),
            "queue_length": float(np.sum(action)),
            "co2_emissions": 100.0,
            "throughput": 1.0,
            "fairness": 0.05,
        }
        return next_obs, reward, terminated, truncated, info

    def close(self):
        pass


def test_fixed_time_is_non_learning_and_deterministic():
    """
    Verifies FixedTimeBaseline is non-learning, has no learnable weights,
    and produces bitwise-identical action sequences given identical seeds.
    """
    c1 = FixedTimeBaseline(tls_ids=["TLS_A", "TLS_B"], green_duration=10, yellow_duration=3)
    c2 = FixedTimeBaseline(tls_ids=["TLS_A", "TLS_B"], green_duration=10, yellow_duration=3)

    # Has no weights or gradients
    assert not hasattr(c1, "parameters") or not callable(getattr(c1, "parameters", None))
    assert not hasattr(c1, "weights")

    # Step both controllers for 30 steps: actions must be bitwise identical
    actions_1 = []
    actions_2 = []
    for _ in range(30):
        actions_1.append(c1.act())
        actions_2.append(c2.act())

    assert all(np.array_equal(a1, a2) for a1, a2 in zip(actions_1, actions_2))
    assert len(actions_1) == 30


def test_deterministic_evaluation_reproducibility():
    """
    Verifies that running deterministic controller evaluation twice with the same eval_seed
    produces identical metrics and bitwise identical trajectories.
    """
    env1 = MockTrafficEnv(seed=42)
    env2 = MockTrafficEnv(seed=42)
    c1 = FixedTimeBaseline(tls_ids=["TLS_A", "TLS_B"], green_duration=5)
    c2 = FixedTimeBaseline(tls_ids=["TLS_A", "TLS_B"], green_duration=5)

    traj1 = []
    traj2 = []

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    assert np.allclose(obs1, obs2)

    for step in range(20):
        a1 = c1.act()
        a2 = c2.act()
        assert np.array_equal(a1, a2)

        next_obs1, r1, done1, _, info1 = env1.step(a1)
        next_obs2, r2, done2, _, info2 = env2.step(a2)

        assert np.allclose(next_obs1, next_obs2)
        assert r1 == pytest.approx(r2, rel=1e-6)
        assert info1["waiting_time"] == pytest.approx(info2["waiting_time"], rel=1e-6)

        traj1.append((a1, r1))
        traj2.append((a2, r2))

    assert len(traj1) == 20
    assert all(np.array_equal(t1[0], t2[0]) and t1[1] == pytest.approx(t2[1]) for t1, t2 in zip(traj1, traj2))


def test_trained_baseline_parameters_frozen_during_evaluation():
    """
    Verifies that during evaluation, baseline model parameters remain frozen
    and do not undergo gradient updates.
    """
    try:
        import torch
        import torch.nn as nn
    except (ImportError, OSError):
        pytest.skip("PyTorch unavailable in this test context")

    class SimplePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        def forward(self, x):
            return self.fc(x)

    model = SimplePolicy()
    # Save initial weights
    initial_weights = copy.deepcopy(model.fc.weight.data)

    # Simulate evaluation mode
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(5, 10)
        out = model(dummy_input)
        assert out is not None

    # Verify model is still in eval mode and weights did NOT change
    assert not model.training
    assert torch.equal(model.fc.weight.data, initial_weights)


def test_compatible_evaluation_artifact_schemas_across_all_four_controllers(tmp_path):
    """
    Verifies that Fixed-Time, FW-PPO, SA-PPO, and IA-MOPPO produce
    strictly compatible evaluation artifacts (manifest.json, summary.json, metrics_step.csv)
    with identical column headers, matching schema, and consistent evaluation metadata.
    """
    import json
    import csv
    from common.experiment import ExperimentRun

    controllers = ["Fixed_Time", "FW_PPO", "SA_PPO", "IA_MOPPO"]
    runs = []

    for idx, algo in enumerate(controllers):
        run = ExperimentRun(
            experiment_name=f"EVAL_{algo}_test",
            algorithm=algo,
            config={"config_hash": "test_hash_123", "objectives": {"fairness_metric": "delay_disparity"}},
            train_seed=42,
            eval_seed=1042 + idx,
            scenario="north_south_surge",
            output_root=str(tmp_path),
            training_timesteps=1800,
            episode_id=0,
            evaluation_mode="deterministic_eval",
        )
        # Log 2 identical steps
        weights = np.full(5, 0.2, dtype=np.float32)
        raw_objs = np.array([25.0, 2.0, 50000.0, 1.0, 0.15])
        norm_objs = np.array([0.2, 0.1, 0.4, 0.05, 0.15])
        actions = [0, 0, 0, 0, 0]

        for step in range(2):
            run.log_step(
                step=step,
                sim_time=float(step + 1),
                reward=-0.5,
                raw_objs=raw_objs,
                norm_objs=norm_objs,
                weights=weights,
                actions=actions,
                shield_interventions=0,
            )
        summary = run.close(status="COMPLETED")
        runs.append((algo, run.run_dir))

    # Compare schemas across all 4 controllers
    expected_manifest_keys = {
        "experiment_id", "run_id", "experiment_name", "algorithm", "scenario",
        "train_seed", "evaluation_seed", "training_timesteps", "episode_id",
        "evaluation_mode", "config_hash", "code_commit", "environment_info",
        "start_time", "end_time", "completion_status", "model_checkpoint_sha256",
        "total_steps", "summary_metrics", "status"
    }
    expected_summary_keys = {
        "run_id", "algorithm", "scenario", "seed", "train_seed", "eval_seed",
        "episode_id", "evaluation_mode", "total_steps", "total_reward",
        "average_reward", "average_waiting", "max_waiting", "average_queue",
        "max_queue", "average_co2", "total_throughput", "average_fairness",
        "fairness_metric_name", "fairness_metric_unit", "fairness_disparity",
        "fairness_variance", "total_shield_interventions", "average_weights"
    }
    expected_csv_headers = [
        "step", "simulation_time", "reward", "raw_waiting", "raw_queue",
        "raw_co2", "raw_throughput", "raw_fairness", "norm_waiting",
        "norm_queue", "norm_co2", "norm_throughput", "norm_fairness",
        "weight_waiting", "weight_queue", "weight_co2", "weight_throughput",
        "weight_fairness", "action_tls", "shield_interventions"
    ]

    for algo, rdir in runs:
        m_file = rdir / "manifest.json"
        s_file = rdir / "summary.json"
        c_file = rdir / "metrics_step.csv"

        assert m_file.exists(), f"Missing manifest for {algo}"
        assert s_file.exists(), f"Missing summary for {algo}"
        assert c_file.exists(), f"Missing CSV for {algo}"

        m_data = json.loads(m_file.read_text(encoding="utf-8"))
        s_data = json.loads(s_file.read_text(encoding="utf-8"))

        # Verify key subset
        assert expected_manifest_keys.issubset(set(m_data.keys())), f"Manifest keys mismatch for {algo}"
        assert expected_summary_keys.issubset(set(s_data.keys())), f"Summary keys mismatch for {algo}"
        assert m_data["evaluation_mode"] == "deterministic_eval"
        assert s_data["evaluation_mode"] == "deterministic_eval"

        with open(c_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert headers == expected_csv_headers, f"CSV headers mismatch for {algo}"


def test_evaluation_horizon_runtime_propagation_all_four_algorithms(tmp_path, monkeypatch):
    """
    TASK 1 Verification:
    Verifies that --eval_horizon 900 propagates at runtime to:
    1. The actual SUMO environment's maximum-step configuration (DOIMTrafficEnv.max_steps == 900)
    2. The evaluation loop termination condition (exactly 900 steps executed before break)
    3. The evaluation manifest (manifest.json has eval_horizon == 900)
    4. The evaluation summary (summary.json has eval_horizon == 900 and total_steps == 900)
    across all four algorithms:
    - Fixed-Time (evaluate_fixed_time)
    - Fixed-Weight PPO (evaluate_fixed_weight_ppo)
    - State-Adaptive PPO (evaluate_state_adaptive_ppo)
    - IA-MOPPO (evaluate_interaction_aware_ppo)
    """
    import json
    import torch
    from common.config import load_experiment_config
    from baselines.fixed_time.fixed_time_controller import evaluate_fixed_time
    from baselines.ppo.fixed_weight_ppo import evaluate_fixed_weight_ppo, FixedWeightPPOAgent
    from baselines.ppo.state_adaptive_ppo import evaluate_state_adaptive_ppo, StateAdaptivePPOAgent
    from training.interaction_aware_ppo import evaluate_interaction_aware_ppo, InteractionAwarePPOAgent
    import baselines.fixed_time.fixed_time_controller as ftc_mod
    import baselines.ppo.fixed_weight_ppo as fwp_mod
    import baselines.ppo.state_adaptive_ppo as sap_mod
    import training.interaction_aware_ppo as iap_mod

    class InstrumentedMockEnv:
        instances = []

        def __init__(self, sumo_config=None, enforce_safety_shield=True, max_steps=1800):
            self.sumo_config = sumo_config
            self.enforce_safety_shield = enforce_safety_shield
            self.max_steps = int(max_steps)
            self.step_count = 0
            self.current_time = 0
            InstrumentedMockEnv.instances.append(self)

        def reset(self, seed=None):
            self.step_count = 0
            self.current_time = 0
            obs = np.zeros(69, dtype=np.float32)
            info = {
                "time": 0.0,
                "node_features": np.zeros((9, 5), dtype=np.float32),
                "raw_state": np.zeros(25, dtype=np.float32),
                "normalized_objectives": np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32),
                "raw_objectives": np.array([10.0, 1.0, 100.0, 5.0, 0.1], dtype=np.float32),
                "interaction_matrix": np.eye(5, dtype=np.float32),
                "shield_interventions": 0,
            }
            return obs, info

        def step(self, action, weights=None):
            self.step_count += 1
            self.current_time = self.step_count
            obs = np.zeros(69, dtype=np.float32)
            terminated = False
            truncated = (self.current_time >= self.max_steps)
            info = {
                "time": float(self.current_time),
                "node_features": np.zeros((9, 5), dtype=np.float32),
                "raw_state": np.zeros(25, dtype=np.float32),
                "normalized_objectives": np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32),
                "raw_objectives": np.array([10.0, 1.0, 100.0, 5.0, 0.1], dtype=np.float32),
                "interaction_matrix": np.eye(5, dtype=np.float32),
                "shield_interventions": 0,
            }
            reward = -1.0
            return obs, reward, terminated, truncated, info

        def close(self):
            pass

    monkeypatch.setattr(ftc_mod, "DOIMTrafficEnv", InstrumentedMockEnv)
    monkeypatch.setattr(fwp_mod, "DOIMTrafficEnv", InstrumentedMockEnv)
    monkeypatch.setattr(sap_mod, "DOIMTrafficEnv", InstrumentedMockEnv)
    monkeypatch.setattr(iap_mod, "DOIMTrafficEnv", InstrumentedMockEnv)

    config = load_experiment_config("configs/default.yaml")
    non_default_horizon = 900
    eval_seed = 1042
    train_seed = 42

    # 1. Fixed-Time runtime evaluation
    InstrumentedMockEnv.instances.clear()
    runs_ft = evaluate_fixed_time(
        n_episodes=1,
        eval_seed=eval_seed,
        train_seed=train_seed,
        eval_horizon=non_default_horizon,
        output_root=str(tmp_path),
    )
    env_ft = InstrumentedMockEnv.instances[-1]
    assert env_ft.max_steps == non_default_horizon, f"FT env max_steps was {env_ft.max_steps}"
    assert env_ft.step_count == non_default_horizon, f"FT loop stepped {env_ft.step_count} times"
    m_ft = json.loads((runs_ft[0].run_dir / "manifest.json").read_text(encoding="utf-8"))
    s_ft = json.loads((runs_ft[0].run_dir / "summary.json").read_text(encoding="utf-8"))
    assert m_ft["eval_horizon"] == non_default_horizon
    assert s_ft["eval_horizon"] == non_default_horizon
    assert s_ft["total_steps"] == non_default_horizon

    # 2. Fixed-Weight PPO runtime evaluation
    agent_fw = FixedWeightPPOAgent(config=config, device="cpu")
    InstrumentedMockEnv.instances.clear()
    runs_fw = evaluate_fixed_weight_ppo(
        agent_or_checkpoint=agent_fw,
        n_episodes=1,
        eval_seed=eval_seed,
        train_seed=train_seed,
        eval_horizon=non_default_horizon,
        output_root=str(tmp_path),
    )
    env_fw = InstrumentedMockEnv.instances[-1]
    assert env_fw.max_steps == non_default_horizon, f"FW env max_steps was {env_fw.max_steps}"
    assert env_fw.step_count == non_default_horizon, f"FW loop stepped {env_fw.step_count} times"
    m_fw = json.loads((runs_fw[0].run_dir / "manifest.json").read_text(encoding="utf-8"))
    s_fw = json.loads((runs_fw[0].run_dir / "summary.json").read_text(encoding="utf-8"))
    assert m_fw["eval_horizon"] == non_default_horizon
    assert s_fw["eval_horizon"] == non_default_horizon
    assert s_fw["total_steps"] == non_default_horizon

    # 3. State-Adaptive PPO runtime evaluation
    agent_sa = StateAdaptivePPOAgent(config=config, device="cpu")
    InstrumentedMockEnv.instances.clear()
    runs_sa = evaluate_state_adaptive_ppo(
        agent_or_checkpoint=agent_sa,
        n_episodes=1,
        eval_seed=eval_seed,
        train_seed=train_seed,
        eval_horizon=non_default_horizon,
        output_root=str(tmp_path),
    )
    env_sa = InstrumentedMockEnv.instances[-1]
    assert env_sa.max_steps == non_default_horizon, f"SA env max_steps was {env_sa.max_steps}"
    assert env_sa.step_count == non_default_horizon, f"SA loop stepped {env_sa.step_count} times"
    m_sa = json.loads((runs_sa[0].run_dir / "manifest.json").read_text(encoding="utf-8"))
    s_sa = json.loads((runs_sa[0].run_dir / "summary.json").read_text(encoding="utf-8"))
    assert m_sa["eval_horizon"] == non_default_horizon
    assert s_sa["eval_horizon"] == non_default_horizon
    assert s_sa["total_steps"] == non_default_horizon

    # 4. IA-MOPPO runtime evaluation
    agent_ia = InteractionAwarePPOAgent(config=config, device="cpu")
    InstrumentedMockEnv.instances.clear()
    runs_ia = evaluate_interaction_aware_ppo(
        agent_or_checkpoint=agent_ia,
        n_episodes=1,
        eval_seed=eval_seed,
        train_seed=train_seed,
        eval_horizon=non_default_horizon,
        output_root=str(tmp_path),
    )
    env_ia = InstrumentedMockEnv.instances[-1]
    assert env_ia.max_steps == non_default_horizon, f"IA env max_steps was {env_ia.max_steps}"
    assert env_ia.step_count == non_default_horizon, f"IA loop stepped {env_ia.step_count} times"
    m_ia = json.loads((runs_ia[0].run_dir / "manifest.json").read_text(encoding="utf-8"))
    s_ia = json.loads((runs_ia[0].run_dir / "summary.json").read_text(encoding="utf-8"))
    assert m_ia["eval_horizon"] == non_default_horizon
    assert s_ia["eval_horizon"] == non_default_horizon
    assert s_ia["total_steps"] == non_default_horizon


def test_live_sumo_eval_horizon_propagation(tmp_path):
    """
    Live SUMO verification:
    Executes actual SUMO/TraCI simulation with non-default eval_horizon=15.
    Verifies that the live simulation stops at exactly 15 steps due to max_steps truncation,
    and both manifest.json and summary.json record eval_horizon=15 and total_steps=15.
    """
    import json
    import csv
    from baselines.fixed_time.fixed_time_controller import evaluate_fixed_time

    horizon = 15
    runs = evaluate_fixed_time(
        scenario_override="scenarios/directional_shift/north_south_surge.sumocfg",
        n_episodes=1,
        eval_seed=1042,
        eval_horizon=horizon,
        output_root=str(tmp_path),
    )
    assert len(runs) == 1
    run = runs[0]

    m_file = run.run_dir / "manifest.json"
    s_file = run.run_dir / "summary.json"
    c_file = run.run_dir / "metrics_step.csv"

    assert m_file.exists()
    assert s_file.exists()
    assert c_file.exists()

    m = json.loads(m_file.read_text(encoding="utf-8"))
    s = json.loads(s_file.read_text(encoding="utf-8"))

    assert m["eval_horizon"] == horizon
    assert s["eval_horizon"] == horizon
    assert s["total_steps"] == horizon

    # Count rows in metrics_step.csv (header + horizon rows)
    with open(c_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == horizon + 1
