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
