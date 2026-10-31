"""
DOIM-Traffic-RL: Full Integration & Regression Test Suite
Validates end-to-end execution of IA-MOPPO, Baselines, and Automated Verifiers.
"""

import os
from pathlib import Path
import pytest
import torch
import numpy as np

from common.config import load_experiment_config
from common.experiment import ExperimentRun
from common.verification import ExperimentVerifier
from environment.sumo_env import DOIMTrafficEnv
from training.interaction_aware_ppo import InteractionAwarePPOAgent, RolloutBuffer
from baselines.fixed_time.fixed_time_controller import FixedTimeController
from baselines.ppo.fixed_weight_ppo import FixedWeightPPOAgent
from baselines.ppo.state_adaptive_ppo import StateAdaptivePPOAgent


def test_ia_moppo_agent_step_and_update():
    """Validates that IA-MOPPO agent performs forward rollout and joint backprop without error."""
    config = load_experiment_config("configs/default.yaml")
    agent = InteractionAwarePPOAgent(config=config, device="cpu")
    buffer = RolloutBuffer(buffer_size=4)

    spatial_hist = torch.randn(1, 10, 64)
    obj_vec = torch.rand(1, 5)
    raw_state = torch.randn(1, 25)

    for i in range(4):
        action, value, log_prob, weights, int_mat = agent.act(
            spatial_history=spatial_hist,
            obj_vec=obj_vec,
            raw_state=raw_state,
        )
        assert len(action) == 5
        assert isinstance(value, float)
        assert np.isclose(weights.sum().item(), 1.0, atol=1e-4)

        buffer.add(
            spatial_hist=spatial_hist,
            obj_vec=obj_vec,
            int_mat=int_mat,
            raw_state=raw_state,
            action=action,
            reward=-0.5,
            value=value,
            log_prob=log_prob,
            done=(i == 3),
            next_obj_vec=torch.rand(1, 5),
            weights=weights,
        )

    loss_dict = agent.update(buffer, last_value=0.0)
    assert "policy_loss" in loss_dict
    assert "aux_loss" in loss_dict
    assert np.isfinite(loss_dict["policy_loss"])
    assert np.isfinite(loss_dict["aux_loss"])


def test_fixed_time_controller_logic():
    """Validates cyclic switching of Fixed-Time controller."""
    controller = FixedTimeController(cycle_time=5, num_controllable_tls=5)
    controller.reset()

    # Steps 1 to 4 should maintain phase (action = 0)
    for _ in range(4):
        act = controller.select_action()
        assert np.all(act == 0)

    # Step 5 triggers transition request (action = 1)
    act = controller.select_action()
    assert np.all(act == 1)


def test_fixed_weight_ppo_agent_step():
    """Validates forward and update step of FixedWeightPPOAgent."""
    config = load_experiment_config("configs/default.yaml")
    agent = FixedWeightPPOAgent(config=config, device="cpu")

    spatial_hist = torch.randn(1, 10, 64)
    raw_state = torch.randn(1, 25)
    action, val, lp = agent.act(spatial_hist, raw_state)
    assert len(action) == 5
    assert np.all(agent.fixed_weights == 0.2)


def test_state_adaptive_ppo_agent_step():
    """Validates state-adaptive weight generation on simplex."""
    config = load_experiment_config("configs/default.yaml")
    agent = StateAdaptivePPOAgent(config=config, device="cpu")

    spatial_hist = torch.randn(1, 10, 64)
    raw_state = torch.randn(1, 25)
    action, val, lp, weights = agent.act(spatial_hist, raw_state)
    assert len(action) == 5
    assert weights.shape == (1, 5)
    assert np.isclose(weights.sum().item(), 1.0, atol=1e-4)


def test_automated_verifier_on_synthetic_data(tmp_path):
    """Validates automated integrity verification logic."""
    # Simplex check
    valid_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    ok, msg = ExperimentVerifier.verify_simplex_weights(valid_weights)
    assert ok is True

    invalid_weights = np.array([0.5, 0.5, 0.5, 0.0, 0.0])
    ok, msg = ExperimentVerifier.verify_simplex_weights(invalid_weights)
    assert ok is False
