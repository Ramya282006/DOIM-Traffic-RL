"""
DOIM-Traffic-RL: Baseline Parity and Operational Tests (Phase G)
Verifies:
  1. Actuated Controller enforces min-green (10s) and triggers gap-out when queues are clear.
  2. Max-Pressure Controller calculates phase pressures and outputs binary transitions.
  3. Parameter-Matched MLP-PPO forward pass matches expected multi-discrete logits and value dimensions.
  4. Interface consistency: all controllers interface with multi-discrete 5-junction action space {0, 1}^5.
"""

import numpy as np
import pytest
import torch

from baselines.actuated.actuated_controller import ActuatedTrafficController
from baselines.max_pressure.max_pressure_controller import MaxPressureTrafficController
from baselines.ppo.mlp_ppo import ParameterMatchedMLPNetwork


def test_actuated_controller_min_green_and_gap_out():
    """Actuated controller must honor min-green before evaluating gap-out."""
    tls_ids = ["A1", "B0", "B1", "B2", "C1"]
    ctrl = ActuatedTrafficController(
        controllable_tls_ids=tls_ids,
        min_green=10,
        max_green=60,
    )
    
    current_phases = {tid: 0 for tid in tls_ids}
    queues = {tid: 0.0 for tid in tls_ids}  # No vehicles -> eligible for gap-out
    
    # Step 0 to 8: min green has NOT elapsed, must return 0 (no transition)
    for _ in range(8):
        act = ctrl.select_actions(current_phases, queues)
        assert (act == 0).all()

    # Step 10: min green has elapsed (10s), queue is 0 -> gap out triggered -> return 1
    for _ in range(3):
        act = ctrl.select_actions(current_phases, queues)
    assert act[0] == 1  # Transition requested!


def test_max_pressure_controller_action_selection():
    """Max-pressure controller outputs valid 5-element actions in {0, 1}."""
    tls_ids = ["A1", "B0", "B1", "B2", "C1"]
    all_tls = ["A0", "A1", "A2", "B0", "B1", "B2", "C0", "C1", "C2"]
    ctrl = MaxPressureTrafficController(controllable_tls_ids=tls_ids)
    
    phases = {tid: 0 for tid in tls_ids}
    mock_feats = np.zeros((9, 5))
    mock_feats[1, 0] = 8.0   # A1 high queue
    mock_feats[1, 1] = 45.0  # A1 high wait
    
    actions = ctrl.select_actions(phases, mock_feats, all_tls)
    assert isinstance(actions, np.ndarray)
    assert actions.shape == (5,)
    assert set(actions).issubset({0, 1})
    assert actions[0] == 0  # Retain green for congested A1


def test_parameter_matched_mlp_network_architecture():
    """MLP network produces 5 action heads (logits for 2 actions each) and scalar value."""
    net = ParameterMatchedMLPNetwork(input_dim=70, hidden_dim=128, num_controllable_tls=5)
    
    x = torch.randn(4, 70)
    out = net(x)
    
    assert "action_logits" in out
    assert "value" in out
    assert len(out["action_logits"]) == 5
    for logits in out["action_logits"]:
        assert logits.shape == (4, 2)
    assert out["value"].shape == (4, 1)
