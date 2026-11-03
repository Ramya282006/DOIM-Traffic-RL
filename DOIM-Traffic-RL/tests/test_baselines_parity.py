"""
DOIM-Traffic-RL: Baseline Parity and Operational Tests (Phase 19)
Verifies:
  1. Actuated Controller enforces min-green (10s) and triggers gap-out when passage timer expires (3s).
  2. Actuated Controller resets passage timer upon vehicle presence.
  3. Max-Pressure Controller calculates phase pressures and outputs binary transitions.
  4. Parameter-Matched MLP-PPO forward pass matches expected multi-discrete logits and value dimensions.
  5. Interface consistency: all controllers interface with multi-discrete 5-junction action space {0, 1}^5.
"""

import numpy as np
import pytest
import torch

from baselines.actuated.actuated_controller import ActuatedTrafficController
from baselines.max_pressure.max_pressure_controller import MaxPressureTrafficController
from baselines.ppo.mlp_ppo import ParameterMatchedMLPNetwork


def test_actuated_controller_min_green_and_gap_out():
    """Actuated controller must honor min-green and NEMA passage timer gap-out."""
    tls_ids = ["A1", "B0", "B1", "B2", "C1"]
    ctrl = ActuatedTrafficController(
        controllable_tls_ids=tls_ids,
        min_green=10,
        max_green=60,
        gap_threshold_sec=3.0,
    )
    
    current_phases = {tid: 0 for tid in tls_ids}
    queues_zero = {tid: 0.0 for tid in tls_ids}
    
    # Steps 0 to 8 (9 steps): min green has NOT elapsed (t_elapsed < 10)
    for _ in range(9):
        act = ctrl.select_actions(current_phases, queues_zero)
        assert (act == 0).all(), "Must not transition during minimum green"

    # Step 9: t_elapsed = 10, gap_timer = 1.0s < 3.0s -> holds green (0)
    act = ctrl.select_actions(current_phases, queues_zero)
    assert act[0] == 0

    # Step 10: t_elapsed = 11, gap_timer = 2.0s < 3.0s -> holds green (0)
    act = ctrl.select_actions(current_phases, queues_zero)
    assert act[0] == 0

    # Step 11: t_elapsed = 12, gap_timer = 3.0s >= 3.0s -> gap-out triggers (1)!
    act = ctrl.select_actions(current_phases, queues_zero)
    assert act[0] == 1, "Must trigger gap-out when passage timer reaches gap_threshold_sec"


def test_actuated_controller_vehicle_extension():
    """Vehicle detection resets passage timer and extends green."""
    tls_ids = ["A1", "B0", "B1", "B2", "C1"]
    ctrl = ActuatedTrafficController(
        controllable_tls_ids=tls_ids,
        min_green=10,
        max_green=60,
        gap_threshold_sec=3.0,
        unit_extension_sec=2.0,
    )
    current_phases = {tid: 0 for tid in tls_ids}
    queues_zero = {tid: 0.0 for tid in tls_ids}
    queues_active = {tid: 3.0 for tid in tls_ids}
    
    # Advance past min-green (10 steps)
    for _ in range(10):
        ctrl.select_actions(current_phases, queues_zero)
        
    # Vehicle arrives at step 10: resets gap timer and extends
    act = ctrl.select_actions(current_phases, queues_active)
    assert act[0] == 0
    assert ctrl.gap_timer["A1"] == 0.0
    assert ctrl.extension_remaining["A1"] == 2.0


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
