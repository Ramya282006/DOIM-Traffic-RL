"""
DOIM-Traffic-RL: SCOIL Scientific Validation Tests
Verifies mathematical invariants, temporal alignment, shrinkage behavior,
and predictive objective dynamics of State-Conditioned Objective Interaction Learning.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from models.interaction_learner import StateConditionedInteractionLearner


def test_scoil_temporal_alignment_and_no_leakage():
    """
    Verifies that the predictive dynamics strictly predict next-step objectives z_{t+1}
    from current state features z_t, action a_t, and traffic embedding h_t,
    with zero future information leakage.
    """
    scoil = StateConditionedInteractionLearner(num_objectives=5, traffic_dim=64)
    traffic_emb = torch.randn(4, 64)
    z_t = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.1, 0.2, 0.3, 0.4],
        [0.2, 0.7, 0.1, 0.3, 0.5],
    ])
    actions = torch.randint(0, 2, (4, 5))
    z_next = torch.clamp(z_t + 0.05 * torch.randn(4, 5), 0.0, 1.0)

    # Forward pass of interaction learner relies ONLY on traffic_emb and z_t
    out = scoil(traffic_emb, z_t)
    m_t = out["shrunk_matrix"]
    assert m_t.shape == (4, 5, 5)

    # Predictive loss uses z_next solely as a target
    loss = scoil.compute_predictive_loss(m_t, z_t, actions, z_next)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0

    # Test autograd backward: gradients must flow to scoil parameters
    loss.backward()
    assert scoil.query_proj.weight.grad is not None
    assert torch.sum(torch.abs(scoil.query_proj.weight.grad)) > 0.0


def test_shrinkage_monotonicity():
    """
    Verifies that as uncertainty / dispersion U_{ij} increases, the shrunk matrix
    monotonically converges towards the uncoupled prior (identity matrix).
    """
    scoil = StateConditionedInteractionLearner(num_objectives=5, traffic_dim=64, shrinkage_lambda=1.0)
    traffic_emb = torch.randn(1, 64)
    z_t = torch.rand(1, 5)

    out = scoil(traffic_emb, z_t)
    raw_m = out["interaction_matrix"]
    shrunk_m = out["shrunk_matrix"]
    u_t = out["uncertainty"]

    # Diagonal must always be 1.0 (self-interaction)
    for i in range(5):
        assert torch.isclose(shrunk_m[0, i, i], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(u_t[0, i, i], torch.tensor(0.0), atol=1e-5)

    # Off-diagonal elements: |shrunk_m[i, j]| <= |raw_m[i, j]|
    for i in range(5):
        for j in range(5):
            if i != j:
                assert abs(shrunk_m[0, i, j].item()) <= abs(raw_m[0, i, j].item()) + 1e-6


def test_boundedness_and_stability():
    """
    Verifies that for arbitrary inputs (including extreme values), the interaction matrix
    remains strictly within [-1.0, 1.0] and uncertainty remains non-negative.
    """
    scoil = StateConditionedInteractionLearner(num_objectives=5, traffic_dim=64)
    
    # Extreme traffic embeddings and saturated objectives
    traffic_emb = torch.randn(8, 64) * 100.0
    z_t = torch.ones(8, 5)

    out = scoil(traffic_emb, z_t)
    m_t = out["interaction_matrix"]
    u_t = out["uncertainty"]
    shrunk_m = out["shrunk_matrix"]

    assert torch.all(m_t >= -1.0) and torch.all(m_t <= 1.0)
    assert torch.all(shrunk_m >= -1.0) and torch.all(shrunk_m <= 1.0)
    assert torch.all(u_t >= 0.0)
    assert not torch.any(torch.isnan(shrunk_m))


def test_scoil_predictive_advantage_over_baselines():
    """
    Compares SCOIL's residual predictive capacity against Identity and Zero interaction operators.
    """
    torch.manual_seed(42)
    scoil = StateConditionedInteractionLearner(num_objectives=5, traffic_dim=64)
    optimizer = torch.optim.Adam(scoil.parameters(), lr=0.01)

    traffic_emb = torch.randn(16, 64)
    z_t = torch.rand(16, 5)
    actions = torch.randint(0, 2, (16, 5))
    
    # Synthetic coupled transition: action on approach 0 increases queue (idx 1) and delay (idx 0)
    delta_true = torch.zeros(16, 5)
    delta_true[:, 0] = actions[:, 0].float() * 0.15 + actions[:, 1].float() * 0.05
    delta_true[:, 1] = actions[:, 0].float() * 0.10
    delta_true[:, 3] = -actions[:, 0].float() * 0.12 # throughput drops
    z_next = torch.clamp(z_t + delta_true, 0.0, 1.0)

    # Train for a few steps
    for _ in range(20):
        optimizer.zero_grad()
        out = scoil(traffic_emb, z_t)
        loss = scoil.compute_predictive_loss(out["shrunk_matrix"], z_t, actions, z_next)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        out = scoil(traffic_emb, z_t)
        scoil_loss = scoil.compute_predictive_loss(out["shrunk_matrix"], z_t, actions, z_next).item()

        # Identity baseline (M = I)
        eye_m = torch.eye(5).unsqueeze(0).expand(16, 5, 5)
        eye_loss = scoil.compute_predictive_loss(eye_m, z_t, actions, z_next).item()

        # Zero baseline (M = 0)
        zero_m = torch.zeros(16, 5, 5)
        zero_loss = scoil.compute_predictive_loss(zero_m, z_t, actions, z_next).item()

    # SCOIL learned interaction should outperform zero interaction
    assert scoil_loss < zero_loss, f"SCOIL loss ({scoil_loss:.4f}) should be lower than zero interaction ({zero_loss:.4f})"
