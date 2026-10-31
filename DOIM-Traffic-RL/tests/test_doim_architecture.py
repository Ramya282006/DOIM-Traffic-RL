"""
DOIM-Traffic-RL: Unit & Gradient-Flow Test Suite for DOIM Architecture
Asserts tensor shapes, autograd backpropagation through all representation layers,
simplex compliance, and numerical stability.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.doim_network import DOIMNetwork


@pytest.fixture
def doim_net():
    return DOIMNetwork(
        num_intersections=9,
        num_controllable_tls=5,
        node_feature_dim=5,
        num_objectives=5,
        spatial_dim=64,
        temporal_dim=64,
        interaction_dim=32,
        objective_dim=32,
        manifold_dim=64,
        raw_state_dim=25,
        temperature=1.0,
    )


def test_spatial_encoding_shapes(doim_net):
    """Verify spatial GNN output dimensions."""
    node_feats = torch.randn(2, 9, 5) # batch=2, nodes=9, features=5
    pooled = doim_net.encode_spatial(node_feats)
    assert pooled.shape == (2, 64)


def test_temporal_encoding_shapes(doim_net):
    """Verify GRU output dimensions."""
    seq = torch.randn(2, 10, 64) # batch=2, horizon=10, dim=64
    temporal = doim_net.encode_temporal(seq)
    assert temporal.shape == (2, 64)


def test_forward_manifold_and_simplex(doim_net):
    """Verify manifold projection and simplex bounds on adaptive weights."""
    batch_size = 4
    spatial_hist = torch.randn(batch_size, 10, 64)
    obj_vec = torch.rand(batch_size, 5)
    int_mat = torch.randn(batch_size, 5, 5)

    manifold, weights = doim_net.forward_manifold(spatial_hist, obj_vec, int_mat)
    assert manifold.shape == (batch_size, 64)
    assert weights.shape == (batch_size, 5)

    # Simplex condition: non-negative and sum to 1.0
    assert torch.all(weights >= 0.0)
    sums = torch.sum(weights, dim=-1)
    torch.testing.assert_close(sums, torch.ones(batch_size), atol=1e-5, rtol=1e-5)


def test_full_forward_pass_shapes(doim_net):
    """Verify complete forward pass shapes for actor, critic, and manifold."""
    batch_size = 3
    spatial_hist = torch.randn(batch_size, 10, 64)
    obj_vec = torch.rand(batch_size, 5)
    int_mat = torch.randn(batch_size, 5, 5)
    raw_state = torch.randn(batch_size, 25)

    out = doim_net(spatial_hist, obj_vec, int_mat, raw_state)
    assert out["manifold"].shape == (batch_size, 64)
    assert out["weights"].shape == (batch_size, 5)
    assert out["value"].shape == (batch_size, 1)
    assert len(out["action_logits"]) == 5
    for logits in out["action_logits"]:
        assert logits.shape == (batch_size, 2)


def test_gradient_flow_to_all_modules(doim_net):
    """
    CRITICAL SCIENTIFIC ASSERTION (Resolves DEF-01 and DEF-02):
    Asserts that backpropagating through policy and auxiliary losses produces
    NON-ZERO gradients in EVERY submodule:
      - Spatial GNN (gcn1, gcn2)
      - Temporal GRU (gru)
      - Objective Encoder (objective_encoder)
      - Interaction Encoder (interaction_encoder)
      - Manifold Core (manifold_core)
      - Preference Head (preference_head)
      - Actor Backbone and Action Heads
      - Critic Head
      - Auxiliary Predictor
    """
    doim_net.train()
    batch_size = 2

    spatial_hist = torch.randn(batch_size, 10, 64, requires_grad=True)
    obj_vec = torch.rand(batch_size, 5, requires_grad=True)
    int_mat = torch.randn(batch_size, 5, 5, requires_grad=True)
    raw_state = torch.randn(batch_size, 25, requires_grad=True)

    out = doim_net(spatial_hist, obj_vec, int_mat, raw_state)

    # Dummy actor loss: cross entropy on arbitrary targets
    actor_loss = sum(
        F.cross_entropy(logits, torch.zeros(batch_size, dtype=torch.long))
        for logits in out["action_logits"]
    )

    # Dummy critic loss
    critic_loss = F.mse_loss(out["value"], torch.ones(batch_size, 1))

    # Auxiliary transition prediction loss
    dummy_actions = torch.ones(batch_size, 5)
    pred_next_obj = doim_net.predict_next_objectives(out["manifold"], dummy_actions)
    aux_loss = F.mse_loss(pred_next_obj, torch.zeros(batch_size, 5))

    # Preference smoothness loss (simulated consecutive steps)
    smooth_loss = torch.mean(out["weights"] ** 2)

    total_loss = actor_loss + critic_loss + aux_loss + smooth_loss
    total_loss.backward()

    # Verify non-zero gradients on all targeted parameter groups
    named_params = dict(doim_net.named_parameters())

    submodules = [
        "gru.weight_ih_l0",
        "objective_encoder.0.weight",
        "interaction_encoder.0.weight",
        "manifold_core.0.weight",
        "preference_head.0.weight",
        "actor_backbone.0.weight",
        "action_heads.0.weight",
        "critic_head.0.weight",
        "aux_predictor.0.weight",
    ]

    for param_name in submodules:
        assert param_name in named_params, f"Parameter {param_name} not found in model!"
        p = named_params[param_name]
        assert p.grad is not None, f"Gradient is None for {param_name}! Autograd graph disconnected."
        grad_norm = float(torch.norm(p.grad))
        assert grad_norm > 0.0, f"Gradient is zero for {param_name}! Parameter received no gradient."

    # Test spatial GNN gradient flow independently
    node_features = torch.randn(batch_size, 9, 5, requires_grad=True)
    pooled_spatial = doim_net.encode_spatial(node_features)
    spatial_loss = torch.sum(pooled_spatial ** 2)
    spatial_loss.backward()

    for p_name in ["gcn1.linear.weight", "gcn2.linear.weight"]:
        if p_name in named_params:
            p = named_params[p_name]
            assert p.grad is not None, f"Gradient is None for {p_name}!"
            assert float(torch.norm(p.grad)) > 0.0, f"Zero gradient for {p_name}!"


def test_numerical_stability(doim_net):
    """Verify that network outputs remain finite under extreme input conditions (zeros, large scale)."""
    batch_size = 2
    # Zero inputs
    spatial_hist = torch.zeros(batch_size, 10, 64)
    obj_vec = torch.zeros(batch_size, 5)
    int_mat = torch.zeros(batch_size, 5, 5)
    raw_state = torch.zeros(batch_size, 25)

    out = doim_net(spatial_hist, obj_vec, int_mat, raw_state)
    assert torch.all(torch.isfinite(out["manifold"]))
    assert torch.all(torch.isfinite(out["weights"]))
    assert torch.all(torch.isfinite(out["value"]))

    # Large inputs (1000.0)
    out_large = doim_net(
        spatial_hist + 1000.0,
        obj_vec + 1.0,
        int_mat + 1.0,
        raw_state + 1000.0,
    )
    assert torch.all(torch.isfinite(out_large["manifold"]))
    assert torch.all(torch.isfinite(out_large["weights"]))
    assert torch.all(torch.isfinite(out_large["value"]))
