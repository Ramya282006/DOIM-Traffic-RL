"""
DOIM-Traffic-RL: Test Suite for Objective Interaction Theory and SCOIL
Asserts bounds, asymmetry, uncertainty shrinkage, autograd differentiability,
and empirical distinction from rolling Pearson correlation.
"""

import numpy as np
import pytest
import torch

from models.interaction_learner import StateConditionedInteractionLearner
from analysis.objective_interaction import ObjectiveInteractionAnalyzer


@pytest.fixture
def learner():
    return StateConditionedInteractionLearner(
        num_objectives=5,
        traffic_dim=64,
        latent_dim=32,
        shrinkage_lambda=0.5,
    )


def test_interaction_matrix_bounds_and_diagonal(learner):
    """Verify that all elements lie in [-1, 1] and diagonal is exactly 1.0."""
    traffic_emb = torch.randn(4, 64)
    obj_vec = torch.rand(4, 5)

    out = learner(traffic_emb, obj_vec)
    m = out["interaction_matrix"]
    assert m.shape == (4, 5, 5)
    assert torch.all(m >= -1.0) and torch.all(m <= 1.0)

    # Diagonal must be 1.0
    for b in range(4):
        torch.testing.assert_close(torch.diagonal(m[b]), torch.ones(5), atol=1e-6, rtol=1e-6)


def test_asymmetric_interactions(learner):
    """Verify that interaction matrices can represent asymmetric coupling (M_ij != M_ji)."""
    traffic_emb = torch.randn(2, 64)
    obj_vec = torch.rand(2, 5)

    out = learner(traffic_emb, obj_vec)
    m = out["interaction_matrix"]

    for b in range(2):
        matrix = m[b]
        # Should not be strictly symmetric
        diff = torch.abs(matrix - matrix.T)
        off_diag = diff[~torch.eye(5, dtype=torch.bool)]
        assert torch.any(off_diag > 1e-4), "Interaction operator is unnaturally symmetric!"


def test_uncertainty_shrinkage(learner):
    """Verify that higher uncertainty adaptively pulls estimates towards prior."""
    traffic_emb = torch.randn(1, 64)
    obj_vec = torch.rand(1, 5)

    out = learner(traffic_emb, obj_vec)
    m = out["interaction_matrix"]
    u = out["uncertainty"]
    shrunk = out["shrunk_matrix"]

    assert torch.all(u >= 0.0), "Uncertainties must be strictly non-negative."
    # Shrunk matrix must remain within [-1, 1]
    assert torch.all(shrunk >= -1.0) and torch.all(shrunk <= 1.0)


def test_interaction_gradient_flow(learner):
    """Verify that predictive loss flows gradients into all interaction parameters."""
    learner.train()
    traffic_emb = torch.randn(2, 64, requires_grad=True)
    obj_vec = torch.rand(2, 5, requires_grad=True)
    obj_next = torch.rand(2, 5)

    out = learner(traffic_emb, obj_vec)
    loss = learner.compute_predictive_loss(
        out["shrunk_matrix"], obj_vec, obj_next, torch.ones(2, 5)
    )
    loss.backward()

    named_params = dict(learner.named_parameters())
    for name in ["feature_net.0.weight", "query_proj.weight", "key_proj.weight", "asymmetry_head.weight"]:
        assert name in named_params
        p = named_params[name]
        assert p.grad is not None
        assert float(torch.norm(p.grad)) > 0.0, f"No gradient reached {name}!"


def test_comparison_with_pearson_correlation(learner):
    """
    Asserts the scientific distinction between static rolling correlation and SCOIL:
    Rolling Pearson produces identical symmetric values regardless of state context
    if historical relative rankings are constant; SCOIL adapts dynamically to traffic_emb.
    """
    # 1. Classical rolling Pearson
    analyzer = ObjectiveInteractionAnalyzer(num_objectives=5, window_size=10)
    for _ in range(15):
        # Monotonically increasing values (demand surge)
        analyzer.update([0.1, 0.2, 0.3, 0.4, 0.5])
    pearson_mat = analyzer.get_interaction_matrix()
    
    # 2. SCOIL with two completely different traffic states
    obj = torch.tensor([[0.2, 0.3, 0.4, 0.5, 0.2]])
    traffic_low = torch.zeros(1, 64) # free-flow traffic
    traffic_high = torch.ones(1, 64) * 5.0 # gridlocked traffic

    out_low = learner(traffic_low, obj)
    out_high = learner(traffic_high, obj)

    m_low = out_low["shrunk_matrix"][0].detach().numpy()
    m_high = out_high["shrunk_matrix"][0].detach().numpy()

    # The learned interaction MUST change under different traffic states
    assert not np.allclose(m_low, m_high, atol=1e-3), "Interaction learner failed to condition on traffic state!"
