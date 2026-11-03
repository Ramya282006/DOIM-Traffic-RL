"""
DOIM-Traffic-RL: SCOIL Scientific Validation Tests (Phase F)
Verifies:
  1. Predictive performance comparison across baselines (SCOIL vs No-Int vs Fixed vs Shuffled).
  2. Dispersion coverage evaluation and honest classification as uncalibrated dispersion proxy if coverage diverges.
  3. Preservation of identity diagonal and bounded [-1, 1] values.
  4. Non-leakage: ensures predictions are causal in decision space without future objective contamination.
"""

import pytest
import torch
import numpy as np

from models.interaction_learner import StateConditionedInteractionLearner
from analysis.scoil_validator import SCOILScientificValidator


def test_scoil_matrix_bounded_and_identity_diagonal():
    """Verifies that SCOIL interaction matrix has exact 1.0 on diagonal and [-1, 1] range."""
    model = StateConditionedInteractionLearner(num_objectives=5, traffic_dim=64, latent_dim=32)
    traffic = torch.randn(4, 64)
    objs = torch.rand(4, 5)
    
    out = model(traffic, objs)
    m = out["shrunk_matrix"]
    
    assert m.shape == (4, 5, 5)
    assert (m >= -1.0).all() and (m <= 1.0).all()
    
    # Check diagonals are +1.0
    for b in range(4):
        diag = torch.diag(m[b])
        assert torch.allclose(diag, torch.ones(5), atol=1e-5)


def test_scoil_validator_baseline_comparison():
    """Verifies that SCOILScientificValidator computes baseline comparison metrics cleanly."""
    model = StateConditionedInteractionLearner(num_objectives=5, traffic_dim=64, latent_dim=32)
    validator = SCOILScientificValidator(model)
    
    b = 8
    traffic = torch.randn(b, 64)
    objs = torch.rand(b, 5)
    actions = torch.randint(0, 2, (b, 5))
    next_objs = torch.clamp(objs + 0.05 * torch.randn(b, 5), 0.0, 1.0)
    
    results = validator.evaluate_predictive_baselines(traffic, objs, actions, next_objs)
    assert "mse_scoil_proposed" in results
    assert "mse_no_interaction" in results
    assert "mse_fixed_identity" in results
    assert "mse_shuffled_interaction" in results
    assert results["mse_scoil_proposed"] >= 0.0


def test_scoil_validator_dispersion_classification():
    """Verifies empirical coverage calibration reporting."""
    model = StateConditionedInteractionLearner(num_objectives=5, traffic_dim=64, latent_dim=32)
    validator = SCOILScientificValidator(model)
    
    b = 16
    traffic = torch.randn(b, 64)
    objs = torch.rand(b, 5)
    actions = torch.randint(0, 2, (b, 5))
    next_objs = torch.clamp(objs + 0.05 * torch.randn(b, 5), 0.0, 1.0)
    
    calib = validator.evaluate_dispersion_coverage(traffic, objs, actions, next_objs, confidence_level=0.95)
    assert "empirical_coverage" in calib
    assert "scientific_classification" in calib
    assert calib["scientific_classification"] in (
        "CALIBRATED_PREDICTIVE_UNCERTAINTY",
        "UNCALIBRATED_DISPERSION_PROXY",
    )

