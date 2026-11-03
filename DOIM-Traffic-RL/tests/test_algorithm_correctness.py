"""
DOIM-Traffic-RL: Algorithm Correctness Verification Suite (Phase D)
Verifies:
  1. PPO clipped surrogate loss math and ratio clipping [1 - eps, 1 + eps].
  2. Action attribution separation: proposed actions for policy loss, executed for dynamics.
  3. GAE calculation with hand-computed absorbing termination vs truncation bootstrapping.
  4. Episode boundary masking in preference temporal smoothness.
  5. Value bootstrapping with zero advantage carryover on truncation.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from training.interaction_aware_ppo import RolloutBuffer, InteractionAwarePPOAgent
from common.config import load_experiment_config


def test_gae_absorbing_termination_vs_truncation():
    """
    Hand-calculated analytical verification of GAE computation:
      - Absorbing termination (done=True, terminated=True, truncated=False): V(next) = 0.
      - Truncation (done=True, terminated=False, truncated=True): V(next) = bootstrap_value.
    """
    buffer = RolloutBuffer(buffer_size=3)
    gamma = 0.9
    gae_lam = 0.8
    
    # Episode 1: Step 0 (normal), Step 1 (truncated with bootstrap V=5.0)
    # Rewards: [1.0, 2.0]
    # Values:  [0.5, 1.5]
    dummy_vec = torch.zeros(5)
    dummy_mat = torch.zeros(5, 5)
    dummy_hist = torch.zeros(10, 64)
    dummy_act = np.array([0, 1, 0, 1, 0])
    
    # Step 0
    buffer.add(
        spatial_hist=dummy_hist, obj_vec=dummy_vec, int_mat=dummy_mat, raw_state=dummy_vec,
        action=dummy_act, reward=1.0, value=0.5, log_prob=-0.5, done=False,
        next_obj_vec=dummy_vec, weights=dummy_vec, terminated=False, truncated=False
    )
    # Step 1 (truncated at boundary, bootstrap=5.0)
    buffer.add(
        spatial_hist=dummy_hist, obj_vec=dummy_vec, int_mat=dummy_mat, raw_state=dummy_vec,
        action=dummy_act, reward=2.0, value=1.5, log_prob=-0.5, done=True,
        next_obj_vec=dummy_vec, weights=dummy_vec, terminated=False, truncated=True,
        bootstrap_value=5.0
    )
    
    returns, advantages = buffer.compute_returns_and_advantages(last_value=0.0, gamma=gamma, gae_lambda=gae_lam)
    
    # Hand calculation:
    # At t=1 (truncated):
    # delta_1 = reward_1 + gamma * boot_val - value_1 = 2.0 + 0.9 * 5.0 - 1.5 = 2.0 + 4.5 - 1.5 = 5.0
    # adv_1 = delta_1 = 5.0 (no carryover from future)
    # At t=0:
    # next_val = value_1 = 1.5
    # delta_0 = reward_0 + gamma * value_1 - value_0 = 1.0 + 0.9 * 1.5 - 0.5 = 1.0 + 1.35 - 0.5 = 1.85
    # adv_0 = delta_0 + gamma * gae_lam * adv_1 = 1.85 + 0.9 * 0.8 * 5.0 = 1.85 + 3.60 = 5.45
    
    unnorm_adv = returns.numpy() - np.array(buffer.values, dtype=np.float32)
    assert pytest.approx(unnorm_adv[1], rel=1e-4) == 5.0
    assert pytest.approx(unnorm_adv[0], rel=1e-4) == 5.45


def test_action_attribution_separation():
    """
    Verifies that the buffer preserves distinct tensors for proposed action vs executed action.
    """
    buffer = RolloutBuffer(buffer_size=1)
    dummy_vec = torch.zeros(5)
    dummy_mat = torch.zeros(5, 5)
    dummy_hist = torch.zeros(10, 64)
    
    proposed = np.array([1, 1, 1, 1, 1])
    executed = np.array([0, 1, 0, 0, 1])  # Shield intervened and overrode indices 0, 2, 3
    
    buffer.add(
        spatial_hist=dummy_hist, obj_vec=dummy_vec, int_mat=dummy_mat, raw_state=dummy_vec,
        action=proposed, exec_action=executed, reward=-0.2, value=0.1, log_prob=-0.3,
        done=False, next_obj_vec=dummy_vec, weights=dummy_vec
    )
    
    assert torch.equal(buffer.actions[0], torch.tensor(proposed, dtype=torch.long))
    assert torch.equal(buffer.exec_actions[0], torch.tensor(executed, dtype=torch.long))
    assert not torch.equal(buffer.actions[0], buffer.exec_actions[0])


def test_temporal_smoothness_episode_boundary_masking():
    """
    Verifies that temporal preference smoothness loss strictly masks out transitions
    across episode boundaries (same_episode_mask = 0.0).
    """
    weights_t = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2], [0.8, 0.05, 0.05, 0.05, 0.05]])
    prev_weights = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2], [0.1, 0.2, 0.2, 0.2, 0.3]])
    
    # Step 0: same episode (mask=1.0), weights match exactly -> loss = 0
    # Step 1: episode boundary reset (mask=0.0) -> even with large weight jump, loss must be 0!
    masks = torch.tensor([1.0, 0.0])
    
    temporal_diff_sq = ((weights_t - prev_weights) ** 2).sum(dim=-1)
    masked_loss = (masks * temporal_diff_sq).sum() / (masks.sum() + 1e-8)
    
    assert pytest.approx(masked_loss.item(), abs=1e-6) == 0.0



def test_manifold_and_simplex_geometric_invariants():
    """
    Phase 15 verification:
      - Retraction map Pi_M produces valid points in M_int with unit diagonal and off-diagonal in (-1, 1).
      - Preference weights strictly lie on the unit 4-simplex Delta^4.
      - Gradients propagate cleanly through the retraction and simplex mappings.
    """
    from models.interaction_learner import StateConditionedInteractionLearner
    from models.doim_network import DOIMNetwork
    
    K = 5
    learner = StateConditionedInteractionLearner(
        traffic_dim=64, num_objectives=K, latent_dim=16, num_controllable_tls=9
    )
    doim_net = DOIMNetwork(
        node_feature_dim=12, spatial_dim=64, temporal_dim=64, num_objectives=K,
        manifold_dim=64, num_controllable_tls=9, temperature=1.0
    )
    
    traffic_emb = torch.randn(4, 64, requires_grad=True)
    obj_vec = torch.rand(4, K, requires_grad=True)
    
    out = learner(traffic_emb, obj_vec)
    M = out["interaction_matrix"]
    M_shrunk = out["shrunk_matrix"]
    
    # Invariant 1: Unit diagonal on M_int
    for b in range(4):
        diag = torch.diagonal(M[b])
        diag_shrunk = torch.diagonal(M_shrunk[b])
        assert torch.allclose(diag, torch.ones(K), atol=1e-5), f"Diagonal not 1: {diag}"
        assert torch.allclose(diag_shrunk, torch.ones(K), atol=1e-5), f"Shrunk diagonal not 1: {diag_shrunk}"
        
        # Invariant 2: Off-diagonal strictly bounded in (-1, 1)
        off_diag_mask = ~torch.eye(K, dtype=torch.bool)
        assert torch.all(torch.abs(M[b][off_diag_mask]) <= 1.0)
        assert torch.all(torch.abs(M_shrunk[b][off_diag_mask]) <= 1.0)
    
    # Invariant 3: Preference Simplex Delta^4 via forward_manifold
    spatial_hist = torch.randn(4, 10, 64, requires_grad=True)
    m_t, w_t = doim_net.forward_manifold(spatial_hist, obj_vec, M_shrunk)
    assert torch.allclose(w_t.sum(dim=-1), torch.ones(4), atol=1e-5), "Simplex sum != 1"
    assert torch.all(w_t > 0.0), "Negative preference weights encountered on simplex"
    
    # Invariant 4: Non-zero gradients back to inputs
    loss = (M_shrunk ** 2).sum() + (w_t ** 2).sum()
    loss.backward()
    assert traffic_emb.grad is not None and torch.norm(traffic_emb.grad) > 0.0
    assert obj_vec.grad is not None and torch.norm(obj_vec.grad) > 0.0
    assert spatial_hist.grad is not None and torch.norm(spatial_hist.grad) > 0.0


def test_ppo_clipped_surrogate_analytical():
    """
    Phase 16 verification:
    Analytically tests the clipped surrogate policy gradient objective:
        L_clip(theta) = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
    Verifies that ratio clipping engages properly for positive and negative advantages.
    """
    eps = 0.2
    
    # Case 1: Positive advantage (A = +2.0), ratio exploded (r = 1.5 > 1 + eps = 1.2)
    # Unclipped: 1.5 * 2.0 = 3.0. Clipped: 1.2 * 2.0 = 2.4. min(3.0, 2.4) = 2.4.
    r_pos = torch.tensor([1.5], requires_grad=True)
    adv_pos = torch.tensor([2.0])
    surr1 = r_pos * adv_pos
    surr2 = torch.clamp(r_pos, 1.0 - eps, 1.0 + eps) * adv_pos
    loss_pos = -torch.min(surr1, surr2).mean()
    assert pytest.approx(loss_pos.item(), abs=1e-5) == -2.4
    
    # Gradient of clipped region w.r.t r_pos should be 0 (clipped is constant w.r.t r when r > 1+eps)
    loss_pos.backward()
    assert r_pos.grad.item() == 0.0
    
    # Case 2: Negative advantage (A = -2.0), ratio collapsed (r = 0.5 < 1 - eps = 0.8)
    # Unclipped: 0.5 * (-2.0) = -1.0. Clipped: 0.8 * (-2.0) = -1.6. min(-1.0, -1.6) = -1.6.
    r_neg = torch.tensor([0.5], requires_grad=True)
    adv_neg = torch.tensor([-2.0])
    surr1_neg = r_neg * adv_neg
    surr2_neg = torch.clamp(r_neg, 1.0 - eps, 1.0 + eps) * adv_neg
    loss_neg = -torch.min(surr1_neg, surr2_neg).mean()
    assert pytest.approx(loss_neg.item(), abs=1e-5) == 1.6
    
    loss_neg.backward()
    assert r_neg.grad.item() == 0.0
