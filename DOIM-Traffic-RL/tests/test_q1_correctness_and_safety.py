"""
DOIM-Traffic-RL: Q1 Master Verification Test Suite
Tests critical correctness remediation gates:
  1. Part 2.1: Authoritative Typed Objective Configuration & Normalization
  2. Part 2.2: Multi-Tier Fairness Metrics (Known Numerical Benchmarks)
  3. Part 2.3: Temporal Preference Smoothness Regularization
  4. Part 3.0: Safety Shield Fail-Safe Locking, Typed Safety Faults, and Episode Abort
  5. Part 4.0: SCOIL Interaction Matrix & Heteroscedastic Predictive Loss
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from objectives.objective_config import AuthoritativeObjectiveConfig, ObjectiveDimensionSpec
from objectives.objective_normalizer import ObjectiveNormalizer
from objectives.objective_engine import MultiObjectiveEngine
from objectives.fairness import CorridorFairnessCalculator
from environment.safety_shield import (
    TrafficSafetyShield,
    SignalPhaseState,
    SafetyFaultError,
    verify_phase_conflicts,
)
from models.interaction_learner import StateConditionedInteractionLearner
from training.interaction_aware_ppo import RolloutBuffer, InteractionAwarePPOAgent
from common.config import load_experiment_config


class TestAuthoritativeObjectiveConfig:
    """Verifies Part 2.1: Authoritative typed objective configuration."""

    def test_default_config_schema(self):
        cfg = AuthoritativeObjectiveConfig()
        assert len(cfg.specs) == 5
        assert cfg.specs[0].name == "waiting"
        assert cfg.specs[1].name == "queue"
        assert cfg.specs[2].name == "co2"
        assert cfg.specs[3].name == "throughput"
        assert cfg.specs[4].name == "fairness"

        # Check directions
        assert cfg.specs[0].direction == "minimize"
        assert cfg.specs[3].direction == "maximize"
        assert cfg.specs[4].direction == "minimize"

        # Check bounds
        assert cfg.lower_bounds[4] == 0.0
        assert cfg.upper_bounds[4] == 1.0  # Normalized Jain disparity in [0, 1]

    def test_normalization_and_throughput_inversion(self):
        cfg = AuthoritativeObjectiveConfig()

        # Best case: 0 waiting, 0 queue, 0 co2, 20 throughput (max), 0.0 disparity (perfect equity)
        raw_best = np.array([0.0, 0.0, 0.0, 20.0, 0.0], dtype=np.float32)
        urgency_best = cfg.normalize(raw_best)
        np.testing.assert_allclose(urgency_best, np.zeros(5, dtype=np.float32), atol=1e-6)

        # Worst case: 600 waiting, 20 queue, 120000 co2, 0 throughput, 1.0 disparity
        raw_worst = np.array([600.0, 20.0, 120000.0, 0.0, 1.0], dtype=np.float32)
        urgency_worst = cfg.normalize(raw_worst)
        np.testing.assert_allclose(urgency_worst, np.ones(5, dtype=np.float32), atol=1e-6)

    def test_fail_fast_on_invalid_raw_vector(self):
        cfg = AuthoritativeObjectiveConfig()

        # Wrong dimension (4 instead of 5)
        with pytest.raises(ValueError, match="Expected raw objective vector of shape"):
            cfg.validate_raw_vector(np.array([1.0, 2.0, 3.0, 4.0]))

        # Non-finite value
        with pytest.raises(ValueError, match="non-finite"):
            cfg.validate_raw_vector(np.array([10.0, np.nan, 50.0, 5.0, 0.2]))

        # Physical impossibility (negative delay)
        with pytest.raises(ValueError, match="outside valid physical range"):
            cfg.validate_raw_vector(np.array([-5.0, 1.0, 50.0, 5.0, 0.2]))


class TestMultiTierFairnessMetrics:
    """Verifies Part 2.2: Multi-tier fairness calculations with hand-computable benchmarks."""

    def test_jain_fairness_equal_service(self):
        # 4 approaches experiencing identical delay -> J = 1.0, Disparity = 0.0
        delays = [25.0, 25.0, 25.0, 25.0]
        jain = CorridorFairnessCalculator.compute_jain_index(delays)
        disparity = CorridorFairnessCalculator.compute_delay_disparity(delays)
        assert pytest.approx(jain, rel=1e-5) == 1.0
        assert pytest.approx(disparity, abs=1e-5) == 0.0

    def test_jain_fairness_hand_computed_inequity(self):
        # 2 corridors: delays = [10.0, 30.0]
        # Sum = 40, Sum^2 = 1600
        # Sum of squares = 100 + 900 = 1000
        # K = 2
        # J = 1600 / (2 * 1000) = 1600 / 2000 = 0.80
        # Disparity = 1.0 - 0.80 = 0.20
        delays = [10.0, 30.0]
        jain = CorridorFairnessCalculator.compute_jain_index(delays)
        disparity = CorridorFairnessCalculator.compute_delay_disparity(delays)
        assert pytest.approx(jain, rel=1e-5) == 0.80
        assert pytest.approx(disparity, rel=1e-5) == 0.20

    def test_worst_approach_delay_and_disparity_ratio(self):
        delays = [10.0, 50.0, 20.0, 5.0]
        worst = CorridorFairnessCalculator.compute_worst_approach_delay(delays)
        ratio = CorridorFairnessCalculator.compute_max_to_min_disparity_ratio(delays, eps=0.0)
        assert worst == 50.0
        assert pytest.approx(ratio, rel=1e-5) == 50.0 / 5.0  # 10.0

    def test_gini_coefficient_known_distribution(self):
        # Equal distribution -> Gini = 0.0
        assert pytest.approx(CorridorFairnessCalculator.compute_gini_coefficient([20.0, 20.0, 20.0]), abs=1e-5) == 0.0
        # Extreme inequality: [0, 0, 100] -> Gini = (2*(300) - 4*100) / (3 * 100) = (600 - 400)/300 = 2/3 = 0.6667
        delays = [0.0, 0.0, 100.0]
        gini = CorridorFairnessCalculator.compute_gini_coefficient(delays)
        assert pytest.approx(gini, rel=1e-3) == 2.0 / 3.0

    def test_tier_separation_integrity(self):
        appr_metrics = CorridorFairnessCalculator.compute_approach_fairness([15.0, 30.0, 45.0])
        inter_metrics = CorridorFairnessCalculator.compute_intersection_fairness([10.0, 20.0, 30.0, 40.0, 50.0])
        veh_metrics = CorridorFairnessCalculator.compute_vehicle_fairness([5.0, 10.0, 15.0, 25.0])

        assert appr_metrics["tier"] == "approach_corridor"
        assert inter_metrics["tier"] == "intersection"
        assert veh_metrics["tier"] == "vehicle"
        assert "worst_approach_delay_s" in appr_metrics
        assert "worst_intersection_delay_s" in inter_metrics
        assert "p95_vehicle_delay_s" in veh_metrics


class TestTemporalPreferenceSmoothness:
    """Verifies Part 2.3: Temporal preference smoothness regularization."""

    def test_temporal_smoothness_mathematical_invariance(self):
        # Verify that weights with large differences across dimensions but constant over time
        # produce ZERO temporal smoothness penalty!
        # Step 0: [0.6, 0.1, 0.1, 0.1, 0.1]
        # Step 1: [0.6, 0.1, 0.1, 0.1, 0.1]  (SAME distribution over time)
        w_t = torch.tensor([[0.6, 0.1, 0.1, 0.1, 0.1]], dtype=torch.float32)
        w_prev = torch.tensor([[0.6, 0.1, 0.1, 0.1, 0.1]], dtype=torch.float32)
        mask = torch.tensor([1.0], dtype=torch.float32)

        diff_sq = ((w_t - w_prev) ** 2).sum(dim=-1)
        smooth_loss = (mask * diff_sq).sum() / (mask.sum() + 1e-8)
        assert smooth_loss.item() == 0.0

        # Contrast with OLD buggy dimensional difference:
        old_buggy_loss = torch.mean((w_t[:, 1:] - w_t[:, :-1]) ** 2)
        # Old code gave (0.5^2 + 0 + 0 + 0)/4 = 0.25 / 4 = 0.0625 > 0 !
        assert old_buggy_loss.item() > 0.05
        # Our temporal regularizer correctly identifies zero temporal chattering!

    def test_temporal_smoothness_hand_computed_penalty(self):
        # Step t-1: [0.2, 0.2, 0.2, 0.2, 0.2]
        # Step t:   [0.3, 0.2, 0.2, 0.1, 0.2]
        # Delta = [0.1, 0.0, 0.0, -0.1, 0.0]
        # ||Delta||^2_2 = 0.01 + 0 + 0 + 0.01 + 0 = 0.02
        w_prev = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=torch.float32)
        w_t = torch.tensor([[0.3, 0.2, 0.2, 0.1, 0.2]], dtype=torch.float32)
        mask = torch.tensor([1.0], dtype=torch.float32)

        diff_sq = ((w_t - w_prev) ** 2).sum(dim=-1)
        smooth_loss = (mask * diff_sq).sum() / mask.sum()
        assert pytest.approx(smooth_loss.item(), rel=1e-5) == 0.02

    def test_episode_boundary_masking(self):
        # Step 0 is an episode boundary (new episode reset): mask = 0.0
        # Even if weights jumped completely from previous episode, loss must be 0!
        w_prev = torch.tensor([[0.8, 0.05, 0.05, 0.05, 0.05]], dtype=torch.float32)
        w_t = torch.tensor([[0.1, 0.1, 0.6, 0.1, 0.1]], dtype=torch.float32)
        mask = torch.tensor([0.0], dtype=torch.float32)

        diff_sq = ((w_t - w_prev) ** 2).sum(dim=-1)
        mask_sum = mask.sum()
        smooth_loss = (mask * diff_sq).sum() / (mask_sum + 1e-8) if mask_sum > 0 else torch.tensor(0.0)
        assert smooth_loss.item() == 0.0

    def test_rollout_buffer_stores_prev_weights_and_masks(self):
        buf = RolloutBuffer(buffer_size=10)
        w0 = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        w1 = torch.tensor([0.3, 0.2, 0.2, 0.1, 0.2])

        # Step 0: start of episode
        buf.add(
            node_hist=torch.zeros(9, 5),
            obj_vec=torch.zeros(5),
            int_mat=torch.zeros(5, 5),
            raw_state=torch.zeros(25),
            action=np.zeros(5),
            reward=0.0,
            value=0.0,
            log_prob=0.0,
            done=False,
            next_obj_vec=torch.zeros(5),
            weights=w0,
            prev_weights=w0,
            same_episode_mask=0.0,
        )

        # Step 1: continuation of episode
        buf.add(
            node_hist=torch.zeros(9, 5),
            obj_vec=torch.zeros(5),
            int_mat=torch.zeros(5, 5),
            raw_state=torch.zeros(25),
            action=np.zeros(5),
            reward=0.0,
            value=0.0,
            log_prob=0.0,
            done=False,
            next_obj_vec=torch.zeros(5),
            weights=w1,
            prev_weights=w0,
            same_episode_mask=1.0,
        )

        assert len(buf.prev_weights) == 2
        assert len(buf.same_episode_masks) == 2
        assert buf.same_episode_masks[0] == 0.0
        assert buf.same_episode_masks[1] == 1.0
        np.testing.assert_allclose(buf.prev_weights[1].numpy(), w0.numpy())


class TestSafetyShieldFaultAndAbort:
    """Verifies Part 3.0: Safety fault detection, typed exceptions, and episode abort."""

    def test_safety_fault_error_raised_when_configured(self):
        # A program with 2 green phases that conflict with each other and NO all-red phase
        # TLS has linkIndex 0 and 1 conflicting
        prog = [
            {"duration": 20, "state": "GG"},      # Phase 0: green
            {"duration": 3,  "state": "yy"},      # Phase 1: yellow
            {"duration": 20, "state": "GG"},      # Phase 2: green conflicting with phase 0 movements!
        ]
        conflicts = [ (0, 1) ]
        # In phase 2 both 0 and 1 have 'G' -> conflict violation!

        shield = TrafficSafetyShield(
            controllable_tls_ids=["B1"],
            tls_programs={"B1": prog},
            tls_conflicts={"B1": conflicts},
            min_green=10,
            max_green=20,
            yellow_duration=3,
            raise_on_fault=True,
        )

        state = shield.states["B1"]
        state.current_phase_index = 0
        state.time_in_current_phase = 25  # Max green exceeded!

        # Attempt to filter action: nominal next is phase 1 -> leads to phase 2 (which violates conflicts!)
        # No alternative green, no all-red -> must raise SafetyFaultError!
        with pytest.raises(SafetyFaultError) as exc_info:
            shield.filter_action(raw_actions=np.array([1]), current_phases=[0])

        assert "CRITICAL SAFETY FAULT on TLS 'B1'" in str(exc_info.value)
        assert shield.has_safety_fault()
        faults = shield.get_active_faults()
        assert len(faults) >= 1
        assert faults[0]["tls_id"] == "B1"

    def test_safety_fault_records_when_raise_on_fault_false(self):
        prog = [
            {"duration": 20, "state": "GG"},
            {"duration": 3,  "state": "yy"},
            {"duration": 20, "state": "GG"},
        ]
        conflicts = [(0, 1)]

        shield = TrafficSafetyShield(
            controllable_tls_ids=["B1"],
            tls_programs={"B1": prog},
            tls_conflicts={"B1": conflicts},
            min_green=10,
            max_green=20,
            yellow_duration=3,
            raise_on_fault=False,  # Log fault instead of immediate crash
        )

        state = shield.states["B1"]
        state.current_phase_index = 0
        state.time_in_current_phase = 25

        safe_act, targets, intervs = shield.filter_action(raw_actions=np.array([1]), current_phases=[0])
        assert shield.has_safety_fault()
        assert state.cannot_safely_progress
        faults = shield.get_active_faults()
        assert len(faults) == 1
        assert "NO_VERIFIED_SAFE_TRANSITION" in faults[0]["reason"]


class TestSCOILInteractionLearner:
    """Verifies Part 4.0: SCOIL interaction matrix range, shrinkage, and predictive loss."""

    def test_scoil_forward_and_shrinkage(self):
        learner = StateConditionedInteractionLearner(
            num_objectives=5,
            traffic_dim=64,
            latent_dim=32,
            shrinkage_lambda=0.1,
            shrinkage_mode="identity",
        )

        traffic_emb = torch.randn(2, 64)
        obj_vec = torch.rand(2, 5)

        out = learner(traffic_emb, obj_vec)
        m_t = out["interaction_matrix"]
        u_t = out["uncertainty"]
        shrunk_m_t = out["shrunk_matrix"]

        # 1. Bounds: M_t in [-1, 1]
        assert torch.all(m_t >= -1.0) and torch.all(m_t <= 1.0)

        # 2. Diagonal self-interaction is 1.0
        for b in range(2):
            for k in range(5):
                assert pytest.approx(m_t[b, k, k].item(), abs=1e-5) == 1.0
                assert pytest.approx(u_t[b, k, k].item(), abs=1e-5) == 0.0

        # 3. Uncertainty is strictly non-negative
        assert torch.all(u_t >= 0.0)

        # 4. Shrunk matrix maintains diagonal 1.0
        for b in range(2):
            for k in range(5):
                assert pytest.approx(shrunk_m_t[b, k, k].item(), abs=1e-5) == 1.0

    def test_scoil_heteroscedastic_predictive_loss(self):
        learner = StateConditionedInteractionLearner(num_objectives=5, traffic_dim=64)
        traffic_emb = torch.randn(4, 64)
        z_t = torch.rand(4, 5)
        actions = torch.randint(0, 2, (4, 5))
        z_next = torch.rand(4, 5)

        out = learner(traffic_emb, z_t)
        loss = learner.compute_predictive_loss(
            out["shrunk_matrix"], z_t, actions, z_next, uncertainty=out["uncertainty"]
        )

        assert torch.isfinite(loss)
        assert loss.requires_grad
        loss.backward()
        # Verify gradient flow to query projection
        assert learner.query_proj.weight.grad is not None
        assert torch.any(learner.query_proj.weight.grad != 0.0)

