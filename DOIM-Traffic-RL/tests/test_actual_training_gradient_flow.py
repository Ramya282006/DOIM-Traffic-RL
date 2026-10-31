"""
DOIM-Traffic-RL: Real Training Gradient Flow Verification Test
Strictly verifies that during an actual PPO mini-batch update, EVERY single intended
trainable module receives non-zero gradients and updates its parameters:
  1. GCN Layer 1 (spatial message passing)
  2. GCN Layer 2 (spatial message passing)
  3. GRU Layer (temporal sequence learning)
  4. Objective Encoder
  5. Interaction Encoder
  6. DOIM Manifold Core (multi-modal fusion)
  7. Adaptive Preference Head (urgency alignment)
  8. Actor Backbone
  9. Action Heads (multi-discrete traffic control)
  10. Critic Head (value function baseline)
  11. Auxiliary Transition Predictor (residual dynamics)
  12. SCOIL Feature Net
  13. SCOIL Query Projection (bilinear coupling)
  14. SCOIL Key Projection (bilinear coupling)
  15. SCOIL Asymmetry Head (directional lags)
  16. SCOIL Uncertainty Head (evidential variance)
  17. SCOIL Action Impact (cross-objective dynamics)
"""

import copy
import pytest
import torch
import numpy as np

from common.config import load_experiment_config
from training.interaction_aware_ppo import InteractionAwarePPOAgent, RolloutBuffer


def test_complete_end_to_end_gradient_flow():
    config = load_experiment_config("configs/default.yaml")
    config.policy.batch_size = 16
    config.policy.n_epochs = 1
    
    agent = InteractionAwarePPOAgent(config=config, device="cpu")
    buffer = RolloutBuffer(buffer_size=32)

    # Clone initial model parameter snapshots before any update
    initial_params = {}
    for name, param in agent.network.named_parameters():
        initial_params[f"network.{name}"] = param.clone().detach()
    for name, param in agent.interaction_learner.named_parameters():
        initial_params[f"scoil.{name}"] = param.clone().detach()

    # Populate buffer with realistic rollout transitions
    torch.manual_seed(42)
    np.random.seed(42)
    
    for t in range(32):
        node_hist = torch.randn(10, 9, 5) # raw graph history
        obj_vec = torch.rand(5) # normalized objectives in [0, 1]
        int_mat = torch.tanh(torch.randn(5, 5)) # initial interaction matrix
        raw_state = torch.randn(25) # local state of 5 controllable intersections
        action = np.random.randint(0, 2, size=5)
        reward = float(np.random.randn() * 0.5)
        value = float(np.random.randn())
        log_prob = -float(np.random.rand() * 2.0)
        done = (t == 31)
        next_obj_vec = torch.clamp(obj_vec + torch.randn(5) * 0.05, 0.0, 1.0)
        weights = torch.softmax(torch.randn(5), dim=-1)

        buffer.add(
            node_hist=node_hist,
            obj_vec=obj_vec,
            int_mat=int_mat,
            raw_state=raw_state,
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done,
            next_obj_vec=next_obj_vec,
            weights=weights,
        )

    # Perform a genuine PPO mini-batch update
    loss_dict = agent.update(buffer, last_value=0.0)

    # Verify every required submodule
    modules_to_verify = {
        "GCN Layer 1": agent.network.gcn1,
        "GCN Layer 2": agent.network.gcn2,
        "Temporal GRU": agent.network.gru,
        "Objective Encoder": agent.network.objective_encoder,
        "Interaction Encoder": agent.network.interaction_encoder,
        "Manifold Core": agent.network.manifold_core,
        "Preference Head": agent.network.preference_head,
        "Actor Backbone": agent.network.actor_backbone,
        "Action Heads": agent.network.action_heads,
        "Critic Head": agent.network.critic_head,
        "Auxiliary Predictor": agent.network.aux_predictor,
        "SCOIL Feature Net": agent.interaction_learner.feature_net,
        "SCOIL Query Proj": agent.interaction_learner.query_proj,
        "SCOIL Key Proj": agent.interaction_learner.key_proj,
        "SCOIL Asymmetry": agent.interaction_learner.asymmetry_head,
        "SCOIL Uncertainty": agent.interaction_learner.uncertainty_head,
        "SCOIL Action Impact": agent.interaction_learner.action_impact,
    }

    print("\n" + "=" * 90)
    print(f"{'Module Name':<25} | {'Param Count':<12} | {'Grad Norm':<16} | {'Param Delta Norm':<18} | {'Status'}")
    print("=" * 90)

    failed_modules = []

    for mod_name, module in modules_to_verify.items():
        grad_norms = []
        delta_norms = []
        p_count = 0

        for name, param in module.named_parameters():
            if param.requires_grad:
                p_count += param.numel()
                assert param.grad is not None, f"Parameter {name} in {mod_name} has NO gradient (grad is None)!"
                g_norm = param.grad.norm().item()
                grad_norms.append(g_norm)

                # Locate corresponding initial param
                matched_initial = None
                for full_name, init_p in initial_params.items():
                    if full_name.endswith(name) and init_p.shape == param.shape:
                        matched_initial = init_p
                        break

                if matched_initial is not None:
                    delta = (param.detach() - matched_initial).norm().item()
                    delta_norms.append(delta)

        avg_grad = float(np.mean(grad_norms)) if grad_norms else 0.0
        avg_delta = float(np.mean(delta_norms)) if delta_norms else 0.0
        status = "PASSED" if (avg_grad > 1e-8 and avg_delta > 1e-8) else "FAILED"

        print(f"{mod_name:<25} | {p_count:<12} | {avg_grad:<16.6e} | {avg_delta:<18.6e} | {status}")

        if status == "FAILED":
            failed_modules.append(mod_name)

    print("=" * 90)
    assert len(failed_modules) == 0, f"The following modules failed gradient flow verification: {failed_modules}"
    print("ALL 17 MODULES VERIFIED: Meaningful gradients and parameter updates confirmed end-to-end!")


if __name__ == "__main__":
    test_complete_end_to_end_gradient_flow()
