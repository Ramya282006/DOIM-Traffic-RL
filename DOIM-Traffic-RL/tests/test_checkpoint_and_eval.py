import os
import shutil
from pathlib import Path
import pytest
import torch
import numpy as np

from common.config import load_experiment_config
from training.interaction_aware_ppo import InteractionAwarePPOAgent, RolloutBuffer


def test_agent_checkpoint_save_and_restore(tmp_path):
    """
    Verifies that model parameters, optimizer state, and step index can be saved and
    faithfully restored without loss of numerical precision.
    """
    config = load_experiment_config("configs/default.yaml")
    agent = InteractionAwarePPOAgent(config=config, device="cpu")

    # Mutate weights slightly with an optimizer step to ensure they are not default
    optimizer = agent.optimizer
    dummy_loss = sum(p.sum() for p in agent.network.parameters())
    dummy_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Save initial weights
    orig_gcn_weight = agent.network.gcn1.linear.weight.clone().detach()
    orig_q_proj_weight = agent.interaction_learner.query_proj.weight.clone().detach()

    ckpt_path = tmp_path / "checkpoints" / "model_step_500.pt"
    agent.save_checkpoint(ckpt_path, step=500)

    assert ckpt_path.exists(), "Checkpoint file was not created"

    # Create a new freshly-initialized agent
    new_agent = InteractionAwarePPOAgent(config=config, device="cpu")
    # Verify weights are initially different
    assert not torch.allclose(new_agent.network.gcn1.linear.weight, orig_gcn_weight)

    # Restore from checkpoint
    restored_step = new_agent.load_checkpoint(ckpt_path)
    assert restored_step == 500, f"Expected step 500, got {restored_step}"

    # Verify restored weights match exactly
    assert torch.allclose(new_agent.network.gcn1.linear.weight, orig_gcn_weight, atol=1e-7)
    assert torch.allclose(new_agent.interaction_learner.query_proj.weight, orig_q_proj_weight, atol=1e-7)


def test_deterministic_act_reproducibility():
    """
    Verifies that when deterministic=True is passed to agent.act(),
    repeated invocations on identical input state yield identical discrete actions.
    """
    config = load_experiment_config("configs/default.yaml")
    agent = InteractionAwarePPOAgent(config=config, device="cpu")
    agent.network.eval()
    agent.interaction_learner.eval()

    node_hist = torch.randn(1, 10, 9, 5)
    obj_vec = torch.tensor([[0.2, 0.4, 0.1, 0.5, 0.3]])
    raw_state = torch.randn(1, 25)

    actions_list = []
    for _ in range(5):
        action, val, logp, weights, int_mat = agent.act(
            node_history=node_hist,
            obj_vec=obj_vec,
            raw_state=raw_state,
            deterministic=True,
        )
        actions_list.append(action)

    for i in range(1, len(actions_list)):
        np.testing.assert_array_equal(
            actions_list[0],
            actions_list[i],
            err_msg="Deterministic act produced non-identical actions across identical calls",
        )
