"""
DOIM-Traffic-RL: Interaction-Aware Multi-Objective PPO (IA-MOPPO)
Couples State-Conditioned Objective Interaction Learning with Policy Gradients,
Preference Adaptation, and Safety-Shielded SUMO Simulation.
"""

import os
import sys
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_experiment_config, ConfigDict
from common.seeding import seed_everything
from common.experiment import ExperimentRun
from common.verification import ExperimentVerifier
from environment.sumo_env import DOIMTrafficEnv
from models.doim_network import DOIMNetwork
from models.interaction_learner import StateConditionedInteractionLearner


class RolloutBuffer:
    """
    Stores on-policy trajectory transitions for PPO training.
    """

    def __init__(self, buffer_size: int, num_controllable_tls: int = 5):
        self.buffer_size = buffer_size
        self.num_controllable = num_controllable_tls
        self.reset()

    def reset(self):
        self.spatial_histories = []
        self.objective_vectors = []
        self.interaction_matrices = []
        self.raw_states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_objectives = []
        self.weights = []
        self.ptr = 0

    def add(
        self,
        spatial_hist: torch.Tensor,
        obj_vec: torch.Tensor,
        int_mat: torch.Tensor,
        raw_state: torch.Tensor,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        next_obj_vec: torch.Tensor,
        weights: torch.Tensor,
    ):
        self.spatial_histories.append(spatial_hist.detach().cpu())
        self.objective_vectors.append(obj_vec.detach().cpu())
        self.interaction_matrices.append(int_mat.detach().cpu())
        self.raw_states.append(raw_state.detach().cpu())
        self.actions.append(torch.tensor(action, dtype=torch.long))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(bool(done))
        self.next_objectives.append(next_obj_vec.detach().cpu())
        self.weights.append(weights.detach().cpu())
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Generalized Advantage Estimation (GAE).
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae_lam = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_val = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_val = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + np.array(self.values, dtype=np.float32)
        # Normalize advantages
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std

        return torch.tensor(returns, dtype=torch.float32), torch.tensor(norm_advantages, dtype=torch.float32)


class InteractionAwarePPOAgent:
    """
    Interaction-Aware Multi-Objective PPO Agent.
    """

    def __init__(
        self,
        config: ConfigDict,
        device: str = "cpu",
        ablation_mode: Optional[str] = None,
    ):
        self.config = config
        self.device = torch.device(device)
        self.ablation_mode = ablation_mode

        # Networks
        self.network = DOIMNetwork(
            num_intersections=config.graph_and_network.num_intersections,
            num_controllable_tls=config.graph_and_network.num_controllable_tls,
            node_feature_dim=config.graph_and_network.node_feature_dim,
            num_objectives=config.objectives.num_objectives,
            spatial_dim=config.graph_and_network.spatial_hidden_dim,
            temporal_dim=config.graph_and_network.spatial_hidden_dim,
            manifold_dim=config.interaction_model.latent_manifold_dim,
            raw_state_dim=25,
            temperature=config.policy.preference_temperature,
        ).to(self.device)

        self.interaction_learner = StateConditionedInteractionLearner(
            num_objectives=config.objectives.num_objectives,
            traffic_dim=config.graph_and_network.spatial_hidden_dim,
            latent_dim=config.interaction_model.interaction_dim,
            shrinkage_lambda=config.interaction_model.shrinkage,
        ).to(self.device)

        # Joint Optimizer
        all_params = list(self.network.parameters()) + list(self.interaction_learner.parameters())
        self.optimizer = optim.Adam(all_params, lr=config.policy.learning_rate)

        # Hyperparameters
        self.clip_range = config.policy.clip_range
        self.ent_coef = config.policy.ent_coef
        self.vf_coef = config.policy.vf_coef
        self.smooth_lambda = config.policy.preference_smoothness_lambda
        self.aux_lambda = 0.1
        self.max_grad_norm = config.policy.max_grad_norm
        self.batch_size = config.policy.batch_size
        self.n_epochs = config.policy.n_epochs

    def act(
        self,
        spatial_history: torch.Tensor,
        obj_vec: torch.Tensor,
        raw_state: torch.Tensor,
        deterministic: bool = False,
        rolling_pearson: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float, torch.Tensor, torch.Tensor]:
        """
        Selects actions conditioned on the dynamic interaction manifold (or ablation matrix).
        """
        spatial_history = spatial_history.to(self.device)
        obj_vec = obj_vec.to(self.device)
        raw_state = raw_state.to(self.device)

        with torch.no_grad():
            traffic_emb = self.network.encode_temporal(spatial_history)
            if self.ablation_mode == "no_interaction":
                int_mat = torch.zeros(1, 5, 5, device=self.device)
            elif self.ablation_mode == "rolling_pearson":
                if rolling_pearson is not None:
                    int_mat = torch.tensor(rolling_pearson, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    int_mat = torch.zeros(1, 5, 5, device=self.device)
            else:
                scoil_out = self.interaction_learner(traffic_emb, obj_vec)
                int_mat = scoil_out["shrunk_matrix"]

            out = self.network(spatial_history, obj_vec, int_mat, raw_state)
            logits_list = out["action_logits"]
            value = out["value"].item()
            weights = out["weights"]

            actions = []
            log_prob_sum = 0.0

            for logits in logits_list:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                if deterministic:
                    a = torch.argmax(probs, dim=-1)
                else:
                    a = dist.sample()
                actions.append(a.item())
                log_prob_sum += dist.log_prob(a).item()

        return np.array(actions, dtype=np.int64), value, log_prob_sum, weights, int_mat

    def update(self, buffer: RolloutBuffer, last_value: float) -> Dict[str, float]:
        """
        Performs joint PPO mini-batch updates on the complete architecture.
        """
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.config.policy.gamma,
            gae_lambda=self.config.policy.gae_lambda,
        )

        n_samples = len(buffer.rewards)
        indices = np.arange(n_samples)

        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_aux_losses = []
        epoch_smooth_losses = []

        spatial_hists = torch.cat(buffer.spatial_histories, dim=0).to(self.device)
        obj_vecs = torch.cat(buffer.objective_vectors, dim=0).to(self.device)
        int_mats = torch.cat(buffer.interaction_matrices, dim=0).to(self.device)
        raw_states = torch.cat(buffer.raw_states, dim=0).to(self.device)
        actions = torch.stack(buffer.actions, dim=0).to(self.device)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32).to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        next_obj_vecs = torch.cat(buffer.next_objectives, dim=0).to(self.device)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                b_idx = indices[start:end]

                b_hist = spatial_hists[b_idx]
                b_obj = obj_vecs[b_idx]
                b_int = int_mats[b_idx]
                b_raw = raw_states[b_idx]
                b_act = actions[b_idx]
                b_old_lp = old_log_probs[b_idx]
                b_ret = returns[b_idx]
                b_adv = advantages[b_idx]
                b_next_obj = next_obj_vecs[b_idx]

                # Forward pass with live interaction computation for full autograd
                if self.ablation_mode == "no_interaction":
                    live_int_mat = torch.zeros_like(b_int)
                elif self.ablation_mode == "rolling_pearson":
                    live_int_mat = b_int
                else:
                    traffic_emb = self.network.encode_temporal(b_hist)
                    scoil_out = self.interaction_learner(traffic_emb, b_obj)
                    live_int_mat = scoil_out["shrunk_matrix"]

                out = self.network(b_hist, b_obj, live_int_mat, b_raw)
                logits_list = out["action_logits"]
                values = out["value"].squeeze(-1)
                weights = out["weights"]
                manifold = out["manifold"]

                # PPO Log Probs and Entropy
                new_log_probs = torch.zeros(len(b_idx), device=self.device)
                entropy = torch.zeros(len(b_idx), device=self.device)

                for k, logits in enumerate(logits_list):
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    new_log_probs += dist.log_prob(b_act[:, k])
                    entropy += dist.entropy()

                # Clipped surrogate policy loss
                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, b_ret)

                # Auxiliary transition loss
                pred_next_obj = self.network.predict_next_objectives(manifold, b_act)
                aux_loss = F.mse_loss(pred_next_obj, b_next_obj)

                # Preference smoothness loss (penalize excessive variance in weights)
                smooth_loss = torch.mean((weights[:, 1:] - weights[:, :-1]) ** 2)

                # Total Multi-Task Loss
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.aux_lambda * aux_loss
                    + self.smooth_lambda * smooth_loss
                    - self.ent_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.network.parameters()) + list(self.interaction_learner.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_aux_losses.append(aux_loss.item())
                epoch_smooth_losses.append(smooth_loss.item())

        return {
            "policy_loss": float(np.mean(epoch_policy_losses)),
            "value_loss": float(np.mean(epoch_value_losses)),
            "aux_loss": float(np.mean(epoch_aux_losses)),
            "smooth_loss": float(np.mean(epoch_smooth_losses)),
        }


def train_interaction_aware_ppo(
    config_path: str = "configs/default.yaml",
    scenario_override: Optional[str] = None,
    total_timesteps: int = 1000,
    seed: int = 42,
    output_root: str = "results/runs",
    ablation_mode: Optional[str] = None,
    enforce_safety_shield: bool = True,
) -> ExperimentRun:
    """
    Executes an interaction-aware PPO experiment with live SUMO and manifest tracking.
    Supports ablations: 'no_interaction', 'rolling_pearson', 'no_shield'.
    """
    config = load_experiment_config(config_path)
    if scenario_override:
        config.simulation.sumo_config = scenario_override
    config.policy.total_timesteps = total_timesteps
    seed_everything(seed)

    algo_name = f"DOIM_{ablation_mode}" if ablation_mode else "IA_MOPPO"
    shield_enabled = False if ablation_mode == "no_shield" else enforce_safety_shield

    run = ExperimentRun(
        experiment_name=f"{algo_name}_{Path(config.simulation.sumo_config).stem}",
        algorithm=algo_name,
        config=config,
        train_seed=seed,
        scenario=Path(config.simulation.sumo_config).stem,
        output_root=output_root,
    )

    env = DOIMTrafficEnv(
        sumo_config=config.simulation.sumo_config,
        enforce_safety_shield=shield_enabled,
        max_steps=1800,
    )

    agent = InteractionAwarePPOAgent(config=config, device="cpu", ablation_mode=ablation_mode)
    rollout_size = config.policy.n_steps
    buffer = RolloutBuffer(buffer_size=rollout_size)

    obs, info = env.reset(seed=seed)
    current_step = 0
    episodes = 0

    print("=" * 80)
    print(f"STARTING {algo_name} TRAINING | RUN: {run.run_id} | SEED: {seed}")
    print(f"Total Steps Target: {total_timesteps} | Rollout Horizon: {rollout_size}")
    if ablation_mode:
        print(f"ACTIVE ABLATION MODE: {ablation_mode}")
    print("=" * 80)

    # Initialize history buffers with real environmental traffic representation
    node_feat = torch.tensor(info["node_features"], dtype=torch.float32)
    with torch.no_grad():
        init_spatial = agent.network.encode_spatial(node_feat).unsqueeze(1) # [1, 1, 64]
    spatial_history = init_spatial.repeat(1, 10, 1)
    raw_state = torch.tensor(info["raw_state"], dtype=torch.float32).unsqueeze(0)

    while current_step < total_timesteps:
        buffer.reset()

        for _ in range(rollout_size):
            if current_step >= total_timesteps:
                break

            obj_vec = torch.tensor(info["normalized_objectives"], dtype=torch.float32).unsqueeze(0)

            action, value, log_prob, weights, int_mat = agent.act(
                spatial_history=spatial_history,
                obj_vec=obj_vec,
                raw_state=raw_state,
                deterministic=False,
                rolling_pearson=info.get("interaction_matrix"),
            )

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            next_obj_vec = torch.tensor(next_info["normalized_objectives"], dtype=torch.float32).unsqueeze(0)

            # Store step in experiment run log
            run.log_step(
                step=current_step,
                sim_time=next_info["time"],
                reward=reward,
                raw_objs=next_info["raw_objectives"],
                norm_objs=next_info["normalized_objectives"],
                weights=weights[0].numpy(),
                actions=action.tolist(),
                shield_interventions=next_info.get("shield_interventions", 0),
            )

            buffer.add(
                spatial_hist=spatial_history,
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

            current_step += 1
            info = next_info

            # Roll spatial history forward with real node features from SUMO
            node_feat = torch.tensor(info["node_features"], dtype=torch.float32)
            with torch.no_grad():
                new_spatial = agent.network.encode_spatial(node_feat).unsqueeze(1)
            spatial_history = torch.cat([spatial_history[:, 1:], new_spatial], dim=1)
            raw_state = torch.tensor(info["raw_state"], dtype=torch.float32).unsqueeze(0)

            if done:
                obs, info = env.reset(seed=seed + current_step)
                episodes += 1
                node_feat = torch.tensor(info["node_features"], dtype=torch.float32)
                with torch.no_grad():
                    init_spatial = agent.network.encode_spatial(node_feat).unsqueeze(1)
                spatial_history = init_spatial.repeat(1, 10, 1)
                raw_state = torch.tensor(info["raw_state"], dtype=torch.float32).unsqueeze(0)

        # Update network parameters
        with torch.no_grad():
            _, last_val, _, _, _ = agent.act(
                spatial_history=spatial_history,
                obj_vec=torch.tensor(info["normalized_objectives"], dtype=torch.float32).unsqueeze(0),
                raw_state=raw_state,
            )
        loss_dict = agent.update(buffer, last_value=last_val)
        print(f"Step {current_step}/{total_timesteps} | Policy Loss: {loss_dict['policy_loss']:.4f} | Aux Loss: {loss_dict['aux_loss']:.4f}")

    env.close()

    # Save final model
    model_save_path = run.checkpoints_dir / "ia_moppo_final.pt"
    torch.save({
        "network_state_dict": agent.network.state_dict(),
        "scoil_state_dict": agent.interaction_learner.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "config": config,
    }, model_save_path)

    summary = run.close(status="COMPLETED", final_model_path=str(model_save_path))
    print("=" * 80)
    print(f"IA-MOPPO TRAINING COMPLETED! Output saved to: {run.run_dir}")
    print(f"Total Steps: {summary['total_steps']}, Total Reward: {summary['total_reward']:.2f}")
    print("=" * 80)
    return run


if __name__ == "__main__":
    train_interaction_aware_ppo(total_timesteps=128)
