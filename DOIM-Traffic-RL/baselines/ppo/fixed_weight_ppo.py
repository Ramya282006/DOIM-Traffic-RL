"""
DOIM-Traffic-RL: Fixed-Weight PPO (FW-PPO) Baseline
Standard Multi-Objective PPO with fixed scalarization weights w = [0.2, 0.2, 0.2, 0.2, 0.2].
Does NOT use the Dynamic Objective Interaction Manifold or adaptive weighting.
"""

import os
import sys
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_experiment_config
from common.experiment import ExperimentRun
from environment.sumo_env import DOIMTrafficEnv
from models.spatial_encoder import get_gcn_layer, create_grid_edge_index


class FixedWeightPPONetwork(nn.Module):
    """
    Standard Actor-Critic network for fixed-weight multi-objective PPO.
    Directly maps spatial-temporal traffic representations to policy and value heads.
    """

    def __init__(
        self,
        node_feature_dim: int = 5,
        spatial_dim: int = 64,
        temporal_dim: int = 64,
        num_controllable_tls: int = 5,
        raw_state_dim: int = 25,
    ):
        super().__init__()
        self.num_controllable = num_controllable_tls

        # Spatial Kipf-Welling GCN
        self.gcn1 = get_gcn_layer(node_feature_dim, spatial_dim)
        self.gcn2 = get_gcn_layer(spatial_dim, spatial_dim)

        # Temporal GRU
        self.gru = nn.GRU(
            input_size=spatial_dim,
            hidden_size=temporal_dim,
            num_layers=1,
            batch_first=True,
        )

        # Policy & Value backbones
        feature_dim = temporal_dim + raw_state_dim
        self.actor_backbone = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.action_heads = nn.ModuleList([
            nn.Linear(64, 2) for _ in range(num_controllable_tls)
        ])

        self.critic_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.register_buffer("default_edge_index", create_grid_edge_index())

    def encode_spatial(self, node_features: torch.Tensor) -> torch.Tensor:
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        batch_size = node_features.shape[0]
        pooled = []
        for b in range(batch_size):
            x = node_features[b]
            h = F.relu(self.gcn1(x, self.default_edge_index))
            h = self.gcn2(h, self.default_edge_index)
            pooled.append(h.mean(dim=0))
        return torch.stack(pooled, dim=0)

    def encode_temporal(self, spatial_seq: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(spatial_seq)
        return hidden[-1]

    def forward(
        self,
        spatial_history: torch.Tensor,
        raw_state: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        temporal_emb = self.encode_temporal(spatial_history)
        features = torch.cat([temporal_emb, raw_state], dim=-1)

        actor_feats = self.actor_backbone(features)
        action_logits = [head(actor_feats) for head in self.action_heads]
        value = self.critic_head(features)
        return action_logits, value


class FixedWeightPPOAgent:
    """
    Fixed-Weight PPO Agent optimizing a scalarized reward with constant weights.
    """

    def __init__(self, config, weights: Optional[np.ndarray] = None, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.fixed_weights = (
            np.array(weights, dtype=np.float32)
            if weights is not None
            else np.full(5, 0.2, dtype=np.float32)
        )

        self.network = FixedWeightPPONetwork(
            node_feature_dim=5,
            spatial_dim=config.graph_and_network.spatial_hidden_dim,
            temporal_dim=config.graph_and_network.temporal_hidden_dim,
            num_controllable_tls=5,
            raw_state_dim=25,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=float(config.policy.learning_rate),
            eps=1e-5,
        )

        self.clip_range = float(config.policy.clip_range)
        self.ent_coef = float(config.policy.ent_coef)
        self.vf_coef = float(config.policy.vf_coef)
        self.max_grad_norm = float(config.policy.max_grad_norm)
        self.batch_size = int(config.policy.batch_size)
        self.n_epochs = int(config.policy.n_epochs)

    def compute_scalar_reward(self, normalized_objectives: np.ndarray) -> float:
        # Scalarization: R = sum_k w_k * (-z_k)
        return float(np.sum(self.fixed_weights * (-normalized_objectives)))

    def act(
        self,
        spatial_history: torch.Tensor,
        raw_state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        spatial_history = spatial_history.to(self.device)
        raw_state = raw_state.to(self.device)

        with torch.no_grad():
            logits_list, value = self.network(spatial_history, raw_state)
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

        return np.array(actions, dtype=np.int64), value.item(), log_prob_sum

    def update(
        self,
        spatial_histories: torch.Tensor,
        raw_states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        n_samples = len(spatial_histories)
        indices = np.arange(n_samples)
        policy_losses, value_losses = [], []

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                b_idx = indices[start : start + self.batch_size]

                b_hist = spatial_histories[b_idx].to(self.device)
                b_raw = raw_states[b_idx].to(self.device)
                b_act = actions[b_idx].to(self.device)
                b_old_lp = old_log_probs[b_idx].to(self.device)
                b_ret = returns[b_idx].to(self.device)
                b_adv = advantages[b_idx].to(self.device)

                logits_list, values = self.network(b_hist, b_raw)
                values = values.squeeze(-1)

                new_log_probs = torch.zeros(len(b_idx), device=self.device)
                entropy = torch.zeros(len(b_idx), device=self.device)

                for k, logits in enumerate(logits_list):
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    new_log_probs += dist.log_prob(b_act[:, k])
                    entropy += dist.entropy()

                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, b_ret)
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
        }

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Loads network parameters and optimizer state from checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.network.load_state_dict(ckpt["network_state_dict"])
        if "optimizer_state_dict" in ckpt and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])


def train_fixed_weight_ppo(
    config_path: str = "configs/default.yaml",
    scenario_override: Optional[str] = None,
    total_timesteps: int = 1000,
    seed: int = 42,
    eval_episodes: int = 0,
    eval_horizon: int = 1800,
    eval_seed: Optional[int] = None,
    output_root: str = "results/runs",
) -> ExperimentRun:
    """
    Executes training for Fixed-Weight PPO.
    """
    config = load_experiment_config(
        default_config_path=config_path,
        scenario_config_path=scenario_override,
    )

    run = ExperimentRun(
        experiment_name=f"FW_PPO_{config.simulation.scenario_name}",
        algorithm="FW_PPO",
        config=config,
        train_seed=seed,
        scenario=config.simulation.scenario_name,
        output_root=output_root,
    )

    env = DOIMTrafficEnv(
        sumo_config=config.simulation.sumo_config,
        enforce_safety_shield=config.safety_and_constraints.enforce_safety_shield,
        max_steps=1800,
    )

    agent = FixedWeightPPOAgent(config=config, device="cpu")
    rollout_size = config.policy.n_steps

    obs, info = env.reset(seed=seed)
    current_step = 0

    print("=" * 80)
    print(f"STARTING FIXED-WEIGHT PPO TRAINING | RUN: {run.run_id} | SEED: {seed}")
    print(f"Total Steps Target: {total_timesteps} | Rollout Horizon: {rollout_size}")
    print("=" * 80)

    # Initialize spatial history
    node_feat = torch.tensor(info["node_features"], dtype=torch.float32)
    with torch.no_grad():
        init_spatial = agent.network.encode_spatial(node_feat).unsqueeze(1)
    spatial_history = init_spatial.repeat(1, 10, 1)
    raw_state = torch.tensor(info["raw_state"], dtype=torch.float32).unsqueeze(0)

    while current_step < total_timesteps:
        # Buffer storage
        buf_hist, buf_raw, buf_act, buf_rew, buf_val, buf_lp, buf_done = [], [], [], [], [], [], []

        for _ in range(rollout_size):
            if current_step >= total_timesteps:
                break

            action, value, log_prob = agent.act(
                spatial_history=spatial_history,
                raw_state=raw_state,
                deterministic=False,
            )

            next_obs, env_reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            # Fixed-weight scalarized reward
            scalar_reward = agent.compute_scalar_reward(next_info["normalized_objectives"])

            run.log_step(
                step=current_step,
                sim_time=next_info["time"],
                reward=scalar_reward,
                raw_objs=next_info["raw_objectives"],
                norm_objs=next_info["normalized_objectives"],
                weights=agent.fixed_weights,
                actions=action.tolist(),
                shield_interventions=next_info.get("shield_interventions", 0),
            )

            buf_hist.append(spatial_history)
            buf_raw.append(raw_state)
            buf_act.append(torch.tensor(action, dtype=torch.long))
            buf_rew.append(scalar_reward)
            buf_val.append(value)
            buf_lp.append(log_prob)
            buf_done.append(done)

            current_step += 1
            info = next_info

            node_feat = torch.tensor(info["node_features"], dtype=torch.float32)
            with torch.no_grad():
                new_spatial = agent.network.encode_spatial(node_feat).unsqueeze(1)
            spatial_history = torch.cat([spatial_history[:, 1:], new_spatial], dim=1)
            raw_state = torch.tensor(info["raw_state"], dtype=torch.float32).unsqueeze(0)

            if done:
                obs, info = env.reset(seed=seed + current_step)
                node_feat = torch.tensor(info["node_features"], dtype=torch.float32)
                with torch.no_grad():
                    init_spatial = agent.network.encode_spatial(node_feat).unsqueeze(1)
                spatial_history = init_spatial.repeat(1, 10, 1)
                raw_state = torch.tensor(info["raw_state"], dtype=torch.float32).unsqueeze(0)

        # Compute GAE
        with torch.no_grad():
            _, last_val, _ = agent.act(spatial_history, raw_state)

        n = len(buf_rew)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae_lam = 0.0
        for t in reversed(range(n)):
            next_non_terminal = 1.0 - float(buf_done[t])
            next_val = last_val if t == n - 1 else buf_val[t + 1]
            delta = buf_rew[t] + config.policy.gamma * next_val * next_non_terminal - buf_val[t]
            last_gae_lam = delta + config.policy.gamma * config.policy.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        returns = advantages + np.array(buf_val, dtype=np.float32)

        # Update
        loss_dict = agent.update(
            spatial_histories=torch.cat(buf_hist, dim=0),
            raw_states=torch.cat(buf_raw, dim=0),
            actions=torch.stack(buf_act, dim=0),
            old_log_probs=torch.tensor(buf_lp, dtype=torch.float32),
            returns=torch.tensor(returns, dtype=torch.float32),
            advantages=torch.tensor(advantages, dtype=torch.float32),
        )

        print(f"Step {current_step}/{total_timesteps} | Policy Loss: {loss_dict['policy_loss']:.4f} | Value Loss: {loss_dict['value_loss']:.4f}")

    env.close()

    model_save_path = run.checkpoints_dir / "fw_ppo_final.pt"
    torch.save({
        "network_state_dict": agent.network.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "fixed_weights": agent.fixed_weights,
    }, model_save_path)

    summary = run.close(status="COMPLETED", final_model_path=str(model_save_path))
    print("=" * 80)
    print(f"FIXED-WEIGHT PPO COMPLETED! Output saved to: {run.run_dir}")
    print(f"Total Steps: {summary['total_steps']}, Total Reward: {summary['total_reward']:.2f}")
    print("=" * 80)

    # Decoupled frozen deterministic evaluation loop if requested
    if eval_episodes > 0:
        actual_eval_seed = eval_seed if eval_seed is not None else seed + 1000
        print(f"\n--- LAUNCHING FROZEN DETERMINISTIC EVALUATION (FW-PPO) (Horizon: {eval_horizon}s) ---")
        evaluate_fixed_weight_ppo(
            agent_or_checkpoint=agent,
            config_path=config_path,
            scenario_override=scenario_override,
            n_episodes=eval_episodes,
            eval_seed=actual_eval_seed,
            train_seed=seed,
            eval_horizon=eval_horizon,
            max_steps_per_episode=eval_horizon,
            output_root=output_root,
        )

    return run


def evaluate_fixed_weight_ppo(
    agent_or_checkpoint: Union[FixedWeightPPOAgent, str, Path],
    config_path: str = "configs/default.yaml",
    scenario_override: Optional[str] = None,
    n_episodes: int = 1,
    eval_seed: int = 1042,
    train_seed: int = 42,
    eval_horizon: int = 1800,
    max_steps_per_episode: Optional[int] = None,
    enforce_safety_shield: bool = True,
    output_root: str = "results/runs",
) -> List[ExperimentRun]:
    """
    Rigorously evaluates a trained Fixed-Weight PPO agent using a deterministic policy (argmax action).
    Guarantees baseline evaluation symmetry with IA-MOPPO: identical evaluation horizons, decoupled seeds,
    and frozen model parameters (torch.no_grad(), eval mode).
    """
    config = load_experiment_config(
        default_config_path=config_path,
        scenario_config_path=scenario_override,
    )
    scenario_name = Path(config.simulation.sumo_config).stem

    if isinstance(agent_or_checkpoint, (str, Path)):
        agent = FixedWeightPPOAgent(config=config, device="cpu")
        agent.load_checkpoint(agent_or_checkpoint)
    else:
        agent = agent_or_checkpoint

    agent.network.eval()
    eval_runs = []
    effective_horizon = int(max_steps_per_episode) if max_steps_per_episode is not None else int(eval_horizon)

    for ep in range(n_episodes):
        current_eval_seed = eval_seed + ep
        run = ExperimentRun(
            experiment_name=f"EVAL_FW_PPO_{scenario_name}_s{train_seed}_e{current_eval_seed}",
            algorithm="FW_PPO",
            config=config,
            train_seed=train_seed,
            eval_seed=current_eval_seed,
            eval_horizon=effective_horizon,
            scenario=scenario_name,
            output_root=output_root,
            training_timesteps=config.policy.total_timesteps,
            episode_id=ep,
            evaluation_mode="deterministic_eval",
        )

        env = DOIMTrafficEnv(
            sumo_config=config.simulation.sumo_config,
            enforce_safety_shield=enforce_safety_shield,
            max_steps=effective_horizon,
        )

        obs, info = env.reset(seed=current_eval_seed)
        node_feat = torch.tensor(info["node_features"], dtype=torch.float32)
        with torch.no_grad():
            init_spatial = agent.network.encode_spatial(node_feat).unsqueeze(1)
        spatial_history = init_spatial.repeat(1, 10, 1)
        raw_state = torch.tensor(info["raw_state"], dtype=torch.float32).unsqueeze(0)

        step = 0
        done = False
        while not done and step < effective_horizon:
            with torch.no_grad():
                action, value, log_prob = agent.act(
                    spatial_history=spatial_history,
                    raw_state=raw_state,
                    deterministic=True,
                )

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            # Scalarized reward using fixed weights
            norm_objs = next_info["normalized_objectives"]
            scalar_reward = float(np.sum(agent.fixed_weights * (-norm_objs)))

            run.log_step(
                step=step,
                sim_time=next_info["time"],
                reward=scalar_reward,
                raw_objs=next_info["raw_objectives"],
                norm_objs=next_info["normalized_objectives"],
                weights=agent.fixed_weights,
                actions=action.tolist(),
                shield_interventions=next_info.get("shield_interventions", 0),
            )

            step += 1
            info = next_info
            node_feat = torch.tensor(info["node_features"], dtype=torch.float32)
            with torch.no_grad():
                new_spatial = agent.network.encode_spatial(node_feat).unsqueeze(1)
            spatial_history = torch.cat([spatial_history[:, 1:], new_spatial], dim=1)
            raw_state = torch.tensor(info["raw_state"], dtype=torch.float32).unsqueeze(0)

        env.close()
        run.close(status="COMPLETED")
        eval_runs.append(run)

    return eval_runs


if __name__ == "__main__":
    train_fixed_weight_ppo(total_timesteps=128)
