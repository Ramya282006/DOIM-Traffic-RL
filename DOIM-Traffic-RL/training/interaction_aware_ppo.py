"""
DOIM-Traffic-RL: Interaction-Aware Multi-Objective PPO (IA-MOPPO)
Couples State-Conditioned Objective Interaction Learning with Policy Gradients,
Preference Adaptation, and Safety-Shielded SUMO Simulation.
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
    Stores on-policy trajectory transitions with raw graph snapshots for end-to-end GNN autograd.
    """

    def __init__(self, buffer_size: int, num_controllable_tls: int = 5):
        self.buffer_size = buffer_size
        self.num_controllable = num_controllable_tls
        self.reset()

    def reset(self):
        self.node_histories = []
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
        node_hist: Optional[torch.Tensor] = None,
        obj_vec: torch.Tensor = None,
        int_mat: torch.Tensor = None,
        raw_state: torch.Tensor = None,
        action: np.ndarray = None,
        reward: float = 0.0,
        value: float = 0.0,
        log_prob: float = 0.0,
        done: bool = False,
        next_obj_vec: torch.Tensor = None,
        weights: torch.Tensor = None,
        spatial_hist: Optional[torch.Tensor] = None,
    ):
        hist = node_hist if node_hist is not None else spatial_hist
        if hist is None:
            raise ValueError("Either node_hist or spatial_hist must be provided.")
        self.node_histories.append(hist.squeeze(0).detach().cpu() if hist.dim() in (4, 3) and hist.size(0) == 1 else hist.detach().cpu())
        self.objective_vectors.append(obj_vec.squeeze(0).detach().cpu() if obj_vec.dim() > 1 and obj_vec.size(0) == 1 else obj_vec.detach().cpu())
        self.interaction_matrices.append(int_mat.squeeze(0).detach().cpu() if int_mat.dim() > 2 and int_mat.size(0) == 1 else int_mat.detach().cpu())
        self.raw_states.append(raw_state.squeeze(0).detach().cpu() if raw_state.dim() > 1 and raw_state.size(0) == 1 else raw_state.detach().cpu())
        self.actions.append(torch.tensor(action, dtype=torch.long))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(bool(done))
        self.next_objectives.append(next_obj_vec.squeeze(0).detach().cpu() if next_obj_vec.dim() > 1 and next_obj_vec.size(0) == 1 else next_obj_vec.detach().cpu())
        self.weights.append(weights.squeeze(0).detach().cpu() if weights.dim() > 1 and weights.size(0) == 1 else weights.detach().cpu())
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
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std

        return torch.tensor(returns, dtype=torch.float32), torch.tensor(norm_advantages, dtype=torch.float32)


class InteractionAwarePPOAgent:
    """
    Interaction-Aware Multi-Objective PPO Agent with end-to-end differentiable GCN, GRU,
    SCOIL interaction manifold, and anti-collapse preference regularization.
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
            num_controllable_tls=config.graph_and_network.num_controllable_tls,
        ).to(self.device)

        # Joint Optimizer encompassing all modules end-to-end
        all_params = list(self.network.parameters()) + list(self.interaction_learner.parameters())
        self.optimizer = optim.Adam(all_params, lr=config.policy.learning_rate)

        # Hyperparameters
        self.clip_range = config.policy.clip_range
        self.ent_coef = config.policy.ent_coef
        self.vf_coef = config.policy.vf_coef
        self.smooth_lambda = config.policy.preference_smoothness_lambda
        self.pref_kl_coef = 0.5
        self.pref_entropy_coef = 0.01
        self.aux_lambda = 0.1
        self.max_grad_norm = config.policy.max_grad_norm
        self.batch_size = config.policy.batch_size
        self.n_epochs = config.policy.n_epochs

    def save_checkpoint(self, path: Union[str, Path], step: int) -> None:
        """Saves model and optimizer checkpoint with weights_only safety."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": int(step),
            "network_state_dict": self.network.state_dict(),
            "scoil_state_dict": self.interaction_learner.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, p)

    def load_checkpoint(self, path: Union[str, Path]) -> int:
        """Loads model and optimizer checkpoint, returning saved step index."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {p}")
        checkpoint = torch.load(p, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.interaction_learner.load_state_dict(checkpoint["scoil_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return int(checkpoint.get("step", 0))

    def act(
        self,
        node_history: Optional[torch.Tensor] = None,
        obj_vec: torch.Tensor = None,
        raw_state: torch.Tensor = None,
        deterministic: bool = False,
        rolling_pearson: Optional[np.ndarray] = None,
        spatial_history: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, float, float, torch.Tensor, torch.Tensor]:
        """
        Selects actions conditioned on the dynamic interaction manifold.
        Supports both raw node histories [B, L, N, D_in] for end-to-end GNN recomputation
        and precomputed spatial sequences [B, L, D_spatial] for legacy compatibility.
        """
        use_precomputed_spatial = False
        if node_history is None and spatial_history is not None:
            spatial_seq = spatial_history
            if spatial_seq.dim() == 2:
                spatial_seq = spatial_seq.unsqueeze(0)
            spatial_seq = spatial_seq.to(self.device)
            use_precomputed_spatial = True
        elif node_history is not None:
            if node_history.dim() == 3:
                node_history = node_history.unsqueeze(0)
            node_history = node_history.to(self.device)
        else:
            raise ValueError("Either node_history or spatial_history must be provided.")

        if obj_vec.dim() == 1:
            obj_vec = obj_vec.unsqueeze(0)
        if raw_state.dim() == 1:
            raw_state = raw_state.unsqueeze(0)

        obj_vec = obj_vec.to(self.device)
        raw_state = raw_state.to(self.device)

        with torch.no_grad():
            if not use_precomputed_spatial:
                spatial_seq = self.network.encode_spatial_sequence(node_history)
            traffic_emb = self.network.encode_temporal(spatial_seq)
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

            out = self.network(spatial_seq, obj_vec, int_mat, raw_state)
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
        Performs joint PPO mini-batch updates on the complete architecture, recomputing
        all spatial GNN and temporal GRU representations live inside the autograd tape.
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
        epoch_pref_losses = []

        node_hists = torch.stack(buffer.node_histories, dim=0).to(self.device)
        obj_vecs = torch.stack(buffer.objective_vectors, dim=0).to(self.device)
        int_mats = torch.stack(buffer.interaction_matrices, dim=0).to(self.device)
        raw_states = torch.stack(buffer.raw_states, dim=0).to(self.device)
        actions = torch.stack(buffer.actions, dim=0).to(self.device)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32).to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        next_obj_vecs = torch.stack(buffer.next_objectives, dim=0).to(self.device)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                b_idx = indices[start:end]

                b_node_hist = node_hists[b_idx]
                b_obj = obj_vecs[b_idx]
                b_int = int_mats[b_idx]
                b_raw = raw_states[b_idx]
                b_act = actions[b_idx]
                b_old_lp = old_log_probs[b_idx]
                b_ret = returns[b_idx]
                b_adv = advantages[b_idx]
                b_next_obj = next_obj_vecs[b_idx]

                # 1. LIVE RECOMPUTATION OF SPATIAL GNN AND TEMPORAL GRU
                # This guarantees that gcn1, gcn2, and gru receive full autograd gradients
                if b_node_hist.shape[-1] != self.network.spatial_dim:
                    b_spatial_seq = self.network.encode_spatial_sequence(b_node_hist)
                else:
                    b_spatial_seq = b_node_hist.squeeze(1) if b_node_hist.dim() == 4 else b_node_hist
                b_traffic_emb = self.network.encode_temporal(b_spatial_seq)

                # 2. LIVE SCOIL INTERACTION LEARNING
                if self.ablation_mode == "no_interaction":
                    live_int_mat = torch.zeros_like(b_int)
                elif self.ablation_mode == "rolling_pearson":
                    live_int_mat = b_int
                else:
                    scoil_out = self.interaction_learner(b_traffic_emb, b_obj)
                    live_int_mat = scoil_out["shrunk_matrix"]

                # 3. LIVE DOIM MANIFOLD & POLICY PASS
                out = self.network(b_spatial_seq, b_obj, live_int_mat, b_raw)
                logits_list = out["action_logits"]
                values = out["value"].squeeze(-1)
                weights = out["weights"]
                manifold = out["manifold"]

                # 4. PPO LOG PROBS AND POLICY SURROGATE LOSS
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

                # 5. VALUE FUNCTION LOSS
                value_loss = F.mse_loss(values, b_ret)

                # 6. AUXILIARY RESIDUAL OBJECTIVE DYNAMICS LOSS
                pred_next_obj = self.network.predict_next_objectives(manifold, b_act, b_obj)
                aux_loss = F.mse_loss(pred_next_obj, b_next_obj)
                scoil_pred_loss = self.interaction_learner.compute_predictive_loss(
                    live_int_mat, b_obj, b_act, b_next_obj
                )

                # 7. URGENCY-REGULATED PREFERENCE ADAPTATION WITH ANTI-COLLAPSE ENTROPY BARRIER
                # Empirical physical urgency vector u_t
                u_t = (b_obj + 1e-4) / (b_obj + 1e-4).sum(dim=-1, keepdim=True)
                kl_pref = F.kl_div(torch.log(weights + 1e-8), u_t, reduction="batchmean")
                pref_entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
                smooth_loss = torch.mean((weights[:, 1:] - weights[:, :-1]) ** 2)
                pref_loss = self.pref_kl_coef * kl_pref - self.pref_entropy_coef * pref_entropy + self.smooth_lambda * smooth_loss

                # 8. JOINT MULTI-OBJECTIVE DIFFERENTIABLE LOSS
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.aux_lambda * (aux_loss + scoil_pred_loss)
                    + pref_loss
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
                epoch_aux_losses.append(aux_loss.item() + scoil_pred_loss.item())
                epoch_pref_losses.append(pref_loss.item())

        return {
            "policy_loss": float(np.mean(epoch_policy_losses)),
            "value_loss": float(np.mean(epoch_value_losses)),
            "aux_loss": float(np.mean(epoch_aux_losses)),
            "pref_loss": float(np.mean(epoch_pref_losses)),
        }


def train_interaction_aware_ppo(
    config_path: str = "configs/default.yaml",
    scenario_override: Optional[str] = None,
    total_timesteps: int = 1000,
    seed: int = 42,
    output_root: str = "results/runs",
    ablation_mode: Optional[str] = None,
    enforce_safety_shield: bool = True,
    checkpoint_interval: int = 2500,
    eval_episodes: int = 0,
    eval_seed: Optional[int] = None,
) -> ExperimentRun:
    """
    Executes an interaction-aware PPO experiment with live SUMO, periodic checkpointing,
    and manifest tracking.
    Supports budgets: 128 (smoke), 5000 (pilot), 10000 (intermediate), 50000 (main).
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
        training_timesteps=total_timesteps,
        evaluation_mode="train_rollout",
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

    # Initialize raw node history buffer with real graph snapshots
    node_feat = torch.tensor(info["node_features"], dtype=torch.float32).unsqueeze(0) # [1, 9, 5]
    node_history = node_feat.unsqueeze(1).repeat(1, 10, 1, 1) # [1, 10, 9, 5]
    raw_state = torch.tensor(info["raw_state"], dtype=torch.float32).unsqueeze(0)

    while current_step < total_timesteps:
        buffer.reset()

        for _ in range(rollout_size):
            if current_step >= total_timesteps:
                break

            obj_vec = torch.tensor(info["normalized_objectives"], dtype=torch.float32).unsqueeze(0)

            action, value, log_prob, weights, int_mat = agent.act(
                node_history=node_history,
                obj_vec=obj_vec,
                raw_state=raw_state,
                deterministic=False,
                rolling_pearson=info.get("interaction_matrix"),
            )

            # Step environment with active preference weights (connecting MORL scalarization)
            next_obs, reward, terminated, truncated, next_info = env.step(
                action, weights=weights[0].numpy()
            )
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
                node_hist=node_history[0],
                obj_vec=obj_vec[0],
                int_mat=int_mat[0],
                raw_state=raw_state[0],
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                next_obj_vec=next_obj_vec[0],
                weights=weights[0],
            )

            current_step += 1
            info = next_info

            # Periodic model checkpoint
            if checkpoint_interval > 0 and current_step % checkpoint_interval == 0:
                ckpt_path = run.checkpoints_dir / f"model_step_{current_step}.pt"
                agent.save_checkpoint(ckpt_path, step=current_step)
                print(f"[CHECKPOINT] Periodic model saved to {ckpt_path.name}")

            # Roll node history forward with real node features from SUMO
            new_node_feat = torch.tensor(info["node_features"], dtype=torch.float32).unsqueeze(0).unsqueeze(1)
            node_history = torch.cat([node_history[:, 1:], new_node_feat], dim=1)
            raw_state = torch.tensor(info["raw_state"], dtype=torch.float32).unsqueeze(0)

            if done:
                obs, info = env.reset(seed=seed + current_step)
                episodes += 1
                node_feat = torch.tensor(info["node_features"], dtype=torch.float32).unsqueeze(0)
                node_history = node_feat.unsqueeze(1).repeat(1, 10, 1, 1)
                raw_state = torch.tensor(info["raw_state"], dtype=torch.float32).unsqueeze(0)

        # Update network parameters
        with torch.no_grad():
            _, last_val, _, _, _ = agent.act(
                node_history=node_history,
                obj_vec=torch.tensor(info["normalized_objectives"], dtype=torch.float32).unsqueeze(0),
                raw_state=raw_state,
            )
        loss_dict = agent.update(buffer, last_value=last_val)
        print(f"Step {current_step}/{total_timesteps} | Policy Loss: {loss_dict['policy_loss']:.4f} | Aux Loss: {loss_dict['aux_loss']:.4f}")

    env.close()

    # Save final model
    model_save_path = run.checkpoints_dir / "ia_moppo_final.pt"
    agent.save_checkpoint(model_save_path, step=current_step)

    summary = run.close(status="COMPLETED", final_model_path=str(model_save_path))
    print("=" * 80)
    print(f"IA-MOPPO TRAINING COMPLETED! Output saved to: {run.run_dir}")
    print(f"Total Steps: {summary['total_steps']}, Total Reward: {summary['total_reward']:.2f}")
    print("=" * 80)

    # Optional deterministic evaluation
    if eval_episodes > 0:
        actual_eval_seed = eval_seed if eval_seed is not None else seed + 1000
        print(f"\n[EVALUATION] Launching {eval_episodes} deterministic evaluation episodes with seed {actual_eval_seed}...")
        evaluate_interaction_aware_ppo(
            agent_or_checkpoint=agent,
            config_path=config_path,
            scenario_override=scenario_override,
            eval_seed=actual_eval_seed,
            train_seed=seed,
            n_episodes=eval_episodes,
            max_steps_per_episode=1800,
            output_root=output_root,
            ablation_mode=ablation_mode,
            enforce_safety_shield=enforce_safety_shield,
        )

    return run


def evaluate_interaction_aware_ppo(
    agent_or_checkpoint: Union[InteractionAwarePPOAgent, str, Path],
    config_path: str = "configs/default.yaml",
    scenario_override: Optional[str] = None,
    eval_seed: int = 1042,
    train_seed: int = 42,
    n_episodes: int = 1,
    max_steps_per_episode: int = 1800,
    output_root: str = "results/runs",
    ablation_mode: Optional[str] = None,
    enforce_safety_shield: bool = True,
) -> List[ExperimentRun]:
    """
    Rigorously evaluates a trained IA-MOPPO agent using a deterministic policy (argmax action).
    Separates evaluation episodes with independent evaluation seeds from training rollouts.
    Ensures model parameters are frozen (torch.no_grad(), eval mode).
    """
    config = load_experiment_config(config_path)
    if scenario_override:
        config.simulation.sumo_config = scenario_override
    scenario_name = Path(config.simulation.sumo_config).stem
    algo_name = f"DOIM_{ablation_mode}" if ablation_mode else "IA_MOPPO"
    shield_enabled = False if ablation_mode == "no_shield" else enforce_safety_shield

    if isinstance(agent_or_checkpoint, (str, Path)):
        agent = InteractionAwarePPOAgent(config=config, device="cpu", ablation_mode=ablation_mode)
        agent.load_checkpoint(agent_or_checkpoint)
    else:
        agent = agent_or_checkpoint

    agent.network.eval()
    agent.interaction_learner.eval()

    eval_runs = []

    for ep in range(n_episodes):
        current_eval_seed = eval_seed + ep
        seed_everything(current_eval_seed)

        run = ExperimentRun(
            experiment_name=f"EVAL_{algo_name}_{scenario_name}_s{train_seed}_e{current_eval_seed}",
            algorithm=algo_name,
            config=config,
            train_seed=train_seed,
            eval_seed=current_eval_seed,
            scenario=scenario_name,
            output_root=output_root,
            training_timesteps=config.policy.total_timesteps,
            episode_id=ep,
            evaluation_mode="deterministic_eval",
        )

        env = DOIMTrafficEnv(
            sumo_config=config.simulation.sumo_config,
            enforce_safety_shield=shield_enabled,
            max_steps=max_steps_per_episode,
        )

        obs, info = env.reset(seed=current_eval_seed)
        node_feat = torch.tensor(info["node_features"], dtype=torch.float32).unsqueeze(0)
        node_history = node_feat.unsqueeze(1).repeat(1, 10, 1, 1)
        raw_state = torch.tensor(info["raw_state"], dtype=torch.float32).unsqueeze(0)

        step = 0
        done = False
        while not done and step < max_steps_per_episode:
            obj_vec = torch.tensor(info["normalized_objectives"], dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action, value, log_prob, weights, int_mat = agent.act(
                    node_history=node_history,
                    obj_vec=obj_vec,
                    raw_state=raw_state,
                    deterministic=True,
                    rolling_pearson=info.get("interaction_matrix"),
                )

            next_obs, reward, terminated, truncated, next_info = env.step(
                action, weights=weights[0].numpy()
            )
            done = terminated or truncated

            run.log_step(
                step=step,
                sim_time=next_info["time"],
                reward=reward,
                raw_objs=next_info["raw_objectives"],
                norm_objs=next_info["normalized_objectives"],
                weights=weights[0].numpy(),
                actions=action.tolist(),
                shield_interventions=next_info.get("shield_interventions", 0),
            )

            step += 1
            info = next_info
            new_node_feat = torch.tensor(info["node_features"], dtype=torch.float32).unsqueeze(0).unsqueeze(1)
            node_history = torch.cat([node_history[:, 1:], new_node_feat], dim=1)
            raw_state = torch.tensor(info["raw_state"], dtype=torch.float32).unsqueeze(0)

        env.close()
        run.close(status="COMPLETED")
        eval_runs.append(run)

    return eval_runs


if __name__ == "__main__":
    train_interaction_aware_ppo(total_timesteps=128)

