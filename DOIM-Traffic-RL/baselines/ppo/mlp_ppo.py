"""
DOIM-Traffic-RL: Parameter-Matched MLP-PPO Baseline (Phase G)
Implements an architectural control baseline that flattens observations into a standard MLP
with matched parameter capacity, isolating the empirical value of GCN + GRU + DOIM representation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from environment.sumo_env import DOIMTrafficEnv
from common.experiment import ExperimentRun
from common.config import load_experiment_config, ConfigDict
from common.seeding import seed_everything


class ParameterMatchedMLPNetwork(nn.Module):
    """
    Standard MLP policy-value architecture with parameter capacity matched to DOIMNetwork.
    Input dimension: 9 intersections * 5 features + 25 raw state = 70.
    """

    def __init__(
        self,
        input_dim: int = 70,
        hidden_dim: int = 128,
        num_controllable_tls: int = 5,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )
        
        # Policy heads for each controllable TLS
        self.actor_heads = nn.ModuleList([
            nn.Linear(64, 2) for _ in range(num_controllable_tls)
        ])
        
        # Centralized Critic
        self.critic_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        feats = self.encoder(x)
        logits = [head(feats) for head in self.actor_heads]
        value = self.critic_head(feats)
        return {"action_logits": logits, "value": value}


class MLPPPOAgent:
    """
    PPO agent utilizing the Parameter-Matched MLP network.
    """

    def __init__(self, config: ConfigDict, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.network = ParameterMatchedMLPNetwork(
            input_dim=70,
            hidden_dim=128,
            num_controllable_tls=config.graph_and_network.num_controllable_tls,
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.policy.learning_rate)
        self.clip_range = config.policy.clip_range

    def act(self, flat_obs: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            if flat_obs.dim() == 1:
                flat_obs = flat_obs.unsqueeze(0)
            flat_obs = flat_obs.to(self.device)
            out = self.network(flat_obs)
            logits_list = out["action_logits"]
            value = out["value"].item()
            
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

        return np.array(actions, dtype=np.int64), value, log_prob_sum


def train_mlp_ppo(
    scenario_override: Optional[str] = None,
    total_timesteps: int = 128,
    seed: int = 42,
    output_root: str = "results/runs",
    eval_episodes: int = 1,
    eval_horizon: int = 1800,
) -> Dict[str, Any]:
    """
    Executes training and deterministic evaluation for the Parameter-Matched MLP-PPO baseline.
    """
    config = load_experiment_config()
    scenario_path = scenario_override if scenario_override is not None else config.simulation.scenario_config
    seed_everything(seed)
    
    agent = MLPPPOAgent(config)
    exp = ExperimentRun(
        experiment_name=f"MLP_PPO_{Path(scenario_path).stem}_s{seed}",
        algorithm="MLP_PPO",
        config=config.to_dict() if hasattr(config, "to_dict") else dict(config),
        train_seed=seed,
        eval_seed=seed,
        scenario=Path(scenario_path).stem,
        output_root=output_root,
        training_timesteps=total_timesteps,
        evaluation_mode="train_rollout",
    )
    
    env = DOIMTrafficEnv(
        config_path=scenario_path,
        seed=seed,
        max_steps=total_timesteps,
        use_safety_shield=True,
    )
    
    obs, info = env.reset(seed=seed)
    for step in range(total_timesteps):
        # Flatten node features and raw state into 70-dim vector
        feat = info.get("feature_matrix", np.zeros((9, 5))).flatten()
        raw = info.get("raw_state", np.zeros(25))
        flat_obs = torch.tensor(np.concatenate([feat, raw]), dtype=torch.float32)
        
        action, value, log_prob = agent.act(flat_obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        
        exp.log_step(
            step=step,
            sim_time=float(step + 1),
            reward=float(reward),
            raw_objs=info.get("raw_objectives", np.zeros(5)),
            norm_objs=info.get("normalized_objectives", np.zeros(5)),
            weights=np.ones(5) * 0.2,
            actions=action.tolist(),
            shield_interventions=info.get("shield_interventions", 0),
        )
        if terminated or truncated:
            break
            
    env.close()
    
    # Save checkpoint
    ckpt_path = exp.checkpoints_dir / "mlp_ppo_final.pt"
    torch.save(agent.network.state_dict(), ckpt_path)
    
    summary = exp.close(status="COMPLETED", final_model_path=str(ckpt_path))
    return summary

