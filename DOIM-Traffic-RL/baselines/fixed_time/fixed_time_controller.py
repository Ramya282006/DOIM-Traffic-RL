"""
DOIM-Traffic-RL: Fixed-Time Traffic Signal Controller Baseline
Classical round-robin fixed-time controller switching phases according to predefined cycle time.
Evaluated on the exact same DOIMTrafficEnv environment and logging infrastructure.
"""

import os
import sys
from pathlib import Path
import time
from typing import Dict, List, Optional
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_experiment_config
from common.experiment import ExperimentRun
from environment.sumo_env import DOIMTrafficEnv


class FixedTimeController:
    """
    Fixed-time traffic signal controller.
    Cycles through signal phases at fixed intervals (e.g., 30s green per phase).
    """

    def __init__(
        self,
        cycle_time: int = 30,
        num_controllable_tls: int = 5,
        tls_ids: Optional[List[str]] = None,
        green_duration: Optional[int] = None,
        yellow_duration: Optional[int] = None,
        **kwargs,
    ):
        if tls_ids is not None:
            self.num_controllable = len(tls_ids)
        else:
            self.num_controllable = int(num_controllable_tls)
        
        if green_duration is not None:
            self.cycle_time = int(green_duration)
        else:
            self.cycle_time = int(cycle_time)
            
        self.step_in_phase = 0

    def select_action(self) -> np.ndarray:
        """
        Action: 0 = keep current phase, 1 = request phase transition.
        Triggers a transition request every cycle_time steps.
        """
        self.step_in_phase += 1
        if self.step_in_phase >= self.cycle_time:
            self.step_in_phase = 0
            return np.ones(self.num_controllable, dtype=np.int64)
        else:
            return np.zeros(self.num_controllable, dtype=np.int64)

    act = select_action

    def reset(self):
        self.step_in_phase = 0


FixedTimeBaseline = FixedTimeController


def run_fixed_time_experiment(
    config_path: str = "configs/default.yaml",
    scenario_override: Optional[str] = None,
    total_timesteps: int = 1000,
    cycle_time: int = 30,
    seed: int = 42,
    eval_seed: Optional[int] = None,
    episode_id: int = 0,
    evaluation_mode: str = "deterministic_eval",
    eval_horizon: Optional[int] = None,
    output_root: str = "results/runs",
) -> ExperimentRun:
    """
    Executes a benchmark evaluation run for the Fixed-Time Controller.
    Fixed-Time is a non-learning, rule-based baseline cycling through signal phases.
    """
    config = load_experiment_config(
        default_config_path=config_path,
        scenario_config_path=scenario_override,
    )

    actual_eval_seed = int(eval_seed if eval_seed is not None else seed)

    run = ExperimentRun(
        experiment_name=f"Fixed_Time_{config.simulation.scenario_name}_s{seed}_e{actual_eval_seed}",
        algorithm="Fixed_Time",
        config=config,
        train_seed=seed,
        eval_seed=actual_eval_seed,
        episode_id=episode_id,
        evaluation_mode=evaluation_mode,
        eval_horizon=eval_horizon,
        scenario=config.simulation.scenario_name,
        output_root=output_root,
        training_timesteps=total_timesteps,
    )

    env = DOIMTrafficEnv(
        sumo_config=config.simulation.sumo_config,
        enforce_safety_shield=config.safety_and_constraints.enforce_safety_shield,
        max_steps=total_timesteps,
    )

    controller = FixedTimeController(cycle_time=cycle_time, num_controllable_tls=5)
    obs, info = env.reset(seed=actual_eval_seed)
    controller.reset()

    # Equal weights for reporting purposes (0.2 for each of 5 objectives)
    fixed_weights = np.full(5, 0.2, dtype=np.float32)

    print("=" * 80)
    print(f"RUNNING FIXED-TIME BASELINE | RUN: {run.run_id} | SEED: {seed} | EVAL_SEED: {actual_eval_seed}")
    print(f"Total Steps Target: {total_timesteps} | Mode: {evaluation_mode}")
    print("=" * 80)

    for step in range(total_timesteps):
        action = controller.select_action()
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated

        run.log_step(
            step=step,
            sim_time=next_info["time"],
            reward=reward,
            raw_objs=next_info["raw_objectives"],
            norm_objs=next_info["normalized_objectives"],
            weights=fixed_weights,
            actions=action.tolist(),
            shield_interventions=next_info.get("shield_interventions", 0),
        )

        if done:
            # Terminate immediately: do not silently chain multiple shorter episodes into one run
            break

    env.close()
    summary = run.close(status="COMPLETED")

    print("=" * 80)
    print(f"FIXED-TIME BASELINE COMPLETED! Output saved to: {run.run_dir}")
    print(f"Total Steps: {summary['total_steps']}, Total Reward: {summary['total_reward']:.2f}")
    print(f"Average Waiting Time: {summary['average_waiting']:.2f}s | Avg Queue: {summary['average_queue']:.2f}")
    print("=" * 80)

    return run


def evaluate_fixed_time(
    config_path: str = "configs/default.yaml",
    scenario_override: Optional[str] = None,
    n_episodes: int = 1,
    eval_seed: int = 1042,
    train_seed: int = 42,
    cycle_time: int = 30,
    eval_horizon: int = 1800,
    max_steps_per_episode: Optional[int] = None,
    output_root: str = "results/runs",
) -> List[ExperimentRun]:
    """
    Evaluates Fixed-Time non-learning baseline across decoupled evaluation seeds.
    Guarantees baseline evaluation symmetry with identical evaluation seeds and horizons.
    """
    eval_runs = []
    effective_horizon = int(max_steps_per_episode) if max_steps_per_episode is not None else int(eval_horizon)

    for ep in range(n_episodes):
        current_eval_seed = eval_seed + ep
        run = run_fixed_time_experiment(
            config_path=config_path,
            scenario_override=scenario_override,
            total_timesteps=effective_horizon,
            cycle_time=cycle_time,
            seed=train_seed,
            eval_seed=current_eval_seed,
            episode_id=ep,
            evaluation_mode="deterministic_eval",
            eval_horizon=effective_horizon,
            output_root=output_root,
        )
        eval_runs.append(run)
    return eval_runs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=None)
    parser.add_argument("--cycle", type=int, default=30)
    parser.add_argument("--scenario", type=str, default=None)
    args = parser.parse_args()

    run_fixed_time_experiment(
        total_timesteps=args.timesteps,
        seed=args.seed,
        eval_seed=args.eval_seed,
        cycle_time=args.cycle,
        scenario_override=args.scenario,
    )
