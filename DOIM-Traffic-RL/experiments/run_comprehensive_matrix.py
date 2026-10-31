"""
DOIM-Traffic-RL: Comprehensive Experimental Matrix Runner
Executes multi-seed, multi-scenario evaluation of Proposed DOIM-Traffic-RL against
all baselines (Fixed-Time, FW-PPO, SA-PPO) and architectural ablations.
Supports configurable budgets (128 smoke, 5000 pilot, 10000 intermediate, 50000 main convergence),
periodic checkpointing, and independent deterministic evaluation loops.
"""

import argparse
import os
import sys
from pathlib import Path
import time
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.interaction_aware_ppo import train_interaction_aware_ppo
from baselines.fixed_time.fixed_time_controller import run_fixed_time_experiment
from baselines.ppo.fixed_weight_ppo import train_fixed_weight_ppo
from baselines.ppo.state_adaptive_ppo import train_state_adaptive_ppo
from analysis.statistical_analyzer import StatisticalAnalyzer
from analysis.plot_generator import PublicationPlotGenerator


def run_experiment_matrix(
    timesteps: int = 128,
    seeds: List[int] = [42],
    include_ablations: bool = True,
    scenarios: List[str] = ["scenarios/directional_shift/north_south_surge.sumocfg"],
    eval_episodes: int = 1,
    eval_horizon: int = 1800,
    checkpoint_interval: int = 2500,
    output_root: str = "results/runs",
):
    print("=" * 80)
    print("STARTING DOIM-TRAFFIC-RL COMPREHENSIVE EXPERIMENT MATRIX")
    print(f"Training Timesteps per run: {timesteps} | Seeds: {seeds} | Scenarios: {scenarios}")
    print(f"Evaluation Episodes: {eval_episodes} (Horizon: {eval_horizon}s) | Checkpoint Interval: {checkpoint_interval}")
    print("=" * 80)

    start_time = time.time()

    for scenario in scenarios:
        for seed in seeds:
            print(f"\n>>> Running Scenario: {scenario} | Seed: {seed}")

            # 1. Proposed DOIM-Traffic-RL (IA-MOPPO)
            print("--- Running Proposed IA-MOPPO ---")
            train_interaction_aware_ppo(
                scenario_override=scenario,
                total_timesteps=timesteps,
                seed=seed,
                output_root=output_root,
                checkpoint_interval=checkpoint_interval,
                eval_episodes=eval_episodes,
            )

            # 2. Fixed-Time Baseline (Rule-based non-learning controller)
            print("--- Running Fixed-Time Baseline ---")
            run_fixed_time_experiment(
                scenario_override=scenario,
                total_timesteps=timesteps,
                seed=seed,
                output_root=output_root,
            )
            if eval_episodes > 0:
                from baselines.fixed_time.fixed_time_controller import evaluate_fixed_time
                print("--- Evaluating Fixed-Time Baseline Deterministically ---")
                evaluate_fixed_time(
                    scenario_override=scenario,
                    n_episodes=eval_episodes,
                    eval_seed=seed + 1000,
                    train_seed=seed,
                    max_steps_per_episode=eval_horizon,
                    output_root=output_root,
                )

            # 3. Fixed-Weight PPO Baseline
            print("--- Running Fixed-Weight PPO Baseline ---")
            train_fixed_weight_ppo(
                scenario_override=scenario,
                total_timesteps=timesteps,
                seed=seed,
                eval_episodes=eval_episodes,
                output_root=output_root,
            )

            # 4. State-Adaptive PPO Baseline (Direct state-to-preference)
            print("--- Running State-Adaptive PPO Baseline ---")
            train_state_adaptive_ppo(
                scenario_override=scenario,
                total_timesteps=timesteps,
                seed=seed,
                eval_episodes=eval_episodes,
                output_root=output_root,
            )

            if include_ablations:
                # 5. Ablation 1: w/o Interaction Manifold (M = 0)
                print("--- Running Ablation 1: No Interaction Manifold ---")
                train_interaction_aware_ppo(
                    scenario_override=scenario,
                    total_timesteps=timesteps,
                    seed=seed,
                    ablation_mode="no_interaction",
                    output_root=output_root,
                    checkpoint_interval=checkpoint_interval,
                    eval_episodes=eval_episodes,
                )

                # 6. Ablation 2: Legacy Rolling Pearson Correlation
                print("--- Running Ablation 2: Rolling Pearson Correlation ---")
                train_interaction_aware_ppo(
                    scenario_override=scenario,
                    total_timesteps=timesteps,
                    seed=seed,
                    ablation_mode="rolling_pearson",
                    output_root=output_root,
                    checkpoint_interval=checkpoint_interval,
                    eval_episodes=eval_episodes,
                )

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS COMPLETED IN {elapsed:.1f}s! RUNNING STATISTICAL ANALYSIS...")
    print("=" * 80)

    analyzer = StatisticalAnalyzer(runs_root=output_root)
    analyzer.export_summary_tables()

    plotter = PublicationPlotGenerator()
    plotter.generate_all_plots()

    print("\n" + "=" * 80)
    print("COMPREHENSIVE MATRIX COMPLETE: DATA, TABLES, AND FIGURES READY!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DOIM-Traffic-RL Experiment Matrix Runner")
    parser.add_argument("--timesteps", type=int, default=128, help="Budget: 128 (smoke), 5000 (pilot), 10000 (intermediate), 50000 (main)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="List of independent training seeds")
    parser.add_argument("--scenarios", nargs="+", type=str, default=["scenarios/directional_shift/north_south_surge.sumocfg"], help="List of scenario sumocfg paths")
    parser.add_argument("--eval_episodes", type=int, default=1, help="Number of frozen deterministic evaluation episodes per model (default: 1)")
    parser.add_argument("--eval_horizon", type=int, default=1800, help="Evaluation horizon in seconds per episode (default: 1800)")
    parser.add_argument("--checkpoint_interval", type=int, default=2500, help="Periodic checkpoint save interval")
    parser.add_argument("--output_root", type=str, default="results/runs", help="Root directory for experiment logs")
    parser.add_argument("--no_ablations", action="store_true", help="Skip ablations")
    args = parser.parse_args()

    run_experiment_matrix(
        timesteps=args.timesteps,
        seeds=args.seeds,
        scenarios=args.scenarios,
        include_ablations=not args.no_ablations,
        eval_episodes=args.eval_episodes,
        eval_horizon=args.eval_horizon,
        checkpoint_interval=args.checkpoint_interval,
        output_root=args.output_root,
    )

