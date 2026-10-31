"""
DOIM-Traffic-RL: Comprehensive Experimental Matrix Runner
Executes multi-seed, multi-scenario evaluation of Proposed DOIM-Traffic-RL against
all baselines (Fixed-Time, FW-PPO, SA-PPO) and architectural ablations.
Automatically generates statistical comparison tables and publication plots.
"""

import argparse
import os
import sys
from pathlib import Path
import time
from typing import List

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
):
    print("=" * 80)
    print("STARTING DOIM-TRAFFIC-RL COMPREHENSIVE EXPERIMENT MATRIX")
    print(f"Timesteps per run: {timesteps} | Seeds: {seeds} | Scenarios: {scenarios}")
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
            )

            # 2. Fixed-Time Baseline
            print("--- Running Fixed-Time Baseline ---")
            run_fixed_time_experiment(
                scenario_override=scenario,
                total_timesteps=timesteps,
                seed=seed,
            )

            # 3. Fixed-Weight PPO Baseline
            print("--- Running Fixed-Weight PPO Baseline ---")
            train_fixed_weight_ppo(
                scenario_override=scenario,
                total_timesteps=timesteps,
                seed=seed,
            )

            # 4. State-Adaptive PPO Baseline (Direct state-to-preference)
            print("--- Running State-Adaptive PPO Baseline ---")
            train_state_adaptive_ppo(
                scenario_override=scenario,
                total_timesteps=timesteps,
                seed=seed,
            )

            if include_ablations:
                # 5. Ablation 1: w/o Interaction Manifold (M = 0)
                print("--- Running Ablation 1: No Interaction Manifold ---")
                train_interaction_aware_ppo(
                    scenario_override=scenario,
                    total_timesteps=timesteps,
                    seed=seed,
                    ablation_mode="no_interaction",
                )

                # 6. Ablation 2: Legacy Rolling Pearson Correlation
                print("--- Running Ablation 2: Rolling Pearson Correlation ---")
                train_interaction_aware_ppo(
                    scenario_override=scenario,
                    total_timesteps=timesteps,
                    seed=seed,
                    ablation_mode="rolling_pearson",
                )

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS COMPLETED IN {elapsed:.1f}s! RUNNING STATISTICAL ANALYSIS...")
    print("=" * 80)

    analyzer = StatisticalAnalyzer()
    analyzer.export_summary_tables()

    plotter = PublicationPlotGenerator()
    plotter.generate_all_plots()

    print("\n" + "=" * 80)
    print("COMPREHENSIVE MATRIX COMPLETE: DATA, TABLES, AND FIGURES READY!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=128)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--scenarios", nargs="+", type=str, default=["scenarios/directional_shift/north_south_surge.sumocfg"])
    parser.add_argument("--no_ablations", action="store_true")
    args = parser.parse_args()

    run_experiment_matrix(
        timesteps=args.timesteps,
        seeds=args.seeds,
        scenarios=args.scenarios,
        include_ablations=not args.no_ablations,
    )
