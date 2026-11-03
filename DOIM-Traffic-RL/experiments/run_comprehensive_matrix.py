"""
DOIM-Traffic-RL: Comprehensive Experimental Matrix Runner (Phase H)
Executes multi-seed, multi-scenario evaluation of Proposed DOIM-Traffic-RL (IA-MOPPO)
against all transportation engineering baselines (Fixed-Time, Actuated, Max-Pressure),
reinforcement learning baselines (FW-PPO, SA-PPO, parameter-matched MLP-PPO), and ablations.

Supports:
  - Experiment Tiers: 'smoke' (128), 'pilot' (5,000), 'intermediate' (10,000), 'main_convergence' (50,000)
  - Strict Compute Gate: Aborts if steps > 5,000 without explicit --approve_expensive flag.
  - Independent training seeds and held-out deterministic evaluation seeds (seed + 1000).
  - Checkpoint provenance logging and fail-closed manifest generation.
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
from baselines.fixed_time.fixed_time_controller import run_fixed_time_experiment, evaluate_fixed_time
from baselines.actuated.actuated_controller import run_actuated_experiment
from baselines.max_pressure.max_pressure_controller import run_max_pressure_experiment
from baselines.ppo.fixed_weight_ppo import train_fixed_weight_ppo
from baselines.ppo.state_adaptive_ppo import train_state_adaptive_ppo
from baselines.ppo.mlp_ppo import train_mlp_ppo
from analysis.statistical_analyzer import StatisticalAnalyzer
from analysis.plot_generator import PublicationPlotGenerator


TIER_BUDGETS = {
    "smoke": 128,
    "pilot": 5000,
    "intermediate": 10000,
    "main_convergence": 50000,
}


def run_experiment_matrix(
    tier: str = "smoke",
    timesteps: Optional[int] = None,
    seeds: List[int] = [42],
    include_ablations: bool = True,
    include_all_baselines: bool = True,
    scenarios: List[str] = ["scenarios/directional_shift/north_south_surge.sumocfg"],
    eval_episodes: int = 1,
    eval_horizon: int = 1800,
    checkpoint_interval: int = 2500,
    output_root: str = "results/runs",
    approve_expensive: bool = False,
):
    step_budget = timesteps if timesteps is not None else TIER_BUDGETS.get(tier.lower(), 128)
    
    # Strict compute approval gate
    if step_budget > 5000 and not approve_expensive:
        raise PermissionError(
            f"COMPUTE APPROVAL BLOCKED: Requested training budget is {step_budget} steps (>5,000 threshold). "
            f"Under IEEE T-ITS experimental protocol, expensive confirmatory experiments require explicit user sign-off. "
            f"Please present compute plan to user and rerun with --approve_expensive."
        )

    num_algos = 7 if include_all_baselines else 4
    if include_ablations:
        num_algos += 2
        
    total_runs = num_algos * len(seeds) * len(scenarios)
    print("=" * 80)
    print(f"STARTING DOIM-TRAFFIC-RL EXPERIMENT MATRIX (TIER: {tier.upper()})")
    print(f"Configurations: {num_algos} algorithms | Total Runs: {total_runs}")
    print(f"Seeds: {seeds} | Scenarios: {scenarios}")
    print(f"Training Steps/Run: {step_budget} | Eval Episodes: {eval_episodes} (Horizon: {eval_horizon}s)")
    print("=" * 80)

    start_time = time.time()

    for scenario in scenarios:
        for seed in seeds:
            print(f"\n=======================================================")
            print(f">>> Scenario: {scenario} | Seed: {seed}")
            print(f"=======================================================")

            # 1. Proposed DOIM-Traffic-RL (IA-MOPPO)
            print("\n--- [1/7] Running Proposed IA-MOPPO ---")
            train_interaction_aware_ppo(
                scenario_override=scenario,
                total_timesteps=step_budget,
                seed=seed,
                output_root=output_root,
                checkpoint_interval=checkpoint_interval,
                eval_episodes=eval_episodes,
                eval_horizon=eval_horizon,
            )

            # 2. Fixed-Time Baseline
            print("\n--- [2/7] Running Fixed-Time Baseline ---")
            run_fixed_time_experiment(
                scenario_override=scenario,
                total_timesteps=step_budget,
                seed=seed,
                output_root=output_root,
            )
            if eval_episodes > 0:
                evaluate_fixed_time(
                    scenario_override=scenario,
                    n_episodes=eval_episodes,
                    eval_seed=seed + 1000,
                    train_seed=seed,
                    eval_horizon=eval_horizon,
                    max_steps_per_episode=eval_horizon,
                    output_root=output_root,
                )

            # 3. Actuated Traffic Controller Baseline
            if include_all_baselines:
                print("\n--- [3/7] Running Actuated Controller Baseline ---")
                run_actuated_experiment(
                    scenario_override=scenario,
                    total_timesteps=step_budget,
                    seed=seed,
                    output_root=output_root,
                )

            # 4. Max-Pressure Controller Baseline
            if include_all_baselines:
                print("\n--- [4/7] Running Max-Pressure Controller Baseline ---")
                run_max_pressure_experiment(
                    scenario_override=scenario,
                    total_timesteps=step_budget,
                    seed=seed,
                    output_root=output_root,
                )

            # 5. Fixed-Weight PPO Baseline
            print("\n--- [5/7] Running Fixed-Weight PPO Baseline ---")
            train_fixed_weight_ppo(
                scenario_override=scenario,
                total_timesteps=step_budget,
                seed=seed,
                eval_episodes=eval_episodes,
                eval_horizon=eval_horizon,
                output_root=output_root,
            )

            # 6. State-Adaptive PPO Baseline
            print("\n--- [6/7] Running State-Adaptive PPO Baseline ---")
            train_state_adaptive_ppo(
                scenario_override=scenario,
                total_timesteps=step_budget,
                seed=seed,
                eval_episodes=eval_episodes,
                eval_horizon=eval_horizon,
                output_root=output_root,
            )

            # 7. Parameter-Matched MLP-PPO Baseline
            if include_all_baselines:
                print("\n--- [7/7] Running Parameter-Matched MLP-PPO Baseline ---")
                train_mlp_ppo(
                    scenario_override=scenario,
                    total_timesteps=step_budget,
                    seed=seed,
                    output_root=output_root,
                    eval_episodes=eval_episodes,
                    eval_horizon=eval_horizon,
                )

            if include_ablations:
                # Ablation 1: No Interaction Manifold (M = 0)
                print("\n--- [Ablation 1] Running IA-MOPPO w/o Interaction Manifold ---")
                train_interaction_aware_ppo(
                    scenario_override=scenario,
                    total_timesteps=step_budget,
                    seed=seed,
                    ablation_mode="no_interaction",
                    output_root=output_root,
                    checkpoint_interval=checkpoint_interval,
                    eval_episodes=eval_episodes,
                    eval_horizon=eval_horizon,
                )

                # Ablation 2: Rolling Pearson Correlation
                print("\n--- [Ablation 2] Running IA-MOPPO w/ Rolling Pearson Correlation ---")
                train_interaction_aware_ppo(
                    scenario_override=scenario,
                    total_timesteps=step_budget,
                    seed=seed,
                    ablation_mode="rolling_pearson",
                    output_root=output_root,
                    checkpoint_interval=checkpoint_interval,
                    eval_episodes=eval_episodes,
                    eval_horizon=eval_horizon,
                )

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"EXPERIMENT MATRIX COMPLETED in {elapsed:.2f} seconds ({elapsed/60.0:.2f} minutes).")
    print("=" * 80)

    # Post-run analysis and reporting
    analyzer = StatisticalAnalyzer(runs_root=output_root)
    df = analyzer.load_run_summaries(reject_incomplete=True)
    print(f"Total valid runs loaded: {len(df)}")
    
    table_csv = Path(output_root).parent / "tables" / f"matrix_{tier}_comparison.csv"
    table_md = Path(output_root).parent / "tables" / f"matrix_{tier}_comparison.md"
    analyzer.export_latex_tables(output_csv=table_csv, output_markdown=table_md)
    print(f"Generated comparison tables: {table_csv} and {table_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DOIM-Traffic-RL Comprehensive Matrix Runner")
    parser.add_argument("--tier", type=str, default="smoke", choices=["smoke", "pilot", "intermediate", "main_convergence"],
                        help="Experiment budget tier")
    parser.add_argument("--timesteps", type=int, default=None, help="Explicit timestep override")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="List of random seeds")
    parser.add_argument("--scenarios", type=str, nargs="+", default=["scenarios/directional_shift/north_south_surge.sumocfg"],
                        help="List of scenario .sumocfg files")
    parser.add_argument("--eval_episodes", type=int, default=1, help="Number of evaluation episodes per run")
    parser.add_argument("--eval_horizon", type=int, default=1800, help="Evaluation horizon in simulated seconds")
    parser.add_argument("--no_ablations", action="store_true", help="Disable ablation variants")
    parser.add_argument("--output_root", type=str, default="results/runs", help="Output directory for runs")
    parser.add_argument("--approve_expensive", action="store_true", help="Explicit user sign-off for >5,000 step runs")

    args = parser.parse_args()

    run_experiment_matrix(
        tier=args.tier,
        timesteps=args.timesteps,
        seeds=args.seeds,
        include_ablations=not args.no_ablations,
        include_all_baselines=True,
        scenarios=args.scenarios,
        eval_episodes=args.eval_episodes,
        eval_horizon=args.eval_horizon,
        output_root=args.output_root,
        approve_expensive=args.approve_expensive,
    )
