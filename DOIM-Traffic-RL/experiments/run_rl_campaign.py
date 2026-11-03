"""
DOIM-Traffic-RL: Confirmatory Benchmark RL Runner (Phase 29)
Executes training and evaluation for all 4 RL algorithms:
  - Algorithms: FW-PPO, SA-PPO, MLP-PPO, DOIM-PPO (IA-MOPPO)
  - Scenarios: grid3x3 (north_south_surge), arterial_corridor, incident_disruption
  - Seeds: 42, 43, 44, 45, 46 (Evaluated on 1042, 1043, 1044, 1045, 1046)
  - Budget: 5,000 training steps (40 PPO updates, 800 gradient steps)
  - Eval Horizon: 1,800 steps (deterministic evaluation)
Idempotent, resumable, and parallelized across worker processes.
"""

import os
import sys
import json
import time
from pathlib import Path
import multiprocessing as mp

PKG_DIR = Path("D:/DOIM-Traffic-RL/DOIM-Traffic-RL").resolve()
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

OUTPUT_ROOT = str(PKG_DIR / "results/runs")

SCENARIOS = {
    "grid3x3": str(PKG_DIR / "scenarios/directional_shift/north_south_surge.sumocfg"),
    "arterial_corridor": str(PKG_DIR / "scenarios/arterial_corridor/arterial_corridor.sumocfg"),
    "incident_disruption": str(PKG_DIR / "scenarios/incident_disruption/incident_disruption.sumocfg"),
}

RL_ALGORITHMS = ["FW_PPO", "SA_PPO", "MLP_PPO", "IA_MOPPO"]
TRAIN_SEEDS = [42, 43, 44, 45, 46]


def is_eval_completed(algo_name: str, scenario_stem: str, eval_seed: int) -> bool:
    runs_dir = Path(OUTPUT_ROOT)
    if not runs_dir.exists():
        return False
    pattern = f"{algo_name}_{scenario_stem}*"
    for d in runs_dir.glob(pattern):
        if not d.is_dir():
            continue
        m_file = d / "manifest.json"
        s_file = d / "summary.json"
        if m_file.exists() and s_file.exists():
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                status = m.get("completion_status") or m.get("status")
                es = m.get("evaluation_seed") or m.get("eval_seed")
                if status == "COMPLETED" and es == eval_seed and m.get("total_steps", 0) >= 1800:
                    return True
            except Exception:
                pass
    return False


def find_existing_checkpoint(algo_name: str, scenario_stem: str, train_seed: int) -> Optional[Path]:
    runs_dir = Path(OUTPUT_ROOT)
    if not runs_dir.exists():
        return None
    pattern = f"{algo_name}_{scenario_stem}*"
    for d in runs_dir.glob(pattern):
        if not d.is_dir():
            continue
        m_file = d / "manifest.json"
        if m_file.exists():
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                ts = m.get("train_seed")
                if ts == train_seed:
                    ckpt_dir = d / "checkpoints"
                    if ckpt_dir.exists():
                        for pt in ckpt_dir.glob("*.pt"):
                            if pt.name.endswith("final.pt") or "5000" in pt.name:
                                return pt
            except Exception:
                pass
    return None


def run_single_rl(task_tuple):
    os.chdir(PKG_DIR)
    algo, sc_key, sc_cfg, train_seed = task_tuple
    eval_seed = train_seed + 1000
    sc_stem = Path(sc_cfg).stem

    if is_eval_completed(algo, sc_stem, eval_seed):
        print(f"[SKIP] {algo} on {sc_key} (seed {train_seed} -> {eval_seed}) eval already completed.")
        return {"algo": algo, "scenario": sc_key, "seed": train_seed, "status": "REUSED"}

    print(f"[START] Processing {algo} on {sc_key} (train seed {train_seed}, eval seed {eval_seed})...")
    t0 = time.time()

    # Step 1: Check if checkpoint exists
    ckpt_path = find_existing_checkpoint(algo, sc_stem, train_seed)

    if ckpt_path is None:
        print(f"  Training {algo} for 5000 steps on {sc_key} (seed {train_seed})...")
        if algo == "IA_MOPPO":
            from training.interaction_aware_ppo import train_interaction_aware_ppo
            train_run = train_interaction_aware_ppo(
                scenario_override=sc_cfg,
                total_timesteps=5000,
                seed=train_seed,
                output_root=OUTPUT_ROOT,
                eval_episodes=1,
                eval_horizon=1800,
                eval_seed=eval_seed,
            )
        elif algo == "FW_PPO":
            from baselines.ppo.fixed_weight_ppo import train_fixed_weight_ppo
            train_run = train_fixed_weight_ppo(
                scenario_override=sc_cfg,
                total_timesteps=5000,
                seed=train_seed,
                output_root=OUTPUT_ROOT,
                eval_episodes=1,
                eval_horizon=1800,
                eval_seed=eval_seed,
            )
        elif algo == "SA_PPO":
            from baselines.ppo.state_adaptive_ppo import train_state_adaptive_ppo
            train_run = train_state_adaptive_ppo(
                scenario_override=sc_cfg,
                total_timesteps=5000,
                seed=train_seed,
                output_root=OUTPUT_ROOT,
                eval_episodes=1,
                eval_horizon=1800,
                eval_seed=eval_seed,
            )
        elif algo == "MLP_PPO":
            from baselines.ppo.mlp_ppo import train_mlp_ppo
            train_run = train_mlp_ppo(
                scenario_override=sc_cfg,
                total_timesteps=5000,
                seed=train_seed,
                output_root=OUTPUT_ROOT,
                eval_episodes=1,
                eval_horizon=1800,
                eval_seed=eval_seed,
            )
        else:
            raise ValueError(f"Unknown RL algo: {algo}")
    else:
        print(f"  Checkpoint found: {ckpt_path}. Running deterministic eval on seed {eval_seed}...")
        if algo == "IA_MOPPO":
            from training.interaction_aware_ppo import evaluate_interaction_aware_ppo
            evaluate_interaction_aware_ppo(
                agent_or_checkpoint=str(ckpt_path),
                scenario_override=sc_cfg,
                eval_seed=eval_seed,
                train_seed=train_seed,
                n_episodes=1,
                eval_horizon=1800,
                output_root=OUTPUT_ROOT,
                training_timesteps=5000,
            )
        elif algo == "FW_PPO":
            from baselines.ppo.fixed_weight_ppo import evaluate_fixed_weight_ppo
            evaluate_fixed_weight_ppo(
                agent_or_checkpoint=str(ckpt_path),
                scenario_override=sc_cfg,
                eval_seed=eval_seed,
                train_seed=train_seed,
                n_episodes=1,
                eval_horizon=1800,
                output_root=OUTPUT_ROOT,
                training_timesteps=5000,
            )
        elif algo == "SA_PPO":
            from baselines.ppo.state_adaptive_ppo import evaluate_state_adaptive_ppo
            evaluate_state_adaptive_ppo(
                agent_or_checkpoint=str(ckpt_path),
                scenario_override=sc_cfg,
                eval_seed=eval_seed,
                train_seed=train_seed,
                n_episodes=1,
                eval_horizon=1800,
                output_root=OUTPUT_ROOT,
                training_timesteps=5000,
            )
        elif algo == "MLP_PPO":
            from baselines.ppo.mlp_ppo import evaluate_mlp_ppo
            evaluate_mlp_ppo(
                agent_or_checkpoint=str(ckpt_path),
                scenario_override=sc_cfg,
                eval_seed=eval_seed,
                train_seed=train_seed,
                n_episodes=1,
                eval_horizon=1800,
                output_root=OUTPUT_ROOT,
                training_timesteps=5000,
            )

    elapsed = time.time() - t0
    print(f"[FINISH] {algo} on {sc_key} (seed {train_seed} -> {eval_seed}) finished in {elapsed:.1f}s.")
    return {"algo": algo, "scenario": sc_key, "seed": train_seed, "status": "COMPLETED", "elapsed": elapsed}


def main():
    tasks = []
    # Order: IA_MOPPO first, then baselines
    for sc_key, sc_cfg in SCENARIOS.items():
        for algo in RL_ALGORITHMS:
            for seed in TRAIN_SEEDS:
                tasks.append((algo, sc_key, sc_cfg, seed))

    print(f"Total RL tasks in confirmatory matrix: {len(tasks)}")
    t_start = time.time()

    with mp.Pool(3) as pool:
        results = pool.map(run_single_rl, tasks)

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL RL RUNS PROCESSED in {total_time:.1f}s ({total_time/60.0:.2f} min)")
    print("=" * 80)
    for r in results:
        print(f"  {r['algo']} | {r['scenario']} | Seed {r['seed']} -> {r['status']}")


if __name__ == "__main__":
    main()
