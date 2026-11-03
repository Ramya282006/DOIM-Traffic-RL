"""
DOIM-Traffic-RL: Confirmatory Benchmark Baseline Runner (Phase 29)
Executes all 45 baseline evaluation runs:
  - Algorithms: Fixed-Time, Actuated, Max-Pressure
  - Scenarios: grid3x3 (north_south_surge), arterial_corridor, incident_disruption
  - Seeds: 1042, 1043, 1044, 1045, 1046
  - Horizon: 1800 steps (deterministic evaluation)
Idempotent and parallelized across worker processes.
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

SEEDS = [1042, 1043, 1044, 1045, 1046]
BASELINES = ["Fixed-Time", "Actuated", "Max-Pressure"]


def is_run_completed(algo_name: str, scenario_stem: str, seed: int) -> bool:
    runs_dir = Path(OUTPUT_ROOT)
    if not runs_dir.exists():
        return False
    # Check matching completed run
    pattern = f"{algo_name}_{scenario_stem}*"
    for d in runs_dir.glob(pattern):
        if not d.is_dir():
            continue
        summary_file = d / "summary.json"
        manifest_file = d / "manifest.json"
        metrics_file = d / "metrics_step.csv"
        if summary_file.exists() and manifest_file.exists() and metrics_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                status = m.get("completion_status") or m.get("status")
                eval_s = m.get("evaluation_seed") or m.get("eval_seed")
                train_s = m.get("train_seed")
                if (
                    status == "COMPLETED"
                    and (eval_s == seed or train_s == seed)
                    and m.get("total_steps", 0) >= 1800
                ):
                    return True
            except Exception:
                pass
    return False


def run_single_baseline(task_tuple):
    os.chdir(PKG_DIR)
    algo, scenario_key, scenario_cfg, seed = task_tuple
    scenario_stem = Path(scenario_cfg).stem

    # Map algo names to directory patterns
    algo_slug = "Fixed_Time" if algo == "Fixed-Time" else ("MaxPressure" if algo == "Max-Pressure" else algo)
    if is_run_completed(algo_slug, scenario_stem, seed):
        print(f"[SKIP] {algo} on {scenario_key} (seed {seed}) already completed.")
        return {"algo": algo, "scenario": scenario_key, "seed": seed, "status": "REUSED"}

    print(f"[START] Running {algo} on {scenario_key} (seed {seed}) for 1800 steps...")
    t0 = time.time()

    if algo == "Fixed-Time":
        from baselines.fixed_time.fixed_time_controller import run_fixed_time_experiment
        summary = run_fixed_time_experiment(
            scenario_override=scenario_cfg,
            total_timesteps=1800,
            seed=seed,
            eval_seed=seed,
            output_root=OUTPUT_ROOT,
            evaluation_mode="deterministic_eval",
        )
    elif algo == "Actuated":
        from baselines.actuated.actuated_controller import run_actuated_experiment
        summary = run_actuated_experiment(
            scenario_override=scenario_cfg,
            total_timesteps=1800,
            seed=seed,
            output_root=OUTPUT_ROOT,
        )
    elif algo == "Max-Pressure":
        from baselines.max_pressure.max_pressure_controller import run_max_pressure_experiment
        summary = run_max_pressure_experiment(
            scenario_override=scenario_cfg,
            total_timesteps=1800,
            seed=seed,
            output_root=OUTPUT_ROOT,
        )
    else:
        raise ValueError(f"Unknown baseline: {algo}")

    elapsed = time.time() - t0
    print(f"[FINISH] {algo} on {scenario_key} (seed {seed}) completed in {elapsed:.1f}s.")
    return {"algo": algo, "scenario": scenario_key, "seed": seed, "status": "COMPLETED", "elapsed": elapsed}


def main():
    tasks = []
    for sc_key, sc_cfg in SCENARIOS.items():
        for algo in BASELINES:
            for seed in SEEDS:
                tasks.append((algo, sc_key, sc_cfg, seed))

    print(f"Total baseline tasks in confirmatory matrix: {len(tasks)}")
    t_start = time.time()

    with mp.Pool(4) as pool:
        results = pool.map(run_single_baseline, tasks)

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL BASELINE RUNS PROCESSED in {total_time:.1f}s ({total_time/60.0:.2f} min)")
    print("=" * 80)
    for r in results:
        print(f"  {r['algo']} | {r['scenario']} | Seed {r['seed']} -> {r['status']}")


if __name__ == "__main__":
    main()
