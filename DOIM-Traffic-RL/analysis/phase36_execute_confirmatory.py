"""
Phase 36 Confirmatory Experiment Matrix Runner
Executes missing confirmatory cells across scenarios and seeds:
- Algorithms: Fixed-Time, Actuated, Max-Pressure, MLP-PPO, DOIM-PPO (IA-MOPPO)
- Scenarios: grid3x3 (north_south_surge), arterial_corridor, incident_disruption
- Evaluation Seeds: 1042, 1043, 1044, 1045, 1046
- Horizon: 1800 steps per run (deterministic evaluation, safety shield active)
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch

PKG_ROOT = Path("D:/DOIM-Traffic-RL/DOIM-Traffic-RL").resolve()
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
os.chdir(PKG_ROOT)

from common.config import load_experiment_config
from common.experiment import ExperimentRun
from common.seeding import seed_everything
from environment.sumo_env import DOIMTrafficEnv
from baselines.fixed_time.fixed_time_controller import run_fixed_time_experiment
from baselines.actuated.actuated_controller import run_actuated_experiment
from baselines.max_pressure.max_pressure_controller import run_max_pressure_experiment
from training.interaction_aware_ppo import evaluate_interaction_aware_ppo
from baselines.ppo.mlp_ppo import MLPPPOAgent

OUTPUT_ROOT = str(PKG_ROOT / "results/runs")

SCENARIOS = {
    "grid3x3": str(PKG_ROOT / "scenarios/directional_shift/north_south_surge.sumocfg"),
    "arterial_corridor": str(PKG_ROOT / "scenarios/arterial_corridor/arterial_corridor.sumocfg"),
    "incident_disruption": str(PKG_ROOT / "scenarios/incident_disruption/incident_disruption.sumocfg"),
}

SEEDS = [1042, 1043, 1044, 1045, 1046]

DOIM_CKPT = str(PKG_ROOT / "results/runs/IA_MOPPO_north_south_surge_s42_20261103_195743/checkpoints/ia_moppo_final.pt")
MLP_CKPT = str(PKG_ROOT / "results/runs/MLP_PPO_north_south_surge_s42_20261103_200832/checkpoints/mlp_ppo_final.pt")


def is_run_already_completed(algo_name: str, scenario_name: str, seed: int) -> bool:
    runs_dir = Path(OUTPUT_ROOT)
    if not runs_dir.exists():
        return False
    # Look for matching completed directory
    patterns = [
        f"{algo_name}_{scenario_name}*",
        f"EVAL_{algo_name}_{scenario_name}*",
        f"*{algo_name}*s*{seed}*",
    ]
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        m_file = d / "manifest.json"
        s_file = d / "summary.json"
        c_file = d / "metrics_step.csv"
        if m_file.exists() and s_file.exists() and c_file.exists():
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                status = m.get("completion_status") or m.get("status")
                eval_s = m.get("evaluation_seed") or m.get("eval_seed")
                train_s = m.get("train_seed")
                scen = m.get("scenario") or ""
                algo = m.get("algorithm") or ""
                if status == "COMPLETED" and (eval_s == seed or (train_s == seed and eval_s is None)):
                    if scenario_name in scen or scen in scenario_name:
                        # Match algo
                        if algo_name in algo or algo in algo_name:
                            return True
            except Exception:
                pass
    return False


def run_eval_mlp_ppo(scenario_path: str, eval_seed: int, train_seed: int = 42, horizon: int = 1800):
    config = load_experiment_config()
    config.simulation.sumo_config = scenario_path
    scenario_name = Path(scenario_path).stem

    agent = MLPPPOAgent(config)
    state_dict = torch.load(MLP_CKPT, map_location="cpu")
    agent.network.load_state_dict(state_dict)
    agent.network.eval()

    seed_everything(eval_seed)
    run = ExperimentRun(
        experiment_name=f"EVAL_MLP_PPO_{scenario_name}_s{train_seed}_e{eval_seed}",
        algorithm="MLP_PPO",
        config=config.to_dict() if hasattr(config, "to_dict") else dict(config),
        train_seed=train_seed,
        eval_seed=eval_seed,
        eval_horizon=horizon,
        scenario=scenario_name,
        output_root=OUTPUT_ROOT,
        training_timesteps=5000,
        episode_id=0,
        evaluation_mode="deterministic_eval",
    )

    env = DOIMTrafficEnv(
        sumo_config=scenario_path,
        max_steps=horizon,
        enforce_safety_shield=True,
    )

    obs, info = env.reset(seed=eval_seed)
    for step in range(horizon):
        feat = info.get("feature_matrix", np.zeros((9, 5))).flatten()
        raw = info.get("raw_state", np.zeros(25))
        flat_obs = torch.tensor(np.concatenate([feat, raw]), dtype=torch.float32)
        with torch.no_grad():
            action, value, log_prob = agent.act(flat_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        run.log_step(
            step=step,
            sim_time=float(step + 1),
            reward=float(reward),
            raw_objs=info.get("raw_objectives", np.zeros(5)),
            norm_objs=info.get("normalized_objectives", np.zeros(5)),
            weights=np.ones(5) * 0.2,
            actions=action.tolist() if hasattr(action, "tolist") else list(action),
            shield_interventions=info.get("shield_interventions", 0),
        )
        if terminated or truncated:
            break
    env.close()
    summary = run.close(status="COMPLETED", final_model_path=MLP_CKPT)
    return summary


def main():
    print("=" * 80)
    print("PHASE 36 CONFIRMATORY EXPERIMENTAL MATRIX EXECUTION")
    print(f"Scenarios: {list(SCENARIOS.keys())}")
    print(f"Seeds: {SEEDS}")
    print(f"Horizon: 1800 steps | Safety Shield: ACTIVE")
    print("=" * 80)

    start_total = time.time()
    results_log = []

    # 1. Transportation Engineering Baselines
    for sc_name, sc_cfg in SCENARIOS.items():
        sc_stem = Path(sc_cfg).stem
        for algo in ["Fixed-Time", "Actuated", "Max-Pressure"]:
            algo_slug = "Fixed_Time" if algo == "Fixed-Time" else ("MaxPressure" if algo == "Max-Pressure" else algo)
            for seed in SEEDS:
                tag = f"{algo} | {sc_name} | Seed {seed}"
                if is_run_already_completed(algo_slug, sc_stem, seed):
                    print(f"[REUSED] {tag}")
                    results_log.append({"algo": algo, "scenario": sc_name, "seed": seed, "status": "REUSED"})
                    continue

                t0 = time.time()
                print(f"[RUNNING] {tag}...")
                try:
                    if algo == "Fixed-Time":
                        res = run_fixed_time_experiment(
                            scenario_override=sc_cfg,
                            total_timesteps=1800,
                            seed=seed,
                            eval_seed=seed,
                            output_root=OUTPUT_ROOT,
                            evaluation_mode="deterministic_eval",
                        )
                    elif algo == "Actuated":
                        res = run_actuated_experiment(
                            scenario_override=sc_cfg,
                            total_timesteps=1800,
                            seed=seed,
                            output_root=OUTPUT_ROOT,
                        )
                    elif algo == "Max-Pressure":
                        res = run_max_pressure_experiment(
                            scenario_override=sc_cfg,
                            total_timesteps=1800,
                            seed=seed,
                            output_root=OUTPUT_ROOT,
                        )
                    elapsed = time.time() - t0
                    print(f"  -> SUCCESS in {elapsed:.1f}s")
                    results_log.append({"algo": algo, "scenario": sc_name, "seed": seed, "status": "SUCCESS", "elapsed": elapsed})
                except Exception as e:
                    print(f"  -> ERROR: {e}")
                    results_log.append({"algo": algo, "scenario": sc_name, "seed": seed, "status": "FAILED", "error": str(e)})

    # 2. RL Agents: DOIM-PPO (IA-MOPPO)
    for sc_name, sc_cfg in SCENARIOS.items():
        sc_stem = Path(sc_cfg).stem
        for seed in SEEDS:
            tag = f"DOIM-PPO | {sc_name} | Seed {seed}"
            if is_run_already_completed("IA_MOPPO", sc_stem, seed) or is_run_already_completed("DOIM_PPO", sc_stem, seed):
                print(f"[REUSED] {tag}")
                results_log.append({"algo": "DOIM-PPO", "scenario": sc_name, "seed": seed, "status": "REUSED"})
                continue

            t0 = time.time()
            print(f"[RUNNING] {tag}...")
            try:
                evaluate_interaction_aware_ppo(
                    agent_or_checkpoint=DOIM_CKPT,
                    scenario_override=sc_cfg,
                    eval_seed=seed,
                    train_seed=42,
                    n_episodes=1,
                    eval_horizon=1800,
                    output_root=OUTPUT_ROOT,
                    training_timesteps=5000,
                )
                elapsed = time.time() - t0
                print(f"  -> SUCCESS in {elapsed:.1f}s")
                results_log.append({"algo": "DOIM-PPO", "scenario": sc_name, "seed": seed, "status": "SUCCESS", "elapsed": elapsed})
            except Exception as e:
                print(f"  -> ERROR: {e}")
                results_log.append({"algo": "DOIM-PPO", "scenario": sc_name, "seed": seed, "status": "FAILED", "error": str(e)})

    # 3. RL Agents: MLP-PPO
    for sc_name, sc_cfg in SCENARIOS.items():
        sc_stem = Path(sc_cfg).stem
        for seed in SEEDS:
            tag = f"MLP-PPO | {sc_name} | Seed {seed}"
            if is_run_already_completed("MLP_PPO", sc_stem, seed):
                print(f"[REUSED] {tag}")
                results_log.append({"algo": "MLP-PPO", "scenario": sc_name, "seed": seed, "status": "REUSED"})
                continue

            t0 = time.time()
            print(f"[RUNNING] {tag}...")
            try:
                run_eval_mlp_ppo(
                    scenario_path=sc_cfg,
                    eval_seed=seed,
                    train_seed=42,
                    horizon=1800,
                )
                elapsed = time.time() - t0
                print(f"  -> SUCCESS in {elapsed:.1f}s")
                results_log.append({"algo": "MLP-PPO", "scenario": sc_name, "seed": seed, "status": "SUCCESS", "elapsed": elapsed})
            except Exception as e:
                print(f"  -> ERROR: {e}")
                results_log.append({"algo": "MLP-PPO", "scenario": sc_name, "seed": seed, "status": "FAILED", "error": str(e)})

    total_time = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"CONFIRMATORY EXPERIMENTAL MATRIX COMPLETED IN {total_time:.1f}s ({total_time/60:.2f} min)")
    print("=" * 80)

    # Save summary of Phase 36 execution log
    out_log = PKG_ROOT / "results/phase36/execution_log.json"
    out_log.parent.mkdir(parents=True, exist_ok=True)
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump({"total_time_s": total_time, "results": results_log}, f, indent=2)
    print(f"Execution log saved to {out_log}")


if __name__ == "__main__":
    main()

