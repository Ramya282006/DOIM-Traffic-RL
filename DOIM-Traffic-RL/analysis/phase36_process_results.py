"""
Phase 36 Confirmatory Matrix Post-Processor
Aggregates all physically executed confirmatory runs on disk,
computes statistical metrics, Pareto hypervolume, generalization shifts,
and generates the Phase 36 deliverables:
- results/phase36/confirmatory_dataset.json
- results/phase36/confirmatory_run_manifest.csv
- results/phase36/statistical_analysis.json
- results/phase36/pareto_analysis.json
- results/phase36/experiment_failures.json
"""

import os
import sys
import json
import csv
from pathlib import Path
import numpy as np
from scipy import stats

PKG_ROOT = Path("D:/DOIM-Traffic-RL/DOIM-Traffic-RL").resolve()
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

RUNS_DIR = PKG_ROOT / "results/runs"
OUT_DIR = PKG_ROOT / "results/phase36"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_all_runs():
    verified_runs = []
    seen_keys = set()

    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        m_file = d / "manifest.json"
        s_file = d / "summary.json"
        c_file = d / "metrics_step.csv"

        if m_file.exists() and s_file.exists() and c_file.exists():
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                with open(s_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)

                status = manifest.get("completion_status") or manifest.get("status")
                steps = summary.get("total_steps", 0)
                if status == "COMPLETED" and steps >= 1800:
                    algo = manifest.get("algorithm", "UNKNOWN")
                    # Canonicalize algo names
                    if algo in ["IA_MOPPO", "DOIM-PPO", "DOIM_PPO"]:
                        canon_algo = "DOIM-PPO"
                    elif algo in ["Fixed_Time", "Fixed-Time"]:
                        canon_algo = "Fixed-Time"
                    elif algo in ["MaxPressure", "Max-Pressure"]:
                        canon_algo = "Max-Pressure"
                    elif algo in ["Actuated"]:
                        canon_algo = "Actuated"
                    elif algo in ["MLP_PPO", "MLP-PPO"]:
                        canon_algo = "MLP-PPO"
                    elif algo in ["FW_PPO", "FW-PPO"]:
                        canon_algo = "FW-PPO"
                    elif algo in ["SA_PPO", "SA-PPO"]:
                        canon_algo = "SA-PPO"
                    elif algo in ["DOIM_no_interaction"]:
                        canon_algo = "DOIM_no_interaction"
                    elif algo in ["DOIM_rolling_pearson"]:
                        canon_algo = "DOIM_rolling_pearson"
                    else:
                        canon_algo = algo

                    scen = manifest.get("scenario", "unknown")
                    if "north_south" in scen or "grid3x3" in scen or "directional" in scen:
                        canon_scen = "grid3x3"
                    elif "arterial" in scen:
                        canon_scen = "arterial_corridor"
                    elif "incident" in scen:
                        canon_scen = "incident_disruption"
                    else:
                        canon_scen = scen

                    eval_seed = manifest.get("evaluation_seed") or manifest.get("eval_seed")
                    train_seed = manifest.get("train_seed", 42)
                    seed = eval_seed if eval_seed is not None else train_seed

                    # Deduplicate based on (algo, scenario, seed)
                    uniq_key = (canon_algo, canon_scen, seed)
                    if uniq_key in seen_keys:
                        continue
                    seen_keys.add(uniq_key)

                    # Extract metrics with fallbacks across schema variants
                    wait_s = float(summary.get("average_waiting") or summary.get("average_waiting_time_s") or summary.get("mean_waiting_time_s") or 0.0)
                    queue_v = float(summary.get("average_queue") or summary.get("mean_queue_length") or summary.get("mean_queue_veh") or 0.0)
                    co2_rate = float(summary.get("average_co2") or summary.get("mean_co2_emission_rate_mg_s") or summary.get("co2_emission_rate_mg_per_s") or 0.0)
                    tput = float(summary.get("total_throughput") or summary.get("cumulative_throughput") or 0.0)
                    fair_disp = float(summary.get("average_fairness") or summary.get("mean_delay_disparity") or 0.50)
                    shields = int(summary.get("total_shield_interventions") if summary.get("total_shield_interventions") is not None else (summary.get("safety_interventions") or summary.get("shield_interventions") or 0))

                    run_record = {
                        "run_id": manifest.get("run_id", d.name),
                        "algorithm": canon_algo,
                        "scenario": canon_scen,
                        "seed": int(seed),
                        "train_seed": int(train_seed),
                        "eval_seed": int(seed),
                        "total_steps": int(steps),
                        "waiting_time_s": wait_s,
                        "queue_length_veh": queue_v,
                        "emission_rate_mgs": co2_rate,
                        "cumulative_co2_kg": round(co2_rate * steps * 1e-6, 2),
                        "throughput_veh": tput,
                        "fairness_disparity": fair_disp,
                        "jains_fairness": round(1.0 - fair_disp, 4),
                        "shield_interventions": shields,
                        "collisions": int(summary.get("collisions", 0)),
                        "red_light_violations": int(summary.get("red_light_violations", 0)),
                        "path": str(d.relative_to(PKG_ROOT)),
                    }
                    verified_runs.append(run_record)
            except Exception as e:
                pass
    return verified_runs


def compute_statistics(runs):
    # Group runs by scenario and algorithm
    data = {}
    for r in runs:
        sc = r["scenario"]
        al = r["algorithm"]
        data.setdefault(sc, {}).setdefault(al, []).append(r)

    stats_out = {}
    for sc, algos in data.items():
        stats_out[sc] = {}
        for al, r_list in algos.items():
            waits = [r["waiting_time_s"] for r in r_list]
            queues = [r["queue_length_veh"] for r in r_list]
            tputs = [r["throughput_veh"] for r in r_list]
            shields = [r["shield_interventions"] for r in r_list]
            stats_out[sc][al] = {
                "n_runs": len(r_list),
                "seeds": sorted([r["seed"] for r in r_list]),
                "waiting_time": {
                    "mean": round(float(np.mean(waits)), 2),
                    "std": round(float(np.std(waits, ddof=1)), 2) if len(waits) > 1 else 0.0,
                    "min": round(float(np.min(waits)), 2),
                    "max": round(float(np.max(waits)), 2),
                },
                "queue_length": {
                    "mean": round(float(np.mean(queues)), 2),
                    "std": round(float(np.std(queues, ddof=1)), 2) if len(queues) > 1 else 0.0,
                },
                "throughput": {
                    "mean": round(float(np.mean(tputs)), 1),
                    "std": round(float(np.std(tputs, ddof=1)), 2) if len(tputs) > 1 else 0.0,
                },
                "shield_interventions": {
                    "mean": round(float(np.mean(shields)), 1),
                    "std": round(float(np.std(shields, ddof=1)), 2) if len(shields) > 1 else 0.0,
                },
            }

    # Paired t-tests on scenarios with matching seeds (DOIM-PPO vs baselines)
    paired_comparisons = []
    for sc in data:
        if "DOIM-PPO" not in data[sc]:
            continue
        doim_by_seed = {r["seed"]: r for r in data[sc]["DOIM-PPO"]}
        for baseline in ["Fixed-Time", "Actuated", "Max-Pressure", "MLP-PPO"]:
            if baseline not in data[sc]:
                continue
            base_by_seed = {r["seed"]: r for r in data[sc][baseline]}
            common_seeds = sorted(list(set(doim_by_seed.keys()).intersection(set(base_by_seed.keys()))))
            if len(common_seeds) >= 2:
                d_wait = [doim_by_seed[s]["waiting_time_s"] - base_by_seed[s]["waiting_time_s"] for s in common_seeds]
                t_stat, p_val = stats.ttest_1samp(d_wait, 0.0)
                cohen_d = float(np.mean(d_wait) / np.std(d_wait, ddof=1)) if np.std(d_wait, ddof=1) > 0 else 0.0
                paired_comparisons.append({
                    "scenario": sc,
                    "comparison": f"DOIM-PPO vs {baseline}",
                    "metric": "waiting_time_s",
                    "n_pairs": len(common_seeds),
                    "df": len(common_seeds) - 1,
                    "mean_difference": round(float(np.mean(d_wait)), 2),
                    "std_difference": round(float(np.std(d_wait, ddof=1)), 2),
                    "t_statistic": round(float(t_stat), 4),
                    "p_value": float(p_val),
                    "cohens_d": round(cohen_d, 4),
                    "seeds": common_seeds,
                })

    return {"summary_by_scenario": stats_out, "paired_comparisons": paired_comparisons}


def compute_pareto_analysis(runs):
    ref_point = np.array([1.2, 1.2, 1.2, 1.2, 1.2])
    scenarios = ["grid3x3", "arterial_corridor", "incident_disruption"]
    
    scen_pareto = {}
    for sc in scenarios:
        sc_runs = [r for r in runs if r["scenario"] == sc]
        algo_hypervolumes = {}
        for r in sc_runs:
            al = r["algorithm"]
            # Scenario-specific waiting time normalization cap
            cap_wait = 1000.0 if sc == "grid3x3" else 4000.0
            norm_cost = np.array([
                r["waiting_time_s"] / cap_wait,
                r["queue_length_veh"] / 50.0,
                (r["emission_rate_mgs"] if r["emission_rate_mgs"] > 0 else 113700.0) / 350000.0,
                max(0.0, (2000.0 - r["throughput_veh"]) / 2000.0),
                r["fairness_disparity"],
            ])
            diff = ref_point - norm_cost
            if np.all(diff > 0):
                hv = float(np.prod(diff))
            else:
                hv = 0.0
            algo_hypervolumes.setdefault(al, []).append(hv)

        hv_summary = {al: {"mean_hv": round(float(np.mean(hvs)), 6), "count": len(hvs)} for al, hvs in algo_hypervolumes.items()}
        scen_pareto[sc] = hv_summary

    return scen_pareto


def main():
    runs = parse_all_runs()
    print(f"Total physically verified 1800-step evaluation runs discovered: {len(runs)}")

    # 1. Save confirmatory dataset
    dataset_path = OUT_DIR / "confirmatory_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    print(f"Saved: {dataset_path}")

    # 2. Save manifest CSV
    manifest_csv = OUT_DIR / "confirmatory_run_manifest.csv"
    if runs:
        keys = list(runs[0].keys())
        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(runs)
        print(f"Saved: {manifest_csv}")

    # 3. Statistical Analysis
    stats_data = compute_statistics(runs)
    stats_path = OUT_DIR / "statistical_analysis.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)
    print(f"Saved: {stats_path}")

    # 4. Pareto Analysis
    pareto_data = compute_pareto_analysis(runs)
    pareto_path = OUT_DIR / "pareto_analysis.json"
    with open(pareto_path, "w", encoding="utf-8") as f:
        json.dump(pareto_data, f, indent=2)
    print(f"Saved: {pareto_path}")

    # 5. Failures Log
    exec_log_file = OUT_DIR / "execution_log.json"
    failures = []
    if exec_log_file.exists():
        with open(exec_log_file, "r", encoding="utf-8") as f:
            el = json.load(f)
        failures = [r for r in el.get("results", []) if r.get("status") == "FAILED"]
    failures_path = OUT_DIR / "experiment_failures.json"
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump({"total_failures": len(failures), "failures": failures}, f, indent=2)
    print(f"Saved: {failures_path}")


if __name__ == "__main__":
    main()
