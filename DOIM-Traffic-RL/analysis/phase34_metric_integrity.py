"""
DOIM-PPO Phase 34.4: Metric Integrity Audit and Recalculation
Audits metrics_step.csv across all 29 verified runs.
Definitively distinguishes:
  - Network Cumulative Waiting Time (W_net in seconds, instantaneous SUMO waiting time)
  - Estimated Per-Vehicle Delay (d_veh in s/veh)
  - Queue Length (mean and peak)
  - CO2 emissions (rate in mg/s and episode total in kg)
  - Delay Disparity (1 - Jain's Index, lower is better) vs Jain's Fairness Index (higher is better)
  - Safety Shield Interventions (total overrides)
"""

import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path

GIT_ROOT = Path("D:/DOIM-Traffic-RL").resolve()
PKG_ROOT = GIT_ROOT / "DOIM-Traffic-RL"
DATASET_PATH = GIT_ROOT / "results/phase34/verified_confirmatory_dataset.json"


def audit_run_metrics(run_path_str: str) -> dict:
    run_path = Path(run_path_str)
    csv_file = run_path / "metrics_step.csv"
    if not csv_file.exists():
        # Fallback to PKG_ROOT if relative path
        csv_file = PKG_ROOT / run_path_str / "metrics_step.csv"
        if not csv_file.exists():
            csv_file = GIT_ROOT / run_path_str / "metrics_step.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"metrics_step.csv not found for {run_path_str}")

    df = pd.read_csv(csv_file)
    steps = len(df)

    mean_wnet = float(df["raw_waiting"].mean())
    max_wnet = float(df["raw_waiting"].max())
    mean_queue = float(df["raw_queue"].mean())
    max_queue = float(df["raw_queue"].max())
    mean_co2_mg = float(df["raw_co2"].mean())
    total_co2_kg = float(df["raw_co2"].sum() * 1e-6)  # SUMO CO2 is mg/s, step is 1s
    
    # Cumulative throughput from summary.json if available
    summary_file = run_path / "summary.json"
    if not summary_file.exists():
        summary_file = PKG_ROOT / run_path_str / "summary.json"
    if not summary_file.exists():
        summary_file = GIT_ROOT / run_path_str / "summary.json"
    
    cumulative_throughput = float("nan")
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as sf:
                s_data = json.load(sf)
                cumulative_throughput = float(s_data.get("total_throughput", s_data.get("throughput", float("nan"))))
        except Exception:
            pass

    if np.isnan(cumulative_throughput):
        cumulative_throughput = float(df["raw_throughput"].sum()) if "raw_throughput" in df.columns else float("nan")
    
    # Fairness: raw_fairness is delay disparity = 1.0 - Jain
    mean_disparity = float(df["raw_fairness"].mean()) if "raw_fairness" in df.columns else float("nan")
    jain_index = float(1.0 - mean_disparity) if not np.isnan(mean_disparity) else float("nan")

    # Safety shield
    shield_interventions = int(df["shield_interventions"].sum()) if "shield_interventions" in df.columns else 0

    # Average active vehicles in network: estimated from throughput + current queue
    # Under Little's Law / steady-state network load:
    # N_veh is approx mean_queue * junctions or throughput / 1800 * travel_time
    # Here we use the empirical reference N_bar = 64.64 on grid3x3 (established in Phase 28.2)
    # or local calculation:
    est_per_veh_delay = mean_wnet / 64.64

    return {
        "steps": steps,
        "mean_w_net_s": round(mean_wnet, 2),
        "max_w_net_s": round(max_wnet, 2),
        "est_delay_per_veh_s": round(est_per_veh_delay, 2),
        "mean_queue_veh": round(mean_queue, 2),
        "max_queue_veh": round(max_queue, 2),
        "mean_co2_mg_s": round(mean_co2_mg, 1),
        "total_co2_kg": round(total_co2_kg, 2),
        "throughput_veh": int(cumulative_throughput) if not np.isnan(cumulative_throughput) else 0,
        "delay_disparity": round(mean_disparity, 4),
        "jain_fairness_index": round(jain_index, 4),
        "shield_interventions": shield_interventions,
    }


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    ds = meta["data"]

    recalculated_records = []

    for sc_name, algos in ds.items():
        for algo_name, seeds in algos.items():
            for seed_str, run_info in seeds.items():
                run_path = run_info["run_path"]
                m = audit_run_metrics(run_path)
                rec = {
                    "scenario": sc_name,
                    "algorithm": algo_name,
                    "seed": int(seed_str),
                    "run_id": run_info["run_id"],
                    **m,
                }
                recalculated_records.append(rec)

    out_csv = GIT_ROOT / "results/phase34/recalculated_metrics.csv"
    keys = list(recalculated_records[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(recalculated_records)

    print(f"Recalculated metrics for {len(recalculated_records)} runs.")
    print(f"Exported to {out_csv}")


if __name__ == "__main__":
    main()
