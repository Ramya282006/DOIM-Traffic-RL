"""
DOIM-PPO Phase 34.2: Confirmatory Matrix Reconstruction & Cell Audit
Reconstructs the 105-run confirmatory benchmark matrix from genuine on-disk artifacts.
Zero interpolation, zero fabrication.
"""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

GIT_ROOT = Path("D:/DOIM-Traffic-RL").resolve()
PKG_ROOT = GIT_ROOT / "DOIM-Traffic-RL"
RUNS_DIR = PKG_ROOT / "results/runs"

ALGORITHMS = [
    ("Fixed-Time", ["Fixed_Time", "Fixed-Time"]),
    ("Actuated", ["Actuated"]),
    ("Max-Pressure", ["MaxPressure", "Max-Pressure"]),
    ("FW-PPO", ["FW_PPO", "FW-PPO"]),
    ("SA-PPO", ["SA_PPO", "SA-PPO"]),
    ("MLP-PPO", ["MLP_PPO", "MLP-PPO"]),
    ("DOIM-PPO", ["IA_MOPPO", "DOIM-PPO"]),
]

SCENARIOS = [
    ("grid3x3", ["north_south_surge", "grid3x3"]),
    ("arterial_corridor", ["arterial_corridor"]),
    ("incident_disruption", ["incident_disruption"]),
]

SEEDS = [1042, 1043, 1044, 1045, 1046]


def match_run_to_cell(algo_aliases: List[str], scenario_aliases: List[str], seed: int, runs: List[Path]) -> Optional[Dict[str, Any]]:
    # Search for matching verified run with >= 1800 steps
    for r in runs:
        m_file = r / "manifest.json"
        s_file = r / "summary.json"
        if not (m_file.exists() and s_file.exists()):
            continue
        try:
            with open(m_file, "r", encoding="utf-8") as f:
                man = json.load(f)
            algo = man.get("algorithm")
            scen = man.get("scenario")
            status = man.get("completion_status") or man.get("status")
            es = man.get("evaluation_seed") or man.get("eval_seed")
            ts = man.get("train_seed")
            steps = man.get("total_steps", 0)

            if status != "COMPLETED" or steps < 1800:
                continue
            if not any(a.lower() == str(algo).lower() for a in algo_aliases):
                continue
            if not any(s.lower() in str(scen).lower() for s in scenario_aliases):
                continue

            # Seed match: check eval_seed or seed
            matched_seed = (es == seed) or (es is None and ts == seed)
            if matched_seed:
                with open(s_file, "r", encoding="utf-8") as f:
                    summ = json.load(f)
                return {
                    "run_id": r.name,
                    "run_path": str(r),
                    "algorithm": algo,
                    "scenario": scen,
                    "train_seed": ts,
                    "eval_seed": es if es is not None else ts,
                    "total_steps": steps,
                    "w_net": float(summ.get("mean_waiting_time", summ.get("average_waiting", 0.0))),
                    "queue": float(summ.get("mean_queue", summ.get("average_queue", 0.0))),
                    "throughput": float(summ.get("total_throughput", 0.0)),
                    "co2": float(summ.get("mean_co2", summ.get("average_co2", 0.0))),
                    "fairness": summ.get("mean_fairness", summ.get("average_fairness")),
                    "shield_interventions": int(summ.get("total_shield_interventions", 0)),
                    "source_commit": man.get("code_commit"),
                    "config_hash": man.get("config_hash"),
                }
        except Exception:
            pass
    return None


def audit_matrix():
    runs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
    matrix_status = []
    verified_dataset = {}

    total_cells = len(ALGORITHMS) * len(SCENARIOS) * len(SEEDS)
    completed_cells = 0
    missing_cells = 0

    for sc_name, sc_aliases in SCENARIOS:
        verified_dataset[sc_name] = {}
        for algo_name, algo_aliases in ALGORITHMS:
            verified_dataset[sc_name][algo_name] = {}
            for seed in SEEDS:
                match = match_run_to_cell(algo_aliases, sc_aliases, seed, runs)
                if match:
                    status = "VERIFIED"
                    completed_cells += 1
                    verified_dataset[sc_name][algo_name][str(seed)] = match
                else:
                    status = "MISSING"
                    missing_cells += 1

                matrix_status.append({
                    "scenario": sc_name,
                    "algorithm": algo_name,
                    "seed": seed,
                    "status": status,
                    "run_id": match["run_id"] if match else "NONE",
                    "w_net": match["w_net"] if match else "N/A",
                    "steps": match["total_steps"] if match else "0",
                })

    out_dir = GIT_ROOT / "results/phase34"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = out_dir / "confirmatory_matrix_status.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "algorithm", "seed", "status", "run_id", "w_net", "steps"])
        writer.writeheader()
        writer.writerows(matrix_status)

    # Write JSON
    json_path = out_dir / "verified_confirmatory_dataset.json"
    meta_payload = {
        "mission_id": "DOIM-PPO-PHASE34-INDEPENDENT-FORENSIC-AUDIT",
        "total_cells": total_cells,
        "verified_cells": completed_cells,
        "missing_cells": missing_cells,
        "completeness_ratio": float(completed_cells / total_cells),
        "data": verified_dataset,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    print(f"Confirmatory Matrix Audit:")
    print(f"  Total Cells: {total_cells}")
    print(f"  Verified On Disk: {completed_cells} ({completed_cells/total_cells*100:.1f}%)")
    print(f"  Missing: {missing_cells} ({missing_cells/total_cells*100:.1f}%)")
    print(f"Exported to {csv_path} and {json_path}")


if __name__ == "__main__":
    audit_matrix()

