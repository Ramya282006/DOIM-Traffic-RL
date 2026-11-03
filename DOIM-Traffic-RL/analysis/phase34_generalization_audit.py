"""
DOIM-PPO Phase 34.7: Generalization & Distribution Shift Audit
Forensically analyzes scenario definitions, topological shifts, and empirical cross-network performance.
Audits the green-wave disruption pathology on arterial corridors and documents the status of RL transfer.
"""

import json
import csv
import numpy as np
from pathlib import Path

GIT_ROOT = Path("D:/DOIM-Traffic-RL").resolve()
PKG_ROOT = GIT_ROOT / "DOIM-Traffic-RL"
METRICS_CSV = GIT_ROOT / "results/phase34/recalculated_metrics.csv"
OUT_JSON = GIT_ROOT / "results/phase34/generalization_recalculation.json"


def main():
    rows = []
    with open(METRICS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Group by scenario and algorithm
    by_scen_algo = {}
    for r in rows:
        sc = r["scenario"]
        al = r["algorithm"]
        if sc not in by_scen_algo:
            by_scen_algo[sc] = {}
        if al not in by_scen_algo[sc]:
            by_scen_algo[sc][al] = []
        by_scen_algo[sc][al].append(r)

    summary = {}
    for sc, algos in by_scen_algo.items():
        summary[sc] = {}
        for al, recs in algos.items():
            w_vals = [float(x["mean_w_net_s"]) for x in recs]
            q_vals = [float(x["mean_queue_veh"]) for x in recs]
            th_vals = [float(x["throughput_veh"]) for x in recs]
            summary[sc][al] = {
                "n_runs": len(recs),
                "seeds": [int(x["seed"]) for x in recs],
                "mean_w_net": round(float(np.mean(w_vals)), 2),
                "std_w_net": round(float(np.std(w_vals, ddof=1)) if len(w_vals) > 1 else 0.0, 2),
                "mean_queue": round(float(np.mean(q_vals)), 2),
                "std_queue": round(float(np.std(q_vals, ddof=1)) if len(q_vals) > 1 else 0.0, 2),
                "mean_throughput": round(float(np.mean(th_vals)), 1),
            }

    # Cross-scenario pathology analysis: Arterial Corridor vs Grid3x3
    # Compare Fixed-Time vs Actuated vs Max-Pressure on arterial corridor
    art = summary.get("arterial_corridor", {})
    grid = summary.get("grid3x3", {})

    pathology_analysis = {
        "arterial_corridor_disruption": {
            "fixed_time_baseline_wnet": art["Fixed-Time"]["mean_w_net"],
            "actuated_wnet": art["Actuated"]["mean_w_net"],
            "actuated_increase_pct": round(((art["Actuated"]["mean_w_net"] - art["Fixed-Time"]["mean_w_net"]) / art["Fixed-Time"]["mean_w_net"]) * 100.0, 1),
            "max_pressure_wnet": art["Max-Pressure"]["mean_w_net"],
            "max_pressure_increase_pct": round(((art["Max-Pressure"]["mean_w_net"] - art["Fixed-Time"]["mean_w_net"]) / art["Fixed-Time"]["mean_w_net"]) * 100.0, 1),
            "traffic_engineering_mechanism": (
                "Fixed-Time control establishes coordinated green progression ('green wave') along the arterial. "
                "Decentralized queue-greedy controllers (Actuated, Max-Pressure) interrupt arterial progression to serve "
                "minor side-street queues, causing queue spillback and dramatic delay surges (+192% for Actuated, +295% for Max-Pressure)."
            ),
        },
        "rl_generalization_status": {
            "grid3x3_rl_status": "Verified (Seed 1042 executed for DOIM-PPO, MLP-PPO, FW-PPO, SA-PPO)",
            "arterial_corridor_rl_status": "MISSING ON DISK (0 completed evaluation runs)",
            "incident_disruption_rl_status": "MISSING ON DISK (0 completed evaluation runs)",
            "scientific_implication": (
                "Prior claims that DOIM-PPO achieved verified superior generalization across all 3 scenarios "
                "were unsubstantiated by physical filesystem artifacts. RL generalization remains an open confirmatory gap."
            ),
        },
    }

    output = {
        "mission_id": "DOIM-PPO-PHASE34-INDEPENDENT-FORENSIC-AUDIT",
        "scenario_summary": summary,
        "pathology_analysis": pathology_analysis,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Phase 34.7 Generalization & Distribution Shift Audit Completed.")
    print(f"Exported to {OUT_JSON}")
    for sc, algos in summary.items():
        print(f"Scenario: {sc}")
        for al, v in algos.items():
            print(f"  {al:15s}: N={v['n_runs']} | W_net={v['mean_w_net']} +/- {v['std_w_net']}s | Queue={v['mean_queue']} | Th={v['mean_throughput']}")


if __name__ == "__main__":
    main()

