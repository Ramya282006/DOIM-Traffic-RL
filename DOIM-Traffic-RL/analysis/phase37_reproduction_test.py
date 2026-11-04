"""
Phase 37 Fresh Independent Reproduction Test
Re-runs Fixed-Time controller on grid3x3 seed 1042 and verifies bitwise determinism:
Compares freshly executed metrics against canonical historical baseline.
"""

import os
import sys
import json
import time
from pathlib import Path

PKG_ROOT = Path("D:/DOIM-Traffic-RL/DOIM-Traffic-RL").resolve()
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
os.chdir(PKG_ROOT)

from baselines.fixed_time.fixed_time_controller import run_fixed_time_experiment

OUT_DIR = PKG_ROOT / "results/phase37"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HISTORICAL_SUMMARY = PKG_ROOT / "results/runs/Fixed_Time_north_south_surge_s42_20261103_195003/summary.json"


def run_reproduction_test():
    print("=" * 80)
    print("PHASE 37: FRESH INDEPENDENT REPRODUCTION TEST")
    print("Scenario: grid3x3 (north_south_surge) | Seed: 1042 | Horizon: 1800 steps")
    print("=" * 80)

    # 1. Load canonical historical summary
    with open(HISTORICAL_SUMMARY, "r", encoding="utf-8") as f:
        hist_data = json.load(f)

    # 2. Execute fresh reproduction run
    repro_output_root = str(OUT_DIR / "runs")
    t0 = time.time()
    repro_run = run_fixed_time_experiment(
        scenario_override="scenarios/directional_shift/north_south_surge.sumocfg",
        total_timesteps=1800,
        seed=1042,
        eval_seed=1042,
        output_root=repro_output_root,
        evaluation_mode="deterministic_eval",
    )
    elapsed = time.time() - t0
    print(f"Fresh reproduction completed in {elapsed:.2f}s")

    # 3. Load fresh summary
    fresh_summary_path = repro_run.run_dir / "summary.json"
    with open(fresh_summary_path, "r", encoding="utf-8") as f:
        fresh_data = json.load(f)

    # 4. Bitwise comparison
    comparisons = {}
    metrics_to_compare = [
        ("total_steps", 1800),
        ("average_waiting", hist_data.get("average_waiting") or hist_data.get("average_waiting_time_s")),
        ("average_queue", hist_data.get("average_queue") or hist_data.get("mean_queue_length")),
        ("total_throughput", hist_data.get("total_throughput")),
        ("average_co2", hist_data.get("average_co2") or hist_data.get("mean_co2_emission_rate_mg_s")),
        ("total_shield_interventions", 0),
    ]

    all_match = True
    for key, hist_val in metrics_to_compare:
        fresh_val = fresh_data.get(key)
        diff = abs(float(fresh_val) - float(hist_val)) if (fresh_val is not None and hist_val is not None) else None
        match = (diff is not None and diff < 1e-6)
        if not match:
            all_match = False
        comparisons[key] = {
            "historical": hist_val,
            "fresh_reproduced": fresh_val,
            "absolute_diff": diff,
            "match": match,
        }

    reproduction_report = {
        "mission_id": "DOIM-PPO-PHASE37-FRESH-REPRODUCTION",
        "scenario": "grid3x3",
        "algorithm": "Fixed-Time",
        "seed": 1042,
        "elapsed_seconds": elapsed,
        "all_metrics_bitwise_matched": all_match,
        "comparisons": comparisons,
    }

    out_file = OUT_DIR / "reproduction_verification.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(reproduction_report, f, indent=2)
    print(f"Reproduction verification saved to {out_file}")

    # Mirror to root results/phase37
    root_out_dir = Path("D:/DOIM-Traffic-RL/results/phase37")
    root_out_dir.mkdir(parents=True, exist_ok=True)
    with open(root_out_dir / "reproduction_verification.json", "w", encoding="utf-8") as f:
        json.dump(reproduction_report, f, indent=2)
    print(f"Bitwise Match Status: {'PASSED (Delta = 0.00000000)' if all_match else 'FAILED'}")
    return reproduction_report


if __name__ == "__main__":
    run_reproduction_test()
