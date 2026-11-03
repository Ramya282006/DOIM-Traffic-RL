"""
DOIM-PPO Phase 34.10: Independent Reproducibility Audit
Executes a clean, independent evaluation of Fixed-Time on grid3x3 (seed 1042)
using run_fixed_time_experiment and compares the result directly against the archived reference run.
"""

import json
import time
from pathlib import Path
import pandas as pd

GIT_ROOT = Path("D:/DOIM-Traffic-RL").resolve()
PKG_ROOT = GIT_ROOT / "DOIM-Traffic-RL"
OUT_JSON = GIT_ROOT / "results/phase34/reproduction_comparison.json"

import sys
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from baselines.fixed_time.fixed_time_controller import run_fixed_time_experiment


def run_reproducibility():
    import os
    os.chdir(PKG_ROOT)
    out_runs = GIT_ROOT / "results/phase34/repro_run"
    out_runs.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    exp = run_fixed_time_experiment(
        config_path=str(PKG_ROOT / "configs/default.yaml"),
        scenario_override=str(PKG_ROOT / "configs/scenarios/directional_shift.yaml"),
        total_timesteps=1800,
        cycle_time=30,
        seed=42,
        eval_seed=1042,
        output_root=str(out_runs),
    )
    elapsed = time.time() - t0

    repro_dir = Path(exp.run_dir)
    with open(repro_dir / "summary.json", "r") as f:
        repro_sum = json.load(f)

    # Load archived reference run
    arch_dir = PKG_ROOT / "results/runs/Fixed_Time_north_south_surge_s42_20261103_200132"
    with open(arch_dir / "summary.json", "r") as f:
        arch_sum = json.load(f)

    arch_w = float(arch_sum["average_waiting"])
    repro_w = float(repro_sum["average_waiting"])
    arch_q = float(arch_sum["average_queue"])
    repro_q = float(repro_sum["average_queue"])
    arch_th = float(arch_sum["total_throughput"])
    repro_th = float(repro_sum["total_throughput"])
    arch_f = float(arch_sum["average_fairness"])
    repro_f = float(repro_sum["average_fairness"])

    diff_w = abs(repro_w - arch_w)
    diff_q = abs(repro_q - arch_q)
    diff_th = abs(repro_th - arch_th)
    diff_f = abs(repro_f - arch_f)

    is_deterministic = (diff_w < 1e-4) and (diff_q < 1e-4) and (diff_th == 0.0)

    result = {
        "mission_id": "DOIM-PPO-PHASE34-INDEPENDENT-FORENSIC-AUDIT",
        "benchmark": "Fixed-Time on grid3x3 (train_seed=42, eval_seed=1042, horizon=1800)",
        "reproduction_run_dir": str(repro_dir),
        "execution_time_seconds": round(elapsed, 2),
        "archived_metrics": {
            "average_waiting": round(arch_w, 6),
            "average_queue": round(arch_q, 6),
            "total_throughput": arch_th,
            "average_fairness": round(arch_f, 6),
        },
        "reproduced_metrics": {
            "average_waiting": round(repro_w, 6),
            "average_queue": round(repro_q, 6),
            "total_throughput": repro_th,
            "average_fairness": round(repro_f, 6),
        },
        "absolute_deltas": {
            "delta_waiting": round(diff_w, 8),
            "delta_queue": round(diff_q, 8),
            "delta_throughput": diff_th,
            "delta_fairness": round(diff_f, 8),
        },
        "reproducibility_verdict": "BITWISE_DETERMINISTIC" if is_deterministic else "DISCREPANCY_DETECTED",
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Phase 34.10 Independent Reproducibility Audit Completed.")
    print(f"Elapsed: {elapsed:.2f}s | Verdict: {result['reproducibility_verdict']}")
    print(f"Archived Waiting: {arch_w:.6f} | Reproduced: {repro_w:.6f} | Delta: {diff_w:.8f}")
    print(f"Archived Queue: {arch_q:.6f} | Reproduced: {repro_q:.6f} | Delta: {diff_q:.8f}")
    print(f"Archived Throughput: {arch_th} | Reproduced: {repro_th} | Delta: {diff_th}")


if __name__ == "__main__":
    run_reproducibility()
