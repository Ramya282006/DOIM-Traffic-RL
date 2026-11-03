"""
DOIM-Traffic-RL: Generalization & Distribution-Shift Benchmark (Phase 31)
Evaluates trained DOIM-PPO and baseline models across out-of-distribution traffic regimes:
  1. Congested Surge (+50% flow rate)
  2. Directional Asymmetry (80/20 flow imbalance)
  3. Incident Bottleneck (lane blockage / capacity drop)
Calculates performance degradation ratios and robustness rankings under distribution shift.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

PKG_DIR = Path("D:/DOIM-Traffic-RL/DOIM-Traffic-RL").resolve()
OUTPUT_ROOT = PKG_DIR / "results/runs"

GENERALIZATION_REGIMES = [
    {
        "name": "nominal_grid",
        "scenario": str(PKG_DIR / "scenarios/directional_shift/north_south_surge.sumocfg"),
        "description": "Nominal balanced flow baseline",
    },
    {
        "name": "arterial_corridor",
        "scenario": str(PKG_DIR / "scenarios/arterial_corridor/arterial_corridor.sumocfg"),
        "description": "High-speed progression with cross-street starvation risk",
    },
    {
        "name": "incident_disruption",
        "scenario": str(PKG_DIR / "scenarios/incident_disruption/incident_disruption.sumocfg"),
        "description": "Severe capacity loss with queue spillback and rerouting",
    },
]


def evaluate_generalization_robustness(runs_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    from analysis.confirmatory_analyzer import collect_confirmatory_dataset

    data = collect_confirmatory_dataset(runs_root)
    scenarios = ["grid3x3", "arterial_corridor", "incident_disruption"]
    algos = ["Fixed_Time", "Actuated", "MaxPressure", "FW_PPO", "SA_PPO", "MLP_PPO", "IA_MOPPO"]

    report = {}
    for algo in algos:
        nominal_w = None
        degradations = {}
        sc_means = {}

        for sc in scenarios:
            sc_data = data.get(sc, {}).get(algo, {})
            if sc_data:
                mean_w = float(np.mean([m["w_net"] for m in sc_data.values()]))
                sc_means[sc] = mean_w
                if sc == "grid3x3":
                    nominal_w = mean_w

        if nominal_w is not None and nominal_w > 0:
            for sc in ["arterial_corridor", "incident_disruption"]:
                if sc in sc_means:
                    deg = (sc_means[sc] - nominal_w) / nominal_w
                    degradations[sc] = float(deg)

        report[algo] = {
            "scenario_means": sc_means,
            "nominal_w_net": nominal_w,
            "degradation_ratios": degradations,
            "average_degradation": float(np.mean(list(degradations.values()))) if degradations else None,
        }

    return report
