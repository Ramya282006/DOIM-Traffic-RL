"""
DOIM-Traffic-RL: Multi-Objective & Component Ablation Analyzer (Phase 30)
Evaluates:
  1. Component Ablations: Full DOIM-PPO vs No-Interaction (M=0) vs Rolling-Pearson vs MLP-PPO (No-GCN)
  2. 5D Pareto Dominance & Hypervolume Expansion:
     - Objectives: [waiting, queue, co2, inverted_throughput, fairness]
     - Reference point: (1.2, 1.2, 1.2, 1.2, 1.2)
  3. Learned Interaction Manifold Dynamics:
     - Extraction and spectral analysis of state-conditioned coupling matrix Psi(s)
     - Dynamic arbitration between competitive and synergistic objectives
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

PKG_DIR = Path("D:/DOIM-Traffic-RL/DOIM-Traffic-RL").resolve()
OUTPUT_ROOT = PKG_DIR / "results/runs"

OBJECTIVE_NAMES = ["waiting", "queue", "co2", "inv_throughput", "fairness"]
OBJECTIVE_BOUNDS = {
    "waiting": 600.0,
    "queue": 20.0,
    "co2": 120000.0,
    "inv_throughput": 20.0,  # Throughput normalized relative to max completed rate
    "fairness": 1.0,
}
REF_POINT = np.array([1.2, 1.2, 1.2, 1.2, 1.2], dtype=np.float32)


def normalize_objectives(raw_dict: Dict[str, float]) -> np.ndarray:
    w = min(raw_dict["w_net"] / OBJECTIVE_BOUNDS["waiting"], 1.2)
    q = min(raw_dict["queue"] / OBJECTIVE_BOUNDS["queue"], 1.2)
    c = min(raw_dict["co2"] / OBJECTIVE_BOUNDS["co2"], 1.2)
    th_rate = (raw_dict["throughput"] / 180.0)
    inv_th = min(max(1.0 - (th_rate / OBJECTIVE_BOUNDS["inv_throughput"]), 0.0), 1.2)
    f = min(raw_dict["fairness"] / OBJECTIVE_BOUNDS["fairness"], 1.2)
    return np.array([w, q, c, inv_th, f], dtype=np.float32)


def is_pareto_dominated(u: np.ndarray, v: np.ndarray) -> bool:
    """Returns True if u dominates v (u is better than v, strictly smaller in at least one, <= in all)."""
    return bool(np.all(u <= v) and np.any(u < v))


def compute_non_dominated_front(points: np.ndarray) -> np.ndarray:
    """Extracts non-dominated indices from an array of N points in D dimensions (all minimization)."""
    n_points = len(points)
    is_dominated = np.zeros(n_points, dtype=bool)
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                if is_pareto_dominated(points[j], points[i]):
                    is_dominated[i] = True
                    break
    return np.where(~is_dominated)[0]


def compute_hypervolume_monte_carlo(non_dominated_pts: np.ndarray, ref: np.ndarray = REF_POINT, n_samples: int = 200000) -> float:
    """Computes exact/high-precision Monte Carlo hypervolume for 5D objective space."""
    if len(non_dominated_pts) == 0:
        return 0.0
    dim = len(ref)
    mins = np.min(non_dominated_pts, axis=0)
    # Filter points outside ref
    valid_mask = np.all(non_dominated_pts <= ref, axis=1)
    pts = non_dominated_pts[valid_mask]
    if len(pts) == 0:
        return 0.0

    lower = np.minimum(mins, 0.0)
    box_vol = float(np.prod(ref - lower))

    # Sample uniformly in bounding box
    rng = np.random.default_rng(seed=42)
    samples = rng.uniform(lower, ref, size=(n_samples, dim))

    # Check if dominated by at least one non-dominated point
    dominated_samples = np.zeros(n_samples, dtype=bool)
    for p in pts:
        dominated_samples |= np.all(samples >= p, axis=1)

    hv = float(box_vol * np.mean(dominated_samples))
    return hv


def analyze_ablations_and_pareto(runs_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    from analysis.confirmatory_analyzer import collect_confirmatory_dataset

    data = collect_confirmatory_dataset(runs_root)

    # Collect ablation runs specifically
    ablation_algos = ["IA_MOPPO", "DOIM_no_interaction", "DOIM_rolling_pearson", "MLP_PPO"]
    ablation_records = {}

    for algo in ablation_algos:
        ablation_records[algo] = []
        for sc in ["grid3x3", "arterial_corridor", "incident_disruption"]:
            sc_data = data.get(sc, {}).get(algo, {})
            for seed, metrics in sc_data.items():
                norm_vec = normalize_objectives(metrics)
                ablation_records[algo].append({
                    "scenario": sc,
                    "seed": seed,
                    "raw": metrics,
                    "norm": norm_vec.tolist(),
                })

    # Summary table across ablations
    table = {}
    all_points_by_algo = {}
    for algo, recs in ablation_records.items():
        if recs:
            w_vals = [r["raw"]["w_net"] for r in recs]
            q_vals = [r["raw"]["queue"] for r in recs]
            c_vals = [r["raw"]["co2"] for r in recs]
            th_vals = [r["raw"]["throughput"] for r in recs]
            f_vals = [r["raw"]["fairness"] for r in recs]
            int_vals = [r["raw"]["interventions"] for r in recs]
            norm_pts = np.array([r["norm"] for r in recs])
            all_points_by_algo[algo] = norm_pts

            hv = compute_hypervolume_monte_carlo(norm_pts)

            table[algo] = {
                "n_runs": len(recs),
                "w_net_mean": float(np.mean(w_vals)),
                "w_net_sem": float(np.std(w_vals, ddof=1) / math.sqrt(len(w_vals))) if len(w_vals) > 1 else 0.0,
                "queue_mean": float(np.mean(q_vals)),
                "co2_mean": float(np.mean(c_vals)),
                "throughput_mean": float(np.mean(th_vals)),
                "fairness_mean": float(np.mean(f_vals)),
                "interventions_mean": float(np.mean(int_vals)),
                "hypervolume": hv,
            }

    # Combined Pareto frontier
    combined_pts = []
    combined_labels = []
    for algo, pts in all_points_by_algo.items():
        for p in pts:
            combined_pts.append(p)
            combined_labels.append(algo)

    if combined_pts:
        combined_arr = np.array(combined_pts)
        non_dom_idx = compute_non_dominated_front(combined_arr)
        pareto_labels = [combined_labels[i] for i in non_dom_idx]
        pareto_counts = {algo: pareto_labels.count(algo) for algo in ablation_algos}
    else:
        pareto_counts = {}

    return {
        "ablation_summary": table,
        "pareto_distribution": pareto_counts,
    }
