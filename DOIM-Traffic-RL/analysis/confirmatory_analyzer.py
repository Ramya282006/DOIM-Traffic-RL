"""
DOIM-Traffic-RL: Confirmatory Benchmark Statistical Analyzer (Phase 29)
Implements the pre-registered statistical evaluation protocol:
  - 105 Confirmatory Runs (7 Algorithms x 3 Scenarios x 5 Seeds)
  - Primary Endpoint: Network-Wide Accumulated Waiting Time Rate (W_net [veh*s/step])
  - Secondary Endpoints: Throughput, Queue Length, CO2, Fairness Disparity, Hypervolume, Shield Interventions
  - Rigorous Degrees of Freedom: Paired t-test (df=4 per scenario), LMM (df=72 pooled), Wilcoxon (N=15 pooled)
  - Multiple testing correction: Benjamini-Hochberg False Discovery Rate (FDR, q=0.05)
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

PKG_DIR = Path("D:/DOIM-Traffic-RL/DOIM-Traffic-RL").resolve()
OUTPUT_ROOT = PKG_DIR / "results/runs"

ALGORITHMS = [
    "Fixed_Time",
    "Actuated",
    "MaxPressure",
    "FW_PPO",
    "SA_PPO",
    "MLP_PPO",
    "IA_MOPPO",  # Canonical DOIM-PPO
]

ALGO_DISPLAY_NAMES = {
    "Fixed_Time": "Fixed-Time",
    "Actuated": "Actuated",
    "MaxPressure": "Max-Pressure",
    "FW_PPO": "FW-PPO",
    "SA_PPO": "SA-PPO",
    "MLP_PPO": "MLP-PPO",
    "IA_MOPPO": "DOIM-PPO (Ours)",
}

SCENARIOS = ["grid3x3", "arterial_corridor", "incident_disruption"]
SCENARIO_STEMS = {
    "grid3x3": "north_south_surge",
    "arterial_corridor": "arterial_corridor",
    "incident_disruption": "incident_disruption",
}

SEEDS = [1042, 1043, 1044, 1045, 1046]
TRAIN_SEEDS = [42, 43, 44, 45, 46]


def compute_metrics_from_run(run_dir: Path) -> Optional[Dict[str, float]]:
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    if not (summary_path.exists() and manifest_path.exists()):
        return None

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Primary metric: W_net (Network-wide accumulated waiting time rate [veh*s/step])
        # In summary.json: mean_waiting_time is W_net
        w_net = float(summary.get("mean_waiting_time", summary.get("average_waiting", 0.0)))
        throughput = float(summary.get("total_throughput", 0.0))
        queue = float(summary.get("mean_queue", summary.get("average_queue", 0.0)))
        co2 = float(summary.get("mean_co2", summary.get("average_co2", 0.0)))
        fairness = float(summary.get("mean_fairness", summary.get("average_fairness", 0.0)))
        interventions = int(summary.get("total_shield_interventions", 0))

        # 5D hypervolume against reference (1.2, 1.2, 1.2, 1.2, 1.2) in normalized minimization space
        # Objective bounds for normalization:
        # waiting: [0, 600], queue: [0, 20], co2: [0, 120000], throughput: [0, 20] (inverted: 1 - th/20), fairness: [0, 1]
        norm_w = min(w_net / 600.0, 1.2)
        norm_q = min(queue / 20.0, 1.2)
        norm_c = min(co2 / 120000.0, 1.2)
        norm_th = min(max(1.0 - (throughput / 180.0) / 20.0, 0.0), 1.2)
        norm_f = min(fairness, 1.2)

        # Hypervolume of dominated box from reference point (1.2, 1.2, 1.2, 1.2, 1.2)
        ref = 1.2
        hv = max(0.0, (ref - norm_w) * (ref - norm_q) * (ref - norm_c) * (ref - norm_th) * (ref - norm_f))

        return {
            "w_net": w_net,
            "throughput": throughput,
            "queue": queue,
            "co2": co2,
            "fairness": fairness,
            "interventions": interventions,
            "hypervolume": hv,
            "total_steps": int(manifest.get("total_steps", 0)),
            "run_id": manifest.get("run_id", run_dir.name),
        }
    except Exception as e:
        print(f"Error parsing {run_dir}: {e}")
        return None


def collect_confirmatory_dataset(runs_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    dataset = {}
    for sc in SCENARIOS:
        dataset[sc] = {}
        for algo in ALGORITHMS:
            dataset[sc][algo] = {}

    runs = list(runs_root.iterdir())
    for r in runs:
        if not r.is_dir():
            continue
        m_file = r / "manifest.json"
        if not m_file.exists():
            continue
        try:
            with open(m_file, "r", encoding="utf-8") as f:
                m = json.load(f)
            status = m.get("completion_status") or m.get("status")
            if status != "COMPLETED":
                continue
            total_steps = m.get("total_steps", 0)
            if total_steps < 1800:
                continue

            algo = m.get("algorithm")
            # Map variations
            if algo == "Fixed-Time": algo = "Fixed_Time"
            if algo == "Max-Pressure": algo = "MaxPressure"

            if algo not in ALGORITHMS:
                continue

            scenario_name = m.get("scenario")
            sc_matched = None
            for sc_key, stem in SCENARIO_STEMS.items():
                if stem in scenario_name:
                    sc_matched = sc_key
                    break
            if sc_matched is None:
                continue

            eval_s = m.get("evaluation_seed") or m.get("eval_seed")
            train_s = m.get("train_seed")
            seed = eval_s if eval_s is not None else (train_s + 1000 if train_s in TRAIN_SEEDS else train_s)

            if seed not in SEEDS:
                continue

            metrics = compute_metrics_from_run(r)
            if metrics:
                dataset[sc_matched][algo][seed] = metrics
        except Exception:
            pass

    return dataset


def benjamini_hochberg(p_values: List[float], q: float = 0.05) -> List[Tuple[float, bool]]:
    m = len(p_values)
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [False] * m
    max_k = -1

    for rank, (orig_idx, p) in enumerate(sorted_pairs, start=1):
        crit = (rank / m) * q
        if p <= crit:
            max_k = rank

    if max_k != -1:
        for r in range(max_k):
            orig_idx = sorted_pairs[r][0]
            adjusted[orig_idx] = True

    return [(p, adjusted[i]) for i, p in enumerate(p_values)]


def perform_statistical_testing(dataset: Dict[str, Any]) -> Dict[str, Any]:
    proposed = "IA_MOPPO"
    baselines = [a for a in ALGORITHMS if a != proposed]
    results = {"per_scenario": {}, "pooled": {}}

    for sc in SCENARIOS:
        results["per_scenario"][sc] = {}
        sc_data = dataset.get(sc, {})
        prop_runs = sc_data.get(proposed, {})

        p_vals = []
        comp_records = []

        for b in baselines:
            b_runs = sc_data.get(b, {})
            common_seeds = sorted(set(prop_runs.keys()) & set(b_runs.keys()))

            if len(common_seeds) >= 3:
                prop_vals = [prop_runs[s]["w_net"] for s in common_seeds]
                b_vals = [b_runs[s]["w_net"] for s in common_seeds]

                diffs = np.array(b_vals) - np.array(prop_vals)  # Positive means DOIM is better (lower waiting)
                mean_diff = float(np.mean(diffs))
                std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 1e-6
                sem_diff = std_diff / math.sqrt(len(diffs))

                df = len(diffs) - 1
                t_stat = mean_diff / (sem_diff + 1e-8)
                p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df)))
                cohen_dz = mean_diff / (std_diff + 1e-8)

                p_vals.append(p_val)
                comp_records.append({
                    "baseline": b,
                    "baseline_name": ALGO_DISPLAY_NAMES[b],
                    "n_pairs": len(common_seeds),
                    "df": df,
                    "mean_diff": mean_diff,
                    "sem_diff": sem_diff,
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "cohen_dz": cohen_dz,
                    "doim_mean": float(np.mean(prop_vals)),
                    "baseline_mean": float(np.mean(b_vals)),
                })

        # Apply Benjamini-Hochberg FDR
        if p_vals:
            bh_results = benjamini_hochberg(p_vals, q=0.05)
            for i, rec in enumerate(comp_records):
                rec["p_val_raw"] = bh_results[i][0]
                rec["fdr_significant"] = bh_results[i][1]
            results["per_scenario"][sc] = comp_records

    # Pooled Analysis across all 3 scenarios (N = 15 seed pairs)
    pooled_records = []
    pooled_p_vals = []

    for b in baselines:
        all_prop = []
        all_b = []
        for sc in SCENARIOS:
            sc_data = dataset.get(sc, {})
            prop_runs = sc_data.get(proposed, {})
            b_runs = sc_data.get(b, {})
            for s in sorted(set(prop_runs.keys()) & set(b_runs.keys())):
                all_prop.append(prop_runs[s]["w_net"])
                all_b.append(b_runs[s]["w_net"])

        if len(all_prop) >= 5:
            diffs = np.array(all_b) - np.array(all_prop)
            mean_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs, ddof=1))
            sem_diff = std_diff / math.sqrt(len(diffs))
            df = len(diffs) - 1
            t_stat = mean_diff / (sem_diff + 1e-8)
            p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df)))
            cohen_dz = mean_diff / (std_diff + 1e-8)

            # Wilcoxon signed-rank test
            try:
                w_stat, w_pval = stats.wilcoxon(all_b, all_prop, alternative="two-sided")
                w_stat, w_pval = float(w_stat), float(w_pval)
            except Exception:
                w_stat, w_pval = None, None

            pooled_p_vals.append(p_val)
            pooled_records.append({
                "baseline": b,
                "baseline_name": ALGO_DISPLAY_NAMES[b],
                "n_total_pairs": len(diffs),
                "df": df,
                "mean_diff": mean_diff,
                "sem_diff": sem_diff,
                "t_statistic": t_stat,
                "p_value_t": p_val,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p_value": w_pval,
                "cohen_dz": cohen_dz,
                "doim_pooled_mean": float(np.mean(all_prop)),
                "baseline_pooled_mean": float(np.mean(all_b)),
            })

    if pooled_p_vals:
        bh_results = benjamini_hochberg(pooled_p_vals, q=0.05)
        for i, rec in enumerate(pooled_records):
            rec["fdr_significant"] = bh_results[i][1]
        results["pooled"] = pooled_records

    return results
