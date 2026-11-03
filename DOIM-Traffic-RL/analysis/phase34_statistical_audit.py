"""
DOIM-PPO Phase 34.3: Statistical Methodology Audit & Recalculation
Performs rigorous, uncorrupted statistical tests on verified on-disk data.
Uses exact paired Student's t-test (df = N - 1), Wilcoxon signed-rank test,
Cohen's d_z, Holm-Bonferroni correction, and exact power calculations.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

GIT_ROOT = Path("D:/DOIM-Traffic-RL").resolve()
PKG_ROOT = GIT_ROOT / "DOIM-Traffic-RL"
DATASET_PATH = GIT_ROOT / "results/phase34/verified_confirmatory_dataset.json"


def cohens_dz(diffs: np.ndarray) -> float:
    """Calculates Cohen's d_z for paired samples: mean(D) / std(D, ddof=1)."""
    s_d = np.std(diffs, ddof=1)
    if s_d == 0:
        return 0.0
    return float(np.mean(diffs) / s_d)


def paired_stats(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str, metric: str):
    """
    Computes exact paired comparison: a vs b.
    Diff is defined as a - b.
    """
    diffs = a - b
    n = len(diffs)
    df = n - 1
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    se_diff = float(std_diff / np.sqrt(n))

    # Paired t-test
    t_res = stats.ttest_rel(a, b)
    t_stat = float(t_res.statistic)
    p_val_t = float(t_res.pvalue)

    # 95% CI
    t_crit = stats.t.ppf(0.975, df)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Cohen's dz
    dz = cohens_dz(diffs)

    # Wilcoxon signed rank test
    try:
        w_res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        w_stat = float(w_res.statistic)
        p_val_w = float(w_res.pvalue)
    except Exception:
        w_stat = float("nan")
        p_val_w = float("nan")

    return {
        "comparison": f"{name_a} vs {name_b}",
        "metric": metric,
        "n_pairs": n,
        "df": df,
        "mean_a": float(np.mean(a)),
        "std_a": float(np.std(a, ddof=1)),
        "mean_b": float(np.mean(b)),
        "std_b": float(np.std(b, ddof=1)),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se_diff": se_diff,
        "t_statistic": t_stat,
        "p_value_t": p_val_t,
        "ci_95": [float(ci_lower), float(ci_upper)],
        "cohens_dz": dz,
        "wilcoxon_stat": w_stat,
        "p_value_wilcoxon": p_val_w,
    }


def holm_bonferroni(p_values: list) -> list:
    """Holm-Bonferroni step-down correction for family-wise error rate control."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (m - rank)
        cum_max = max(cum_max, adj)
        adjusted[orig_idx] = min(cum_max, 1.0)
    return adjusted


def power_and_mde(n: int, alpha: float = 0.05, target_power: float = 0.80):
    """
    Computes exact statistical power profile for paired t-test with n pairs.
    Finds the Minimum Detectable Effect (MDE) dz at target_power.
    """
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    # Binary search for MDE dz
    low = 0.01
    high = 10.0
    for _ in range(100):
        mid = (low + high) / 2
        ncp = mid * np.sqrt(n)
        # Power = 1 - Pr(-t_crit <= t <= t_crit | ncp)
        pw = 1.0 - (stats.nct.cdf(t_crit, df, ncp) - stats.nct.cdf(-t_crit, df, ncp))
        if pw < target_power:
            low = mid
        else:
            high = mid
    mde_dz = (low + high) / 2
    return mde_dz


def calculate_power_for_dz(n: int, dz: float, alpha: float = 0.05) -> float:
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ncp = abs(dz) * np.sqrt(n)
    power = 1.0 - (stats.nct.cdf(t_crit, df, ncp) - stats.nct.cdf(-t_crit, df, ncp))
    return float(power)


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        ds = json.load(f)["data"]

    grid = ds["grid3x3"]
    arterial = ds["arterial_corridor"]

    comparisons = []

    # 1. grid3x3: Fixed-Time vs Actuated vs Max-Pressure (N = 5)
    seeds = ["1042", "1043", "1044", "1045", "1046"]

    ft_wnet = np.array([grid["Fixed-Time"][s]["w_net"] for s in seeds])
    act_wnet = np.array([grid["Actuated"][s]["w_net"] for s in seeds])
    mp_wnet = np.array([grid["Max-Pressure"][s]["w_net"] for s in seeds])

    ft_q = np.array([grid["Fixed-Time"][s]["queue"] for s in seeds])
    act_q = np.array([grid["Actuated"][s]["queue"] for s in seeds])
    mp_q = np.array([grid["Max-Pressure"][s]["queue"] for s in seeds])

    ft_th = np.array([grid["Fixed-Time"][s]["throughput"] for s in seeds])
    act_th = np.array([grid["Actuated"][s]["throughput"] for s in seeds])
    mp_th = np.array([grid["Max-Pressure"][s]["throughput"] for s in seeds])

    # Comparisons for W_net
    comparisons.append(paired_stats(ft_wnet, act_wnet, "Fixed-Time", "Actuated", "w_net"))
    comparisons.append(paired_stats(act_wnet, mp_wnet, "Actuated", "Max-Pressure", "w_net"))
    comparisons.append(paired_stats(ft_wnet, mp_wnet, "Fixed-Time", "Max-Pressure", "w_net"))

    # Comparisons for Queue
    comparisons.append(paired_stats(ft_q, act_q, "Fixed-Time", "Actuated", "queue"))
    comparisons.append(paired_stats(act_q, mp_q, "Actuated", "Max-Pressure", "queue"))
    comparisons.append(paired_stats(ft_q, mp_q, "Fixed-Time", "Max-Pressure", "queue"))

    # Comparisons for Throughput
    comparisons.append(paired_stats(act_th, ft_th, "Actuated", "Fixed-Time", "throughput"))
    comparisons.append(paired_stats(mp_th, act_th, "Max-Pressure", "Actuated", "throughput"))
    comparisons.append(paired_stats(mp_th, ft_th, "Max-Pressure", "Fixed-Time", "throughput"))

    # 2. arterial_corridor: Fixed-Time vs Actuated (4 matching seeds: 1042, 1043, 1044, 1046)
    art_seeds = ["1042", "1043", "1044", "1046"]
    art_ft_wnet = np.array([arterial["Fixed-Time"][s]["w_net"] for s in art_seeds])
    art_act_wnet = np.array([arterial["Actuated"][s]["w_net"] for s in art_seeds])
    comparisons.append(paired_stats(art_ft_wnet, art_act_wnet, "Fixed-Time (Arterial)", "Actuated (Arterial)", "w_net"))

    # Apply Holm-Bonferroni correction
    raw_p_values = [c["p_value_t"] for c in comparisons]
    adj_p_values = holm_bonferroni(raw_p_values)
    for c, adj_p in zip(comparisons, adj_p_values):
        c["p_value_holm_bonferroni"] = float(adj_p)
        c["power_achieved"] = calculate_power_for_dz(c["n_pairs"], c["cohens_dz"])

    # Power & MDE analysis
    mde_n5 = power_and_mde(5, alpha=0.05, target_power=0.80)
    mde_n4 = power_and_mde(4, alpha=0.05, target_power=0.80)

    # 3. Seed 1042 Head-to-Head Comparison on grid3x3 (All 7 algorithms)
    seed1042_bench = {}
    for algo in ["Fixed-Time", "Actuated", "Max-Pressure", "FW-PPO", "SA-PPO", "MLP-PPO", "DOIM-PPO"]:
        if "1042" in grid[algo]:
            seed1042_bench[algo] = {
                "w_net": grid[algo]["1042"]["w_net"],
                "queue": grid[algo]["1042"]["queue"],
                "throughput": grid[algo]["1042"]["throughput"],
                "shield_interventions": grid[algo]["1042"]["shield_interventions"],
            }

    doim_w = seed1042_bench["DOIM-PPO"]["w_net"]
    rel_improvements = {}
    for algo, vals in seed1042_bench.items():
        if algo == "DOIM-PPO":
            continue
        rel_diff = ((vals["w_net"] - doim_w) / vals["w_net"]) * 100.0
        rel_improvements[algo] = {
            "w_net_baseline": vals["w_net"],
            "w_net_doim": doim_w,
            "reduction_pct": float(rel_diff),
            "queue_baseline": vals["queue"],
            "queue_doim": seed1042_bench["DOIM-PPO"]["queue"],
            "throughput_baseline": vals["throughput"],
            "throughput_doim": seed1042_bench["DOIM-PPO"]["throughput"],
        }

    output = {
        "mission_id": "DOIM-PPO-PHASE34-INDEPENDENT-FORENSIC-AUDIT",
        "methodology": {
            "audited_error": "Prior reports claimed pooled Linear Mixed Models with df=72 across all 105 planned cells. Only 29 cells exist on disk, making df=72 non-estimable and fraudulent.",
            "corrected_approach": "Exact paired Student's t-test with df = N - 1 (df=4 for N=5, df=3 for N=4), exact Wilcoxon signed-rank tests, Cohen's dz effect size, Holm-Bonferroni correction, and exact power calculations.",
            "mde_n5_80pct_power": float(mde_n5),
            "mde_n4_80pct_power": float(mde_n4),
        },
        "paired_baseline_tests": comparisons,
        "seed_1042_head_to_head_grid3x3": {
            "raw_metrics": seed1042_bench,
            "relative_improvements_vs_doim": rel_improvements,
        },
    }

    out_file = GIT_ROOT / "results/phase34/statistical_recalculation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Phase 34.3 Statistical Recalculation Completed.")
    print(f"Exported to {out_file}")
    for c in comparisons:
        print(f"[{c['comparison']} | {c['metric']}] df={c['df']}, MeanDiff={c['mean_diff']:.2f}, dz={c['cohens_dz']:.2f}, p_raw={c['p_value_t']:.4e}, p_holm={c['p_value_holm_bonferroni']:.4e}")


if __name__ == "__main__":
    main()

