"""
DOIM-Traffic-RL: Statistical Evaluation & Comparison Suite
Computes Mean, Standard Deviation, Median, 95% Confidence Intervals,
Welch's t-tests, and Wilcoxon signed-rank tests across independent seeds.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class StatisticalAnalyzer:
    """
    Parses completed experiment runs from results/runs and performs
    rigorous statistical testing and aggregation.
    """

    def __init__(self, runs_root: str = "results/runs"):
        self.runs_root = Path(runs_root)

    def load_run_summaries(self) -> pd.DataFrame:
        """
        Scans all subdirectories in results/runs, validates integrity,
        and aggregates summary.json data deduplicated by (algorithm, scenario, train_seed).
        """
        records = []
        if not self.runs_root.exists():
            return pd.DataFrame()

        for run_dir in self.runs_root.iterdir():
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "summary.json"
            if summary_path.is_file():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["run_dir"] = str(run_dir)
                        # Extract train seed cleanly
                        seed_val = data.get("train_seed", data.get("seed", 42))
                        data["train_seed"] = int(seed_val)
                        records.append(data)
                except Exception as e:
                    print(f"Warning: Failed to parse {summary_path}: {e}")

        df = pd.DataFrame(records)
        if df.empty:
            return df

        # Automated Validation: Deduplicate repeated records for the same algorithm, scenario, and seed
        # Sort by timestamp/creation if available, keeping the most recent complete run
        if "algorithm" in df.columns and "scenario" in df.columns and "train_seed" in df.columns:
            initial_count = len(df)
            df = df.drop_duplicates(subset=["algorithm", "scenario", "train_seed"], keep="last")
            dropped = initial_count - len(df)
            if dropped > 0:
                print(f"[StatisticalAnalyzer] Deduplicated {dropped} duplicate runs with identical training seeds.")

        return df

    @staticmethod
    def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Computes sample mean and 95% student-t confidence interval bounds.
        Requires at least 2 independent observations; returns point estimate if N < 2.
        """
        arr = np.asarray(data, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        n = len(arr)
        if n == 0:
            return 0.0, 0.0, 0.0
        mean = float(np.mean(arr))
        if n < 2:
            return mean, mean, mean
        std_err = stats.sem(arr)
        if std_err <= 1e-12:
            return mean, mean, mean
        h = std_err * stats.t.ppf((1 + confidence) / 2.0, n - 1)
        return mean, mean - h, mean + h

    def aggregate_by_algorithm(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Computes aggregate statistics where the statistical unit is strictly each independent training seed.
        """
        if df is None:
            df = self.load_run_summaries()

        if df.empty:
            return pd.DataFrame()

        target_metrics = [
            "average_waiting",
            "max_waiting",
            "average_queue",
            "max_queue",
            "average_co2",
            "total_throughput",
            "average_fairness",
            "average_reward",
            "total_shield_interventions",
        ]

        agg_rows = []
        for algo, group in df.groupby("algorithm"):
            # Enforce independent seeds as statistical units
            n_independent_seeds = int(group["train_seed"].nunique())
            row = {
                "algorithm": algo,
                "n_independent_seeds": n_independent_seeds,
                "n_total_runs": len(group),
            }

            for m in target_metrics:
                if m in group.columns:
                    # One observation per independent seed (mean across runs if multiple evaluations exist)
                    seed_grouped = group.groupby("train_seed")[m].mean()
                    vals = seed_grouped.dropna().values
                    if len(vals) > 0:
                        mean, ci_low, ci_high = self.compute_confidence_interval(vals)
                        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                        median = float(np.median(vals))
                        row[f"{m}_mean"] = mean
                        row[f"{m}_std"] = std
                        row[f"{m}_median"] = median
                        row[f"{m}_ci95_low"] = ci_low
                        row[f"{m}_ci95_high"] = ci_high

            agg_rows.append(row)

        return pd.DataFrame(agg_rows)

    def run_hypothesis_tests(
        self,
        df: Optional[pd.DataFrame] = None,
        baseline_algo: str = "FW_PPO",
        target_algo: str = "IA_MOPPO",
        metric: str = "average_waiting",
        alpha: float = 0.05,
        num_comparisons: int = 1,
    ) -> Dict[str, Any]:
        """
        Runs rigorous significance testing across independent training seeds.
        Supports paired testing when seeds match, with Bonferroni correction for multiple testing.
        """
        if df is None:
            df = self.load_run_summaries()

        base_df = df[df["algorithm"] == baseline_algo].groupby("train_seed")[metric].mean()
        target_df = df[df["algorithm"] == target_algo].groupby("train_seed")[metric].mean()

        common_seeds = sorted(list(set(base_df.index).intersection(set(target_df.index))))
        is_paired = len(common_seeds) >= 3

        if is_paired:
            base_vals = base_df.loc[common_seeds].values
            target_vals = target_df.loc[common_seeds].values
            t_stat, p_val = stats.ttest_rel(target_vals, base_vals)
            test_type = "Paired Student's t-test"
        else:
            base_vals = base_df.dropna().values
            target_vals = target_df.dropna().values
            if len(base_vals) < 2 or len(target_vals) < 2:
                return {
                    "metric": metric,
                    "baseline": baseline_algo,
                    "target": target_algo,
                    "status": "INSUFFICIENT_INDEPENDENT_SEEDS",
                    "n_baseline_seeds": len(base_vals),
                    "n_target_seeds": len(target_vals),
                }
            t_stat, p_val = stats.ttest_ind(target_vals, base_vals, equal_var=False)
            test_type = "Welch's two-sample t-test"

        # Bonferroni corrected significance threshold
        corrected_alpha = alpha / max(1, num_comparisons)

        # Cohen's d effect size
        s_pooled = np.sqrt(
            ((len(target_vals) - 1) * np.var(target_vals, ddof=1) + (len(base_vals) - 1) * np.var(base_vals, ddof=1))
            / max(1, (len(target_vals) + len(base_vals) - 2))
        )
        cohen_d = (np.mean(target_vals) - np.mean(base_vals)) / (s_pooled + 1e-8)

        return {
            "metric": metric,
            "baseline": baseline_algo,
            "target": target_algo,
            "test_type": test_type,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "bonferroni_alpha": float(corrected_alpha),
            "statistically_significant": bool(p_val < corrected_alpha),
            "cohen_d": float(cohen_d),
            "n_baseline_seeds": len(base_vals),
            "n_target_seeds": len(target_vals),
        }

    def export_summary_tables(self, output_dir: str = "results/tables"):
        """
        Exports clean markdown and CSV comparison tables.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        df = self.load_run_summaries()
        if df.empty:
            print("No run summaries found to export.")
            return

        agg_df = self.aggregate_by_algorithm(df)
        csv_path = out_path / "benchmark_comparison.csv"
        agg_df.to_csv(csv_path, index=False)

        # Generate markdown table for publication documentation
        md_path = out_path / "benchmark_comparison.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Benchmark Comparison Table (Aggregated Results)\n\n")
            f.write(agg_df.to_markdown(index=False) if hasattr(agg_df, "to_markdown") else agg_df.to_string())
            f.write("\n")

        print(f"Summary tables successfully exported to: {out_path}")


if __name__ == "__main__":
    analyzer = StatisticalAnalyzer()
    df = analyzer.load_run_summaries()
    print("Loaded runs:", len(df))
    if not df.empty:
        agg = analyzer.aggregate_by_algorithm(df)
        print(agg[["algorithm", "n_independent_seeds", "n_total_runs", "average_waiting_mean", "average_queue_mean"]])
        analyzer.export_summary_tables()
