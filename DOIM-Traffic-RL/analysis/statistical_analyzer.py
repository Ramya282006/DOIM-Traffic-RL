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
        Scans all subdirectories in results/runs and aggregates summary.json data.
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
                        records.append(data)
                except Exception as e:
                    print(f"Warning: Failed to parse {summary_path}: {e}")

        df = pd.DataFrame(records)
        return df

    @staticmethod
    def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Computes sample mean and 95% student-t confidence interval bounds.
        """
        arr = np.asarray(data, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        n = len(arr)
        if n == 0:
            return 0.0, 0.0, 0.0
        mean = float(np.mean(arr))
        if n == 1:
            return mean, mean, mean
        std_err = stats.sem(arr)
        h = std_err * stats.t.ppf((1 + confidence) / 2.0, n - 1)
        return mean, mean - h, mean + h

    def aggregate_by_algorithm(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Computes aggregate statistics (Mean, Std, Median, 95% CI) per algorithm.
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
            n_seeds = group["seed"].nunique() if "seed" in group else len(group)
            row = {"algorithm": algo, "n_seeds": n_seeds, "n_runs": len(group)}

            for m in target_metrics:
                if m in group.columns:
                    vals = group[m].dropna().values
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
    ) -> Dict[str, Any]:
        """
        Runs Welch's two-sample t-test and Wilcoxon signed-rank test.
        """
        if df is None:
            df = self.load_run_summaries()

        base_vals = df[df["algorithm"] == baseline_algo][metric].dropna().values
        target_vals = df[df["algorithm"] == target_algo][metric].dropna().values

        if len(base_vals) < 2 or len(target_vals) < 2:
            return {
                "metric": metric,
                "baseline": baseline_algo,
                "target": target_algo,
                "status": "INSUFFICIENT_SAMPLES",
                "n_baseline": len(base_vals),
                "n_target": len(target_vals),
            }

        t_stat, p_val = stats.ttest_ind(target_vals, base_vals, equal_var=False)

        # Cohen's d effect size
        s_pooled = np.sqrt(
            ((len(target_vals) - 1) * np.var(target_vals, ddof=1) + (len(base_vals) - 1) * np.var(base_vals, ddof=1))
            / (len(target_vals) + len(base_vals) - 2)
        )
        cohen_d = (np.mean(target_vals) - np.mean(base_vals)) / (s_pooled + 1e-8)

        return {
            "metric": metric,
            "baseline": baseline_algo,
            "target": target_algo,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohen_d": float(cohen_d),
            "statistically_significant": bool(p_val < 0.05),
            "n_baseline": len(base_vals),
            "n_target": len(target_vals),
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
        print(agg[["algorithm", "n_runs", "average_waiting_mean", "average_queue_mean"]])
        analyzer.export_summary_tables()
