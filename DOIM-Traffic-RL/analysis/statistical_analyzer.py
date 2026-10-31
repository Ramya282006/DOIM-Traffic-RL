"""
DOIM-Traffic-RL: Scenario-Specific Statistical Evaluation & Hypothesis Testing Suite
Strictly isolates experimental scenarios; treats independent training seeds as primary statistical units;
averages repeated evaluation episodes within trained models; supports Holm-Bonferroni corrections,
paired Student's t-test, Wilcoxon signed-rank test, and Welch's t-test.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from common.experiment import validate_experiment_manifest


class StatisticalAnalyzer:
    """
    Parses completed experiment runs from results/runs and performs
    rigorous statistical testing and aggregation adhering to IEEE T-ITS standards.
    """

    def __init__(self, runs_root: Union[str, Path] = "results/runs"):
        self.runs_root = Path(runs_root)

    def load_run_summaries(self, reject_incomplete: bool = True) -> pd.DataFrame:
        """
        Scans all subdirectories in results/runs, validates manifest and CSV integrity,
        and aggregates run records. Rejects incomplete, failed, or corrupted runs.
        Sorts records by timestamp before deduplicating.
        """
        records = []
        if not self.runs_root.exists():
            return pd.DataFrame()

        for run_dir in sorted(self.runs_root.iterdir()):
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "summary.json"
            manifest_path = run_dir / "manifest.json"
            metrics_path = run_dir / "metrics_step.csv"

            if not summary_path.is_file():
                continue

            # Strict completion verification
            if reject_incomplete:
                if manifest_path.is_file():
                    is_valid, msg = validate_experiment_manifest(manifest_path)
                    if not is_valid:
                        # Skip incomplete/failed runs
                        continue
                if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
                    continue

            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                data["run_dir"] = str(run_dir)
                # Seed extraction
                seed_val = data.get("train_seed", data.get("seed", 42))
                data["train_seed"] = int(seed_val)
                data["eval_seed"] = int(data.get("eval_seed", data.get("evaluation_seed", seed_val)))
                data["episode_id"] = int(data.get("episode_id", 0))
                data["evaluation_mode"] = str(data.get("evaluation_mode", "train_rollout"))
                data["scenario"] = str(data.get("scenario", "unknown_scenario"))
                data["algorithm"] = str(data.get("algorithm", "unknown_algo"))
                data["start_time"] = str(data.get("start_time", run_dir.name))

                records.append(data)
            except Exception as e:
                print(f"Warning: Failed to load summary from {run_dir}: {e}")

        df = pd.DataFrame(records)
        if df.empty:
            return df

        # Sort explicitly by start_time / run_dir before deduplication
        df = df.sort_values(by=["algorithm", "scenario", "train_seed", "episode_id", "start_time"])

        # Deduplicate records that have identical algorithm, scenario, train_seed, and episode_id
        initial_len = len(df)
        df = df.drop_duplicates(
            subset=["algorithm", "scenario", "train_seed", "episode_id"],
            keep="last",
        ).reset_index(drop=True)
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"[StatisticalAnalyzer] Deduplicated {dropped} redundant run records.")

        return df

    @staticmethod
    def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Computes sample mean and 95% Student-t confidence interval bounds.
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

    @staticmethod
    def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Dict[str, Any]]:
        """
        Applies Holm-Bonferroni step-down correction to a list of p-values.
        Guarantees strong family-wise error rate (FWER) control at level alpha.
        """
        m = len(p_values)
        if m == 0:
            return []

        # Sort by p-value ascending while preserving original index
        indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
        results = [None] * m

        running_max_adj = 0.0
        rejected = True
        for rank, (orig_idx, p_val) in enumerate(indexed_p):
            k = rank + 1
            threshold = alpha / (m - k + 1)
            adj_p = min(1.0, max(running_max_adj, p_val * (m - k + 1)))
            running_max_adj = adj_p

            if p_val > threshold:
                rejected = False

            results[orig_idx] = {
                "rank": k,
                "raw_p_value": float(p_val),
                "adjusted_p_value": float(adj_p),
                "threshold": float(threshold),
                "significant": bool(rejected and (p_val <= threshold)),
            }

        return results

    def aggregate_by_scenario_and_algorithm(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Strict Phase 2 scenario-specific aggregation:
        Grouping strictly by (scenario, algorithm).
        Statistical unit is strictly each independent training seed.
        Repeated evaluation episodes within each trained model are averaged before cross-model statistics.
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
        # Strictly separate scenarios and algorithms
        for (scenario, algo), group in df.groupby(["scenario", "algorithm"]):
            n_independent_seeds = int(group["train_seed"].nunique())
            n_eval_episodes = len(group)

            row = {
                "scenario": scenario,
                "algorithm": algo,
                "n_independent_seeds": n_independent_seeds,
                "n_evaluation_episodes": n_eval_episodes,
            }

            for m in target_metrics:
                if m in group.columns:
                    # Intra-model averaging: average repeated evaluation episodes for each trained seed
                    seed_level_series = group.groupby("train_seed")[m].mean().dropna()
                    vals = seed_level_series.values

                    if len(vals) > 0:
                        mean, ci_low, ci_high = self.compute_confidence_interval(vals)
                        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                        median = float(np.median(vals))
                        row[f"{m}_mean"] = mean
                        row[f"{m}_std"] = std
                        row[f"{m}_median"] = median
                        row[f"{m}_ci95_low"] = ci_low
                        row[f"{m}_ci95_high"] = ci_high
                        row[f"{m}_min"] = float(np.min(vals))
                        row[f"{m}_max"] = float(np.max(vals))

            agg_rows.append(row)

        res_df = pd.DataFrame(agg_rows)
        if not res_df.empty:
            res_df = res_df.sort_values(by=["scenario", "algorithm"]).reset_index(drop=True)
        return res_df

    def aggregate_by_algorithm(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Backward-compatible wrapper for single-scenario legacy scripts.
        """
        return self.aggregate_by_scenario_and_algorithm(df)

    def run_hypothesis_tests(
        self,
        df: Optional[pd.DataFrame] = None,
        scenario: Optional[str] = None,
        baseline_algo: str = "FW_PPO",
        target_algo: str = "IA_MOPPO",
        metric: str = "average_waiting",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Runs rigorous significance testing strictly within a specified scenario across independent training seeds.
        - Uses paired Student's t-test or Wilcoxon signed-rank test when seeds match.
        - Uses Welch's two-sample t-test when seeds do not match.
        - Explicitly reports insufficient sample size.
        """
        if df is None:
            df = self.load_run_summaries()

        if scenario is not None:
            df = df[df["scenario"] == scenario]

        if df.empty:
            return {"status": "EMPTY_DATA"}

        # Intra-model averaging per independent seed
        base_df = df[df["algorithm"] == baseline_algo].groupby("train_seed")[metric].mean().dropna()
        target_df = df[df["algorithm"] == target_algo].groupby("train_seed")[metric].mean().dropna()

        common_seeds = sorted(list(set(base_df.index).intersection(set(target_df.index))))
        is_paired = len(common_seeds) >= 3

        if is_paired:
            base_vals = base_df.loc[common_seeds].values
            target_vals = target_df.loc[common_seeds].values
            n_obs = len(common_seeds)

            t_stat, p_val = stats.ttest_rel(target_vals, base_vals)
            test_type = "Paired Student's t-test"

            # Wilcoxon signed-rank test for paired non-normal data if n >= 5
            wilcoxon_stat, wilcoxon_p = None, None
            if n_obs >= 5:
                try:
                    diffs = target_vals - base_vals
                    if not np.all(diffs == 0):
                        w_res = stats.wilcoxon(target_vals, base_vals)
                        wilcoxon_stat = float(w_res.statistic)
                        wilcoxon_p = float(w_res.pvalue)
                except Exception:
                    pass

        else:
            base_vals = base_df.values
            target_vals = target_df.values
            if len(base_vals) < 2 or len(target_vals) < 2:
                return {
                    "scenario": scenario,
                    "metric": metric,
                    "baseline": baseline_algo,
                    "target": target_algo,
                    "status": "INSUFFICIENT_INDEPENDENT_SEEDS",
                    "n_baseline_seeds": len(base_vals),
                    "n_target_seeds": len(target_vals),
                }
            t_stat, p_val = stats.ttest_ind(target_vals, base_vals, equal_var=False)
            test_type = "Welch's two-sample t-test"
            wilcoxon_stat, wilcoxon_p = None, None

        # Cohen's d effect size
        s_pooled = np.sqrt(
            ((len(target_vals) - 1) * np.var(target_vals, ddof=1) + (len(base_vals) - 1) * np.var(base_vals, ddof=1))
            / max(1, (len(target_vals) + len(base_vals) - 2))
        )
        cohen_d = (np.mean(target_vals) - np.mean(base_vals)) / (s_pooled + 1e-8)

        return {
            "scenario": scenario,
            "metric": metric,
            "baseline": baseline_algo,
            "target": target_algo,
            "test_type": test_type,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "wilcoxon_statistic": wilcoxon_stat,
            "wilcoxon_p_value": wilcoxon_p,
            "statistically_significant": bool(p_val < alpha),
            "cohen_d": float(cohen_d),
            "n_baseline_seeds": len(base_vals),
            "n_target_seeds": len(target_vals),
            "baseline_mean": float(np.mean(base_vals)),
            "target_mean": float(np.mean(target_vals)),
        }

    def export_summary_tables(self, output_dir: Union[str, Path] = "results/tables"):
        """
        Exports clean markdown and CSV comparison tables separated strictly by scenario.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        df = self.load_run_summaries()
        if df.empty:
            print("No run summaries found to export.")
            return

        agg_df = self.aggregate_by_scenario_and_algorithm(df)
        csv_path = out_path / "benchmark_comparison.csv"
        agg_df.to_csv(csv_path, index=False)

        # Generate markdown table for paper documentation
        md_path = out_path / "benchmark_comparison.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Benchmark Comparison Table (Scenario-Specific Aggregation)\n\n")
            f.write(agg_df.to_markdown(index=False) if hasattr(agg_df, "to_markdown") else agg_df.to_string())
            f.write("\n")

        print(f"Summary tables successfully exported to: {out_path}")


if __name__ == "__main__":
    analyzer = StatisticalAnalyzer()
    df = analyzer.load_run_summaries()
    print("Loaded runs:", len(df))
    if not df.empty:
        agg = analyzer.aggregate_by_scenario_and_algorithm(df)
        print(agg[["scenario", "algorithm", "n_independent_seeds", "n_evaluation_episodes", "average_waiting_mean", "average_queue_mean"]])
        analyzer.export_summary_tables()
