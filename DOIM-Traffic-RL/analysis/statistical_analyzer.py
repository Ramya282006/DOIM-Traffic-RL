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

from common.experiment import (
    validate_experiment_manifest,
    classify_experiment_manifest,
    ManifestStatus,
)


class StatisticalAnalyzer:
    """
    Parses completed experiment runs from results/runs and performs
    rigorous statistical testing and aggregation adhering to IEEE T-ITS standards.
    """

    def __init__(
        self,
        runs_root: Union[str, Path] = "results/runs",
        runs_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        target = runs_dir if runs_dir is not None else runs_root
        self.runs_root = Path(target)
        self.rejected_runs: List[Dict[str, Any]] = []
        self.legacy_runs: List[str] = []
        self.valid_runs: List[str] = []

    def load_completed_runs(self, reject_incomplete: bool = True) -> pd.DataFrame:
        """Alias for load_run_summaries."""
        return self.load_run_summaries(reject_incomplete=reject_incomplete)

    def load_run_summaries(self, reject_incomplete: bool = True) -> pd.DataFrame:
        """
        Scans all subdirectories in results/runs, validates manifest and CSV integrity,
        and aggregates run records. Rejects incomplete, failed, or corrupted runs.
        Maintains an audit trail of rejected runs with explicit rejection reasons.
        Sorts records by timestamp before deduplicating.
        """
        records = []
        self.rejected_runs = []
        self.legacy_runs = []
        self.valid_runs = []

        if not self.runs_root.exists():
            return pd.DataFrame()

        for run_dir in sorted(self.runs_root.iterdir()):
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "summary.json"
            manifest_path = run_dir / "manifest.json"
            metrics_path = run_dir / "metrics_step.csv"

            # Check manifest and classify
            status, reason, migrated_manifest = classify_experiment_manifest(manifest_path, allow_legacy=True)

            if status in (ManifestStatus.INCOMPLETE, ManifestStatus.INVALID):
                self.rejected_runs.append({
                    "run_dir": str(run_dir.name),
                    "status": status,
                    "reason": reason,
                })
                if reject_incomplete:
                    continue
            elif status == ManifestStatus.LEGACY_VALID:
                self.legacy_runs.append(str(run_dir.name))
            elif status == ManifestStatus.VALID:
                self.valid_runs.append(str(run_dir.name))

            # Verify metrics CSV file and count recorded steps
            if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
                self.rejected_runs.append({
                    "run_dir": str(run_dir.name),
                    "status": "INCOMPLETE_CSV",
                    "reason": "Missing or empty metrics_step.csv",
                })
                if reject_incomplete:
                    continue

            try:
                with open(metrics_path, "r", encoding="utf-8") as f_csv:
                    # Count non-header non-empty rows
                    csv_lines = [line.strip() for line in f_csv if line.strip()]
                    csv_step_count = max(0, len(csv_lines) - 1)
            except Exception as e:
                self.rejected_runs.append({
                    "run_dir": str(run_dir.name),
                    "status": "CORRUPT_CSV",
                    "reason": f"Failed to read metrics_step.csv: {e}",
                })
                if reject_incomplete:
                    continue

            if not summary_path.is_file():
                self.rejected_runs.append({
                    "run_dir": str(run_dir.name),
                    "status": "MISSING_SUMMARY",
                    "reason": "summary.json file does not exist",
                })
                if reject_incomplete:
                    continue

            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                data["run_dir"] = str(run_dir)

                # Cross-Artifact Consistency Checks (Summary vs Manifest vs Metrics CSV)
                inconsistencies = []

                # 1. Step count consistency
                sum_steps = data.get("total_steps")
                if sum_steps is not None and int(sum_steps) != csv_step_count:
                    inconsistencies.append(
                        f"Step count mismatch: summary says {sum_steps} steps, but metrics_step.csv has {csv_step_count} rows"
                    )

                # 2. Manifest vs Summary identity consistency
                if migrated_manifest:
                    man_algo = migrated_manifest.get("algorithm")
                    sum_algo = data.get("algorithm")
                    if sum_algo and man_algo and sum_algo != man_algo:
                        inconsistencies.append(f"Algorithm mismatch: summary='{sum_algo}' vs manifest='{man_algo}'")

                    man_scen = migrated_manifest.get("scenario")
                    sum_scen = data.get("scenario")
                    if sum_scen and man_scen and sum_scen != man_scen:
                        inconsistencies.append(f"Scenario mismatch: summary='{sum_scen}' vs manifest='{man_scen}'")

                    sum_train = data.get("train_seed", data.get("seed"))
                    man_train = migrated_manifest.get("train_seed")
                    if sum_train is not None and man_train is not None and int(sum_train) != int(man_train):
                        inconsistencies.append(f"Train seed mismatch: summary={sum_train} vs manifest={man_train}")

                    sum_eval = data.get("eval_seed")
                    man_eval = migrated_manifest.get("evaluation_seed", migrated_manifest.get("eval_seed"))
                    if sum_eval is not None and man_eval is not None and int(sum_eval) != int(man_eval):
                        inconsistencies.append(f"Eval seed mismatch: summary={sum_eval} vs manifest={man_eval}")

                    sum_mode = data.get("evaluation_mode")
                    man_mode = migrated_manifest.get("evaluation_mode")
                    if sum_mode and man_mode and man_mode != "UNKNOWN" and sum_mode != man_mode:
                        inconsistencies.append(f"Evaluation mode mismatch: summary='{sum_mode}' vs manifest='{man_mode}'")

                if inconsistencies:
                    self.rejected_runs.append({
                        "run_dir": str(run_dir.name),
                        "status": "DATA_INCONSISTENCY",
                        "reason": "; ".join(inconsistencies),
                    })
                    if reject_incomplete:
                        continue

                # Seed & provenance extraction without invented defaults
                sum_train = data.get("train_seed", data.get("seed"))
                man_train = migrated_manifest.get("train_seed") if migrated_manifest else None
                train_seed = int(sum_train) if sum_train is not None else (int(man_train) if man_train is not None else None)
                data["train_seed"] = train_seed

                sum_eval = data.get("eval_seed")
                man_eval = migrated_manifest.get("evaluation_seed", migrated_manifest.get("eval_seed")) if migrated_manifest else None
                eval_seed = int(sum_eval) if sum_eval is not None else (int(man_eval) if man_eval is not None else None)
                data["eval_seed"] = eval_seed

                data["episode_id"] = int(data.get("episode_id", migrated_manifest.get("episode_id", 0) if migrated_manifest else 0))

                sum_mode = data.get("evaluation_mode")
                man_mode = migrated_manifest.get("evaluation_mode") if migrated_manifest else None
                data["evaluation_mode"] = str(sum_mode or man_mode or "UNKNOWN")
                data["scenario"] = str(data.get("scenario", migrated_manifest.get("scenario", "unknown_scenario") if migrated_manifest else "unknown_scenario"))
                data["algorithm"] = str(data.get("algorithm", migrated_manifest.get("algorithm", "unknown_algo") if migrated_manifest else "unknown_algo"))
                data["start_time"] = str(data.get("start_time", run_dir.name))
                data["reproducibility_eligible"] = bool(migrated_manifest.get("reproducibility_eligible", False) if migrated_manifest else False)

                # Fairness metric identification strictly via explicit metadata (NO numerical thresholding!)
                raw_fairness = float(data.get("average_fairness", 0.0))
                metric_name = data.get("fairness_metric_name")
                if not metric_name and migrated_manifest:
                    metric_name = migrated_manifest.get("summary_metrics", {}).get("fairness_metric_name")

                # Also inspect config_snapshot.yaml if available
                if not metric_name:
                    cfg_path = run_dir / "config_snapshot.yaml"
                    if cfg_path.is_file():
                        try:
                            import yaml
                            with open(cfg_path, "r", encoding="utf-8") as cf:
                                loaded_cfg = yaml.safe_load(cf)
                                if isinstance(loaded_cfg, dict):
                                    metric_name = loaded_cfg.get("objectives", {}).get("fairness_metric")
                        except Exception:
                            pass

                metric_name_str = str(metric_name).lower() if metric_name else ""
                if metric_name_str in ("delay_disparity", "jain_index", "jain_index_or_delay_disparity", "intersection_fairness_disparity", "vehicle_fairness_disparity"):
                    data["fairness_metric_name"] = "delay_disparity"
                    data["fairness_metric_unit"] = "dimensionless"
                    data["fairness_disparity"] = raw_fairness
                    data["fairness_variance"] = np.nan
                    data["fairness_eligible"] = True
                elif metric_name_str in ("delay_variance", "variance", "delay_var"):
                    data["fairness_metric_name"] = "delay_variance"
                    data["fairness_metric_unit"] = "s^2"
                    data["fairness_disparity"] = np.nan
                    data["fairness_variance"] = raw_fairness
                    data["fairness_eligible"] = True
                else:
                    # Unidentifiable / ambiguous fairness metric: classify as UNKNOWN / ineligible rather than guessing!
                    data["fairness_metric_name"] = "UNKNOWN"
                    data["fairness_metric_unit"] = "UNKNOWN"
                    data["fairness_disparity"] = np.nan
                    data["fairness_variance"] = np.nan
                    data["fairness_eligible"] = False

                # Primary algorithm vs ablation identification
                algo_str = str(data.get("algorithm", ""))
                is_ablation = bool(
                    data.get("ablation_mode") is not None
                    or (migrated_manifest and migrated_manifest.get("ablation_mode") is not None)
                    or algo_str.startswith("DOIM_no_")
                    or algo_str.startswith("DOIM_rolling_")
                    or "ablation" in algo_str.lower()
                )
                data["is_ablation"] = is_ablation
                data["configuration_type"] = "Ablation" if is_ablation else "Primary Algorithm"

                records.append(data)
            except Exception as e:
                self.rejected_runs.append({
                    "run_dir": str(run_dir.name),
                    "status": "PARSE_ERROR",
                    "reason": f"Failed to load summary.json: {e}",
                })
                print(f"Warning: Failed to load summary from {run_dir}: {e}")

        df = pd.DataFrame(records)
        if df.empty:
            return df

        # Sort explicitly by start_time / run_dir before deduplication
        df["train_seed_sort"] = df["train_seed"].fillna(-1)
        df = df.sort_values(by=["algorithm", "scenario", "train_seed_sort", "episode_id", "start_time"])
        df = df.drop(columns=["train_seed_sort"])

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
            "fairness_disparity",
            "fairness_variance",
            "average_reward",
            "total_shield_interventions",
        ]

        agg_rows = []
        # Strictly separate scenarios and algorithms
        for (scenario, algo), group in df.groupby(["scenario", "algorithm"]):
            n_independent_seeds = int(group["train_seed"].nunique())
            n_eval_episodes = len(group)
            is_ablation = bool(group["is_ablation"].iloc[0]) if "is_ablation" in group.columns else False
            config_type = "Ablation" if is_ablation else "Primary Algorithm"

            row = {
                "scenario": scenario,
                "algorithm": algo,
                "configuration_type": config_type,
                "is_ablation": is_ablation,
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
        Also exports rejected_runs.json documenting all rejected runs and reasons.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        df = self.load_run_summaries()

        # Export rejected runs audit trail
        rejected_path = out_path / "rejected_runs.json"
        with open(rejected_path, "w", encoding="utf-8") as f:
            json.dump({
                "rejected_count": len(self.rejected_runs),
                "legacy_valid_count": len(self.legacy_runs),
                "canonical_valid_count": len(self.valid_runs),
                "rejected_runs": self.rejected_runs,
            }, f, indent=2)

        if df.empty:
            print("No valid run summaries found to export.")
            return

        agg_df = self.aggregate_by_scenario_and_algorithm(df)
        csv_path = out_path / "benchmark_comparison.csv"
        agg_df.to_csv(csv_path, index=False)

        def _to_md_safe(d):
            try:
                return d.to_markdown(index=False)
            except Exception:
                return d.to_string()

        # Generate markdown table for paper documentation
        md_path = out_path / "benchmark_comparison.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Benchmark Comparison Table (Scenario-Specific Aggregation)\n\n")
            f.write(_to_md_safe(agg_df))
            f.write("\n")

        # Export per-scenario isolated tables
        for scenario_name, scen_group in agg_df.groupby("scenario"):
            scen_slug = scenario_name.replace("/", "_").replace(".sumocfg", "")
            scen_csv = out_path / f"{scen_slug}_benchmark_comparison.csv"
            scen_md = out_path / f"{scen_slug}_benchmark_comparison.md"
            scen_group.to_csv(scen_csv, index=False)
            with open(scen_md, "w", encoding="utf-8") as f:
                f.write(f"# Benchmark Comparison: Scenario `{scenario_name}`\n\n")
                f.write(_to_md_safe(scen_group))
                f.write("\n")

        # Export isolated primary vs ablation tables if ablations exist
        if "is_ablation" in agg_df.columns:
            primary_df = agg_df[agg_df["is_ablation"] == False].reset_index(drop=True)
            ablation_df = agg_df[agg_df["is_ablation"] == True].reset_index(drop=True)
            if not primary_df.empty:
                primary_df.to_csv(out_path / "primary_benchmark_comparison.csv", index=False)
                with open(out_path / "primary_benchmark_comparison.md", "w", encoding="utf-8") as f:
                    f.write("# Primary Benchmark Comparison Table (Excluding Ablations)\n\n")
                    f.write(_to_md_safe(primary_df))
                    f.write("\n")
            if not ablation_df.empty:
                ablation_df.to_csv(out_path / "ablation_study_comparison.csv", index=False)
                with open(out_path / "ablation_study_comparison.md", "w", encoding="utf-8") as f:
                    f.write("# Ablation Study Comparison Table\n\n")
                    f.write(_to_md_safe(ablation_df))
                    f.write("\n")

        print(f"Summary tables successfully exported to: {out_path} (including rejected runs audit)")


if __name__ == "__main__":
    analyzer = StatisticalAnalyzer()
    df = analyzer.load_run_summaries()
    print("Loaded runs:", len(df))
    if not df.empty:
        agg = analyzer.aggregate_by_scenario_and_algorithm(df)
        print(agg[["scenario", "algorithm", "n_independent_seeds", "n_evaluation_episodes", "average_waiting_mean", "average_queue_mean"]])
        analyzer.export_summary_tables()
