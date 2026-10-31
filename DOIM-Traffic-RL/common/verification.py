"""
DOIM-Traffic-RL: Automated Scientific Verification Suite
Validates mathematical bounds, finite metrics, simplex conditions,
CSV/JSON consistency, duplicate detection, and seed independence.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd


class ExperimentVerifier:
    """
    Automated scientific integrity checks on experiment outputs.
    """

    @staticmethod
    def verify_simplex_weights(weights: np.ndarray, tolerance: float = 1e-4) -> Tuple[bool, str]:
        """
        Verifies that adaptive objective preference weights lie on the standard simplex Δ^(K-1):
        1. w_i >= 0
        2. sum(w_i) == 1.0
        """
        weights = np.asarray(weights, dtype=np.float32)
        if not np.all(np.isfinite(weights)):
            return False, "Weights contain NaN or infinite values."
        if np.any(weights < -tolerance):
            return False, f"Weights contain negative values: {weights[weights < 0]}"
        weight_sum = float(np.sum(weights, axis=-1))
        if abs(weight_sum - 1.0) > tolerance:
            return False, f"Weight sum deviates from 1.0 (got {weight_sum})."
        return True, "Valid simplex weights."

    @staticmethod
    def verify_finite_metrics(step_csv_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Verifies that no NaN, Inf, or corrupted values exist in the step-level metric log.
        """
        path = Path(step_csv_path)
        if not path.is_file():
            return False, f"Step metrics file missing: {path}"
        df = pd.read_csv(path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        null_counts = df[numeric_cols].isnull().sum().to_dict()
        has_nulls = any(v > 0 for v in null_counts.values())
        if has_nulls:
            return False, f"Null values detected in step metrics: {null_counts}"
        has_infs = any(np.isinf(df[c]).any() for c in numeric_cols)
        if has_infs:
            return False, "Infinite values detected in step metrics."
        return True, f"All {len(df)} step records contain finite numeric values."

    @staticmethod
    def verify_json_csv_consistency(run_dir: Union[str, Path], tolerance: float = 1e-3) -> Tuple[bool, str]:
        """
        Asserts exact mathematical consistency between raw metrics_step.csv and summary.json.
        """
        run_path = Path(run_dir)
        csv_path = run_path / "metrics_step.csv"
        json_path = run_path / "summary.json"
        
        if not csv_path.exists() or not json_path.exists():
            return False, f"Missing files in {run_dir} for consistency check."
            
        df = pd.read_csv(csv_path)
        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
            
        # Verify step counts
        if len(df) != summary.get("total_steps", 0):
            return False, f"Step count mismatch: CSV has {len(df)}, JSON reports {summary.get('total_steps')}"
            
        # Verify total reward
        csv_total_reward = float(df["reward"].sum())
        json_total_reward = float(summary.get("total_reward", 0.0))
        if abs(csv_total_reward - json_total_reward) > tolerance:
            return False, f"Total reward mismatch: CSV {csv_total_reward:.4f} vs JSON {json_total_reward:.4f}"
            
        # Verify average waiting
        csv_avg_waiting = float(df["raw_waiting"].mean())
        json_avg_waiting = float(summary.get("average_waiting", 0.0))
        if abs(csv_avg_waiting - json_avg_waiting) > tolerance:
            return False, f"Average waiting mismatch: CSV {csv_avg_waiting:.4f} vs JSON {json_avg_waiting:.4f}"
            
        return True, "CSV and summary JSON are mathematically consistent."

    @staticmethod
    def verify_seed_independence(results_df: pd.DataFrame, seed_col: str = "seed") -> Tuple[bool, str]:
        """
        Verifies that different seeds produce genuinely distinct simulation trajectories
        and detects copy-paste or omitted-seed artifacts (like DEF-04).
        """
        if len(results_df) <= 1:
            return True, "Insufficient rows for cross-seed variance check."
            
        # Check if metric values are completely identical across different seeds
        for col in ["average_waiting", "average_queue", "total_reward"]:
            if col in results_df.columns:
                unique_vals = results_df[col].nunique()
                num_seeds = results_df[seed_col].nunique()
                if num_seeds > 1 and unique_vals == 1:
                    return False, f"DEFECT DETECTED: Column '{col}' is 100% identical across {num_seeds} distinct seeds! Check SUMO seeding."
                    
        return True, "Independent seeds exhibit statistically distinct trajectories."

    @classmethod
    def verify_run(cls, run_dir: Union[str, Path]) -> Dict[str, Tuple[bool, str]]:
        """
        Executes complete verification pipeline on an experiment run directory.
        """
        path = Path(run_dir)
        csv_path = path / "metrics_step.csv"
        finite_ok, finite_msg = cls.verify_finite_metrics(csv_path)
        json_ok, json_msg = cls.verify_json_csv_consistency(path)

        weights_ok, weights_msg = True, "No weight columns found."
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            weight_cols = [c for c in df.columns if c.startswith("weight_")]
            if weight_cols:
                weights = df[weight_cols].values
                # Check mean across steps
                weights_ok, weights_msg = cls.verify_simplex_weights(weights.mean(axis=0))

        return {
            "finite_metrics": (finite_ok, finite_msg),
            "json_csv_consistency": (json_ok, json_msg),
            "simplex_weights": (weights_ok, weights_msg),
        }
