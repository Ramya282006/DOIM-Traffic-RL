"""
DOIM-Traffic-RL: Structured Experiment Execution and Run Manifest Manager
Provides immutable run artifact generation, per-step structured logging,
manifest tracking, and metadata validation.
"""

import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from common.config import snapshot_config


def get_git_commit_sha() -> str:
    """
    Retrieves current Git commit hash if in a git repository.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return sha
    except Exception:
        return "UNKNOWN_COMMIT"


REQUIRED_MANIFEST_SCHEMA_FIELDS = [
    "experiment_id",
    "algorithm",
    "scenario",
    "train_seed",
    "evaluation_seed",
    "training_timesteps",
    "episode_id",
    "evaluation_mode",
    "config_hash",
    "code_commit",
    "start_time",
    "completion_status",
]


def validate_experiment_manifest(manifest: Union[Dict[str, Any], str, Path]) -> Tuple[bool, str]:
    """
    Strict IEEE-standard experiment schema validator.
    Rejects incomplete, corrupted, or inconsistent experiment records.
    """
    if isinstance(manifest, (str, Path)):
        p = Path(manifest)
        if not p.is_file():
            return False, f"Manifest file does not exist: {p}"
        try:
            with open(p, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as e:
            return False, f"Failed to parse manifest JSON: {e}"
    elif isinstance(manifest, dict):
        manifest_data = manifest
    else:
        return False, f"Invalid manifest object type: {type(manifest)}"

    # Check all required schema fields
    for field in REQUIRED_MANIFEST_SCHEMA_FIELDS:
        if field not in manifest_data:
            return False, f"Missing required schema field: '{field}'"
        if manifest_data[field] is None:
            return False, f"Schema field '{field}' cannot be None"

    # Validate completion status
    status = str(manifest_data.get("completion_status", "")).upper()
    if status != "COMPLETED":
        return False, f"Incomplete or unverified experiment record (completion_status='{status}')"

    # Validate seed types
    try:
        train_seed = int(manifest_data["train_seed"])
        eval_seed = int(manifest_data["evaluation_seed"])
    except (ValueError, TypeError):
        return False, "train_seed and evaluation_seed must be valid integers"

    # Validate evaluation mode
    valid_modes = {"train_rollout", "deterministic_eval", "stochastic_eval", "smoke_test"}
    eval_mode = manifest_data.get("evaluation_mode", "")
    if eval_mode not in valid_modes:
        return False, f"Invalid evaluation_mode '{eval_mode}'. Must be one of {valid_modes}"

    # Validate training timesteps
    try:
        timesteps = int(manifest_data["training_timesteps"])
        if timesteps < 0:
            return False, "training_timesteps cannot be negative"
    except (ValueError, TypeError):
        return False, "training_timesteps must be a valid integer"

    return True, "Valid experiment manifest schema."


class ExperimentRun:
    """
    Manages the lifecycle, directory structure, logging, and metadata of a single experiment run.
    """

    def __init__(
        self,
        experiment_name: str,
        algorithm: str,
        config: Dict[str, Any],
        train_seed: int,
        eval_seed: Optional[int] = None,
        scenario: str = "directional_shift",
        output_root: Union[str, Path] = "results/runs",
        training_timesteps: int = 128,
        episode_id: int = 0,
        evaluation_mode: str = "train_rollout",
        experiment_id: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.algorithm = algorithm
        self.config = config
        self.train_seed = int(train_seed)
        self.eval_seed = int(eval_seed if eval_seed is not None else train_seed)
        self.scenario = scenario
        self.training_timesteps = int(training_timesteps)
        self.episode_id = int(episode_id)
        self.evaluation_mode = str(evaluation_mode)
        self.start_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Unique run ID / Experiment ID
        timestamp_slug = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_slug = f"s{self.train_seed}"
        self.run_id = f"{algorithm}_{scenario}_{seed_slug}_{timestamp_slug}"
        self.experiment_id = experiment_id if experiment_id is not None else self.run_id
        
        # Directory structure
        self.run_dir = Path(output_root) / self.run_id
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.logs_dir = self.run_dir / "logs"
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Snapshot configuration
        snapshot_config(config, self.run_dir)
        
        # Manifest
        self.manifest_path = self.run_dir / "manifest.json"
        self.metrics_csv_path = self.run_dir / "metrics_step.csv"
        self.summary_json_path = self.run_dir / "summary.json"
        
        self.csv_file = None
        self.csv_writer = None
        self.step_records: List[Dict[str, Any]] = []
        
        self.init_manifest()

    def init_manifest(self) -> None:
        """
        Initializes manifest.json with runtime metadata adhering strictly to Phase 1 schema.
        """
        manifest = {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "algorithm": self.algorithm,
            "scenario": self.scenario,
            "train_seed": self.train_seed,
            "evaluation_seed": self.eval_seed,
            "training_timesteps": self.training_timesteps,
            "episode_id": self.episode_id,
            "evaluation_mode": self.evaluation_mode,
            "config_hash": self.config.get("config_hash", "UNKNOWN"),
            "code_commit": get_git_commit_sha(),
            "start_time": self.start_timestamp,
            "end_time": None,
            "completion_status": "INITIALIZED",
            "model_checkpoint_sha256": None,
            "total_steps": 0,
            "summary_metrics": {},
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def open_step_logger(self) -> None:
        """
        Opens CSV writer for streaming step metrics.
        """
        fieldnames = [
            "step",
            "simulation_time",
            "reward",
            "raw_waiting",
            "raw_queue",
            "raw_co2",
            "raw_throughput",
            "raw_fairness",
            "norm_waiting",
            "norm_queue",
            "norm_co2",
            "norm_throughput",
            "norm_fairness",
            "weight_waiting",
            "weight_queue",
            "weight_co2",
            "weight_throughput",
            "weight_fairness",
            "action_tls",
            "shield_interventions",
        ]
        self.csv_file = open(self.metrics_csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()

    def log_step(
        self,
        step: int,
        sim_time: float,
        reward: float,
        raw_objs: Union[List[float], np.ndarray],
        norm_objs: Union[List[float], np.ndarray],
        weights: Union[List[float], np.ndarray],
        actions: Any,
        shield_interventions: int = 0,
    ) -> None:
        """
        Streams a single simulation step record.
        """
        if self.csv_writer is None:
            self.open_step_logger()

        record = {
            "step": step,
            "simulation_time": float(sim_time),
            "reward": float(reward),
            "raw_waiting": float(raw_objs[0]),
            "raw_queue": float(raw_objs[1]),
            "raw_co2": float(raw_objs[2]),
            "raw_throughput": float(raw_objs[3]),
            "raw_fairness": float(raw_objs[4]),
            "norm_waiting": float(norm_objs[0]),
            "norm_queue": float(norm_objs[1]),
            "norm_co2": float(norm_objs[2]),
            "norm_throughput": float(norm_objs[3]),
            "norm_fairness": float(norm_objs[4]),
            "weight_waiting": float(weights[0]),
            "weight_queue": float(weights[1]),
            "weight_co2": float(weights[2]),
            "weight_throughput": float(weights[3]),
            "weight_fairness": float(weights[4]),
            "action_tls": str(actions),
            "shield_interventions": int(shield_interventions),
        }
        self.csv_writer.writerow(record)
        self.step_records.append(record)

    def close(self, status: str = "COMPLETED", final_model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Closes log streams, computes summary statistics, updates manifest, and computes model checksum.
        """
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()

        # Compute summary metrics from step records
        summary = self._compute_summary()
        
        # Model checksum if provided
        model_hash = None
        if final_model_path and Path(final_model_path).exists():
            with open(final_model_path, "rb") as f:
                model_hash = hashlib.sha256(f.read()).hexdigest()

        # Save summary JSON
        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Update manifest
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        manifest["end_time"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        manifest["completion_status"] = status
        manifest["status"] = status
        manifest["total_steps"] = len(self.step_records)
        manifest["summary_metrics"] = summary
        manifest["model_checkpoint_sha256"] = model_hash

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return summary

    def _compute_summary(self) -> Dict[str, Any]:
        """
        Computes mean, std, min, max, and quantiles across recorded steps.
        """
        if not self.step_records:
            return {}

        rewards = [r["reward"] for r in self.step_records]
        waitings = [r["raw_waiting"] for r in self.step_records]
        queues = [r["raw_queue"] for r in self.step_records]
        co2s = [r["raw_co2"] for r in self.step_records]
        throughputs = [r["raw_throughput"] for r in self.step_records]
        fairnesses = [r["raw_fairness"] for r in self.step_records]
        shields = [r["shield_interventions"] for r in self.step_records]

        weights_matrix = np.array([
            [r["weight_waiting"], r["weight_queue"], r["weight_co2"], r["weight_throughput"], r["weight_fairness"]]
            for r in self.step_records
        ])

        summary = {
            "run_id": self.run_id,
            "algorithm": self.algorithm,
            "scenario": self.scenario,
            "seed": self.train_seed,
            "total_steps": len(self.step_records),
            "total_reward": float(np.sum(rewards)),
            "average_reward": float(np.mean(rewards)),
            "average_waiting": float(np.mean(waitings)),
            "max_waiting": float(np.max(waitings)),
            "average_queue": float(np.mean(queues)),
            "max_queue": float(np.max(queues)),
            "average_co2": float(np.mean(co2s)),
            "total_throughput": float(np.sum(throughputs)),
            "average_fairness": float(np.mean(fairnesses)),
            "total_shield_interventions": int(np.sum(shields)),
            "average_weights": [float(w) for w in np.mean(weights_matrix, axis=0)],
        }
        return summary
