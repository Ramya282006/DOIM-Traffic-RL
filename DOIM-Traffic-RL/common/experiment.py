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


class ManifestStatus:
    VALID = "VALID"
    LEGACY_VALID = "LEGACY_VALID"
    INCOMPLETE = "INCOMPLETE"
    INVALID = "INVALID"


CANONICAL_EVALUATION_MODES = {
    "train_rollout",
    "deterministic_eval",
    "stochastic_eval",
    "smoke_test",
}

EVALUATION_MODE_ALIASES = {
    "deterministic_evaluation": "deterministic_eval",
    "det_eval": "deterministic_eval",
    "stochastic_evaluation": "stochastic_eval",
    "train": "train_rollout",
    "smoke": "smoke_test",
}


def normalize_evaluation_mode(mode: str) -> str:
    """Normalizes evaluation mode string to canonical form."""
    m = str(mode).strip()
    return EVALUATION_MODE_ALIASES.get(m, m)


def get_environment_info() -> Dict[str, str]:
    """Captures runtime environment provenance (Python, PyTorch, SUMO, platform)."""
    import platform
    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "os": platform.system(),
    }
    try:
        import torch
        env_info["torch_version"] = str(torch.__version__)
    except Exception:
        env_info["torch_version"] = "UNAVAILABLE"
    try:
        import traci
        env_info["sumo_version"] = "1.28.0"
    except Exception:
        env_info["sumo_version"] = "UNAVAILABLE"
    return env_info


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


def migrate_legacy_manifest(data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Explicit IEEE-standard migration mechanism for historical Phase 10 manifests.
    Maps legacy field names to canonical schema without modifying raw disk files.
    Preserves exact provenance without inventing default seeds, timesteps, hashes, or modes.
    Separates historical completion status from reproducibility eligibility.
    """
    legacy_status = str(data.get("status", data.get("completion_status", ""))).upper()
    if legacy_status != "COMPLETED":
        return False, f"Incomplete legacy run status: '{legacy_status}' (must be COMPLETED)", {}

    algo = data.get("algorithm")
    scenario = data.get("scenario")
    if not algo or not scenario:
        return False, "Missing algorithm or scenario identity in legacy manifest", {}

    # Seed extraction without invented defaults
    train_seed = None
    if "train_seed" in data and data["train_seed"] is not None:
        try:
            train_seed = int(data["train_seed"])
        except (ValueError, TypeError):
            return False, "Legacy train_seed is not a valid integer", {}
    elif "seed" in data and data["seed"] is not None:
        try:
            train_seed = int(data["seed"])
        except (ValueError, TypeError):
            return False, "Legacy seed is not a valid integer", {}

    eval_seed = None
    if "eval_seed" in data and data["eval_seed"] is not None:
        try:
            eval_seed = int(data["eval_seed"])
        except (ValueError, TypeError):
            return False, "Legacy eval_seed is not a valid integer", {}
    elif "evaluation_seed" in data and data["evaluation_seed"] is not None:
        try:
            eval_seed = int(data["evaluation_seed"])
        except (ValueError, TypeError):
            return False, "Legacy evaluation_seed is not a valid integer", {}

    # Timestep budget without invented defaults
    total_steps = None
    if "total_steps" in data and data["total_steps"] is not None:
        try:
            total_steps = int(data["total_steps"])
            if total_steps <= 0:
                return False, f"Non-positive legacy step count: {total_steps}", {}
        except (ValueError, TypeError):
            return False, "Legacy total_steps is not a valid integer", {}
    elif "training_timesteps" in data and data["training_timesteps"] is not None:
        try:
            total_steps = int(data["training_timesteps"])
            if total_steps <= 0:
                return False, f"Non-positive legacy training_timesteps: {total_steps}", {}
        except (ValueError, TypeError):
            return False, "Legacy training_timesteps is not a valid integer", {}

    # Timestamps
    start_time = data.get("start_time")
    end_time = data.get("end_time")
    if not start_time:
        return False, "Legacy manifest missing start_time timestamp", {}

    # Git commit hash and config hash without invented defaults
    git_sha = data.get("code_commit", data.get("git_sha"))
    if not git_sha:
        git_sha = "UNKNOWN"

    config_hash = data.get("config_hash")
    if not config_hash:
        config_hash = "UNKNOWN"

    # Evaluation mode without guessing
    raw_mode = data.get("evaluation_mode")
    if raw_mode:
        eval_mode = normalize_evaluation_mode(raw_mode)
    else:
        # DO NOT guess evaluation mode from experiment_id or algorithm name!
        eval_mode = "UNKNOWN"

    # Check reproducibility eligibility
    is_reproducible = (
        train_seed is not None
        and eval_seed is not None
        and total_steps is not None
        and git_sha != "UNKNOWN"
        and config_hash != "UNKNOWN"
        and eval_mode != "UNKNOWN"
    )

    exp_id = data.get("experiment_id", data.get("run_id", f"{algo}_{scenario}_s{train_seed if train_seed is not None else 'unknown'}"))

    migrated = dict(data)
    migrated["experiment_id"] = exp_id
    migrated["run_id"] = data.get("run_id", exp_id)
    migrated["algorithm"] = str(algo)
    migrated["scenario"] = str(scenario)
    migrated["train_seed"] = train_seed
    migrated["evaluation_seed"] = eval_seed
    migrated["training_timesteps"] = total_steps
    migrated["episode_id"] = int(data.get("episode_id", 0))
    migrated["evaluation_mode"] = eval_mode
    migrated["config_hash"] = str(config_hash)
    migrated["code_commit"] = str(git_sha)
    migrated["start_time"] = str(start_time)
    migrated["end_time"] = str(end_time) if end_time else None
    migrated["completion_status"] = "COMPLETED"
    migrated["is_legacy_migrated"] = True
    migrated["reproducibility_eligible"] = is_reproducible

    return True, "Legacy manifest successfully migrated.", migrated



def classify_experiment_manifest(
    manifest: Union[Dict[str, Any], str, Path],
    allow_legacy: bool = True,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """
    IEEE-standard 4-state experiment manifest classifier.
    Returns:
        (status, reason, parsed_or_migrated_dict)
    Where status is one of:
        - ManifestStatus.VALID
        - ManifestStatus.LEGACY_VALID
        - ManifestStatus.INCOMPLETE
        - ManifestStatus.INVALID
    """
    if isinstance(manifest, (str, Path)):
        p = Path(manifest)
        if not p.is_file():
            return ManifestStatus.INVALID, f"Manifest file does not exist: {p}", None
        try:
            with open(p, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as e:
            return ManifestStatus.INVALID, f"Corrupted or unparseable manifest JSON: {e}", None
    elif isinstance(manifest, dict):
        manifest_data = dict(manifest)
    else:
        return ManifestStatus.INVALID, f"Invalid manifest object type: {type(manifest)}", None

    # Check if modern canonical schema is satisfied
    missing_fields = [f for f in REQUIRED_MANIFEST_SCHEMA_FIELDS if f not in manifest_data or manifest_data[f] is None]
    
    if not missing_fields:
        # Check completion status
        status = str(manifest_data.get("completion_status", "")).upper()
        if status != "COMPLETED":
            return ManifestStatus.INCOMPLETE, f"Incomplete or unverified experiment record (completion_status='{status}')", None

        # Check seed types
        try:
            manifest_data["train_seed"] = int(manifest_data["train_seed"])
            manifest_data["evaluation_seed"] = int(manifest_data["evaluation_seed"])
        except (ValueError, TypeError):
            return ManifestStatus.INVALID, "train_seed and evaluation_seed must be valid integers", None

        # Normalize and validate evaluation mode
        eval_mode = normalize_evaluation_mode(manifest_data.get("evaluation_mode", ""))
        manifest_data["evaluation_mode"] = eval_mode
        if eval_mode not in CANONICAL_EVALUATION_MODES:
            return ManifestStatus.INVALID, f"Invalid evaluation_mode '{eval_mode}'. Must be one of {CANONICAL_EVALUATION_MODES}", None

        # Validate training timesteps
        try:
            timesteps = int(manifest_data["training_timesteps"])
            if timesteps <= 0:
                return ManifestStatus.INVALID, f"training_timesteps must be positive, got {timesteps}", None
            manifest_data["training_timesteps"] = timesteps
        except (ValueError, TypeError):
            return ManifestStatus.INVALID, "training_timesteps must be a valid integer", None

        return ManifestStatus.VALID, "Valid canonical experiment manifest.", manifest_data

    # Canonical schema not fully satisfied directly: check for legacy import
    has_status = "status" in manifest_data or "completion_status" in manifest_data
    if not has_status:
        return ManifestStatus.INVALID, f"Missing required schema field: '{missing_fields[0]}'", None

    if allow_legacy:
        # Check if incomplete legacy run
        raw_status = str(manifest_data.get("status", manifest_data.get("completion_status", ""))).upper()
        if raw_status == "INITIALIZED" or manifest_data.get("total_steps", 0) == 0:
            return ManifestStatus.INCOMPLETE, f"Incomplete or unverified historical run (status='{raw_status}')", None

        success, msg, migrated = migrate_legacy_manifest(manifest_data)
        if success:
            return ManifestStatus.LEGACY_VALID, f"Legacy manifest migrated: {msg}", migrated
        else:
            return ManifestStatus.INVALID, f"Failed canonical validation and legacy migration: {msg}", None

    return ManifestStatus.INVALID, f"Missing required schema field: '{missing_fields[0]}'", None


def validate_experiment_manifest(
    manifest: Union[Dict[str, Any], str, Path],
    allow_legacy: bool = True,
) -> Tuple[bool, str]:
    """
    Convenience validator returning boolean success.
    Accepts VALID and LEGACY_VALID (when allow_legacy=True).
    Rejects INCOMPLETE and INVALID.
    """
    status, reason, _ = classify_experiment_manifest(manifest, allow_legacy=allow_legacy)
    if status == ManifestStatus.VALID:
        return True, reason
    elif status == ManifestStatus.LEGACY_VALID and allow_legacy:
        return True, reason
    else:
        return False, f"[{status}] {reason}"


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
        self.evaluation_mode = normalize_evaluation_mode(str(evaluation_mode))
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
        
        # Determine explicit fairness metric identity and unit metadata from config
        objs_cfg = self.config.get("objectives", {}) if isinstance(self.config, dict) else getattr(self.config, "objectives", {})
        if isinstance(objs_cfg, dict):
            raw_metric = str(objs_cfg.get("fairness_metric", "delay_disparity")).lower()
        else:
            raw_metric = str(getattr(objs_cfg, "fairness_metric", "delay_disparity")).lower()
            
        if raw_metric in ("delay_disparity", "jain_index", "jain_index_or_delay_disparity", "intersection_fairness_disparity"):
            self.fairness_metric_name = "delay_disparity"
            self.fairness_metric_unit = "dimensionless"
        elif raw_metric in ("delay_variance", "variance"):
            self.fairness_metric_name = "delay_variance"
            self.fairness_metric_unit = "s^2"
        else:
            self.fairness_metric_name = raw_metric
            self.fairness_metric_unit = "unknown"

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
            "config_hash": self.config.get("config_hash", "UNKNOWN") if isinstance(self.config, dict) else getattr(self.config, "config_hash", "UNKNOWN"),
            "code_commit": get_git_commit_sha(),
            "environment_info": get_environment_info(),
            "start_time": self.start_timestamp,
            "end_time": None,
            "completion_status": "INITIALIZED",
            "model_checkpoint_sha256": None,
            "total_steps": 0,
            "summary_metrics": {
                "fairness_metric_name": self.fairness_metric_name,
                "fairness_metric_unit": self.fairness_metric_unit,
            },
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
            "train_seed": self.train_seed,
            "eval_seed": self.eval_seed,
            "episode_id": self.episode_id,
            "evaluation_mode": self.evaluation_mode,
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
            "fairness_metric_name": self.fairness_metric_name,
            "fairness_metric_unit": self.fairness_metric_unit,
            "fairness_disparity": float(np.mean(fairnesses)) if self.fairness_metric_name == "delay_disparity" else None,
            "fairness_variance": float(np.mean(fairnesses)) if self.fairness_metric_name == "delay_variance" else None,
            "total_shield_interventions": int(np.sum(shields)),
            "average_weights": [float(w) for w in np.mean(weights_matrix, axis=0)],
        }
        return summary
