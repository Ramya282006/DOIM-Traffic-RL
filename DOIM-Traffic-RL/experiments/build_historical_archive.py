"""
DOIM-Traffic-RL: Historical Artifact Archive Manifest Builder (Phase A)
Indexes every historical experiment run, checkpoint, CSV, and summary in results/
with SHA-256 integrity checksums, run metadata classification, and duplicate detection.
"""

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file in binary mode."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def scan_historical_archive(results_root: Path) -> Dict[str, Any]:
    """
    Scans the results directory and produces a comprehensive provenance manifest.
    """
    manifest_records: List[Dict[str, Any]] = []
    checkpoint_hashes: Dict[str, List[str]] = {}
    trajectory_hashes: Dict[str, List[str]] = {}
    missing_artifacts: List[Dict[str, str]] = []
    
    runs_dir = results_root / "runs"
    
    # 1. Scan results/runs/
    if runs_dir.exists():
        for run_path in sorted(runs_dir.iterdir()):
            if not run_path.is_dir():
                continue
                
            run_id = run_path.name
            manifest_file = run_path / "manifest.json"
            summary_file = run_path / "summary.json"
            metrics_file = run_path / "metrics_step.csv"
            config_file = run_path / "config_snapshot.yaml"
            checkpoints_dir = run_path / "checkpoints"
            
            # Read manifest metadata if available
            manifest_data = {}
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        manifest_data = json.load(f)
                except Exception as e:
                    manifest_data = {"error": str(e)}
            else:
                missing_artifacts.append({"run_id": run_id, "missing": "manifest.json"})
                
            # Classify run
            total_steps = manifest_data.get("training_timesteps") or manifest_data.get("total_steps") or 0
            algo = manifest_data.get("algorithm") or run_id.split("_")[0]
            scenario = manifest_data.get("scenario") or "unknown"
            train_seed = manifest_data.get("train_seed") or manifest_data.get("seed")
            eval_seed = manifest_data.get("evaluation_seed") or manifest_data.get("eval_seed")
            eval_mode = manifest_data.get("evaluation_mode") or "train_rollout"
            
            classification = "CONFIRMATORY_CANDIDATE" if total_steps >= 5000 else "DIAGNOSTIC_LEGACY_128STEP"
            
            # Index all files in this run
            run_files = []
            for item in sorted(run_path.rglob("*")):
                if item.is_file():
                    rel_p = str(item.relative_to(results_root)).replace("\\", "/")
                    sha = compute_sha256(item)
                    size = item.stat().st_size
                    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(item.stat().st_mtime))
                    
                    file_info = {
                        "relative_path": rel_p,
                        "file_size_bytes": size,
                        "sha256": sha,
                        "last_modified_utc": mtime,
                        "run_id": run_id,
                        "algorithm": algo,
                        "scenario": scenario,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "step_count": total_steps,
                        "evaluation_mode": eval_mode,
                        "classification": classification,
                    }
                    manifest_records.append(file_info)
                    run_files.append((rel_p, sha))
                    
                    # Track checkpoints and trajectories
                    if item.suffix in (".pt", ".pth"):
                        checkpoint_hashes.setdefault(sha, []).append(rel_p)
                    elif item.name == "metrics_step.csv":
                        trajectory_hashes.setdefault(sha, []).append(rel_p)

    # 2. Scan other top-level folders in results (evaluation, figures, tables, etc.)
    for sub in ["evaluation", "figures", "tables", "objective_interactions", "ppo", "ppo_fixed"]:
        target_dir = results_root / sub
        if target_dir.exists():
            for item in sorted(target_dir.rglob("*")):
                if item.is_file():
                    rel_p = str(item.relative_to(results_root)).replace("\\", "/")
                    sha = compute_sha256(item)
                    size = item.stat().st_size
                    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(item.stat().st_mtime))
                    
                    manifest_records.append({
                        "relative_path": rel_p,
                        "file_size_bytes": size,
                        "sha256": sha,
                        "last_modified_utc": mtime,
                        "run_id": "GLOBAL_AGGREGATE",
                        "algorithm": "AGGREGATE",
                        "scenario": "AGGREGATE",
                        "train_seed": None,
                        "eval_seed": None,
                        "step_count": None,
                        "evaluation_mode": "AGGREGATE_REPORT",
                        "classification": "DERIVED_REPORT_OR_FIGURE",
                    })

    # Detect duplicate checkpoints (same weights across different seeds/runs)
    duplicate_checkpoints = {
        sha: paths for sha, paths in checkpoint_hashes.items() if len(paths) > 1
    }
    
    # Detect identical trajectories (pseudoreplication)
    duplicate_trajectories = {
        sha: paths for sha, paths in trajectory_hashes.items() if len(paths) > 1
    }

    archive_manifest = {
        "manifest_version": "1.0.0",
        "archive_generation_timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "total_files_indexed": len(manifest_records),
        "total_runs_indexed": len([r for r in manifest_records if r["run_id"] != "GLOBAL_AGGREGATE"]),
        "duplicate_checkpoints_detected": duplicate_checkpoints,
        "duplicate_trajectories_detected": duplicate_trajectories,
        "missing_artifacts_detected": missing_artifacts,
        "records": manifest_records,
    }
    
    return archive_manifest


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    
    manifest = scan_historical_archive(results_dir)
    
    out_file = results_dir / "historical_archive_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Archive manifest generated successfully with {manifest['total_files_indexed']} files indexed.")
    print(f"Saved to: {out_file}")
    if manifest["duplicate_checkpoints_detected"]:
        print(f"WARNING: {len(manifest['duplicate_checkpoints_detected'])} duplicate checkpoint sets detected!")
    if manifest["duplicate_trajectories_detected"]:
        print(f"WARNING: {len(manifest['duplicate_trajectories_detected'])} duplicate trajectory sets detected!")
