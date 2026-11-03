"""
DOIM-PPO Phase 34.1: Forensic Repository & Experiment Inventory Auditor
Audits every experiment directory across the repository with zero assumptions.
Reconstructs truth directly from filesystem artifacts.
"""

import os
import sys
import json
import csv
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GIT_ROOT = Path("D:/DOIM-Traffic-RL").resolve()
PKG_ROOT = GIT_ROOT / "DOIM-Traffic-RL"
RUNS_DIRS = [
    PKG_ROOT / "results/runs",
    GIT_ROOT / "results/runs",
]


def hash_file_sha256(filepath: Path) -> Optional[str]:
    if not filepath.exists() or not filepath.is_file():
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_metrics_csv(csv_path: Path) -> Dict[str, Any]:
    if not csv_path.exists():
        return {"exists": False, "rows": 0, "headers": [], "valid": False}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            row_count = sum(1 for _ in reader)
        expected_cols = {"step", "reward", "raw_waiting", "raw_queue", "raw_co2", "raw_throughput", "raw_fairness"}
        is_valid = expected_cols.issubset(set(headers)) and row_count > 0
        return {
            "exists": True,
            "rows": row_count,
            "headers": headers,
            "valid": is_valid,
        }
    except Exception as e:
        return {"exists": True, "rows": 0, "headers": [], "valid": False, "error": str(e)}


def inspect_run_directory(run_dir: Path) -> Dict[str, Any]:
    manifest_file = run_dir / "manifest.json"
    summary_file = run_dir / "summary.json"
    metrics_file = run_dir / "metrics_step.csv"
    ckpt_dir = run_dir / "checkpoints"

    record = {
        "run_id": run_dir.name,
        "run_path": str(run_dir),
        "parent_dir": run_dir.parent.name,
        "algorithm": None,
        "scenario": None,
        "train_seed": None,
        "eval_seed": None,
        "evaluation_mode": None,
        "source_sha": None,
        "config_hash": None,
        "sumo_version": None,
        "total_steps": 0,
        "completion_status": "MISSING_MANIFEST",
        "has_summary": summary_file.exists(),
        "has_metrics": metrics_file.exists(),
        "metrics_rows": 0,
        "metrics_valid": False,
        "checkpoints": [],
        "provenance_status": "UNVERIFIED",
        "notes": [],
    }

    # Inspect manifest
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                man = json.load(f)
            record["algorithm"] = man.get("algorithm")
            record["scenario"] = man.get("scenario")
            record["train_seed"] = man.get("train_seed")
            record["eval_seed"] = man.get("evaluation_seed") or man.get("eval_seed")
            record["evaluation_mode"] = man.get("evaluation_mode")
            record["source_sha"] = man.get("code_commit")
            record["config_hash"] = man.get("config_hash")
            env_info = man.get("environment_info", {})
            record["sumo_version"] = env_info.get("sumo_version")
            record["total_steps"] = man.get("total_steps", 0)
            record["completion_status"] = man.get("completion_status") or man.get("status", "UNKNOWN")
        except Exception as e:
            record["notes"].append(f"Manifest parse error: {e}")
            record["completion_status"] = "CORRUPT_MANIFEST"

    # Inspect summary
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                summ = json.load(f)
            if not record["algorithm"]:
                record["algorithm"] = summ.get("algorithm")
            if not record["scenario"]:
                record["scenario"] = summ.get("scenario")
            if record["train_seed"] is None:
                record["train_seed"] = summ.get("train_seed") or summ.get("seed")
            if record["eval_seed"] is None:
                record["eval_seed"] = summ.get("eval_seed") or summ.get("seed")
            record["summary_total_steps"] = summ.get("total_steps", 0)
            record["mean_waiting"] = summ.get("mean_waiting_time", summ.get("average_waiting"))
            record["mean_queue"] = summ.get("mean_queue", summ.get("average_queue"))
            record["throughput"] = summ.get("total_throughput")
            record["shield_interventions"] = summ.get("total_shield_interventions")
        except Exception as e:
            record["notes"].append(f"Summary parse error: {e}")

    # Inspect metrics CSV
    csv_info = audit_metrics_csv(metrics_file)
    record["metrics_rows"] = csv_info["rows"]
    record["metrics_valid"] = csv_info["valid"]

    # Inspect checkpoints
    if ckpt_dir.exists():
        for pt in ckpt_dir.glob("*.pt"):
            record["checkpoints"].append({
                "name": pt.name,
                "size_bytes": pt.stat().st_size,
                "sha256": hash_file_sha256(pt),
            })

    # Provenance Classification
    steps_match = (record["total_steps"] == record["metrics_rows"]) or (record["total_steps"] == record["metrics_rows"] + 1)
    status_completed = record["completion_status"] == "COMPLETED"

    if status_completed and record["metrics_valid"] and record["has_summary"] and steps_match:
        record["provenance_status"] = "VERIFIED"
    elif status_completed and not steps_match:
        record["provenance_status"] = "CONFLICTED"
        record["notes"].append(f"Step count mismatch: manifest={record['total_steps']}, csv={record['metrics_rows']}")
    elif not status_completed:
        record["provenance_status"] = "INCOMPLETE"
    else:
        record["provenance_status"] = "UNVERIFIED"

    return record


def run_forensic_inventory() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    seen_dirs = set()
    inventory = []

    for r_dir in RUNS_DIRS:
        if not r_dir.exists():
            continue
        for child in sorted(r_dir.iterdir()):
            if child.is_dir() and child.name not in seen_dirs:
                seen_dirs.add(child.name)
                rec = inspect_run_directory(child)
                inventory.append(rec)

    # Detect duplicates
    run_id_counts = {}
    for rec in inventory:
        rid = rec["run_id"]
        run_id_counts[rid] = run_id_counts.get(rid, 0) + 1

    summary = {
        "total_directories_scanned": len(inventory),
        "provenance_breakdown": {
            "VERIFIED": sum(1 for r in inventory if r["provenance_status"] == "VERIFIED"),
            "CONFLICTED": sum(1 for r in inventory if r["provenance_status"] == "CONFLICTED"),
            "INCOMPLETE": sum(1 for r in inventory if r["provenance_status"] == "INCOMPLETE"),
            "UNVERIFIED": sum(1 for r in inventory if r["provenance_status"] == "UNVERIFIED"),
        },
        "evaluation_runs_ge_1800": sum(1 for r in inventory if r["metrics_rows"] >= 1800 and r["provenance_status"] == "VERIFIED"),
        "training_runs_ge_5000": sum(1 for r in inventory if r["metrics_rows"] >= 5000 and r["provenance_status"] == "VERIFIED"),
        "smoke_runs_le_500": sum(1 for r in inventory if r["metrics_rows"] <= 500),
        "algorithms_detected": sorted(list(set(r["algorithm"] for r in inventory if r["algorithm"]))),
        "scenarios_detected": sorted(list(set(r["scenario"] for r in inventory if r["scenario"]))),
        "total_checkpoints_found": sum(len(r["checkpoints"]) for r in inventory),
    }

    return inventory, summary


if __name__ == "__main__":
    inv, summ = run_forensic_inventory()
    print("Forensic Inventory Summary:")
    print(json.dumps(summ, indent=2))

    out_dir = GIT_ROOT / "results/phase34"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export CSV
    csv_file = out_dir / "forensic_run_inventory.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_id", "algorithm", "scenario", "train_seed", "eval_seed",
            "evaluation_mode", "total_steps", "metrics_rows", "completion_status",
            "provenance_status", "source_sha", "config_hash", "sumo_version",
            "has_summary", "has_metrics", "num_checkpoints", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in inv:
            writer.writerow({
                "run_id": r["run_id"],
                "algorithm": r["algorithm"],
                "scenario": r["scenario"],
                "train_seed": r["train_seed"],
                "eval_seed": r["eval_seed"],
                "evaluation_mode": r["evaluation_mode"],
                "total_steps": r["total_steps"],
                "metrics_rows": r["metrics_rows"],
                "completion_status": r["completion_status"],
                "provenance_status": r["provenance_status"],
                "source_sha": r["source_sha"],
                "config_hash": r["config_hash"],
                "sumo_version": r["sumo_version"],
                "has_summary": r["has_summary"],
                "has_metrics": r["has_metrics"],
                "num_checkpoints": len(r["checkpoints"]),
                "notes": "; ".join(r["notes"]),
            })

    # Export JSON
    json_file = out_dir / "forensic_inventory_summary.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)

    print(f"\nInventory exported to: {csv_file}")
    print(f"Summary exported to: {json_file}")
