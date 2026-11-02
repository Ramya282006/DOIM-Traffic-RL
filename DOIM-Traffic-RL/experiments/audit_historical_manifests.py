"""
DOIM-Traffic-RL: Historical Manifest Revalidation and Migration Audit
Audits all historical experiment records without modifying original files on disk.
Rigorously separates historical execution completion from reproducibility eligibility.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.experiment import classify_experiment_manifest, ManifestStatus

def run_historical_audit(runs_root: Path = Path("results/runs"), output_path: Path = Path("results/manifest_migration_audit.json")):
    audit = {
        "total_historical_records": 0,
        "completed_on_disk": 0,
        "incomplete_on_disk": 0,
        "reproducibility_eligible": 0,
        "reproducibility_ineligible": 0,
        "records": []
    }

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        audit["total_historical_records"] += 1
        m_path = run_dir / "manifest.json"
        s_path = run_dir / "summary.json"
        c_path = run_dir / "metrics_step.csv"

        record_audit = {
            "run_dir": run_dir.name,
            "manifest_exists": m_path.exists(),
            "summary_exists": s_path.exists(),
            "metrics_csv_exists": c_path.exists(),
            "classification_status": None,
            "classification_reason": None,
            "historical_completion": "INCOMPLETE",
            "reproducibility_eligibility": "INELIGIBLE",
            "recovered_fields": {},
            "inferred_fields": {},
            "unavailable_fields": [],
            "ineligibility_reasons": [],
        }

        if not m_path.exists():
            record_audit["classification_status"] = "MISSING_MANIFEST"
            record_audit["classification_reason"] = "manifest.json file does not exist"
            record_audit["ineligibility_reasons"].append("Missing manifest.json")
            audit["incomplete_on_disk"] += 1
            audit["reproducibility_ineligible"] += 1
            audit["records"].append(record_audit)
            continue

        try:
            raw_manifest = json.loads(m_path.read_text(encoding="utf-8"))
        except Exception as e:
            record_audit["classification_status"] = "CORRUPTED_JSON"
            record_audit["classification_reason"] = str(e)
            record_audit["ineligibility_reasons"].append(f"Corrupted JSON: {e}")
            audit["incomplete_on_disk"] += 1
            audit["reproducibility_ineligible"] += 1
            audit["records"].append(record_audit)
            continue

        status, reason, migrated = classify_experiment_manifest(raw_manifest, allow_legacy=True)
        record_audit["classification_status"] = status
        record_audit["classification_reason"] = reason

        if status in (ManifestStatus.VALID, ManifestStatus.LEGACY_VALID):
            record_audit["historical_completion"] = "COMPLETED"
            audit["completed_on_disk"] += 1
        else:
            audit["incomplete_on_disk"] += 1

        if migrated:
            for field in ["algorithm", "scenario", "train_seed", "evaluation_seed", "training_timesteps", "code_commit", "config_hash"]:
                val = migrated.get(field)
                if val is not None and val != "UNKNOWN":
                    record_audit["recovered_fields"][field] = val
                else:
                    record_audit["unavailable_fields"].append(field)
                    record_audit["ineligibility_reasons"].append(f"Field '{field}' is missing or UNKNOWN")

            eval_mode = migrated.get("evaluation_mode")
            if eval_mode and eval_mode != "UNKNOWN":
                record_audit["recovered_fields"]["evaluation_mode"] = eval_mode
            else:
                record_audit["unavailable_fields"].append("evaluation_mode")
                record_audit["ineligibility_reasons"].append("Evaluation mode is UNKNOWN (not explicitly declared in historical run)")

            # Fairness metric
            sm = raw_manifest.get("summary_metrics", {})
            if isinstance(sm, dict) and sm.get("fairness_metric_name"):
                record_audit["recovered_fields"]["fairness_metric_name"] = sm.get("fairness_metric_name")
            else:
                record_audit["unavailable_fields"].append("fairness_metric_name")
                record_audit["ineligibility_reasons"].append("Explicit fairness metric identity unavailable in historical manifest")

            if migrated.get("reproducibility_eligible", False):
                record_audit["reproducibility_eligibility"] = "ELIGIBLE"
                audit["reproducibility_eligible"] += 1
            else:
                audit["reproducibility_ineligible"] += 1
        else:
            audit["reproducibility_ineligible"] += 1

        audit["records"].append(record_audit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("=" * 80)
    print("HISTORICAL MANIFEST AUDIT COMPLETE")
    print(f"Total Historical Records: {audit['total_historical_records']}")
    print(f"Completed on Disk: {audit['completed_on_disk']}")
    print(f"Incomplete on Disk: {audit['incomplete_on_disk']}")
    print(f"Reproducibility Eligible: {audit['reproducibility_eligible']}")
    print(f"Reproducibility Ineligible: {audit['reproducibility_ineligible']}")
    print(f"Audit report saved to: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_historical_audit()
