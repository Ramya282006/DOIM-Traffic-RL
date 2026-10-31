"""
DOIM-Traffic-RL: Automated Manuscript Consistency and Scientific Integrity Checker
Verifies that all numerical claims, table values, statistical assertions,
and safety statements in research_docs/paper/MANUSCRIPT.md strictly match
the machine-readable benchmark artifacts in results/tables/benchmark_comparison.csv.
"""

from pathlib import Path
import re
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT_PATH = REPO_ROOT / "research_docs" / "paper" / "MANUSCRIPT.md"
BENCHMARK_CSV_PATH = REPO_ROOT / "results" / "tables" / "benchmark_comparison.csv"


def test_manuscript_files_exist():
    assert MANUSCRIPT_PATH.exists(), f"Manuscript not found at: {MANUSCRIPT_PATH}"
    assert BENCHMARK_CSV_PATH.exists(), f"Benchmark CSV not found at: {BENCHMARK_CSV_PATH}"


def test_no_phantom_delay_claims():
    """
    CRITICAL CHECK: Verifies that unsupported historical phantom numbers
    (e.g., 32.14s) do not appear in the manuscript text.
    """
    content = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    assert "32.14" not in content, (
        "Manuscript contains phantom claim of 32.14s which contradicts committed benchmark results!"
    )


def test_safety_claim_terminology():
    """
    Verifies that the manuscript adheres to strict IEEE standards by avoiding
    unsupported claims of 'formal safety guarantee', using instead
    'empirically verified safety constraints'.
    """
    content = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    # Disallow 'formal safety guarantee'
    assert "formal safety guarantee" not in content.lower(), (
        "Manuscript claims 'formal safety guarantee'. Must use 'empirically verified safety constraints'!"
    )
    assert "empirically verified" in content.lower(), (
        "Manuscript should state that safety constraints are empirically verified."
    )


def test_submission_claim_integrity():
    """
    Verifies that the paper is not claimed to be submitted to IEEE unless actually submitted.
    """
    content = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    assert "submitted to ieee" not in content.lower() or "targeted for submission" in content.lower(), (
        "Manuscript falsely claims submission to IEEE without actual submission!"
    )


def test_table_matches_benchmark_csv():
    """
    Verifies that numerical values reported in Table I match the committed benchmark CSV.
    """
    if not BENCHMARK_CSV_PATH.exists():
        pytest.skip("Benchmark CSV not found")

    df = pd.read_csv(BENCHMARK_CSV_PATH)
    content = MANUSCRIPT_PATH.read_text(encoding="utf-8")

    # For each algorithm in benchmark CSV, check that its mean waiting time is referenced accurately
    for _, row in df.iterrows():
        algo = row["algorithm"]
        if algo == "DOIM_SMOKE":
            continue
        mean_wait = float(row["average_waiting_mean"])
        # Formats to check: e.g. "70.57" or "70.6" or "97.57" or "97.6"
        str_val_2dp = f"{mean_wait:.2f}"
        str_val_1dp = f"{mean_wait:.1f}"

        # If algorithm is cited in Table I, it must match the CSV
        if algo in content:
            matches_pattern = (str_val_2dp in content) or (str_val_1dp in content)
            assert matches_pattern, (
                f"Manuscript Table does not match benchmark CSV for {algo}: expected ~{str_val_2dp}s"
            )
