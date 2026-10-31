"""
Tests for Traffic Signal Safety Verifier and Conflict Matrix Extraction.
"""

from pathlib import Path
import pytest
from environment.safety_verifier import SignalSafetyVerifier, extract_junction_conflicts_from_net


def test_conflict_matrix_extraction_research_grid():
    """
    Verifies that the network conflict extractor successfully parses foes bitmasks
    for all 5 controllable intersections from grid.net.xml.
    """
    net_path = Path("environment/network/research_3x3/grid.net.xml")
    assert net_path.exists(), "grid.net.xml network file not found"

    conflicts = extract_junction_conflicts_from_net(net_path)
    assert set(conflicts.keys()) == {"A1", "B0", "B1", "B2", "C1"}

    # Central 4-way intersection B1 has 20 links and 128 conflicting movement pairs
    assert len(conflicts["B1"]) > 50
    # 3-way border intersections have 12 links and conflicting pairs
    assert len(conflicts["A1"]) > 10
    assert len(conflicts["B0"]) > 10


def test_no_conflicts_in_legal_phases():
    """
    Verifies that all standard legal phases defined in SUMO contain zero
    simultaneous conflicting priority greens ('G').
    """
    verifier = SignalSafetyVerifier()

    # Known legal phase strings from research_3x3
    legal_phase_samples = {
        "A1": ["GGggrrrrGGGg", "yyyyrrrrGyyy", "rrrrGGGgGrrr", "rrrryyyyGrrr"],
        "B0": ["rrrrGGGgGGgg", "rrrrGyyyyyyy", "GGGgGrrrrrrr", "yyyyGrrrrrrr"],
        "B1": ["GGGggrrrrrGGGggrrrrr", "yyyyyrrrrryyyyyrrrrr", "rrrrrGGGggrrrrrGGGgg", "rrrrryyyyyrrrrryyyyy"],
        "B2": ["GGggrrrrGGGg", "yyyyrrrrGyyy", "rrrrGGGgGrrr", "rrrryyyyGrrr"],
        "C1": ["GGGgGGggrrrr", "Gyyyyyyyrrrr", "GrrrrrrrGGGg", "Grrrrrrryyyy"],
    }

    for tid, phases in legal_phase_samples.items():
        for phase_str in phases:
            viols = verifier.verify_signal_string(tid, phase_str)
            assert len(viols) == 0, f"False positive conflict detected in legal phase {phase_str} on TLS {tid}"


def test_adversarial_conflicting_greens_detected():
    """
    Verifies that an adversarial state displaying simultaneous green on conflicting links
    is immediately caught and flagged.
    """
    verifier = SignalSafetyVerifier()

    # For B1, link 1 (North straight) and link 6 (East straight) conflict.
    # Craft an illegal state where both have 'G'
    illegal_b1_state = list("rrrrrrrrrrrrrrrrrrrr")
    illegal_b1_state[1] = "G"
    illegal_b1_state[6] = "G"
    illegal_str = "".join(illegal_b1_state)

    viols = verifier.verify_signal_string("B1", illegal_str, sim_time=100.0)
    assert len(viols) > 0, "Failed to detect simultaneous conflicting major greens!"
    assert any(v["type"] == "CONFLICTING_MAJOR_GREENS" for v in viols)


def test_rapid_fluttering_min_green_violation():
    """
    Verifies that rapid phase switching under 10s min green triggers MIN_GREEN_VIOLATION.
    """
    verifier = SignalSafetyVerifier(min_green=10)

    # Initial registration: phase 0 at t=0
    verifier.record_step(sim_time=0, current_phases={"B1": 0})
    # Advance to t=4 (4s in phase 0)
    for t in range(1, 5):
        verifier.record_step(sim_time=t, current_phases={"B1": 0})

    # Premature switch to phase 1 after only 5 seconds (less than 10s min green)
    step_viols = verifier.record_step(sim_time=5, current_phases={"B1": 1})
    assert len(step_viols) > 0
    assert any(v["type"] == "MIN_GREEN_VIOLATION" for v in step_viols)


def test_direct_green_jump_violation():
    """
    Verifies that jumping directly from green phase 0 to green phase 2 without yellow clearance triggers DIRECT_GREEN_JUMP_NO_YELLOW.
    """
    verifier = SignalSafetyVerifier(min_green=10)

    # Stay in phase 0 for 15s (legal min green satisfied)
    for t in range(15):
        verifier.record_step(sim_time=t, current_phases={"B1": 0})

    # Jump directly from phase 0 (green) to phase 2 (green) skipping phase 1 (yellow)
    step_viols = verifier.record_step(sim_time=15, current_phases={"B1": 2})
    assert any(v["type"] == "DIRECT_GREEN_JUMP_NO_YELLOW" for v in step_viols)
    assert any(v["type"] == "ILLEGAL_PHASE_SEQUENCE" for v in step_viols)


def test_trace_verification_summary():
    """
    Verifies that verify_trace generates a complete summary report.
    """
    verifier = SignalSafetyVerifier()
    trace = [
        {"sim_time": 0, "phases": {"B1": 0}},
        {"sim_time": 1, "phases": {"B1": 0}},
        {"sim_time": 2, "phases": {"B1": 1}}, # illegal min green switch
    ]
    summary = verifier.verify_trace(trace)
    assert summary["total_violations"] > 0
    assert summary["min_green_violations"] == 1
    assert summary["verified_safe"] is False
