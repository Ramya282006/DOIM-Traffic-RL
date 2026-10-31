"""
DOIM-Traffic-RL: Unit tests for Traffic Safety Shield & Transition Semantics
Tests actual SUMO TLS program parsing, multi-stage clearance intervals,
single-phase junction handling, transition clearance checking, and conflict verification.
"""

from pathlib import Path
import numpy as np
import pytest

from environment.safety_shield import (
    TrafficSafetyShield,
    SignalPhaseState,
    extract_tls_programs_from_net,
    validate_signal_state_string,
    verify_phase_conflicts,
    verify_transition_clearance,
)


def test_signal_state_string_validation():
    assert validate_signal_state_string("GGggrrrrGGggrrrr") is True
    assert validate_signal_state_string("yyyyrrrryyyyrrrr") is True
    assert validate_signal_state_string("rrrrrrrrrrrrrrrr") is True
    # Invalid characters
    assert validate_signal_state_string("GGggrrrrX") is False
    assert validate_signal_state_string("") is False
    assert validate_signal_state_string(123) is False


def test_movement_conflict_verification():
    # Example junction where movement 0 (North-South through) conflicts with movement 4 (East-West through)
    conflict_pairs = [(0, 4)]
    
    # Valid state: Movement 0 has green 'G', movement 4 has red 'r'
    valid_state = "GGggrrrr"
    is_valid, violations = verify_phase_conflicts(valid_state, conflict_pairs)
    assert is_valid is True
    assert len(violations) == 0

    # Conflicting state: Both movement 0 and movement 4 have 'G'
    conflict_state = "GGggGrrr"
    is_valid, violations = verify_phase_conflicts(conflict_state, conflict_pairs)
    assert is_valid is False
    assert len(violations) == 1
    assert "Conflict violation" in violations[0]


def test_transition_clearance_verification():
    # Normal transition: Phase 0 (green) -> Phase 1 (yellow) -> Phase 2 (red)
    from_green = "GGggrrrr"
    clearance_yellow = ["yyyyrrrr"]
    to_red = "rrrrGGgg"
    
    is_valid, violations = verify_transition_clearance(from_green, to_red, clearance_yellow)
    assert is_valid is True
    assert len(violations) == 0

    # Illegal abrupt transition: Turns red without yellow clearance
    illegal_clearance = ["rrrrrrrr"] # all-red without yellow
    is_valid, violations = verify_transition_clearance(from_green, to_red, illegal_clearance)
    assert is_valid is False
    assert len(violations) > 0
    assert "Clearance violation on movement" in violations[0]


def test_extract_programs_from_real_sumo_network():
    grid_net = Path("environment/network/research_3x3/grid.net.xml")
    if not grid_net.exists():
        pytest.skip(f"Network file not found: {grid_net}")
        
    programs = extract_tls_programs_from_net(grid_net)
    assert len(programs) >= 9
    assert "A0" in programs
    assert "B1" in programs
    
    # A0 is a perimeter 1-phase junction
    assert len(programs["A0"]) == 1
    assert programs["A0"][0]["state"] == "GGGG"
    
    # B1 is a 4-phase interior junction with 20 controlled signals
    assert len(programs["B1"]) == 4
    assert len(programs["B1"][0]["state"]) == 20


def test_single_phase_junction_stability():
    # Junction with only 1 green phase (perimeter boundary nodes)
    prog_1phase = [{"duration": 90.0, "state": "GGGG"}]
    shield = TrafficSafetyShield(
        controllable_tls_ids=["A0"],
        tls_programs={"A0": prog_1phase},
        min_green=10,
        max_green=60,
    )
    
    # Attempting to request a phase switch should be safely blocked as an intervention
    raw_actions = np.array([1])
    current_phases = [0]
    safe_actions, target_phases, interventions = shield.filter_action(raw_actions, current_phases)
    
    assert safe_actions[0] == 0
    assert target_phases[0] is None
    assert interventions == 1  # Shield intervened to block illegal switch on 1-phase node


def test_multi_stage_clearance_transition():
    # Program with multi-stage clearance: Green(0) -> Yellow(1) -> All-Red(2) -> Green(3)
    prog = [
        {"duration": 30.0, "state": "GGGG"}, # phase 0: Green
        {"duration": 2.0, "state": "yyyy"},  # phase 1: Yellow
        {"duration": 1.0, "state": "rrrr"},  # phase 2: All-Red
        {"duration": 30.0, "state": "GGGG"}, # phase 3: Green
        {"duration": 2.0, "state": "yyyy"},  # phase 4: Yellow
        {"duration": 1.0, "state": "rrrr"},  # phase 5: All-Red
    ]
    
    shield = TrafficSafetyShield(
        controllable_tls_ids=["TLS_MS"],
        tls_programs={"TLS_MS": prog},
        min_green=5,
        max_green=30,
        yellow_duration=2,
    )
    
    # Step past min_green
    for _ in range(6):
        shield.filter_action(np.array([0]), [0])
        
    # Request switch: should transition to stage 1 (Yellow, phase 1)
    safe_act, target, interv = shield.filter_action(np.array([1]), [0])
    assert safe_act[0] == 1
    assert target[0] == 1  # Phase 1 (Yellow)
    
    # Step through yellow duration (2s)
    shield.filter_action(np.array([0]), [1])
    safe_act, target, _ = shield.filter_action(np.array([0]), [1])
    # Should transition to stage 2 (All-Red, phase 2)
    assert target[0] == 2
    
    # Step through all-red duration (1s)
    safe_act, target, _ = shield.filter_action(np.array([0]), [2])
    # Should enter next green phase (Phase 3)
    assert target[0] == 3
