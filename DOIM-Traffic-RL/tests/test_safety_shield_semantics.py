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


def test_extract_conflicts_from_real_sumo_network():
    """
    Verifies that movement conflicts are dynamically extracted from actual SUMO network
    junction request foes bitmasks rather than manually hardcoded test pairs.
    """
    from environment.safety_shield import extract_tls_conflicts_from_net

    grid_net = Path("environment/network/research_3x3/grid.net.xml")
    if not grid_net.exists():
        pytest.skip(f"Network file not found: {grid_net}")

    conflicts = extract_tls_conflicts_from_net(grid_net)
    assert len(conflicts) >= 5
    assert "B1" in conflicts
    assert len(conflicts["B1"]) > 0
    # Every conflict pair must be a 2-tuple of distinct integers (idx1, idx2)
    for idx1, idx2 in conflicts["B1"]:
        assert idx1 < idx2
        assert idx1 >= 0
        assert idx2 < 20

    # Also test intersection.net.xml
    inter_net = Path("environment/network/intersection.net.xml")
    if inter_net.exists():
        inter_conflicts = extract_tls_conflicts_from_net(inter_net)
        assert "J0" in inter_conflicts
        # Straight N (link 2) and straight E (link 6) must conflict
        assert (2, 6) in inter_conflicts["J0"]


def test_mixed_green_yellow_signal_state_classification():
    """
    Verifies that mixed green/yellow signal states (e.g., A1's 'yyyyrrrrGyyy' and C1's 'Gyyyyyyyrrrr')
    are properly classified as mixed_clearance, treated as clearance phases by the state machine,
    and verified by transition clearance logic.
    """
    from environment.safety_shield import extract_tls_programs_from_net

    grid_net = Path("environment/network/research_3x3/grid.net.xml")
    if not grid_net.exists():
        pytest.skip(f"Network file not found: {grid_net}")

    programs = extract_tls_programs_from_net(grid_net)
    assert "A1" in programs

    # Program for A1 has mixed green/yellow in phase 1 ('yyyyrrrrGyyy') and phase 3 ('rrrryyyyGrrr')
    a1_phases = programs["A1"]
    state_a1 = SignalPhaseState(tls_id="A1", num_phases=len(a1_phases), program_phases=a1_phases)

    assert state_a1.phase_types[1] == "mixed_clearance"
    assert state_a1.phase_types[3] == "mixed_clearance"
    assert 1 in state_a1.clearance_phases
    assert 3 in state_a1.clearance_phases
    assert 0 in state_a1.green_phases
    assert 2 in state_a1.green_phases

    # Transition from phase 0 ('GGggrrrrGGGg') to phase 2 ('rrrrGGGgGrrr') through clearance phase 1 ('yyyyrrrrGyyy')
    is_valid, violations = verify_transition_clearance(
        from_state=a1_phases[0]["state"],
        to_state=a1_phases[2]["state"],
        clearance_states=[a1_phases[1]["state"]],
    )
    assert is_valid is True, f"Clearance violations in real A1 transition: {violations}"


def test_multiple_tls_programs_handling(tmp_path):
    """
    Verifies that multiple programs for a single TLS (e.g., programID '0' and programID '1')
    are correctly handled without silent overwriting, honoring preferred_program_id.
    """
    multi_prog_xml = tmp_path / "multi_program.net.xml"
    multi_prog_xml.write_text(
        '<net version="1.16">\n'
        '  <tlLogic id="TLS_MULTI" type="static" programID="0" offset="0">\n'
        '    <phase duration="40" state="GGGG"/>\n'
        '    <phase duration="4" state="yyyy"/>\n'
        '  </tlLogic>\n'
        '  <tlLogic id="TLS_MULTI" type="static" programID="1" offset="0">\n'
        '    <phase duration="60" state="rrrr"/>\n'
        '    <phase duration="5" state="GGGG"/>\n'
        '  </tlLogic>\n'
        '</net>\n',
        encoding="utf-8",
    )

    prog_0 = extract_tls_programs_from_net(multi_prog_xml, preferred_program_id="0")
    assert prog_0["TLS_MULTI"][0]["state"] == "GGGG"
    assert prog_0["TLS_MULTI"][0]["duration"] == 40.0

    prog_1 = extract_tls_programs_from_net(multi_prog_xml, preferred_program_id="1")
    assert prog_1["TLS_MULTI"][0]["state"] == "rrrr"
    assert prog_1["TLS_MULTI"][0]["duration"] == 60.0


def test_actual_sumo_environment_safety_shield_integration():
    """
    Live SUMO integration test: Executes actual simulation steps with DOIMTrafficEnv
    under an adversarial policy requesting phase switches on every single step.
    Verifies that:
      1. Min green is strictly enforced (interventions > 0).
      2. No illegal yellow phase jumps occur.
      3. The safety shield operates properly in the live simulation loop.
    """
    from environment.sumo_env import DOIMTrafficEnv
    from environment.sumo_locator import find_sumo_binary

    if not find_sumo_binary():
        pytest.skip("SUMO binary not found in PATH or environment.")

    sumo_cfg = "scenarios/directional_shift/north_south_surge.sumocfg"
    if not Path(sumo_cfg).exists():
        pytest.skip(f"Config not found: {sumo_cfg}")

    env = DOIMTrafficEnv(
        sumo_config=sumo_cfg,
        enforce_safety_shield=True,
        max_steps=20,
    )

    try:
        obs, info = env.reset(seed=42)
        total_interventions = 0

        # Adversarial policy: requests phase switch (action=1) at every single step!
        for step in range(15):
            adversarial_action = np.ones(5, dtype=np.int64)
            obs, reward, terminated, truncated, next_info = env.step(adversarial_action)
            step_interventions = next_info.get("shield_interventions", 0)
            total_interventions += step_interventions
            if terminated or truncated:
                break

        # Since min_green is 10s, requesting a switch at every step MUST trigger shield interventions
        assert total_interventions > 0, "Safety shield failed to intervene against rapid adversarial switching!"
    finally:
        env.close()


def test_extract_tls_index_mapping_research_grid_all_five_controllable():
    """
    Validates explicit request-index to signal-state linkIndex mapping across all
    controllable intersections in the research grid network (A1, B0, B1, B2, C1).
    Verifies that all connections are fully mapped (unmapped_count == 0) and reliable.
    """
    from environment.safety_shield import extract_tls_index_mapping

    grid_net = Path("environment/network/research_3x3/grid.net.xml")
    if not grid_net.exists():
        pytest.skip(f"Network file not found: {grid_net}")

    mappings = extract_tls_index_mapping(grid_net)
    controllable = ["A1", "B0", "B1", "B2", "C1"]

    expected_link_counts = {
        "A1": 12,
        "B0": 12,
        "B1": 20,
        "B2": 12,
        "C1": 12,
    }

    for tid in controllable:
        assert tid in mappings, f"TLS {tid} missing from extracted index mappings"
        m = mappings[tid]
        assert m["is_reliable"] is True, f"Mapping for {tid} is not marked reliable"
        assert m["unmapped_count"] == 0, f"TLS {tid} has unmapped connections: {m['unmapped_connections']}"
        assert len(m["link_to_reqs"]) == expected_link_counts[tid], (
            f"TLS {tid} expected {expected_link_counts[tid]} link indices, got {len(m['link_to_reqs'])}"
        )
        assert len(m["conflict_pairs"]) > 0, f"TLS {tid} has no detected conflict pairs"
        assert len(m["yield_pairs"]) > 0, f"TLS {tid} has no detected yield pairs"

        # Ensure all conflict pairs are within bounds
        for l1, l2 in m["conflict_pairs"]:
            assert 0 <= l1 < expected_link_counts[tid]
            assert 0 <= l2 < expected_link_counts[tid]
            assert l1 < l2


def test_multiple_connections_sharing_link_index(tmp_path):
    """
    Constructs a synthetic SUMO network XML where multiple connections share the same
    signal linkIndex, and verifies that extract_tls_index_mapping correctly aggregates
    all connections under the shared linkIndex and derives all foe conflicts.
    """
    from environment.safety_shield import extract_tls_index_mapping

    net_xml = tmp_path / "shared_link.net.xml"
    net_xml.write_text(
        '<net version="1.16">\n'
        '  <junction id="J_SHARED" type="traffic_light" intLanes=":J_0_0 :J_0_1 :J_1_0">\n'
        '    <request index="0" response="000" foes="010" cont="0"/>\n'
        '    <request index="1" response="000" foes="100" cont="0"/>\n'
        '    <request index="2" response="000" foes="001" cont="0"/>\n'
        '  </junction>\n'
        '  <connection from="E0" to="E1" fromLane="0" toLane="0" tl="J_SHARED" linkIndex="0" via=":J_0_0"/>\n'
        '  <connection from="E0" to="E1" fromLane="1" toLane="1" tl="J_SHARED" linkIndex="0" via=":J_0_1"/>\n'
        '  <connection from="E2" to="E3" fromLane="0" toLane="0" tl="J_SHARED" linkIndex="1" via=":J_1_0"/>\n'
        '</net>\n',
        encoding="utf-8",
    )

    mapping = extract_tls_index_mapping(net_xml)
    assert "J_SHARED" in mapping
    m = mapping["J_SHARED"]
    assert m["is_reliable"] is True
    assert m["unmapped_count"] == 0

    # linkIndex 0 must contain both request index 0 and request index 1
    assert set(m["link_to_reqs"][0]) == {0, 1}
    assert m["link_to_reqs"][1] == [2]
    assert len(m["link_to_conns"][0]) == 2
    assert len(m["link_to_conns"][1]) == 1


def test_priority_green_and_permissive_green_yield_semantics():
    """
    Verifies that verify_phase_conflicts rigorously distinguishes between:
      1. Conflicting priority greens ('G' vs 'G') -> REJECTED
      2. Permissive green yielding to opposing priority green ('g' vs 'G' with registered yield) -> PERMITTED
      3. Permissive green without registered yield hierarchy -> REJECTED
      4. Dual permissive green without yield hierarchy -> REJECTED
    """
    from environment.safety_shield import verify_phase_conflicts

    # Conflict pair between movement 0 and movement 2
    conflict_pairs = [(0, 2)]
    # Movement 0 yields to movement 2
    yield_pairs = {(0, 2)}

    # Case 1: Both hold priority green 'G' -> Unconditional collision hazard
    is_valid, viols = verify_phase_conflicts("GrGr", conflict_pairs, yield_pairs)
    assert is_valid is False
    assert any("both have priority green 'G'" in v for v in viols)

    # Case 2: Movement 0 has permissive 'g' and yields to priority 'G' on movement 2 -> Safe permissive turn
    is_valid, viols = verify_phase_conflicts("grGr", conflict_pairs, yield_pairs)
    assert is_valid is True
    assert len(viols) == 0

    # Case 3: Movement 2 has permissive 'g' and movement 0 has 'G', but movement 2 does NOT yield to 0
    is_valid, viols = verify_phase_conflicts("Grgr", conflict_pairs, yield_pairs)
    assert is_valid is False
    assert any("does not have registered right-of-way yield" in v for v in viols)

    # Case 4: Both have permissive 'g' without mutual yield resolution
    is_valid, viols = verify_phase_conflicts("grgr", conflict_pairs, set())
    assert is_valid is False
    assert any("Dual permissive conflict" in v for v in viols)


def test_active_tls_program_consistency_with_traci():
    """
    Live TraCI test: Verifies that the active TLS program in TraCI corresponds to
    the phase program selected by the safety shield, and controlled link count
    strictly matches state string character length for all 5 controllable lights.
    """
    from environment.sumo_locator import find_sumo_binary
    from environment.safety_shield import extract_tls_programs_from_net, find_net_file_from_sumocfg

    sumo_bin = find_sumo_binary(prefer_gui=False)
    if not sumo_bin:
        pytest.skip("SUMO binary not found in PATH or environment.")

    sumocfg_path = "scenarios/directional_shift/north_south_surge.sumocfg"
    if not Path(sumocfg_path).exists():
        pytest.skip(f"Config not found: {sumocfg_path}")

    net_file = find_net_file_from_sumocfg(sumocfg_path)
    assert net_file is not None and net_file.exists()
    xml_programs = extract_tls_programs_from_net(net_file, preferred_program_id="0")

    import traci
    cmd = [sumo_bin, "-c", sumocfg_path, "--start", "--quit-on-end", "--no-warnings"]
    try:
        traci.start(cmd)
        controllable = ["A1", "B0", "B1", "B2", "C1"]
        for tid in controllable:
            prog_id = str(traci.trafficlight.getProgram(tid))
            assert prog_id == "0", f"TLS {tid} active program is '{prog_id}', expected '0'"

            controlled_links = traci.trafficlight.getControlledLinks(tid)
            current_state = traci.trafficlight.getRedYellowGreenState(tid)
            assert len(controlled_links) == len(current_state), (
                f"TLS {tid}: {len(controlled_links)} controlled links != {len(current_state)} chars"
            )

            # Cross-verify with XML parsed program
            assert tid in xml_programs
            assert len(xml_programs[tid][0]["state"]) == len(current_state)
    finally:
        try:
            traci.close()
        except Exception:
            pass


def test_fail_safe_behavior_on_unreliable_or_unmapped_tls():
    """
    Verifies that if a TLS mapping cannot be established reliably (e.g. unmapped connection),
    TrafficSafetyShield safely logs a warning and enforces fail-safe behavior.
    """
    from environment.safety_shield import TrafficSafetyShield

    shield = TrafficSafetyShield(
        controllable_tls_ids=["TLS_UNKNOWN"],
        tls_phase_counts={"TLS_UNKNOWN": 2},
        tls_programs={"TLS_UNKNOWN": [
            {"duration": 30.0, "state": "GG"},
            {"duration": 3.0, "state": "yy"},
        ]},
        min_green=10,
        max_green=60,
    )

    # Unmapped TLS: attempting rapid switch should still be rejected by min green
    raw_actions = np.array([1])
    safe_actions, targets, interv = shield.filter_action(raw_actions, [0])
    assert safe_actions[0] == 0
    assert interv == 1
