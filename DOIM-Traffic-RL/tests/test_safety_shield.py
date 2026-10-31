"""
DOIM-Traffic-RL: Unit Test Suite for Traffic Signal Safety Shield
Verifies min-green compliance, yellow clearance state machine,
max-green protection, and zero illegal rapid fluttering.
"""

import numpy as np
import pytest

from environment.safety_shield import TrafficSafetyShield, SignalPhaseState


def test_signal_phase_state_timing():
    """Verify state clock ticking and min/max green thresholds."""
    state = SignalPhaseState(
        tls_id="TLS_1",
        num_phases=4,
        min_green_seconds=10,
        max_green_seconds=60,
        yellow_seconds=3,
    )

    assert not state.can_switch_phase()
    for _ in range(9):
        state.step_clock()
    assert not state.can_switch_phase()

    state.step_clock() # 10th second
    assert state.can_switch_phase()
    assert not state.is_max_green_exceeded()

    for _ in range(50):
        state.step_clock() # 60th second
    assert state.is_max_green_exceeded()


def test_safety_shield_min_green_rejection():
    """Verify that the shield rejects rapid switching attempts under min green."""
    tls_ids = ["A1", "B0", "B1", "B2", "C1"]
    phase_counts = {t: 4 for t in tls_ids}
    shield = TrafficSafetyShield(
        controllable_tls_ids=tls_ids,
        tls_phase_counts=phase_counts,
        min_green=10,
        max_green=60,
        yellow_duration=3,
    )

    current_phases = [0, 0, 0, 0, 0]

    # Step 1: Agent tries to switch all 5 lights immediately
    raw_action = np.ones(5, dtype=np.int64)
    safe_action, targets, interventions = shield.filter_action(raw_action, current_phases)

    # All 5 switches must be REJECTED!
    assert np.all(safe_action == 0)
    assert interventions == 5
    assert all(t is None for t in targets)

    # Step 2-9: Keep trying to switch, all must be rejected
    for _ in range(8):
        safe_act, _, interv = shield.filter_action(raw_action, current_phases)
        assert np.all(safe_act == 0)
        assert interv == 5

    # Step 10: 10 seconds elapsed -> switches are GRANTED to yellow!
    safe_act, targets, interv = shield.filter_action(raw_action, current_phases)
    assert np.all(safe_act == 1)
    assert interv == 0
    # Targets should be intermediate yellow phase (phase 1)
    assert targets == [1, 1, 1, 1, 1]


def test_safety_shield_yellow_interval_protection():
    """Verify that during yellow interval, conflicting agent commands are ignored."""
    tls_ids = ["A1"]
    shield = TrafficSafetyShield(
        controllable_tls_ids=tls_ids,
        tls_phase_counts={"A1": 4},
        min_green=5,
        yellow_duration=3,
    )

    # Advance past min green
    for _ in range(5):
        shield.filter_action([0], [0])

    # Trigger switch -> enters yellow (phase 1)
    safe_act, targets, _ = shield.filter_action([1], [0])
    assert safe_act[0] == 1
    assert targets[0] == 1

    # During yellow (next 2 seconds), agent tries to switch again -> ignored
    safe_act_y1, targ_y1, _ = shield.filter_action([1], [1])
    assert safe_act_y1[0] == 0
    assert targ_y1[0] is None

    safe_act_y2, targ_y2, _ = shield.filter_action([1], [1])
    assert safe_act_y2[0] == 0
    assert targ_y2[0] is None

    # After 3s yellow, transitions automatically to next green (phase 2)
    safe_act_g, targ_g, _ = shield.filter_action([0], [1])
    assert targ_g[0] == 2
