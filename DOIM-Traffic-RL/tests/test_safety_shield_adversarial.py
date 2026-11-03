"""
DOIM-Traffic-RL: Independent Reference Validation Tests for Safety Shield and Adversarial Edge Cases.
Validates:
  1. Conflict detection on simultaneous conflicting green phases
  2. Signal state character validation
  3. Minimum green enforcement against premature switch attacks
  4. Maximum green enforcement against starvation / indefinite hold attacks
  5. Suppression of rapid toggling requests during clearance intervals
  6. Fail-safe lock behavior preventing unsafe progression
"""

import pytest
import numpy as np

from environment.safety_shield import (
    validate_signal_state_string,
    verify_phase_conflicts,
    SignalPhaseState,
    TrafficSafetyShield,
)


class TestSafetyShieldAdversarial:
    """Adversarial stress-testing suite for safety-critical traffic signal control."""

    def test_validate_signal_state_string_valid_and_invalid(self):
        """Validates SUMO character legality check."""
        assert validate_signal_state_string("GGrrrrGGrrrr") is True
        assert validate_signal_state_string("yyrrryyyrrrr") is True
        assert validate_signal_state_string("rrrrrrrrrrrr") is True
        assert validate_signal_state_string("gGrruo") is True

        # Invalid characters
        assert validate_signal_state_string("GGXrrr") is False
        assert validate_signal_state_string("123456") is False
        assert validate_signal_state_string("") is False

    def test_verify_phase_conflicts_simultaneous_priority_greens(self):
        """
        Simultaneous priority green ('G') on conflicting movements must be detected as an illegal conflict.
        """
        conflict_pairs = [(0, 3), (1, 4)]

        # Adversarial state: both link 0 and link 3 have priority green 'G'
        illegal_state = "GGrGGr"
        is_valid, violations = verify_phase_conflicts(illegal_state, conflict_pairs)
        assert is_valid is False
        assert len(violations) > 0
        assert any("movements 0 and 3" in v for v in violations)

        # Legal state: link 0 has 'G', but conflicting link 3 has red 'r'
        legal_state = "GGrrrr"
        is_valid_legal, violations_legal = verify_phase_conflicts(legal_state, conflict_pairs)
        assert is_valid_legal is True
        assert len(violations_legal) == 0

    def test_adversarial_premature_switch_suppressed_by_min_green(self):
        """
        When agent sends action=1 before min_green has elapsed, the safety shield
        must suppress the action to 0, maintain current phase, and log an intervention.
        """
        programs = {
            "tls_0": [
                {"state": "GGGrrr", "duration": 30.0, "type": "green"},
                {"state": "yyyrrr", "duration": 3.0, "type": "yellow"},
                {"state": "rrrGGG", "duration": 30.0, "type": "green"},
                {"state": "rrryyy", "duration": 3.0, "type": "yellow"},
            ]
        }
        shield = TrafficSafetyShield(
            controllable_tls_ids=["tls_0"],
            tls_programs=programs,
            min_green=10,
            max_green=60,
        )

        # Current phase is 0 (green). Time in phase = 0.
        # Agent attacks with action=1 at t=0, t=1, ..., t=9
        for t in range(9):
            safe_act, target_phases, interventions = shield.filter_action(
                raw_actions=np.array([1]),
                current_phases=[0],
            )
            # Must suppress to 0
            assert safe_act[0] == 0
            assert interventions == 1
            assert target_phases[0] is None

        # At t=10, min_green is satisfied. Agent action=1 is now permitted.
        safe_act, target_phases, interventions = shield.filter_action(
            raw_actions=np.array([1]),
            current_phases=[0],
        )
        assert safe_act[0] == 1
        assert interventions == 0
        # Target phase transitions to phase 1 (yellow)
        assert target_phases[0] == 1

    def test_adversarial_indefinite_hold_forced_by_max_green(self):
        """
        When agent attempts to hold green indefinitely (action=0) beyond max_green,
        the safety shield must force a switch (action=1) to prevent starvation.
        """
        programs = {
            "tls_0": [
                {"state": "GGGrrr", "duration": 30.0, "type": "green"},
                {"state": "yyyrrr", "duration": 3.0, "type": "yellow"},
                {"state": "rrrGGG", "duration": 30.0, "type": "green"},
                {"state": "rrryyy", "duration": 3.0, "type": "yellow"},
            ]
        }
        shield = TrafficSafetyShield(
            controllable_tls_ids=["tls_0"],
            tls_programs=programs,
            min_green=10,
            max_green=30,
        )

        state = shield.states["tls_0"]
        # Advance time to max_green without switching
        state.time_in_current_phase = 30

        # Agent attacks with action=0 (hold forever)
        safe_act, target_phases, _ = shield.filter_action(
            raw_actions=np.array([0]),
            current_phases=[0],
        )

        # Must force action=1 and target yellow clearance phase 1
        assert safe_act[0] == 1
        assert target_phases[0] == 1

    def test_rapid_toggle_suppression_during_clearance(self):
        """
        During yellow / clearance intervals, rapid toggle attempts (action=1)
        must be completely suppressed to protect vehicular clearance.
        """
        programs = {
            "tls_0": [
                {"state": "GGGrrr", "duration": 30.0, "type": "green"},
                {"state": "yyyrrr", "duration": 3.0, "type": "yellow"},
                {"state": "rrrGGG", "duration": 30.0, "type": "green"},
                {"state": "rrryyy", "duration": 3.0, "type": "yellow"},
            ]
        }
        shield = TrafficSafetyShield(
            controllable_tls_ids=["tls_0"],
            tls_programs=programs,
            min_green=10,
            max_green=60,
            yellow_duration=3,
        )

        state = shield.states["tls_0"]
        # Allow initial transition to yellow
        state.time_in_current_phase = 15
        safe_act, target_phases, _ = shield.filter_action(
            raw_actions=np.array([1]),
            current_phases=[0],
        )
        assert target_phases[0] == 1
        assert state.is_yellow is True

        # Now in yellow. Agent aggressively sends action=1 repeatedly.
        for _ in range(2):
            safe_act, target_phases, _ = shield.filter_action(
                raw_actions=np.array([1]),
                current_phases=[1],
            )
            # Action suppressed to 0 while clearance is underway
            assert safe_act[0] == 0
            assert state.is_yellow is True

        # When yellow duration (3s) elapses, target_phase becomes green phase 2
        safe_act, target_phases, _ = shield.filter_action(
            raw_actions=np.array([0]),
            current_phases=[1],
        )
        assert target_phases[0] == 2
        assert state.is_yellow is False

    def test_fail_safe_lock_blocks_all_actions(self):
        """
        When a TLS is locked in fail-safe mode, all actions are suppressed to 0,
        and the designated safe hold phase is continuously maintained.
        """
        programs = {
            "tls_0": [
                {"state": "GGGrrr", "duration": 30.0, "type": "green"},
                {"state": "yyyrrr", "duration": 3.0, "type": "yellow"},
            ]
        }
        shield = TrafficSafetyShield(
            controllable_tls_ids=["tls_0"],
            tls_programs=programs,
        )
        state = shield.states["tls_0"]
        state.is_fail_safe_locked = True
        state.fail_safe_hold_phase = 1  # Hold yellow/safe phase

        safe_act, target_phases, interventions = shield.filter_action(
            raw_actions=np.array([1]),
            current_phases=[1],
        )
        assert safe_act[0] == 0
        assert target_phases[0] == 1
        assert interventions == 1
