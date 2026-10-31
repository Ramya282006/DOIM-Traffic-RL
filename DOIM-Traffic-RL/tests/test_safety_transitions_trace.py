"""
DOIM-Traffic-RL: Actual SUMO Safety Transitions Trace Verification
Stress-tests the safety shield under adversarial high-frequency switching requests
and verifies through SignalSafetyVerifier that zero illegal transitions occur.
"""

import pytest
import numpy as np

from environment.sumo_env import DOIMTrafficEnv
from environment.safety_verifier import SignalSafetyVerifier


def test_adversarial_safety_shield_sumo_trace():
    """
    Feeds continuous switch requests (action=[1, 1, 1, 1, 1] at every step) to attempt
    forcing illegal transitions and phase fluttering. The safety shield must reject all
    illegal requests and maintain strict compliance with min green, max green, and yellow clearance.
    """
    env = DOIMTrafficEnv(
        sumo_config="scenarios/directional_shift/north_south_surge.sumocfg",
        enforce_safety_shield=True,
        max_steps=150,
    )

    verifier = SignalSafetyVerifier(
        controllable_tls_ids=["A1", "B0", "B1", "B2", "C1"],
        min_green=10,
        max_green=60,
        yellow_duration=3,
    )

    obs, info = env.reset(seed=42)
    controllable = env.controllable_tls

    initial_phases = dict(zip(controllable, env.signal_controller.get_current_phases()))
    verifier.record_step(sim_time=0.0, current_phases=initial_phases)

    total_interventions_logged = 0

    for step in range(120):
        # Adversarial action: constantly request phase switch every second
        adversarial_action = np.ones(5, dtype=np.int64)

        obs, reward, term, trunc, info = env.step(adversarial_action)
        current_phases = dict(zip(controllable, env.signal_controller.get_current_phases()))

        step_viols = verifier.record_step(
            sim_time=info["time"],
            current_phases=current_phases,
        )

        assert len(step_viols) == 0, f"Safety violation detected at t={info['time']}: {step_viols}"
        total_interventions_logged += info.get("shield_interventions", 0)

        if term or trunc:
            break

    env.close()

    summary = verifier.get_summary()
    print("\n" + "=" * 80)
    print("SUMO TRAFFIC SAFETY SHIELD VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total Transitions Checked:       {summary['total_transitions']}")
    print(f"Adversarial Interventions Logged:{total_interventions_logged}")
    print(f"Minimum Green Violations (<10s): {summary['min_green_violations']}")
    print(f"Maximum Green Violations (>60s): {summary['max_green_violations']}")
    print(f"Yellow Clearance Violations:     {summary['yellow_clearance_violations']}")
    print(f"Illegal Sequence Violations:     {summary['illegal_sequence_violations']}")
    print(f"Overall Trace Status:            {'VERIFIED SAFE' if summary['verified_safe'] else 'FAILED'}")
    print("=" * 80)

    assert summary["verified_safe"] is True
    assert summary["total_transitions"] > 0
    assert total_interventions_logged > 0
    print("TRAFFIC SAFETY GUARANTEE VERIFIED ON LIVE SUMO TRACE!")


if __name__ == "__main__":
    test_adversarial_safety_shield_sumo_trace()
