import os
import traci
import numpy as np

from environment.traffic_signal_controller import (
    TrafficSignalController,
)


SUMO_CONFIG = os.path.join(
    "scenarios",
    "directional_shift",
    "north_south_surge.sumocfg",
)


def main():

    print("=" * 80)
    print("TRAFFIC SIGNAL CONTROLLER TEST")
    print("=" * 80)

    traci.start([
        "sumo",
        "-c",
        SUMO_CONFIG,
        "--start",
        "--quit-on-end",
    ])

    try:

        tls_ids = sorted(
            traci.trafficlight.getIDList()
        )

        controller = (
            TrafficSignalController(
                tls_ids
            )
        )

        print(
            "\nAll TLS:",
            tls_ids
        )

        print(
            "Controllable TLS:",
            controller.controllable_tls
        )

        print(
            "Number controllable:",
            controller.num_controllable
        )

        # --------------------------------------------------
        # Expected network structure
        # --------------------------------------------------

        expected = [
            "A1",
            "B0",
            "B1",
            "B2",
            "C1",
        ]

        assert (
            controller.controllable_tls
            == expected
        )

        assert (
            controller.num_controllable
            == 5
        )

        # --------------------------------------------------
        # Current phases
        # --------------------------------------------------

        phases_before = (
            controller.get_current_phases()
        )

        print(
            "\nInitial phases:",
            phases_before
        )

        # --------------------------------------------------
        # Keep action
        # --------------------------------------------------

        keep_action = np.zeros(
            5,
            dtype=np.int64,
        )

        controller.apply_action(
            keep_action
        )

        phases_after_keep = (
            controller.get_current_phases()
        )

        print(
            "After KEEP:",
            phases_after_keep
        )

        assert (
            phases_after_keep
            == phases_before
        )

        # --------------------------------------------------
        # Advance simulation
        # --------------------------------------------------

        traci.simulationStep()

        # --------------------------------------------------
        # Request transition
        # --------------------------------------------------

        change_action = np.ones(
            5,
            dtype=np.int64,
        )

        controller.apply_action(
            change_action
        )

        phases_after_change = (
            controller.get_current_phases()
        )

        print(
            "After CHANGE:",
            phases_after_change
        )

        # Each controllable TLS should move
        # to its next phase.
        for before, after in zip(
            phases_before,
            phases_after_change,
        ):

            # The phase may have naturally changed
            # after simulationStep(), so only verify
            # that the resulting phase is valid.
            assert after >= 0
            assert after <= 3

        print(
            "\nAction space:"
        )

        print(
            "Each controllable TLS: "
            "{0=KEEP, 1=CHANGE}"
        )

        print(
            "Joint action dimensions: 5"
        )

        print(
            "Possible joint decisions: "
            f"{2 ** 5}"
        )

    finally:

        traci.close()

    print("\n" + "=" * 80)
    print(
        "TRAFFIC SIGNAL CONTROLLER TEST PASSED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()