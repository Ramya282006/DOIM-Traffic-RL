import traci
import shutil
import numpy as np

from environment.state_extractor import TrafficStateExtractor
from objectives.objective_engine import MultiObjectiveEngine


CONFIG = (
    "scenarios/directional_shift/"
    "north_south_surge.sumocfg"
)


def main():

    sumo = shutil.which("sumo")

    if sumo is None:
        raise RuntimeError("SUMO was not found in PATH.")

    # ------------------------------------------------------
    # Start SUMO
    # ------------------------------------------------------

    traci.start([
        sumo,
        "-c",
        CONFIG,
        "--start",
        "--quit-on-end",
    ])

    try:

        # --------------------------------------------------
        # Discover traffic lights
        # --------------------------------------------------

        tls_ids = traci.trafficlight.getIDList()

        print("\n" + "=" * 80)
        print("DOIM MULTI-OBJECTIVE ENGINE TEST")
        print("=" * 80)

        print(f"\nTraffic-light nodes: {len(tls_ids)}")

        # --------------------------------------------------
        # Create state extractor
        # --------------------------------------------------

        extractor = TrafficStateExtractor(tls_ids)

        # --------------------------------------------------
        # Create objective engine
        # --------------------------------------------------

        objective_engine = MultiObjectiveEngine()

        # --------------------------------------------------
        # Run simulation
        # --------------------------------------------------

        print("\n")
        print(
            "Time | Waiting | Queue | CO2 | "
            "Throughput | Fairness"
        )
        print("-" * 80)

        for step in range(300):

            traci.simulationStep()

            # ----------------------------------------------
            # Network state
            # ----------------------------------------------

            network_states = (
                extractor.get_network_state()
            )

            # ----------------------------------------------
            # Number of completed vehicles
            # ----------------------------------------------

            completed = (
                traci.simulation.getArrivedNumber()
            )

            # ----------------------------------------------
            # Objective vector
            # ----------------------------------------------

            objectives = (
                objective_engine.compute_objectives(
                    network_states,
                    completed,
                )
            )

            current_time = traci.simulation.getTime()

            # Print every 10 seconds
            if step % 10 == 0:

                print(
                    f"{current_time:4.0f} | "
                    f"{objectives[0]:8.2f} | "
                    f"{objectives[1]:5.0f} | "
                    f"{objectives[2]:8.2f} | "
                    f"{objectives[3]:10.0f} | "
                    f"{objectives[4]:8.2f}"
                )

            # Stop if SUMO has no more vehicles
            if (
                traci.simulation.getMinExpectedNumber()
                == 0
            ):
                break

    finally:

        traci.close()

    print("\n" + "=" * 80)
    print("MULTI-OBJECTIVE TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()