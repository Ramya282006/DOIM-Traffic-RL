import traci
import shutil
import numpy as np

from environment.state_extractor import TrafficStateExtractor
from objectives.objective_engine import MultiObjectiveEngine
from objectives.objective_interaction import (
    ObjectiveInteractionAnalyzer,
)


CONFIG = (
    "scenarios/directional_shift/"
    "north_south_surge.sumocfg"
)


def main():

    sumo = shutil.which("sumo")

    if sumo is None:
        raise RuntimeError(
            "SUMO was not found in PATH."
        )

    traci.start([
        sumo,
        "-c",
        CONFIG,
        "--start",
        "--quit-on-end",
    ])

    try:

        tls_ids = traci.trafficlight.getIDList()

        extractor = TrafficStateExtractor(
            tls_ids
        )

        objective_engine = MultiObjectiveEngine()

        analyzer = ObjectiveInteractionAnalyzer(
            window_size=30
        )

        print("\n" + "=" * 80)
        print("DYNAMIC OBJECTIVE INTERACTION TEST")
        print("=" * 80)

        for step in range(300):

            traci.simulationStep()

            # ------------------------------------------
            # Extract traffic state
            # ------------------------------------------

            network_states = (
                extractor.get_network_state()
            )

            # ------------------------------------------
            # Completed vehicles
            # ------------------------------------------

            completed = (
                traci.simulation.getArrivedNumber()
            )

            # ------------------------------------------
            # Compute objectives
            # ------------------------------------------

            objectives = (
                objective_engine.compute_objectives(
                    network_states,
                    completed,
                )
            )

            # ------------------------------------------
            # Update interaction analyzer
            # ------------------------------------------

            analyzer.update(objectives)

            # ------------------------------------------
            # Print interaction matrix periodically
            # ------------------------------------------

            if step in [29, 59, 89, 119, 149, 179, 209, 239, 269, 299]:

                matrix = (
                    analyzer.get_interaction_matrix()
                )

                print("\n" + "-" * 80)

                print(
                    f"Simulation time: "
                    f"{traci.simulation.getTime():.0f}s"
                )

                print(
                    "\nObjective interaction matrix:"
                )

                print(
                    np.array2string(
                        matrix,
                        precision=2,
                        suppress_small=True,
                    )
                )

                print(
                    "\nInteraction vector:"
                )

                print(
                    np.array2string(
                        analyzer.get_interaction_vector(),
                        precision=2,
                        suppress_small=True,
                    )
                )

            if (
                traci.simulation.getMinExpectedNumber()
                == 0
            ):
                break

    finally:

        traci.close()

    print("\n" + "=" * 80)
    print("DYNAMIC INTERACTION TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()