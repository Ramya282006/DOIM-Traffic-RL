import traci
import shutil
import numpy as np

from environment.state_extractor import TrafficStateExtractor


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
        # Get the 9 traffic-light controllers
        # --------------------------------------------------

        tls_ids = traci.trafficlight.getIDList()

        print("\n" + "=" * 70)
        print("DOIM TRAFFIC STATE EXTRACTION TEST")
        print("=" * 70)

        print("\nTraffic lights:")
        print(tls_ids)

        # --------------------------------------------------
        # Create state extractor
        # --------------------------------------------------

        extractor = TrafficStateExtractor(tls_ids)

        # --------------------------------------------------
        # Advance simulation
        # --------------------------------------------------

        for step in range(10):

            traci.simulationStep()

            feature_matrix = extractor.get_feature_matrix()

            print("\n" + "-" * 70)
            print(
                f"Simulation time: "
                f"{traci.simulation.getTime():.1f} s"
            )

            print(
                f"Feature matrix shape: "
                f"{feature_matrix.shape}"
            )

            print("\n[Queue, Waiting, Speed, Density, CO2]")

            print(
                np.array2string(
                    feature_matrix,
                    precision=2,
                    suppress_small=True,
                )
            )

    finally:

        traci.close()

    print("\n" + "=" * 70)
    print("STATE EXTRACTION TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()