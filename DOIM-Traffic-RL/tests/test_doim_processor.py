import os
import numpy as np
import traci

from environment.doim_processor import (
    DOIMProcessor,
)


SUMO_CONFIG = os.path.join(
    "scenarios",
    "directional_shift",
    "north_south_surge.sumocfg",
)


def main():

    print("=" * 80)
    print("DOIM PROCESSOR TEST")
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

        processor = DOIMProcessor(
            tls_ids
        )

        processor.reset()

        for step in range(10):

            traci.simulationStep()

            completed = (
                traci.simulation
                .getArrivedNumber()
            )

            result = processor.process(
                completed
            )

            print(
                f"\nStep {step + 1}"
            )

            print(
                "Feature matrix:",
                result[
                    "feature_matrix"
                ].shape
            )

            print(
                "Normalized objectives:",
                np.round(
                    result[
                        "normalized_objectives"
                    ],
                    3,
                )
            )

            print(
                "Traffic embedding:",
                result[
                    "traffic_embedding"
                ].shape
            )

            print(
                "Interaction matrix:",
                result[
                    "interaction_matrix"
                ].shape
            )

            print(
                "Manifold:",
                result[
                    "manifold"
                ].shape
            )

            print(
                "Weights:",
                np.round(
                    result[
                        "weights"
                    ],
                    4,
                )
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            assert (
                result[
                    "feature_matrix"
                ].shape
                == (9, 5)
            )

            assert (
                result[
                    "normalized_objectives"
                ].shape
                == (5,)
            )

            assert (
                result[
                    "interaction_matrix"
                ].shape
                == (5, 5)
            )

            assert (
                result[
                    "traffic_embedding"
                ].shape
                == (64,)
            )

            assert (
                result[
                    "manifold"
                ].shape
                == (64,)
            )

            assert (
                result[
                    "weights"
                ].shape
                == (5,)
            )

            assert np.isfinite(
                result["weights"]
            ).all()

            assert abs(
                result["weights"].sum()
                - 1.0
            ) < 1e-5

    finally:

        traci.close()

    print("\n" + "=" * 80)
    print(
        "DOIM PROCESSOR TEST PASSED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()