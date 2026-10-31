import os
import sys
from pathlib import Path
import traci

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SUMO_CONFIG = os.path.join(
    "scenarios",
    "directional_shift",
    "north_south_surge.sumocfg",
)


def main():

    print("=" * 80)
    print("TRAFFIC LIGHT PHASE INSPECTION")
    print("=" * 80)

    from environment.sumo_locator import find_sumo_binary
    sumo_bin = find_sumo_binary(prefer_gui=False)

    traci.start([
        sumo_bin,
        "-c",
        SUMO_CONFIG,
        "--start",
        "--quit-on-end",
    ])

    try:

        tls_ids = sorted(
            traci.trafficlight.getIDList()
        )

        print(
            f"\nNumber of traffic lights: "
            f"{len(tls_ids)}"
        )

        for tls_id in tls_ids:

            logic = (
                traci.trafficlight
                .getAllProgramLogics(tls_id)
            )

            print("\n" + "-" * 70)

            print(
                f"Traffic Light: {tls_id}"
            )

            for program in logic:

                print(
                    f"Program ID: "
                    f"{program.programID}"
                )

                print(
                    f"Number of phases: "
                    f"{len(program.phases)}"
                )

                for i, phase in enumerate(
                    program.phases
                ):

                    print(
                        f"  Phase {i}: "
                        f"duration="
                        f"{phase.duration:.1f}s "
                        f"state="
                        f"{phase.state}"
                    )

    finally:

        traci.close()

    print("\n" + "=" * 80)
    print(
        "TRAFFIC LIGHT PHASE INSPECTION COMPLETED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()