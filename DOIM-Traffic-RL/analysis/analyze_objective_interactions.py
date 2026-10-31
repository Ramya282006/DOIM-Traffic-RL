import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import traci
import shutil

from environment.state_extractor import TrafficStateExtractor
from objectives.objective_engine import MultiObjectiveEngine


CONFIG = (
    "scenarios/directional_shift/"
    "north_south_surge.sumocfg"
)

OUTPUT_DIR = "results/objective_interactions"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


OBJECTIVE_NAMES = [
    "Waiting",
    "Queue",
    "CO2",
    "Throughput",
    "Fairness",
]


def compute_safe_correlation(data):

    """
    Compute correlation while handling
    zero-variance objectives safely.
    """

    data = np.asarray(
        data,
        dtype=np.float64
    )

    n_objectives = data.shape[1]

    correlation = np.eye(
        n_objectives,
        dtype=np.float64
    )

    for i in range(n_objectives):

        for j in range(i + 1, n_objectives):

            x = data[:, i]
            y = data[:, j]

            # Correlation is undefined if either
            # objective has no variation.
            if (
                np.std(x) < 1e-8
                or np.std(y) < 1e-8
            ):
                value = 0.0

            else:

                value = np.corrcoef(
                    x,
                    y
                )[0, 1]

                if not np.isfinite(value):
                    value = 0.0

            correlation[i, j] = value
            correlation[j, i] = value

    return correlation


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

    objective_history = []

    try:

        tls_ids = traci.trafficlight.getIDList()

        extractor = TrafficStateExtractor(
            tls_ids
        )

        objective_engine = MultiObjectiveEngine()

        print("\n" + "=" * 80)
        print(
            "DOIM OBJECTIVE INTERACTION ANALYSIS"
        )
        print("=" * 80)

        # --------------------------------------------------
        # Run simulation
        # --------------------------------------------------

        for step in range(300):

            traci.simulationStep()

            network_states = (
                extractor.get_network_state()
            )

            completed = (
                traci.simulation.getArrivedNumber()
            )

            objectives = (
                objective_engine.compute_objectives(
                    network_states,
                    completed
                )
            )

            objective_history.append(
                objectives.copy()
            )

            if (
                traci.simulation.getMinExpectedNumber()
                == 0
            ):
                break

    finally:

        traci.close()

    objective_history = np.asarray(
        objective_history,
        dtype=np.float64
    )

    # ======================================================
    # Save raw objective history
    # ======================================================

    csv_path = os.path.join(
        OUTPUT_DIR,
        "objective_history.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "time",
                "waiting",
                "queue",
                "co2",
                "throughput",
                "fairness",
            ]
        )

        for t, row in enumerate(
            objective_history,
            start=1
        ):

            writer.writerow(
                [
                    t,
                    *row
                ]
            )

    print(
        f"\nSaved objective history:"
        f"\n{csv_path}"
    )

    # ======================================================
    # Rolling correlation analysis
    # ======================================================

    window = 30

    interaction_history = []

    interaction_times = []

    for end in range(
        window,
        len(objective_history) + 1
    ):

        window_data = objective_history[
            end - window:end
        ]

        correlation = (
            compute_safe_correlation(
                window_data
            )
        )

        interaction_history.append(
            correlation
        )

        interaction_times.append(
            end
        )

    interaction_history = np.asarray(
        interaction_history
    )

    # ======================================================
    # Plot selected objective interactions
    # ======================================================

    pairs = [
        (0, 1, "Waiting vs Queue"),
        (0, 2, "Waiting vs CO2"),
        (0, 3, "Waiting vs Throughput"),
        (1, 2, "Queue vs CO2"),
        (1, 4, "Queue vs Fairness"),
    ]

    for i, j, title in pairs:

        values = (
            interaction_history[
                :, i, j
            ]
        )

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            interaction_times,
            values,
            linewidth=2
        )

        plt.axhline(
            0,
            linestyle="--",
            linewidth=1
        )

        plt.ylim(
            -1.05,
            1.05
        )

        plt.xlabel(
            "Simulation Time (s)"
        )

        plt.ylabel(
            "Rolling Correlation"
        )

        plt.title(
            title
        )

        plt.grid(
            alpha=0.3
        )

        filename = (
            title
            .lower()
            .replace(" ", "_")
            .replace("vs", "vs")
            + ".png"
        )

        path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=300
        )

        plt.close()

        print(
            f"Saved plot: {path}"
        )

    # ======================================================
    # Save final interaction matrix
    # ======================================================

    final_matrix = (
        interaction_history[-1]
    )

    matrix_path = os.path.join(
        OUTPUT_DIR,
        "final_interaction_matrix.csv"
    )

    with open(
        matrix_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "",
                *OBJECTIVE_NAMES
            ]
        )

        for name, row in zip(
            OBJECTIVE_NAMES,
            final_matrix
        ):

            writer.writerow(
                [
                    name,
                    *row
                ]
            )

    print(
        f"Saved final interaction matrix:"
        f"\n{matrix_path}"
    )

    # ======================================================
    # Print summary
    # ======================================================

    print("\n" + "=" * 80)
    print(
        "FINAL OBJECTIVE INTERACTION MATRIX"
    )
    print("=" * 80)

    print(
        OBJECTIVE_NAMES
    )

    print(
        np.array2string(
            final_matrix,
            precision=3,
            suppress_small=True
        )
    )

    print("\n" + "=" * 80)
    print(
        "ANALYSIS COMPLETED SUCCESSFULLY"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()