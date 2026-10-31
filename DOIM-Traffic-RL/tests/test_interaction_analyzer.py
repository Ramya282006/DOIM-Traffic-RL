import numpy as np

from analysis.objective_interaction import (
    ObjectiveInteractionAnalyzer
)


def main():

    print("=" * 80)
    print("OBJECTIVE INTERACTION ANALYZER TEST")
    print("=" * 80)

    analyzer = ObjectiveInteractionAnalyzer(
        num_objectives=5,
        window_size=5,
    )

    # --------------------------------------------------
    # Synthetic objective history
    # --------------------------------------------------

    observations = [

        [0.10, 0.12, 0.15, 0.90, 0.10],

        [0.20, 0.22, 0.25, 0.80, 0.20],

        [0.30, 0.32, 0.35, 0.70, 0.30],

        [0.40, 0.42, 0.45, 0.60, 0.40],

        [0.50, 0.52, 0.55, 0.50, 0.50],

        [0.60, 0.62, 0.65, 0.40, 0.60],

    ]

    # --------------------------------------------------
    # Update analyzer
    # --------------------------------------------------

    for step, values in enumerate(
        observations,
        start=1
    ):

        analyzer.update(
            np.array(
                values,
                dtype=np.float32
            )
        )

        matrix = (
            analyzer
            .get_interaction_matrix()
        )

        print("\n" + "-" * 80)

        print(
            f"Step {step}"
        )

        print(
            "History length:",
            len(analyzer.history)
        )

        print(
            "Interaction matrix:"
        )

        print(
            np.round(
                matrix,
                3
            )
        )

    # --------------------------------------------------
    # Final vector
    # --------------------------------------------------

    vector = (
        analyzer
        .get_interaction_vector()
    )

    print("\n" + "=" * 80)

    print(
        "Unique interaction vector:"
    )

    print(
        np.round(
            vector,
            3
        )
    )

    print(
        "Vector length:",
        len(vector)
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    assert matrix.shape == (
        5,
        5
    )

    assert vector.shape == (
        10,
    )

    assert np.allclose(
        matrix,
        matrix.T
    )

    assert np.allclose(
        np.diag(matrix),
        np.ones(5)
    )

    print("\n" + "=" * 80)

    print(
        "OBJECTIVE INTERACTION ANALYZER TEST PASSED"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()