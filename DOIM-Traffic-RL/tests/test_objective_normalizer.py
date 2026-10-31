import numpy as np

from objectives.objective_normalizer import ObjectiveNormalizer


def main():

    print("=" * 80)
    print("OBJECTIVE NORMALIZER TEST")
    print("=" * 80)

    normalizer = ObjectiveNormalizer()

    # --------------------------------------------------
    # Simulated traffic observations
    # --------------------------------------------------

    observations = [

        # waiting, queue, CO2, throughput, fairness
        [0,   0,  0,     10,  0],

        [100, 2,  20000, 8,   100],

        [300, 5,  50000, 5,   500],

        [600, 10, 90000, 2,   1500],

        [900, 15, 120000, 0, 3000],

    ]

    # --------------------------------------------------
    # Process observations
    # --------------------------------------------------

    for step, values in enumerate(
        observations,
        start=1
    ):

        raw = np.array(
            values,
            dtype=np.float32
        )

        normalized = normalizer.normalize(
            raw
        )

        print("\n" + "-" * 80)

        print(
            f"Step {step}"
        )

        print(
            "Raw:",
            raw
        )

        print(
            "Normalized urgency:",
            np.round(
                normalized,
                4
            )
        )

        print(
            "Range:",
            np.round(
                normalizer.running_max
                - normalizer.running_min,
                4
            )
        )

    # --------------------------------------------------
    # Validate output
    # --------------------------------------------------

    final = normalizer.normalize(
        np.array(
            [450, 7, 70000, 4, 1000],
            dtype=np.float32
        )
    )

    assert final.shape == (5,)

    assert np.all(
        final >= 0.0
    )

    assert np.all(
        final <= 1.0
    )

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    print("\n" + "=" * 80)

    print(
        "Final normalized objective:"
    )

    print(
        np.round(
            final,
            4
        )
    )

    print("\n" + "=" * 80)

    print(
        "OBJECTIVE NORMALIZER TEST PASSED"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()