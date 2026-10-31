import numpy as np

from objectives.objective_normalizer import (
    ObjectiveNormalizer,
)


def main():

    print("=" * 80)
    print("STABLE OBJECTIVE NORMALIZER TEST")
    print("=" * 80)

    normalizer = ObjectiveNormalizer()

    # ======================================================
    # Test 1: minimum state
    # ======================================================

    objectives = np.array(
        [
            0.0,    # waiting
            0.0,    # queue
            0.0,    # CO2
            20.0,   # throughput
            0.0,    # fairness
        ],
        dtype=np.float32,
    )

    result = normalizer.normalize(
        objectives
    )

    print("\nMinimum-urgency state:")
    print("Raw:", objectives)
    print(
        "Urgency:",
        np.round(result, 4)
    )

    # Throughput is excellent, everything else is zero
    assert np.allclose(
        result,
        [0, 0, 0, 0, 0],
    )

    # ======================================================
    # Test 2: maximum-urgency state
    # ======================================================

    objectives = np.array(
        [
            600.0,
            20.0,
            120000.0,
            0.0,
            3000.0,
        ],
        dtype=np.float32,
    )

    result = normalizer.normalize(
        objectives
    )

    print("\nMaximum-urgency state:")
    print("Raw:", objectives)
    print(
        "Urgency:",
        np.round(result, 4)
    )

    assert np.allclose(
        result,
        [1, 1, 1, 1, 1],
    )

    # ======================================================
    # Test 3: midpoint
    # ======================================================

    objectives = np.array(
        [
            300.0,
            10.0,
            60000.0,
            10.0,
            1500.0,
        ],
        dtype=np.float32,
    )

    result = normalizer.normalize(
        objectives
    )

    print("\nMidpoint state:")
    print("Raw:", objectives)
    print(
        "Urgency:",
        np.round(result, 4)
    )

    assert np.allclose(
        result,
        [0.5, 0.5, 0.5, 0.5, 0.5],
    )

    # ======================================================
    # Test 4: throughput direction
    # ======================================================

    low_throughput = np.array(
        [
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=np.float32,
    )

    high_throughput = np.array(
        [
            0,
            0,
            0,
            20,
            0,
        ],
        dtype=np.float32,
    )

    low_result = normalizer.normalize(
        low_throughput
    )

    high_result = normalizer.normalize(
        high_throughput
    )

    print("\nThroughput direction test:")

    print(
        "Zero throughput:",
        np.round(low_result, 4)
    )

    print(
        "Maximum throughput:",
        np.round(high_result, 4)
    )

    assert low_result[3] == 1.0
    assert high_result[3] == 0.0

    # ======================================================
    # Test 5: clipping
    # ======================================================

    objectives = np.array(
        [
            1000.0,
            100.0,
            500000.0,
            -10.0,
            10000.0,
        ],
        dtype=np.float32,
    )

    result = normalizer.normalize(
        objectives
    )

    print("\nOut-of-range state:")
    print("Raw:", objectives)
    print(
        "Urgency:",
        np.round(result, 4)
    )

    assert np.all(
        result >= 0.0
    )

    assert np.all(
        result <= 1.0
    )

    # ======================================================
    # Final
    # ======================================================

    print("\n" + "=" * 80)
    print(
        "STABLE OBJECTIVE NORMALIZER TEST PASSED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()