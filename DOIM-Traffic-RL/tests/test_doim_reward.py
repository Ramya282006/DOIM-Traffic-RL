import numpy as np

from objectives.doim_reward import (
    DOIMReward,
)


def main():

    print("=" * 80)
    print("DOIM ADAPTIVE REWARD TEST")
    print("=" * 80)

    reward_engine = DOIMReward()

    # ======================================================
    # TEST 1 — Perfect traffic state
    # ======================================================

    urgency = np.array(
        [
            0.0,  # waiting
            0.0,  # queue
            0.0,  # CO2
            0.0,  # throughput urgency
            0.0,  # fairness
        ],
        dtype=np.float32,
    )

    weights = np.array(
        [
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
        ],
        dtype=np.float32,
    )

    reward = reward_engine.compute(
        urgency,
        weights,
    )

    print(
        "\nPerfect state:"
    )

    print(
        "Urgency:",
        urgency
    )

    print(
        "Weights:",
        weights
    )

    print(
        "Reward:",
        reward
    )

    assert reward == 0.0

    # ======================================================
    # TEST 2 — Worst state
    # ======================================================

    urgency = np.ones(
        5,
        dtype=np.float32,
    )

    reward = reward_engine.compute(
        urgency,
        weights,
    )

    print(
        "\nWorst state:"
    )

    print(
        "Urgency:",
        urgency
    )

    print(
        "Reward:",
        reward
    )

    assert np.isclose(
        reward,
        -1.0,
    )

    # ======================================================
    # TEST 3 — Adaptive weights
    # ======================================================

    urgency = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    weights = np.array(
        [
            0.8,
            0.05,
            0.05,
            0.05,
            0.05,
        ],
        dtype=np.float32,
    )

    reward = reward_engine.compute(
        urgency,
        weights,
    )

    print(
        "\nWaiting-critical state:"
    )

    print(
        "Urgency:",
        urgency
    )

    print(
        "Weights:",
        weights
    )

    print(
        "Reward:",
        reward
    )

    assert np.isclose(
        reward,
        -0.8,
    )

    # ======================================================
    # TEST 4 — Throughput direction
    # ======================================================

    low_throughput = np.array(
        [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        dtype=np.float32,
    )

    high_throughput = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    weights = np.array(
        [
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
        ],
        dtype=np.float32,
    )

    low_reward = reward_engine.compute(
        low_throughput,
        weights,
    )

    high_reward = reward_engine.compute(
        high_throughput,
        weights,
    )

    print(
        "\nThroughput direction test:"
    )

    print(
        "Low throughput urgency:",
        low_throughput
    )

    print(
        "Low throughput reward:",
        low_reward
    )

    print(
        "High throughput urgency:",
        high_throughput
    )

    print(
        "High throughput reward:",
        high_reward
    )

    assert (
        high_reward
        > low_reward
    )

    # ======================================================
    # TEST 5 — Contributions
    # ======================================================

    urgency = np.array(
        [
            0.5,
            0.4,
            0.3,
            0.2,
            0.1,
        ],
        dtype=np.float32,
    )

    weights = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.2,
            0.2,
        ],
        dtype=np.float32,
    )

    contributions = (
        reward_engine
        .get_contributions(
            urgency,
            weights,
        )
    )

    print(
        "\nObjective contributions:"
    )

    print(
        contributions
    )

    print(
        "Contribution sum:",
        contributions.sum()
    )

    assert contributions.shape == (
        5,
    )

    assert np.isfinite(
        contributions
    ).all()

    print("\n" + "=" * 80)
    print(
        "DOIM ADAPTIVE REWARD TEST PASSED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()