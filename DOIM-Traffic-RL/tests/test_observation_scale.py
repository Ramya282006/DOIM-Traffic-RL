import numpy as np

from environment.sumo_env import DOIMTrafficEnv


SUMO_CONFIG = (
    "scenarios/directional_shift/"
    "north_south_surge.sumocfg"
)


def main():

    print("=" * 80)
    print("DOIM OBSERVATION SCALE TEST")
    print("=" * 80)

    env = DOIMTrafficEnv(SUMO_CONFIG)

    observation, info = env.reset()

    observations = [observation.copy()]

    print("\nInitial observation:")
    print("Shape:", observation.shape)
    print("NaN:", np.isnan(observation).any())
    print("Inf:", np.isinf(observation).any())

    for step in range(20):

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        observations.append(observation.copy())

        if terminated or truncated:
            break

    env.close()

    observations = np.asarray(observations)

    print("\n" + "-" * 80)
    print("OBSERVATION STATISTICS")
    print("-" * 80)

    print("Collected observations:", len(observations))
    print("Shape:", observations.shape)

    print("\nGlobal minimum:", observations.min())
    print("Global maximum:", observations.max())
    print("Global mean:", observations.mean())
    print("Global std:", observations.std())

    print("\nNaN present:", np.isnan(observations).any())
    print("Inf present:", np.isinf(observations).any())

    # --------------------------------------------------
    # DOIM manifold
    # --------------------------------------------------

    manifold = observations[:, :64]

    print("\n" + "-" * 80)
    print("64-D DOIM MANIFOLD")
    print("-" * 80)

    print("Minimum:", manifold.min())
    print("Maximum:", manifold.max())
    print("Mean:", manifold.mean())
    print("Std:", manifold.std())

    # --------------------------------------------------
    # Adaptive weights
    # --------------------------------------------------

    weights = observations[:, 64:]

    print("\n" + "-" * 80)
    print("5-D ADAPTIVE WEIGHTS")
    print("-" * 80)

    print("Minimum:", weights.min())
    print("Maximum:", weights.max())
    print("Mean:", weights.mean())
    print("Std:", weights.std())

    print("\nWeight sums:")

    for i, w in enumerate(weights):
        print(
            f"Step {i:02d}: "
            f"{w.sum():.6f}"
        )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert observations.shape[1] == 69

    assert not np.isnan(observations).any()

    assert not np.isinf(observations).any()

    assert np.allclose(
        weights.sum(axis=1),
        1.0,
        atol=1e-5,
    )

    print("\n" + "=" * 80)
    print("OBSERVATION SCALE TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()