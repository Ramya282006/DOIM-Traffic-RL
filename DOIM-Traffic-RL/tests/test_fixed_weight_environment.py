from environment.fixed_weight_env import (
    FixedWeightTrafficEnv
)

import numpy as np


SUMO_CONFIG = (
    "scenarios/directional_shift/"
    "north_south_surge.sumocfg"
)


def main():

    print("=" * 80)
    print("FIXED-WEIGHT ENVIRONMENT TEST")
    print("=" * 80)

    env = FixedWeightTrafficEnv(
        SUMO_CONFIG
    )

    observation, info = env.reset()

    print("\nObservation shape:")
    print(observation.shape)

    print("\nFixed weights:")
    print(info["fixed_weights"])

    print("\nWeight sum:")
    print(np.sum(info["fixed_weights"]))

    for step in range(5):

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        print(f"\nStep {step + 1}")

        print("Action:")
        print(action)

        print("Reward:")
        print(round(float(reward), 5))

        print("Fixed weights:")
        print(info["fixed_weights"])

        print("Weight sum:")
        print(np.sum(info["fixed_weights"]))

        if terminated or truncated:
            break

    env.close()

    print("\n" + "=" * 80)
    print("FIXED-WEIGHT ENVIRONMENT TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()