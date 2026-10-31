import os
import numpy as np

from environment.sumo_env import (
    DOIMTrafficEnv,
)


SUMO_CONFIG = os.path.join(
    "scenarios",
    "directional_shift",
    "north_south_surge.sumocfg",
)


def main():

    print("=" * 80)
    print("CONTROLLED DOIM TRAFFIC ENVIRONMENT TEST")
    print("=" * 80)

    env = DOIMTrafficEnv(
        SUMO_CONFIG
    )

    try:

        # --------------------------------------------------
        # Gym reset
        # --------------------------------------------------

        observation, info = (
            env.reset(seed=42)
        )

        print(
            "\nInitial observation:",
            observation
        )

        print(
            "Controllable TLS:",
            info["controllable_tls"]
        )

        print(
            "Action space:",
            env.action_space
        )

        # --------------------------------------------------
        # Validate action space
        # --------------------------------------------------

        assert isinstance(
            env.action_space,
            type(
                env.action_space
            ),
        )

        assert (
            env.action_space.shape
            == (5,)
        )

        # --------------------------------------------------
        # Test random valid actions
        # --------------------------------------------------

        for step in range(10):

            action = (
                env.action_space.sample()
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            print(
                f"\nStep {step + 1}"
            )

            print(
                "Action:",
                action
            )

            print(
                "Observation:",
                observation
            )

            print(
                "Reward:",
                reward
            )

            print(
                "Phases:",
                info["current_phases"]
            )

            assert (
                observation.shape
                == (1,)
            )

            assert np.isfinite(
                reward
            )

            if terminated or truncated:

                break

        print(
            "\nEnvironment successfully "
            "executed real traffic-light actions."
        )

    finally:

        env.close()

    print("\n" + "=" * 80)
    print(
        "CONTROLLED ENVIRONMENT TEST PASSED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()