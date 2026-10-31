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
    print("DOIM-AWARE TRAFFIC ENVIRONMENT TEST")
    print("=" * 80)

    env = DOIMTrafficEnv(
        SUMO_CONFIG
    )

    try:

        observation, info = (
            env.reset(seed=42)
        )

        print(
            "\nInitial observation shape:",
            observation.shape
        )

        print(
            "Observation:",
            np.round(
                observation,
                3,
            )
        )

        print(
            "Action space:",
            env.action_space
        )

        # --------------------------------------------------
        # Observation validation
        # --------------------------------------------------

        assert (
            observation.shape
            == (69,)
        )

        assert (
            env.observation_space.shape
            == (69,)
        )

        assert np.isfinite(
            observation
        ).all()

        # --------------------------------------------------
        # Execute several actions
        # --------------------------------------------------

        for step in range(10):

            action = (
                env.action_space.sample()
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            print(
                f"\nStep {step + 1}"
            )

            print(
                "Action:",
                action
            )

            print(
                "Observation shape:",
                observation.shape
            )

            print(
                "Manifold sample:",
                np.round(
                    observation[:5],
                    3,
                )
            )

            print(
                "Weights:",
                np.round(
                    observation[64:],
                    4,
                )
            )

            print(
                "Weight sum:",
                observation[64:].sum()
            )

            print(
                "Reward:",
                reward
            )

            print(
                "Phases:",
                info[
                    "current_phases"
                ]
            )

            assert (
                observation.shape
                == (69,)
            )

            assert np.isfinite(
                observation
            ).all()

            assert abs(
                observation[64:].sum()
                - 1.0
            ) < 1e-4

            if terminated or truncated:
                break

    finally:

        env.close()

    print("\n" + "=" * 80)
    print(
        "DOIM-AWARE ENVIRONMENT TEST PASSED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()