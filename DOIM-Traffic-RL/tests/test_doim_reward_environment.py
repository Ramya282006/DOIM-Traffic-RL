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
    print("DOIM REWARD ENVIRONMENT INTEGRATION TEST")
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
            observation.shape,
        )

        assert (
            observation.shape
            == (69,)
        )

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
                action,
            )

            print(
                "Reward:",
                round(
                    reward,
                    5,
                ),
            )

            print(
                "Raw objectives:",
                np.round(
                    info[
                        "raw_objectives"
                    ],
                    2,
                ),
            )

            print(
                "Normalized objectives:",
                np.round(
                    info[
                        "normalized_objectives"
                    ],
                    3,
                ),
            )

            print(
                "DOIM weights:",
                np.round(
                    info[
                        "doim_weights"
                    ],
                    4,
                ),
            )

            print(
                "Contributions:",
                np.round(
                    info[
                        "objective_contributions"
                    ],
                    4,
                ),
            )

            print(
                "Phases:",
                info[
                    "current_phases"
                ],
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            assert np.isfinite(
                reward
            )

            assert (
                info[
                    "normalized_objectives"
                ].shape
                == (5,)
            )

            assert (
                info[
                    "doim_weights"
                ].shape
                == (5,)
            )

            assert (
                info[
                    "objective_contributions"
                ].shape
                == (5,)
            )

            assert np.isfinite(
                info[
                    "doim_weights"
                ]
            ).all()

            assert abs(
                np.sum(
                    info[
                        "doim_weights"
                    ]
                ) - 1.0
            ) < 1e-4

            assert np.isfinite(
                info[
                    "objective_contributions"
                ]
            ).all()

            if terminated or truncated:

                break

    finally:

        env.close()

    print("\n" + "=" * 80)
    print(
        "DOIM REWARD ENVIRONMENT "
        "INTEGRATION TEST PASSED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()