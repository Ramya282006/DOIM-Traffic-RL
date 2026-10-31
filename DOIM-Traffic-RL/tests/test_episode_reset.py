from environment.sumo_env import DOIMTrafficEnv


CONFIG = (
    "scenarios/directional_shift/"
    "north_south_surge.sumocfg"
)


def main():

    print("=" * 80)
    print("DOIM EPISODE RESET TEST")
    print("=" * 80)

    env = DOIMTrafficEnv(CONFIG)

    # --------------------------------------------------
    # Episode 1
    # --------------------------------------------------

    obs, info = env.reset()

    print("\nEpisode 1")
    print("Observation shape:", obs.shape)
    print(
        "DOIM history length:",
        len(env.doim_processor.history)
    )

    for _ in range(5):

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        if terminated or truncated:
            break

    print(
        "History after 5 steps:",
        len(env.doim_processor.history)
    )

    # --------------------------------------------------
    # Episode 2 reset
    # --------------------------------------------------

    obs, info = env.reset()

    print("\nEpisode 2 after reset")
    print("Observation shape:", obs.shape)
    print(
        "DOIM history length:",
        len(env.doim_processor.history)
    )

    # --------------------------------------------------
    # Verify reset
    # --------------------------------------------------

    assert len(
        env.doim_processor.history
    ) == 1, (
        "DOIM history was not reset correctly."
    )

    assert (
        env.doim_processor
        .objective_engine
        .previous_completed
        == 0
    ), (
        "Objective engine was not reset."
    )

    assert (
        len(
            env.doim_processor
            .interaction_analyzer
            .history
        ) == 1
    ), (
        "Interaction history was not reset correctly."
    )

    print("\nAll reset checks passed.")

    env.close()

    print("=" * 80)
    print("DOIM EPISODE RESET TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()