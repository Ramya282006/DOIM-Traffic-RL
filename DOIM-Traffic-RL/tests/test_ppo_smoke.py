from stable_baselines3 import PPO

from environment.sumo_env import DOIMTrafficEnv


CONFIG = (
    "scenarios/directional_shift/"
    "north_south_surge.sumocfg"
)


def main():

    print("=" * 80)
    print("PPO + DOIM ENVIRONMENT SMOKE TEST")
    print("=" * 80)

    # --------------------------------------------------
    # Create environment
    # --------------------------------------------------

    env = DOIMTrafficEnv(CONFIG)

    print("\nObservation space:")
    print(env.observation_space)

    print("\nAction space:")
    print(env.action_space)

    # --------------------------------------------------
    # Create PPO
    # --------------------------------------------------

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=16,
        batch_size=16,
        learning_rate=3e-4,
        gamma=0.99,
        seed=42,
    )

    print("\nPPO model created successfully.")

    # --------------------------------------------------
    # Tiny learning run
    # --------------------------------------------------

    print("\nStarting PPO smoke training...")

    model.learn(
        total_timesteps=32
    )

    print("\nPPO learning completed.")

    # --------------------------------------------------
    # Test prediction
    # --------------------------------------------------

    observation, info = env.reset()

    action, _ = model.predict(
        observation,
        deterministic=True,
    )

    print("\nPredicted action:")
    print(action)

    print("\nAction shape:")
    print(action.shape)

    print("\nAction space contains action:")
    print(env.action_space.contains(action))

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    env.close()

    print("\n" + "=" * 80)
    print("PPO + DOIM ENVIRONMENT SMOKE TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()