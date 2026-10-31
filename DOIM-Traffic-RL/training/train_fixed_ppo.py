import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment.fixed_weight_env import (
    FixedWeightTrafficEnv
)


SUMO_CONFIG = (
    "scenarios/directional_shift/"
    "north_south_surge.sumocfg"
)

TOTAL_TIMESTEPS = 1000
SEED = 42

MODEL_DIR = "results/ppo_fixed"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fixed_ppo_final"
)


def main():

    print("=" * 80)
    print("FIXED-WEIGHT PPO TRAINING")
    print("=" * 80)

    print("\nSUMO configuration:")
    print(SUMO_CONFIG)

    print("\nTotal timesteps:")
    print(TOTAL_TIMESTEPS)

    print("\nRandom seed:")
    print(SEED)

    # --------------------------------------------------
    # Environment
    # --------------------------------------------------

    env = FixedWeightTrafficEnv(
    SUMO_CONFIG
)

    env = Monitor(env)

    print("\nObservation space:")
    print(env.observation_space)

    print("\nAction space:")
    print(env.action_space)

    # --------------------------------------------------
    # PPO
    # --------------------------------------------------

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=SEED,
        tensorboard_log="results/ppo_fixed/tensorboard",
    )

    print("\nPPO model created successfully.")

    print("\n" + "=" * 80)
    print("STARTING FIXED-WEIGHT PPO TRAINING")
    print("=" * 80)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=False,
        tb_log_name="fixed_ppo",
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model.save(
        MODEL_PATH
    )

    print("\n" + "=" * 80)
    print("FIXED-WEIGHT PPO MODEL SAVED")
    print("=" * 80)

    print("\nFinal model:")
    print(MODEL_PATH)

    env.close()


if __name__ == "__main__":
    main()