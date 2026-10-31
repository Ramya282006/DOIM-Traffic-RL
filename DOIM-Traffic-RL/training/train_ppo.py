import os
import random

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from environment.sumo_env import DOIMTrafficEnv


# ============================================================
# CONFIGURATION
# ============================================================

SUMO_CONFIG = (
    "scenarios/directional_shift/"
    "north_south_surge.sumocfg"
)

MODEL_DIR = "results/ppo"
LOG_DIR = "results/ppo/tensorboard"
CHECKPOINT_DIR = "results/ppo/checkpoints"

TOTAL_TIMESTEPS = 1000
SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# ENVIRONMENT
# ============================================================

def make_environment():

    env = DOIMTrafficEnv(
        SUMO_CONFIG
    )

    env = Monitor(env)

    return env


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 80)
    print("DOIM-PPO TRAINING")
    print("=" * 80)

    print()
    print("SUMO configuration:")
    print(SUMO_CONFIG)

    print()
    print("Total timesteps:")
    print(TOTAL_TIMESTEPS)

    print()
    print("Random seed:")
    print(SEED)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    os.makedirs(
        LOG_DIR,
        exist_ok=True,
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = make_environment()

    print()
    print("Observation space:")
    print(env.observation_space)

    print()
    print("Action space:")
    print(env.action_space)

    # --------------------------------------------------------
    # Check environment
    # --------------------------------------------------------

    observation, info = env.reset(
        seed=SEED
    )

    print()
    print("Initial observation shape:")
    print(observation.shape)

    print()
    print("Initial observation finite:")
    print(
        np.all(
            np.isfinite(observation)
        )
    )

    # --------------------------------------------------------
    # PPO
    # --------------------------------------------------------

    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=3e-4,

        n_steps=128,

        batch_size=64,

        n_epochs=10,

        gamma=0.99,

        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.01,

        vf_coef=0.5,

        max_grad_norm=0.5,

        verbose=1,

        seed=SEED,

        tensorboard_log=LOG_DIR,

        device="cpu",
    )

    print()
    print("PPO model created successfully.")

    # --------------------------------------------------------
    # Checkpoint callback
    # --------------------------------------------------------

    checkpoint_callback = (
        CheckpointCallback(
            save_freq=500,
            save_path=CHECKPOINT_DIR,
            name_prefix="doim_ppo",
        )
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    try:

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True,
            tb_log_name="doim_ppo",
        )

    finally:

        # ----------------------------------------------------
        # Save final model
        # ----------------------------------------------------

        final_path = os.path.join(
            MODEL_DIR,
            "doim_ppo_final",
        )

        model.save(
            final_path
        )

        print()
        print("=" * 80)
        print("MODEL SAVED")
        print("=" * 80)

        print()
        print(
            "Final model:",
            final_path,
        )

        env.close()

    print()
    print("=" * 80)
    print("DOIM-PPO TRAINING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":

    main()