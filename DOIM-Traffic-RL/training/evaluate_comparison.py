import os
import numpy as np

from stable_baselines3 import PPO

from environment.sumo_env import DOIMTrafficEnv


SUMO_CONFIG = (
    "scenarios/directional_shift/"
    "north_south_surge.sumocfg"
)

DOIM_MODEL = (
    "results/ppo/doim_ppo_final"
)

FIXED_MODEL = (
    "results/ppo_fixed/fixed_ppo_final"
)

EVAL_STEPS = 1000
SEED = 42


def evaluate_model(model_path, model_name):

    print("\n" + "=" * 80)
    print(f"EVALUATING: {model_name}")
    print("=" * 80)

    env = DOIMTrafficEnv(
        SUMO_CONFIG
    )

    model = PPO.load(
        model_path,
        env=env
    )

    observation, info = env.reset(
        seed=SEED
    )

    rewards = []

    waiting_values = []
    queue_values = []
    co2_values = []
    throughput_values = []
    fairness_values = []

    for step in range(EVAL_STEPS):

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        rewards.append(
            float(reward)
        )

        # ----------------------------------------------
        # Collect objective values
        # ----------------------------------------------

        if "raw_objectives" in info:

            raw = np.asarray(
                info["raw_objectives"],
                dtype=np.float32
            )

            if len(raw) == 5:

                waiting_values.append(raw[0])
                queue_values.append(raw[1])
                co2_values.append(raw[2])
                throughput_values.append(raw[3])
                fairness_values.append(raw[4])

        if terminated or truncated:
            break

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    results = {

        "model": model_name,

        "steps": len(rewards),

        "total_reward":
            float(np.sum(rewards)),

        "mean_reward":
            float(np.mean(rewards))
            if rewards else 0.0,

        "waiting":
            float(np.mean(waiting_values))
            if waiting_values else 0.0,

        "queue":
            float(np.mean(queue_values))
            if queue_values else 0.0,

        "co2":
            float(np.mean(co2_values))
            if co2_values else 0.0,

        "throughput":
            float(np.mean(throughput_values))
            if throughput_values else 0.0,

        "fairness":
            float(np.mean(fairness_values))
            if fairness_values else 0.0,
    }

    print("\nEvaluation results:")

    for key, value in results.items():
        print(f"{key:20s}: {value}")

    env.close()

    return results


def main():

    print("=" * 80)
    print("DOIM-PPO vs FIXED-WEIGHT PPO EVALUATION")
    print("=" * 80)

    doim_results = evaluate_model(
        DOIM_MODEL,
        "DOIM-PPO"
    )

    fixed_results = evaluate_model(
        FIXED_MODEL,
        "Fixed-Weight PPO"
    )

    # ----------------------------------------------
    # Comparison
    # ----------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    metrics = [
        "total_reward",
        "mean_reward",
        "waiting",
        "queue",
        "co2",
        "throughput",
        "fairness",
    ]

    print(
        f"\n{'Metric':<20}"
        f"{'DOIM-PPO':>15}"
        f"{'Fixed PPO':>15}"
    )

    print("-" * 50)

    for metric in metrics:

        print(
            f"{metric:<20}"
            f"{doim_results[metric]:>15.4f}"
            f"{fixed_results[metric]:>15.4f}"
        )


if __name__ == "__main__":
    main()