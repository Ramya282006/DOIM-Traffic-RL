import os
import csv
import numpy as np

from stable_baselines3 import PPO

from environment.sumo_env import DOIMTrafficEnv
from environment.fixed_weight_env import FixedWeightTrafficEnv


# ============================================================
# CONFIGURATION
# ============================================================

SUMO_CONFIG = (
    "scenarios/directional_shift/"
    "north_south_surge.sumocfg"
)

SEEDS = [42, 123, 456, 789, 2024]

EVALUATION_STEPS = 1000

MODEL_PATHS = {
    "doim_ppo": "results/ppo/doim_ppo_final",
    "fixed_ppo": "results/ppo_fixed/fixed_ppo_final",
}

RESULTS_DIR = "results/evaluation"

CSV_PATH = os.path.join(
    RESULTS_DIR,
    "multiseed_results.csv"
)


# ============================================================
# EVALUATE ONE MODEL / ONE SEED
# ============================================================

def evaluate_model(
    model_name,
    model_path,
    seed,
):

    print("\n" + "=" * 80)
    print(
        f"EVALUATING: {model_name.upper()} | "
        f"SEED: {seed}"
    )
    print("=" * 80)

    if not os.path.exists(
        model_path + ".zip"
    ):

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{model_path}.zip"
        )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    if model_name == "doim_ppo":
        env = DOIMTrafficEnv(
            SUMO_CONFIG
        )

    elif model_name == "fixed_ppo":
        env = FixedWeightTrafficEnv(
            SUMO_CONFIG
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = PPO.load(
        model_path,
        env=env,
        device="cpu",
    )

    # --------------------------------------------------------
    # Reset with experiment seed
    # --------------------------------------------------------

    observation, info = env.reset(
        seed=seed
    )

    total_reward = 0.0

    rewards = []

    waiting_history = []
    queue_history = []
    co2_history = []
    throughput_history = []
    fairness_history = []

    weight_history = []

    action_history = []

    step_count = 0

    # --------------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------------

    for step in range(
        EVALUATION_STEPS
    ):

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=np.int64,
        )

        action_history.append(
            action.copy()
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        reward = float(reward)

        total_reward += reward

        rewards.append(
            reward
        )

        step_count += 1

        # ----------------------------------------------------
        # Objectives
        # ----------------------------------------------------

        if "raw_objectives" in info:

            raw = np.asarray(
                info["raw_objectives"],
                dtype=np.float32,
            )

            if len(raw) == 5:

                waiting_history.append(
                    float(raw[0])
                )

                queue_history.append(
                    float(raw[1])
                )

                co2_history.append(
                    float(raw[2])
                )

                throughput_history.append(
                    float(raw[3])
                )

                fairness_history.append(
                    float(raw[4])
                )

        # ----------------------------------------------------
        # DOIM weights
        # ----------------------------------------------------

        if "doim_weights" in info:

            weights = np.asarray(
                info["doim_weights"],
                dtype=np.float32,
            )

            if len(weights) == 5:

                weight_history.append(
                    weights.copy()
                )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            step_count == 1
            or step_count % 100 == 0
        ):

            print(
                f"Step {step_count:4d} | "
                f"Reward {reward: .5f} | "
                f"Time {info.get('time', 'N/A')}"
            )

        if terminated or truncated:

            print(
                f"Episode terminated at "
                f"step {step_count}"
            )

            break

    # ========================================================
    # Calculate statistics
    # ========================================================

    average_reward = (
        total_reward / step_count
        if step_count > 0
        else 0.0
    )

    if waiting_history:

        average_waiting = float(
            np.mean(waiting_history)
        )

        maximum_waiting = float(
            np.max(waiting_history)
        )

        average_queue = float(
            np.mean(queue_history)
        )

        maximum_queue = float(
            np.max(queue_history)
        )

        average_co2 = float(
            np.mean(co2_history)
        )

        total_throughput = float(
            np.sum(throughput_history)
        )

        average_fairness = float(
            np.mean(fairness_history)
        )

    else:

        average_waiting = 0.0
        maximum_waiting = 0.0
        average_queue = 0.0
        maximum_queue = 0.0
        average_co2 = 0.0
        total_throughput = 0.0
        average_fairness = 0.0

    # --------------------------------------------------------
    # Action statistics
    # --------------------------------------------------------

    if action_history:

        actions = np.asarray(
            action_history
        )

        total_actions = int(
            actions.size
        )

        keep_decisions = int(
            np.sum(actions == 0)
        )

        change_decisions = int(
            np.sum(actions == 1)
        )

        change_ratio = float(
            np.mean(actions)
        )

    else:

        total_actions = 0
        keep_decisions = 0
        change_decisions = 0
        change_ratio = 0.0

    # --------------------------------------------------------
    # Average DOIM weights
    # --------------------------------------------------------

    if weight_history:

        weights_array = np.asarray(
            weight_history
        )

        average_weights = np.mean(
            weights_array,
            axis=0,
        )

    else:

        average_weights = np.array(
            [0.2] * 5,
            dtype=np.float32,
        )

    # ========================================================
    # Print results
    # ========================================================

    print("\n" + "-" * 80)
    print(
        f"{model_name.upper()} | SEED {seed} RESULTS"
    )
    print("-" * 80)

    print(
        "Evaluation steps:",
        step_count
    )

    print(
        "Total reward:",
        round(total_reward, 4)
    )

    print(
        "Average reward:",
        round(average_reward, 4)
    )

    print(
        "Average waiting:",
        round(average_waiting, 4)
    )

    print(
        "Average queue:",
        round(average_queue, 4)
    )

    print(
        "Average CO2:",
        round(average_co2, 4)
    )

    print(
        "Total throughput:",
        round(total_throughput, 4)
    )

    print(
        "Average fairness:",
        round(average_fairness, 4)
    )

    print(
        "CHANGE ratio:",
        round(change_ratio, 4)
    )

    print(
        "Average DOIM weights:",
        np.round(
            average_weights,
            4
        )
    )

    # ========================================================
    # Close environment
    # ========================================================

    env.close()

    # ========================================================
    # Return result
    # ========================================================

    return {

        "model": model_name,

        "seed": seed,

        "evaluation_steps": step_count,

        "total_reward": total_reward,

        "average_reward": average_reward,

        "average_waiting": average_waiting,

        "maximum_waiting": maximum_waiting,

        "average_queue": average_queue,

        "maximum_queue": maximum_queue,

        "average_co2": average_co2,

        "total_throughput": total_throughput,

        "average_fairness": average_fairness,

        "total_actions": total_actions,

        "keep_decisions": keep_decisions,

        "change_decisions": change_decisions,

        "change_ratio": change_ratio,

        "weight_waiting":
            float(average_weights[0]),

        "weight_queue":
            float(average_weights[1]),

        "weight_co2":
            float(average_weights[2]),

        "weight_throughput":
            float(average_weights[3]),

        "weight_fairness":
            float(average_weights[4]),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DOIM vs FIXED PPO MULTI-SEED EVALUATION")
    print("=" * 80)

    print("\nScenario:")
    print(SUMO_CONFIG)

    print("\nSeeds:")
    print(SEEDS)

    print("\nEvaluation steps per run:")
    print(EVALUATION_STEPS)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    all_results = []

    # --------------------------------------------------------
    # Run every model on every seed
    # --------------------------------------------------------

    for seed in SEEDS:

        for model_name, model_path in MODEL_PATHS.items():

            result = evaluate_model(
                model_name,
                model_path,
                seed,
            )

            all_results.append(
                result
            )

    # ========================================================
    # Save CSV
    # ========================================================

    fieldnames = [
        "model",
        "seed",
        "evaluation_steps",
        "total_reward",
        "average_reward",
        "average_waiting",
        "maximum_waiting",
        "average_queue",
        "maximum_queue",
        "average_co2",
        "total_throughput",
        "average_fairness",
        "total_actions",
        "keep_decisions",
        "change_decisions",
        "change_ratio",
        "weight_waiting",
        "weight_queue",
        "weight_co2",
        "weight_throughput",
        "weight_fairness",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    # ========================================================
    # Final summary
    # ========================================================

    print("\n" + "=" * 80)
    print("MULTI-SEED EVALUATION COMPLETED")
    print("=" * 80)

    print("\nTotal runs:")
    print(len(all_results))

    print("\nExpected:")
    print(len(SEEDS) * len(MODEL_PATHS))

    print("\nResults saved to:")
    print(CSV_PATH)


if __name__ == "__main__":
    main()