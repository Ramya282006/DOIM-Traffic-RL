import os
import json
import numpy as np

from stable_baselines3 import PPO

from environment.sumo_env import DOIMTrafficEnv


# ============================================================
# CONFIGURATION
# ============================================================

SUMO_CONFIG = (
    "scenarios/directional_shift/"
    "north_south_surge.sumocfg"
)

# ------------------------------------------------------------
# Select model here
# ------------------------------------------------------------

MODEL_NAME = "fixed_ppo"

MODEL_PATHS = {
    "doim_ppo": "results/ppo/doim_ppo_final",
    "fixed_ppo": "results/ppo_fixed/fixed_ppo_final",
}

MODEL_PATH = MODEL_PATHS[MODEL_NAME]

EVALUATION_STEPS = 1000

RESULTS_DIR = "results/evaluation"


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 80)
    print(f"{MODEL_NAME.upper()} EVALUATION")
    print("=" * 80)

    print("\nSUMO configuration:")
    print(SUMO_CONFIG)

    print("\nModel:")
    print(MODEL_PATH)

    if not os.path.exists(MODEL_PATH + ".zip"):

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}.zip"
        )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = DOIMTrafficEnv(
        SUMO_CONFIG
    )

    print("\nObservation space:")
    print(env.observation_space)

    print("\nAction space:")
    print(env.action_space)

    # --------------------------------------------------------
    # Load PPO
    # --------------------------------------------------------

    print("\nLoading trained PPO model...")

    model = PPO.load(
        MODEL_PATH,
        env=env,
        device="cpu",
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    observation, info = env.reset()

    print("\nInitial observation shape:")
    print(observation.shape)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_reward = 0.0
    step_count = 0

    rewards = []

    waiting_history = []
    queue_history = []
    co2_history = []
    throughput_history = []
    fairness_history = []

    weight_history = []

    action_history = []

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("STARTING EVALUATION")
    print("=" * 80)

    for step in range(EVALUATION_STEPS):

        # ----------------------------------------------------
        # Deterministic policy
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        step_count += 1

        total_reward += float(reward)

        rewards.append(
            float(reward)
        )

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
                f"\nStep: {step_count}"
            )

            print(
                "Simulation time:",
                info.get("time", "N/A")
            )

            print(
                "Action:",
                action.tolist()
            )

            print(
                "Reward:",
                round(
                    float(reward),
                    5,
                )
            )

            if "raw_objectives" in info:

                print(
                    "Raw objectives:",
                    np.round(
                        info["raw_objectives"],
                        3,
                    )
                )

            if "normalized_objectives" in info:

                print(
                    "Normalized objectives:",
                    np.round(
                        info["normalized_objectives"],
                        3,
                    )
                )

            if "doim_weights" in info:

                print(
                    "DOIM weights:",
                    np.round(
                        info["doim_weights"],
                        3,
                    )
                )

        # ----------------------------------------------------
        # Episode end
        # ----------------------------------------------------

        if terminated or truncated:

            print(
                "\nEpisode terminated at step:",
                step_count
            )

            break

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("=" * 80)
    print(f"{MODEL_NAME.upper()} EVALUATION RESULTS")
    print("=" * 80)

    print(
        "\nEvaluation steps:",
        step_count
    )

    print(
        "Total reward:",
        round(
            total_reward,
            4,
        )
    )

    if step_count > 0:

        average_reward = (
            total_reward / step_count
        )

        print(
            "Average reward:",
            round(
                average_reward,
                4,
            )
        )

    else:

        average_reward = 0.0

    # --------------------------------------------------------
    # Objective statistics
    # --------------------------------------------------------

    if waiting_history:

        print("\nObjective statistics:")
        print("-" * 80)

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

        print(
            "Average waiting:",
            round(average_waiting, 4)
        )

        print(
            "Maximum waiting:",
            round(maximum_waiting, 4)
        )

        print(
            "Average queue:",
            round(average_queue, 4)
        )

        print(
            "Maximum queue:",
            round(maximum_queue, 4)
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

        print("\nAction statistics:")
        print("-" * 80)

        print(
            "Total decisions:",
            total_actions
        )

        print(
            "KEEP decisions:",
            keep_decisions
        )

        print(
            "CHANGE decisions:",
            change_decisions
        )

        print(
            "CHANGE ratio:",
            round(
                change_ratio,
                4,
            )
        )

    else:

        total_actions = 0
        keep_decisions = 0
        change_decisions = 0
        change_ratio = 0.0

    # --------------------------------------------------------
    # Weight statistics
    # --------------------------------------------------------

    if weight_history:

        weights_array = np.asarray(
            weight_history
        )

        average_weights = np.mean(
            weights_array,
            axis=0,
        )

        print("\nAverage DOIM weights:")
        print(
            np.round(
                average_weights,
                4,
            )
        )

    else:

        average_weights = np.array(
            [0.2] * 5,
            dtype=np.float32,
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {

        "model": MODEL_NAME,

        "model_path": MODEL_PATH,

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

        "average_doim_weights":
            average_weights.tolist(),
    }

    summary_path = os.path.join(
        RESULTS_DIR,
        f"{MODEL_NAME}_summary.json"
    )

    with open(
        summary_path,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    print("\nSaved summary:")
    print(summary_path)

    # ========================================================
    # CLOSE
    # ========================================================

    env.close()

    print("\n")
    print("=" * 80)
    print(f"{MODEL_NAME.upper()} EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()