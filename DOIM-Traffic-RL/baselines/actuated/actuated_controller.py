"""
DOIM-Traffic-RL: Actuated Traffic Signal Controller Baseline (Phase 19)
Implements standard NEMA TS 2 vehicle-actuated signal control adhering to transportation engineering standards:
  - Enforces minimum green time (g_min = 10s).
  - Evaluates approach detector vehicle gaps (passage timer / gap-out logic):
    * While arriving vehicles pulse the detector, the passage timer is reset and green is extended by unit_extension_sec.
    * If no vehicle is detected for elapsed time >= gap_threshold_sec (gap-out), requests phase transition.
  - Enforces maximum green time (g_max = 60s): forces transition (max-out) regardless of ongoing gap extensions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from environment.sumo_env import DOIMTrafficEnv
from common.experiment import ExperimentRun, SCHEMA_VERSION_CURRENT
from common.config import load_experiment_config


class ActuatedTrafficController:
    """
    Standard NEMA-style vehicle-actuated traffic signal controller across multiple intersections.
    """

    def __init__(
        self,
        controllable_tls_ids: List[str],
        min_green: int = 10,
        max_green: int = 60,
        gap_threshold_sec: float = 3.0,
        unit_extension_sec: float = 2.0,
        step_length_sec: float = 1.0,
    ):
        self.controllable_tls_ids = list(controllable_tls_ids)
        self.num_controllers = len(self.controllable_tls_ids)
        self.min_green = int(min_green)
        self.max_green = int(max_green)
        self.gap_threshold_sec = float(gap_threshold_sec)
        self.unit_extension_sec = float(unit_extension_sec)
        self.step_length_sec = float(step_length_sec)

        # State tracking per intersection
        self.time_in_phase = {tid: 0 for tid in self.controllable_tls_ids}
        self.last_phase = {tid: 0 for tid in self.controllable_tls_ids}
        self.gap_timer = {tid: 0.0 for tid in self.controllable_tls_ids}
        self.extension_remaining = {tid: 0.0 for tid in self.controllable_tls_ids}

    def reset(self):
        for tid in self.controllable_tls_ids:
            self.time_in_phase[tid] = 0
            self.last_phase[tid] = 0
            self.gap_timer[tid] = 0.0
            self.extension_remaining[tid] = 0.0

    def select_actions(
        self,
        current_phases: Union[Dict[str, int], List[int]],
        approach_queues: Dict[str, float],
        approach_vehicles: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Determines binary transition decisions a_j in {0, 1} for each controllable intersection
        using NEMA gap-out passage timers and min/max green constraints.
        """
        if isinstance(current_phases, (list, tuple)):
            current_phases_dict = {tid: current_phases[i] for i, tid in enumerate(self.controllable_tls_ids)}
        else:
            current_phases_dict = current_phases or {}
        actions = []
        for tid in self.controllable_tls_ids:
            c_phase = current_phases_dict.get(tid, 0)
            if c_phase != self.last_phase.get(tid, 0):
                self.time_in_phase[tid] = 0
                self.last_phase[tid] = c_phase
                self.gap_timer[tid] = 0.0
                self.extension_remaining[tid] = 0.0
            else:
                self.time_in_phase[tid] += 1

            t_elapsed = self.time_in_phase[tid]

            # Rule 1: Minimum green constraint (must remain in phase)
            if t_elapsed < self.min_green:
                self.gap_timer[tid] = 0.0
                self.extension_remaining[tid] = 0.0
                actions.append(0)
                continue

            # Rule 2: Maximum green constraint (forces transition / max-out)
            if t_elapsed >= self.max_green:
                actions.append(1)
                continue

            # Rule 3: NEMA Passage Timer & Gap-Out Logic
            queue = approach_queues.get(tid, 0.0)
            num_veh = approach_vehicles.get(tid, int(queue > 0.0)) if approach_vehicles is not None else int(queue > 0.0)

            if num_veh > 0:
                # Vehicle detected: reset passage timer and add unit extension
                self.gap_timer[tid] = 0.0
                self.extension_remaining[tid] = max(self.extension_remaining[tid], self.unit_extension_sec)
                actions.append(0)
            else:
                # No arriving vehicle on this step
                if self.extension_remaining[tid] > 0.0:
                    # Still serving previously earned vehicle extension
                    self.extension_remaining[tid] = max(0.0, self.extension_remaining[tid] - self.step_length_sec)
                    actions.append(0)
                else:
                    # Increment gap timer
                    self.gap_timer[tid] += self.step_length_sec
                    if self.gap_timer[tid] >= self.gap_threshold_sec:
                        # Gap-out condition met: no vehicles within threshold, yield right-of-way
                        actions.append(1)
                    else:
                        # Allowable passage gap has not yet expired
                        actions.append(0)

        return np.array(actions, dtype=np.int64)


def run_actuated_experiment(
    scenario_override: Optional[str] = None,
    total_timesteps: int = 1800,
    seed: int = 42,
    output_root: str = "results/runs",
    min_green: int = 10,
    max_green: int = 60,
    gap_threshold_sec: float = 3.0,
    unit_extension_sec: float = 2.0,
) -> Dict[str, Any]:
    """
    Executes a standard Actuated Controller experiment in SUMO.
    """
    config = load_experiment_config()
    scenario_path = scenario_override if scenario_override is not None else config.simulation.scenario_config

    exp = ExperimentRun(
        experiment_name=f"Actuated_{Path(scenario_path).stem}_s{seed}",
        algorithm="Actuated",
        config=config.to_dict() if hasattr(config, "to_dict") else dict(config),
        train_seed=seed,
        eval_seed=seed,
        scenario=Path(scenario_path).stem,
        output_root=output_root,
        training_timesteps=total_timesteps,
        evaluation_mode="deterministic_eval",
        eval_horizon=total_timesteps,
    )

    env = DOIMTrafficEnv(
        sumo_config=scenario_path,
        max_steps=total_timesteps,
        enforce_safety_shield=True,
    )

    obs, info = env.reset(seed=seed)

    controller = ActuatedTrafficController(
        controllable_tls_ids=env.controllable_tls,
        min_green=min_green,
        max_green=max_green,
        gap_threshold_sec=gap_threshold_sec,
        unit_extension_sec=unit_extension_sec,
    )
    controller.reset()

    total_reward = 0.0
    for step in range(total_timesteps):
        current_phases = env.signal_controller.get_current_phases()
        feature_matrix = info.get("feature_matrix", np.zeros((9, 5)))
        tls_list = sorted(env.doim_processor.tls_ids) if env.doim_processor else []

        queues = {}
        for tid in env.controllable_tls:
            if tid in tls_list:
                idx = tls_list.index(tid)
                queues[tid] = float(feature_matrix[idx, 0])
            else:
                queues[tid] = 0.0

        action = controller.select_actions(current_phases, queues)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        exp.log_step(
            step=step,
            sim_time=float(step + 1),
            reward=float(reward),
            raw_objs=info.get("raw_objectives", np.zeros(5)),
            norm_objs=info.get("normalized_objectives", np.zeros(5)),
            weights=info.get("doim_weights", np.ones(5) * 0.2),
            actions=action.tolist(),
            shield_interventions=info.get("shield_interventions", 0),
        )

        if terminated or truncated:
            break

    env.close()
    summary = exp.close(status="COMPLETED")
    return summary
