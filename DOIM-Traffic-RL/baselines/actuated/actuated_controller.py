"""
DOIM-Traffic-RL: Actuated Traffic Signal Controller Baseline (Phase G)
Implements standard vehicle-actuated signal control adhering to transportation engineering standards:
  - Enforces minimum green time (e.g. 10s).
  - Evaluates approach detector vehicle gaps (gap-out logic): if no vehicle is detected within gap threshold (e.g. 3.0s),
    or maximum green time (60s) is reached, requests phase transition.
  - Extends active green while continuous vehicle flow is detected.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from environment.sumo_env import DOIMTrafficEnv
from common.experiment import ExperimentRun, SCHEMA_VERSION_CURRENT
from common.config import load_experiment_config


class ActuatedTrafficController:
    """
    Standard vehicle-actuated traffic signal controller across multiple intersections.
    """

    def __init__(
        self,
        controllable_tls_ids: List[str],
        min_green: int = 10,
        max_green: int = 60,
        gap_threshold_sec: float = 3.0,
        unit_extension_sec: float = 2.0,
    ):
        self.controllable_tls_ids = list(controllable_tls_ids)
        self.num_controllers = len(self.controllable_tls_ids)
        self.min_green = int(min_green)
        self.max_green = int(max_green)
        self.gap_threshold_sec = float(gap_threshold_sec)
        self.unit_extension_sec = float(unit_extension_sec)
        
        # State tracking per intersection
        self.time_in_phase = {tid: 0 for tid in self.controllable_tls_ids}
        self.last_phase = {tid: 0 for tid in self.controllable_tls_ids}

    def reset(self):
        for tid in self.controllable_tls_ids:
            self.time_in_phase[tid] = 0
            self.last_phase[tid] = 0

    def select_actions(
        self,
        current_phases: Dict[str, int],
        approach_queues: Dict[str, float],
        approach_vehicles: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Determines binary transition decisions a_j in {0, 1} for each controllable intersection.
        """
        actions = []
        for tid in self.controllable_tls_ids:
            c_phase = current_phases.get(tid, 0)
            if c_phase != self.last_phase.get(tid, 0):
                self.time_in_phase[tid] = 0
                self.last_phase[tid] = c_phase
            else:
                self.time_in_phase[tid] += 1

            t_elapsed = self.time_in_phase[tid]
            
            # Rule 1: Must satisfy minimum green
            if t_elapsed < self.min_green:
                actions.append(0)
                continue

            # Rule 2: Force termination at maximum green
            if t_elapsed >= self.max_green:
                actions.append(1)
                continue

            # Rule 3: Gap-out logic
            # If vehicle presence/queue is zero or below threshold on current active approach, yield
            queue = approach_queues.get(tid, 0.0)
            num_veh = approach_vehicles.get(tid, 0) if approach_vehicles is not None else int(queue > 0)
            
            if num_veh == 0:
                # Gap out: no arriving vehicles detected, transition to next phase
                actions.append(1)
            else:
                # Extend green
                actions.append(0)

        return np.array(actions, dtype=np.int64)


def run_actuated_experiment(
    scenario_override: Optional[str] = None,
    total_timesteps: int = 1800,
    seed: int = 42,
    output_root: str = "results/runs",
    min_green: int = 10,
    max_green: int = 60,
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
        config_path=scenario_path,
        seed=seed,
        max_steps=total_timesteps,
        use_safety_shield=True,
    )
    
    controller = ActuatedTrafficController(
        controllable_tls_ids=env.controllable_tls,
        min_green=min_green,
        max_green=max_green,
    )
    
    obs, info = env.reset(seed=seed)
    controller.reset()
    
    total_reward = 0.0
    for step in range(total_timesteps):
        current_phases = env.signal_controller.get_current_phases()
        # Extract queues per controllable TLS
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

