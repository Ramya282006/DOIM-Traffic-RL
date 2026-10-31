import gymnasium as gym
from gymnasium import spaces
from environment.doim_processor import (
    DOIMProcessor,
)
from objectives.doim_reward import (
    DOIMReward,
)
import numpy as np
import traci
import shutil

from environment.sumo_locator import find_sumo_binary
from environment.safety_shield import TrafficSafetyShield
from environment.traffic_signal_controller import (
    TrafficSignalController,
)


class DOIMTrafficEnv(gym.Env):

    metadata = {
        "render_modes": ["human"]
    }

    def __init__(self, sumo_config, enforce_safety_shield=True, max_steps=1800):

        super().__init__()

        self.sumo_config = sumo_config
        self.enforce_safety_shield = bool(enforce_safety_shield)
        self.max_steps = int(max_steps)
        self.safety_shield = None

        # --------------------------------------------------
        # Find SUMO robustly
        # --------------------------------------------------

        self.sumo_binary = find_sumo_binary(prefer_gui=False)

        # --------------------------------------------------
        # Traffic signal controller
        #
        # Created after SUMO starts because TraCI
        # is required to inspect the TLS programs.
        # --------------------------------------------------

        self.signal_controller = None

        self.controllable_tls = []

        # --------------------------------------------------
        # Action space
        #
        # One binary decision per controllable TLS:
        #
        # 0 = keep current phase
        # 1 = request next phase
        #
        # Five controllable intersections:
        #
        # A1, B0, B1, B2, C1
        # --------------------------------------------------

        self.action_space = spaces.MultiDiscrete(
            [2, 2, 2, 2, 2]
        )

        # --------------------------------------------------
        # Temporary observation
        #
        # We will replace this with the
        # GNN/GRU/DOIM representation later.
        # --------------------------------------------------

        self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(69,),
        dtype=np.float32,
    )

        self.current_time = 0
        self.doim_processor = None
        self.reward_engine = DOIMReward()    # ======================================================
        self.latest_doim_result = None
  # RESET
    # ======================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )

        # --------------------------------------------------
        # Close previous SUMO instance
        # --------------------------------------------------

        try:

            traci.close()

        except Exception:

            pass

        self.current_time = 0

        # --------------------------------------------------
        # Start SUMO
        # --------------------------------------------------

        command = [
            self.sumo_binary,
            "-c",
            self.sumo_config,
            "--start",
            "--quit-on-end",
            "--seed",
            str(seed if seed is not None else 42),
        ]

        traci.start(
            command
        )

        # --------------------------------------------------
        # Discover traffic lights
        # --------------------------------------------------

        tls_ids = sorted(
            traci.trafficlight.getIDList()
        )

        self.signal_controller = (
            TrafficSignalController(
                tls_ids
            )
        )

        self.controllable_tls = (
            self.signal_controller
            .controllable_tls
        )
        # --------------------------------------------------
        # DOIM processor
        # --------------------------------------------------

        self.doim_processor = DOIMProcessor(
            tls_ids
        )

        self.doim_processor.reset()
        self.latest_doim_result = None

        # --------------------------------------------------
        # Safety check
        # --------------------------------------------------

        if len(
            self.controllable_tls
        ) != 5:

            raise RuntimeError(
                "Expected 5 controllable "
                "traffic lights, found "
                f"{len(self.controllable_tls)}: "
                f"{self.controllable_tls}"
            )

        # --------------------------------------------------
        # Initialize Safety Shield
        # --------------------------------------------------

        if self.enforce_safety_shield:
            phase_counts = {}
            for tid in self.controllable_tls:
                logics = traci.trafficlight.getAllProgramLogics(tid)
                phase_counts[tid] = len(logics[0].phases) if logics else 4
            self.safety_shield = TrafficSafetyShield(
                controllable_tls_ids=self.controllable_tls,
                tls_phase_counts=phase_counts,
                min_green=10,
                max_green=60,
                yellow_duration=3,
            )

        # --------------------------------------------------
        # Initial observation
        # --------------------------------------------------

        observation = (
            self._get_observation()
        )

        vehicle_count = traci.vehicle.getIDCount()
        info = self._build_info(vehicle_count, interventions=0)

        return (
            observation,
            info,
        )

    def _build_info(self, vehicle_count, interventions=0):
        feature_matrix = self.latest_doim_result.get("feature_matrix", np.zeros((9, 5), dtype=np.float32))
        interaction_matrix = self.latest_doim_result.get("interaction_matrix", np.zeros((5, 5), dtype=np.float32))
        
        # Raw state: extract features of controllable intersections
        tls_id_list = sorted(self.doim_processor.tls_ids) if self.doim_processor else []
        controllable_idx = [tls_id_list.index(t) for t in self.controllable_tls if t in tls_id_list]
        if controllable_idx and len(feature_matrix) > max(controllable_idx):
            raw_state = feature_matrix[controllable_idx].flatten()
        else:
            raw_state = np.zeros(25, dtype=np.float32)

        return {
            "time": self.current_time,
            "shield_interventions": interventions,
            "vehicle_count": int(vehicle_count),
            "controllable_tls": self.controllable_tls,
            "current_phases": self.signal_controller.get_current_phases(),
            "raw_objectives": self.latest_doim_result["raw_objectives"],
            "normalized_objectives": self.latest_doim_result["normalized_objectives"],
            "doim_weights": self.latest_doim_result["weights"],
            "feature_matrix": feature_matrix,
            "node_features": feature_matrix,
            "interaction_matrix": interaction_matrix,
            "raw_state": raw_state,
            "objective_contributions": self.reward_engine.get_contributions(
                self.latest_doim_result["normalized_objectives"],
                self.latest_doim_result["weights"],
            ),
        }

    # ======================================================
    # STEP
    # ======================================================

    def step(
        self,
        action,
    ):

        # --------------------------------------------------
        # Validate action
        # --------------------------------------------------

        action = np.asarray(
            action,
            dtype=np.int64,
        )

        if action.shape != (5,):

            raise ValueError(
                "Expected action shape "
                "(5,), got "
                f"{action.shape}"
            )

        if np.any(
            (action < 0)
            | (action > 1)
        ):

            raise ValueError(
                "Actions must contain "
                "only 0 or 1."
            )

        # --------------------------------------------------
        # Apply traffic signal action with Safety Shield
        # --------------------------------------------------

        interventions = 0
        current_phases = self.signal_controller.get_current_phases()

        if self.enforce_safety_shield and self.safety_shield is not None:
            safe_action, target_phases, interventions = self.safety_shield.filter_action(
                action, current_phases
            )
            for tid, target_phase in zip(self.controllable_tls, target_phases):
                if target_phase is not None:
                    traci.trafficlight.setPhase(tid, target_phase)
        else:
            self.signal_controller.apply_action(
                action
            )

        # --------------------------------------------------
        # Advance SUMO
        # --------------------------------------------------

        traci.simulationStep()

        self.current_time = (
            traci.simulation.getTime()
        )

        # --------------------------------------------------
        # Observation
        # --------------------------------------------------

        observation = (
            self._get_observation()
        )

        # --------------------------------------------------
        # Temporary reward
        #
        # We are NOT using this as the final
        # research reward yet.
        # --------------------------------------------------

        # --------------------------------------------------
        # DOIM adaptive reward
        # --------------------------------------------------

        normalized_objectives = (
            self.latest_doim_result[
                "normalized_objectives"
            ]
        )

        weights = (
            self.latest_doim_result[
                "weights"
            ]
        )

        reward = self.reward_engine.compute(
            normalized_objectives,
            weights,
        )

        vehicle_count = (
            traci.vehicle.getIDCount()
        )

        # --------------------------------------------------
        # Episode termination
        # --------------------------------------------------

        simulation_end = (
            traci.simulation
            .getMinExpectedNumber()
            == 0
        )

        terminated = simulation_end
        truncated = (self.current_time >= self.max_steps)

        # --------------------------------------------------
        # Information
        # --------------------------------------------------

        info = self._build_info(vehicle_count, interventions=interventions)

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # ======================================================
    # OBSERVATION
    # ======================================================

    def _get_observation(self):

        completed_vehicles = (
            traci.simulation.getArrivedNumber()
        )

        self.latest_doim_result = (
            self.doim_processor.process(
                completed_vehicles
            )
        )

        manifold = self.latest_doim_result[
            "manifold"
        ]

        weights = self.latest_doim_result[
            "weights"
        ]

        observation = np.concatenate(
            [
                manifold,
                weights,
            ]
        ).astype(
            np.float32
        )

        return observation    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):

        try:

            traci.close()

        except Exception:

            pass