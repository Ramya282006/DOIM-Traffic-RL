import gymnasium as gym
from gymnasium import spaces

from environment.doim_processor import DOIMProcessor
from objectives.doim_reward import DOIMReward
from environment.traffic_signal_controller import TrafficSignalController

import numpy as np
import traci
import shutil


class FixedWeightTrafficEnv(gym.Env):

    metadata = {
        "render_modes": ["human"]
    }

    def __init__(
        self,
        sumo_config,
    ):

        super().__init__()

        self.sumo_config = sumo_config

        # --------------------------------------------------
        # Find SUMO
        # --------------------------------------------------

        self.sumo_binary = shutil.which("sumo")

        if self.sumo_binary is None:
            raise RuntimeError(
                "SUMO was not found in PATH."
            )

        # --------------------------------------------------
        # Fixed objective weights
        #
        # Waiting
        # Queue
        # CO2
        # Throughput
        # Fairness
        # --------------------------------------------------

        self.fixed_weights = np.array(
            [
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
            ],
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Action space
        # --------------------------------------------------

        self.action_space = spaces.MultiDiscrete(
            [2, 2, 2, 2, 2]
        )

        # --------------------------------------------------
        # Observation space
        #
        # 64-D manifold + 5-D weights
        # --------------------------------------------------

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(69,),
            dtype=np.float32,
        )

        self.current_time = 0

        self.signal_controller = None
        self.controllable_tls = []

        self.doim_processor = None
        self.reward_engine = DOIMReward()

        self.latest_doim_result = None

    # ======================================================
    # RESET
    # ======================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        super().reset(seed=seed)

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
        ]

        traci.start(command)

        # --------------------------------------------------
        # Traffic lights
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
            self.signal_controller.controllable_tls
        )

        # --------------------------------------------------
        # DOIM processor
        #
        # Used to generate the traffic representation
        # and objective information.
        # --------------------------------------------------

        self.doim_processor = DOIMProcessor(
            tls_ids
        )

        self.doim_processor.reset()

        # --------------------------------------------------
        # Initial observation
        # --------------------------------------------------

        observation = self._get_observation()

        vehicle_count = (
            traci.vehicle.getIDCount()
        )

        info = {
            "time": self.current_time,

            "vehicle_count":
                int(vehicle_count),

            "controllable_tls":
                self.controllable_tls,

            "current_phases":
                self.signal_controller
                .get_current_phases(),

            "fixed_weights":
                self.fixed_weights.copy(),

            "raw_objectives":
                self.latest_doim_result[
                    "raw_objectives"
                ],

            "normalized_objectives":
                self.latest_doim_result[
                    "normalized_objectives"
                ],

            "objective_contributions":
                self.reward_engine
                .get_contributions(
                    self.latest_doim_result[
                        "normalized_objectives"
                    ],
                    self.fixed_weights,
                ),
        }

        return observation, info

    # ======================================================
    # STEP
    # ======================================================

    def step(
        self,
        action,
    ):

        action = np.asarray(
            action,
            dtype=np.int64,
        )

        if action.shape != (5,):

            raise ValueError(
                "Expected action shape "
                f"(5,), got {action.shape}"
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
        # Apply traffic signal action
        # --------------------------------------------------

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
        # Process DOIM representation
        # --------------------------------------------------

        observation = self._get_observation()

        normalized_objectives = (
            self.latest_doim_result[
                "normalized_objectives"
            ]
        )

        # --------------------------------------------------
        # IMPORTANT:
        #
        # Fixed baseline DOES NOT use adaptive DOIM
        # weights.
        #
        # It always uses:
        #
        # [0.2, 0.2, 0.2, 0.2, 0.2]
        # --------------------------------------------------

        reward = self.reward_engine.compute(
            normalized_objectives,
            self.fixed_weights,
        )

        vehicle_count = (
            traci.vehicle.getIDCount()
        )

        # --------------------------------------------------
        # Termination
        # --------------------------------------------------

        simulation_end = (
            traci.simulation
            .getMinExpectedNumber()
            == 0
        )

        terminated = simulation_end
        truncated = False

        # --------------------------------------------------
        # Information
        # --------------------------------------------------

        info = {

            "time":
                self.current_time,

            "vehicle_count":
                int(vehicle_count),

            "controllable_tls":
                self.controllable_tls,

            "current_phases":
                self.signal_controller
                .get_current_phases(),

            "raw_objectives":
                self.latest_doim_result[
                    "raw_objectives"
                ],

            "normalized_objectives":
                normalized_objectives,

            "fixed_weights":
                self.fixed_weights.copy(),

            "objective_contributions":
                self.reward_engine
                .get_contributions(
                    normalized_objectives,
                    self.fixed_weights,
                ),
        }

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
            traci.simulation
            .getArrivedNumber()
        )

        self.latest_doim_result = (
            self.doim_processor.process(
                completed_vehicles
            )
        )

        manifold = (
            self.latest_doim_result[
                "manifold"
            ]
        )

        # --------------------------------------------------
        # IMPORTANT:
        #
        # Observation contains fixed weights rather than
        # adaptive DOIM weights.
        # --------------------------------------------------

        observation = np.concatenate(
            [
                manifold,
                self.fixed_weights,
            ]
        ).astype(
            np.float32
        )

        return observation

    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):

        try:
            traci.close()
        except Exception:
            pass