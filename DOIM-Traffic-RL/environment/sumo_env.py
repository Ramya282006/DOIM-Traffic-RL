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

from environment.traffic_signal_controller import (
    TrafficSignalController,
)


class DOIMTrafficEnv(gym.Env):

    metadata = {
        "render_modes": ["human"]
    }

    def __init__(self, sumo_config):

        super().__init__()

        self.sumo_config = sumo_config

        # --------------------------------------------------
        # Find SUMO
        # --------------------------------------------------

        self.sumo_binary = shutil.which(
            "sumo"
        )

        if self.sumo_binary is None:

            raise RuntimeError(
                "SUMO was not found in PATH."
            )

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
        # Initial observation
        # --------------------------------------------------

        observation = (
            self._get_observation()
        )

        vehicle_count = traci.vehicle.getIDCount()

        info = {
            "time": self.current_time,

            "vehicle_count": int(
                vehicle_count
            ),

            "controllable_tls":
                self.controllable_tls,

            "current_phases":
                self.signal_controller.get_current_phases(),

            "raw_objectives":
                self.latest_doim_result[
                    "raw_objectives"
                ],

            "normalized_objectives":
                self.latest_doim_result[
                    "normalized_objectives"
                ],

            "doim_weights":
                self.latest_doim_result[
                    "weights"
                ],

            "objective_contributions":
                self.reward_engine.get_contributions(
                    self.latest_doim_result[
                        "normalized_objectives"
                    ],
                    self.latest_doim_result[
                        "weights"
                    ],
                ),
        }

        return (
            observation,
            info,
        )

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

        truncated = False

        # --------------------------------------------------
        # Information
        # --------------------------------------------------

        info = {
            "time": self.current_time,

            "vehicle_count": int(
                vehicle_count
            ),

            "controllable_tls":
                self.controllable_tls,

            "current_phases":
                self.signal_controller.get_current_phases(),

            "raw_objectives":
                self.latest_doim_result[
                    "raw_objectives"
                ],

            "normalized_objectives":
                self.latest_doim_result[
                    "normalized_objectives"
                ],

            "doim_weights":
                self.latest_doim_result[
                    "weights"
                ],

            "objective_contributions":
                self.reward_engine.get_contributions(
                    self.latest_doim_result[
                        "normalized_objectives"
                    ],
                    self.latest_doim_result[
                        "weights"
                    ],
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