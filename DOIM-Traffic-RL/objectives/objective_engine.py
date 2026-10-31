"""
DOIM-Traffic-RL: Unified Multi-Objective Engine
Aggregates and computes the 5 core traffic objectives:
  1. Total Waiting Time (vehicle-seconds)
  2. Total Queue Length (halting vehicles)
  3. Integrated CO2 Emissions (milligrams)
  4. Step Throughput (completed vehicles)
  5. Corridor Delay Fairness (Jain's disparity index)
"""

from typing import Dict, List, Optional
import numpy as np

from objectives.fairness import CorridorFairnessCalculator
from objectives.emissions import compute_step_co2
from objectives.queue import compute_total_queue
from objectives.waiting_time import compute_total_waiting
from objectives.throughput import ThroughputTracker


class MultiObjectiveEngine:
    """
    Computes the 5-dimensional objective vector from current traffic simulation states.
    """

    def __init__(self, fairness_mode: str = "jain_disparity"):
        self.fairness_mode = str(fairness_mode)
        self.throughput_tracker = ThroughputTracker()
        self.fairness_calculator = CorridorFairnessCalculator()

    @property
    def cumulative_completed(self) -> int:
        return self.throughput_tracker.cumulative_completed

    def compute_objectives(
        self,
        network_states: Dict[str, Dict[str, float]],
        completed_vehicles: int,
    ) -> np.ndarray:
        """
        Computes the 5-objective vector from network intersection states.
        
        Args:
            network_states: Dictionary mapping TLS ID -> state metrics
            completed_vehicles: Number of vehicles arriving at destination this step
            
        Returns:
            np.ndarray of shape (5,) with float32 values:
                [waiting, queue, co2, throughput, fairness]
        """
        queues = []
        waiting = []
        emissions = []

        for tls_id, state in network_states.items():
            queues.append(state.get("queue", 0.0))
            waiting.append(state.get("waiting", 0.0))
            emissions.append(state.get("co2", 0.0))

        # Objective 1: Total waiting time (veh·s)
        total_waiting = compute_total_waiting(waiting)

        # Objective 2: Total queue length (vehicles)
        total_queue = compute_total_queue(queues)

        # Objective 3: Total instantaneous CO2 (mg)
        total_co2 = compute_step_co2(emissions, step_length_sec=1.0)

        # Objective 4: Step throughput (completed vehicles)
        step_throughput = float(self.throughput_tracker.update(completed_vehicles))

        # Objective 5: Fairness Disparity (1.0 - Jain Index across intersections in [0, 1])
        if self.fairness_mode == "jain_disparity":
            fairness_val = float(CorridorFairnessCalculator.compute_delay_disparity(waiting))
        else:
            # Fallback to normalized variance
            fairness_val = float(np.var(waiting) if waiting else 0.0)

        return np.array(
            [
                total_waiting,
                total_queue,
                total_co2,
                step_throughput,
                fairness_val,
            ],
            dtype=np.float32,
        )