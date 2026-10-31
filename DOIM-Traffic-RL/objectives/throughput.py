"""
DOIM-Traffic-RL: Throughput Metric Module
Tracks completed vehicle discharge rates and cumulative network arrivals.
"""

from typing import Tuple


class ThroughputTracker:
    """
    Maintains step-wise and cumulative vehicle completion statistics.
    """

    def __init__(self):
        self.cumulative_completed = 0
        self.step_history = []

    def update(self, arrived_in_step: int) -> int:
        """
        Records arrived vehicles from traci.simulation.getArrivedNumber().
        """
        count = max(0, int(arrived_in_step))
        self.cumulative_completed += count
        self.step_history.append(count)
        return count

    def get_completion_rate_per_hour(self, current_sim_time_sec: float) -> float:
        """
        Extrapolates throughput to vehicles completed per hour.
        """
        if current_sim_time_sec <= 0:
            return 0.0
        return float((self.cumulative_completed / current_sim_time_sec) * 3600.0)
