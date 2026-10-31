"""
DOIM-Traffic-RL: Waiting Time Metric Module
Accumulates vehicle-seconds of delay for all active vehicles on incoming lanes.
"""

from typing import List
import numpy as np


def compute_total_waiting(vehicle_waiting_times_sec: List[float]) -> float:
    """
    Computes total accumulated waiting time in vehicle-seconds.
    """
    if not vehicle_waiting_times_sec:
        return 0.0
    return float(np.sum(vehicle_waiting_times_sec))
