"""
DOIM-Traffic-RL: Queue Metric Module
Computes halting vehicles and queue spatial occupancy across incoming approaches.
"""

from typing import List
import numpy as np


def compute_total_queue(halting_counts_per_lane: List[int]) -> float:
    """
    Computes aggregate count of halting vehicles (speed < 0.1 m/s) on incoming lanes.
    """
    if not halting_counts_per_lane:
        return 0.0
    return float(np.sum(halting_counts_per_lane))
