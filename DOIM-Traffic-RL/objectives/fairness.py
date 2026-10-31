"""
DOIM-Traffic-RL: Rigorous Approach-Level Delay Fairness Metrics
Implements Jain's Fairness Index, Gini disparity, and corridor-level delay equity.
"""

from typing import Dict, List, Optional
import numpy as np


class CorridorFairnessCalculator:
    """
    Computes mathematically rigorous equity metrics across competing arterial corridors:
      - North-South incoming approaches
      - East-West incoming approaches
      - Turn movements
    """

    def __init__(self, num_corridors: int = 4):
        self.num_corridors = int(num_corridors)

    @staticmethod
    def compute_jain_index(delays: List[float]) -> float:
        """
        Computes Jain's Fairness Index:
            J(x) = (∑ x_i)^2 / (K * ∑ x_i^2)
        Normalized in [1/K, 1.0], where 1.0 is perfectly equitable service.
        """
        arr = np.asarray(delays, dtype=np.float64)
        if len(arr) == 0:
            return 1.0
        # If all delays are zero, service is perfectly equal
        if np.all(arr <= 1e-6):
            return 1.0
            
        sum_x = np.sum(arr)
        sum_sq = np.sum(arr ** 2)
        if sum_sq <= 1e-8:
            return 1.0
            
        k = len(arr)
        jain = (sum_x ** 2) / (k * sum_sq)
        return float(np.clip(jain, 1.0 / k, 1.0))

    @staticmethod
    def compute_delay_disparity(corridor_delays: List[float]) -> float:
        """
        Computes disparity penalty in [0, 1]:
            Disparity = 1.0 - JainIndex
        0.0 = perfect equity (desirable).
        1.0 = maximum starvation / inequity (undesirable).
        """
        jain = CorridorFairnessCalculator.compute_jain_index(corridor_delays)
        return float(1.0 - jain)

    @staticmethod
    def compute_gini_coefficient(delays: List[float]) -> float:
        """
        Computes the Gini coefficient of inequality across vehicle delays.
        0.0 = absolute equality, 1.0 = absolute inequality.
        """
        arr = np.asarray(delays, dtype=np.float64)
        if len(arr) <= 1 or np.all(arr <= 1e-6):
            return 0.0
        arr = np.sort(arr)
        n = len(arr)
        index = np.arange(1, n + 1)
        gini = (2.0 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))
        return float(np.clip(gini, 0.0, 1.0))
