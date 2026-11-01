"""
DOIM-Traffic-RL: Rigorous Approach-Level Delay Fairness Metrics
Implements mathematically rigorous equity metrics across competing arterial corridors:
  - Primary Metric: Approach Delay Disparity (1.0 - Jain's Fairness Index in [0, 1], dimensionless, lower is better)
  - Secondary Metric: Delay Variance (in seconds squared s^2, lower is better)
  - Gini disparity coefficient in [0, 1]
"""

from typing import Dict, List, Optional
import numpy as np


class CorridorFairnessCalculator:
    """
    Computes mathematically rigorous equity metrics across competing arterial corridors:
      - North-South incoming approaches
      - East-West incoming approaches
      - Turn movements

    METRIC DEFINITIONS:
    1. Primary Metric: Delay Disparity
       - Formula: D(x) = 1.0 - J(x) = 1.0 - [ (∑ x_i)^2 / (K * ∑ x_i^2) ]
       - Domain: Dimensionless, bounded in [0.0, 1.0 - 1/K] ⊂ [0.0, 1.0]
       - Interpretation: 0.0 represents absolute equity (all approaches experience identical delay).
                         Values close to 1.0 indicate severe starvation of one or more corridors.
       - Direction of Improvement: Lower is better (minimization objective).

    2. Secondary Metric: Delay Variance
       - Formula: Var(x) = (1/K) * ∑ (x_i - x̄)^2
       - Domain: Seconds squared (s^2), bounded in [0.0, ∞)
       - Interpretation: Classical spread of waiting times around the mean.
       - Direction of Improvement: Lower is better (minimization objective).
       - Note: Must NEVER be mathematically averaged or conflated with dimensionless Delay Disparity.
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
        # If all delays are zero or negligible, service is perfectly equal
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
    def compute_delay_disparity(delays: List[float]) -> float:
        """
        Primary Fairness Metric: Delay Disparity in [0, 1].
            Disparity = 1.0 - JainIndex
        0.0 = perfect equity (desirable).
        1.0 = maximum starvation / inequity (undesirable).
        Direction: Minimization (lower is better).
        Unit: Dimensionless.
        """
        jain = CorridorFairnessCalculator.compute_jain_index(delays)
        return float(1.0 - jain)

    @staticmethod
    def compute_delay_variance(delays: List[float], ddof: int = 0) -> float:
        """
        Secondary Fairness Metric: Delay Variance in seconds squared (s^2).
            Var = (1 / (N - ddof)) * ∑ (x_i - mean)^2
        Direction: Minimization (lower is better).
        Unit: s^2 (seconds squared).
        """
        arr = np.asarray(delays, dtype=np.float64)
        if len(arr) <= ddof:
            return 0.0
        return float(np.var(arr, ddof=ddof))

    @staticmethod
    def compute_intersection_fairness(intersection_delays: List[float]) -> Dict[str, float]:
        """
        Computes intersection-level equity metrics:
          - Jain's Fairness Index (dimensionless in [1/K, 1])
          - Disparity penalty (1.0 - Jain in [0, 1])
          - Delay variance (in s^2)
          - Gini coefficient (dimensionless in [0, 1])
        """
        jain = CorridorFairnessCalculator.compute_jain_index(intersection_delays)
        disparity = float(1.0 - jain)
        variance = CorridorFairnessCalculator.compute_delay_variance(intersection_delays)
        gini = CorridorFairnessCalculator.compute_gini_coefficient(intersection_delays)
        return {
            "jain_index": float(jain),
            "disparity": disparity,
            "variance_s2": variance,
            "gini": float(gini),
        }

    @staticmethod
    def compute_vehicle_fairness_disparity(vehicle_waits: List[float]) -> float:
        """
        Computes vehicle-level delay disparity across all active vehicles in the network:
            Disparity_veh = 1.0 - JainIndex(vehicle_waiting_times)
        """
        if not vehicle_waits:
            return 0.0
        return float(1.0 - CorridorFairnessCalculator.compute_jain_index(vehicle_waits))

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

