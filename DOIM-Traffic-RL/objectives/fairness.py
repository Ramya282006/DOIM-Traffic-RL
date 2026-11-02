"""
DOIM-Traffic-RL: Rigorous Multi-Tier Delay Fairness Metrics
Implements mathematically rigorous equity metrics across three distinct hierarchical tiers:
  1. Approach/Corridor-Level Fairness: equity across competing arterial approaches (N-S vs E-W)
  2. Intersection-Level Fairness: equity across the 9 network junctions
  3. Vehicle-Level Fairness: equity across individual vehicle delay distributions

METRIC SUITE:
  - Primary Metric: Delay Disparity (1.0 - Jain's Fairness Index in [0, 1], dimensionless, lower is better)
  - Jain's Fairness Index (J(x) in [1/K, 1], dimensionless, higher is better)
  - Worst-Approach Delay (in seconds, lower is better)
  - Max-to-Min Delay Disparity Ratio (max(x) / (min(x) + eps))
  - Starvation Duration (consecutive steps with halting vehicles but no green phase)
  - Gini Inequality Coefficient in [0, 1] (dimensionless, lower is better)
  - Delay Variance in seconds squared (s^2, lower is better)
"""

from typing import Dict, List, Optional, Union
import numpy as np


class CorridorFairnessCalculator:
    """
    Computes mathematically rigorous equity metrics across competing arterial corridors,
    intersections, and individual vehicles.
    """

    def __init__(self, num_corridors: int = 4):
        self.num_corridors = int(num_corridors)

    @staticmethod
    def compute_jain_index(delays: Union[List[float], np.ndarray]) -> float:
        """
        Computes Jain's Fairness Index:
            J(x) = (∑ x_i)^2 / (K * ∑ x_i^2)
        Normalized in [1/K, 1.0], where 1.0 indicates perfectly equitable service.
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
    def compute_delay_disparity(delays: Union[List[float], np.ndarray]) -> float:
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
    def compute_delay_variance(delays: Union[List[float], np.ndarray], ddof: int = 0) -> float:
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
    def compute_gini_coefficient(delays: Union[List[float], np.ndarray]) -> float:
        """
        Computes the Gini coefficient of inequality across delay distributions.
        0.0 = absolute equality, 1.0 = absolute inequality.
        """
        arr = np.asarray(delays, dtype=np.float64)
        if len(arr) <= 1 or np.all(arr <= 1e-6):
            return 0.0
        arr = np.sort(arr)
        n = len(arr)
        index = np.arange(1, n + 1)
        sum_arr = np.sum(arr)
        if sum_arr <= 1e-8:
            return 0.0
        gini = (2.0 * np.sum(index * arr) - (n + 1) * sum_arr) / (n * sum_arr)
        return float(np.clip(gini, 0.0, 1.0))

    @staticmethod
    def compute_worst_approach_delay(approach_delays: Union[List[float], np.ndarray]) -> float:
        """
        Returns the maximum delay experienced by any single approach / corridor:
            WorstDelay = max(x_i)
        Unit: seconds.
        """
        arr = np.asarray(approach_delays, dtype=np.float64)
        if len(arr) == 0:
            return 0.0
        return float(np.max(arr))

    @staticmethod
    def compute_max_to_min_disparity_ratio(delays: Union[List[float], np.ndarray], eps: float = 1e-4) -> float:
        """
        Computes the ratio between maximum and minimum observed delays:
            Ratio = max(x_i) / (min(x_i) + eps)
        Unit: Dimensionless ratio.
        """
        arr = np.asarray(delays, dtype=np.float64)
        if len(arr) == 0:
            return 1.0
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        return float(max_val / (min_val + eps))

    @staticmethod
    def compute_approach_fairness(approach_delays: Union[List[float], np.ndarray]) -> Dict[str, float]:
        """
        Computes corridor/approach-level equity metrics:
          - Jain Index
          - Delay Disparity (1 - Jain)
          - Worst Approach Delay
          - Max-to-Min Disparity Ratio
          - Delay Variance (s^2)
          - Gini Coefficient
        """
        arr = np.asarray(approach_delays, dtype=np.float64)
        jain = CorridorFairnessCalculator.compute_jain_index(arr)
        return {
            "tier": "approach_corridor",
            "jain_index": float(jain),
            "delay_disparity": float(1.0 - jain),
            "worst_approach_delay_s": CorridorFairnessCalculator.compute_worst_approach_delay(arr),
            "max_to_min_ratio": CorridorFairnessCalculator.compute_max_to_min_disparity_ratio(arr),
            "variance_s2": CorridorFairnessCalculator.compute_delay_variance(arr),
            "gini": CorridorFairnessCalculator.compute_gini_coefficient(arr),
        }

    @staticmethod
    def compute_intersection_fairness(intersection_delays: Union[List[float], np.ndarray]) -> Dict[str, float]:
        """
        Computes intersection-level equity metrics across the 9 network junctions.
        """
        arr = np.asarray(intersection_delays, dtype=np.float64)
        jain = CorridorFairnessCalculator.compute_jain_index(arr)
        return {
            "tier": "intersection",
            "jain_index": float(jain),
            "delay_disparity": float(1.0 - jain),
            "worst_intersection_delay_s": float(np.max(arr)) if len(arr) > 0 else 0.0,
            "max_to_min_ratio": CorridorFairnessCalculator.compute_max_to_min_disparity_ratio(arr),
            "variance_s2": CorridorFairnessCalculator.compute_delay_variance(arr),
            "gini": CorridorFairnessCalculator.compute_gini_coefficient(arr),
        }

    @staticmethod
    def compute_vehicle_fairness_disparity(vehicle_waits: Union[List[float], np.ndarray]) -> float:
        """
        Computes vehicle-level delay disparity across individual vehicles:
            Disparity_veh = 1.0 - JainIndex(vehicle_waiting_times)
        """
        if len(vehicle_waits) == 0:
            return 0.0
        return float(1.0 - CorridorFairnessCalculator.compute_jain_index(vehicle_waits))

    @staticmethod
    def compute_vehicle_fairness(vehicle_waits: Union[List[float], np.ndarray]) -> Dict[str, float]:
        """
        Computes vehicle-level equity metrics across all active vehicles in the network.
        """
        arr = np.asarray(vehicle_waits, dtype=np.float64)
        jain = CorridorFairnessCalculator.compute_jain_index(arr)
        return {
            "tier": "vehicle",
            "jain_index": float(jain),
            "delay_disparity": float(1.0 - jain),
            "max_vehicle_delay_s": float(np.max(arr)) if len(arr) > 0 else 0.0,
            "p95_vehicle_delay_s": float(np.percentile(arr, 95)) if len(arr) > 0 else 0.0,
            "variance_s2": CorridorFairnessCalculator.compute_delay_variance(arr),
            "gini": CorridorFairnessCalculator.compute_gini_coefficient(arr),
        }
