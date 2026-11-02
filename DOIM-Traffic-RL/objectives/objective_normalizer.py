"""
DOIM-Traffic-RL: Stable Objective Normalizer
Normalizes the 5 traffic objectives into urgency values in [0.0, 1.0].
Coupled directly with AuthoritativeObjectiveConfig.
"""

from typing import Optional, Tuple, Union
import numpy as np
from objectives.objective_config import AuthoritativeObjectiveConfig


class ObjectiveNormalizer:
    """
    Stable normalization of the five traffic objectives.

    Objective order:
        0. Waiting (minimize: higher raw value -> higher urgency)
        1. Queue (minimize: higher raw value -> higher urgency)
        2. CO2 (minimize: higher raw value -> higher urgency)
        3. Throughput (maximize: lower raw value -> higher urgency)
        4. Fairness (minimize disparity: higher disparity -> higher urgency)

    Output convention:
        0 = low urgency / desirable
        1 = high urgency / undesirable
    """

    def __init__(
        self,
        lower_bounds: Optional[Union[np.ndarray, list]] = None,
        upper_bounds: Optional[Union[np.ndarray, list]] = None,
        config: Optional[AuthoritativeObjectiveConfig] = None,
    ):
        if config is not None:
            self.lower_bounds = config.lower_bounds.copy()
            self.upper_bounds = config.upper_bounds.copy()
            self.config = config
        else:
            if lower_bounds is None:
                lower_bounds = np.array(
                    [
                        0.0,       # waiting
                        0.0,       # queue
                        0.0,       # CO2
                        0.0,       # throughput
                        0.0,       # fairness disparity
                    ],
                    dtype=np.float32,
                )

            if upper_bounds is None:
                upper_bounds = np.array(
                    [
                        600.0,      # waiting
                        20.0,       # queue
                        120000.0,   # CO2
                        20.0,       # throughput
                        1.0,        # fairness disparity in [0, 1]
                    ],
                    dtype=np.float32,
                )

            self.lower_bounds = np.asarray(lower_bounds, dtype=np.float32)
            self.upper_bounds = np.asarray(upper_bounds, dtype=np.float32)
            self.config = None

        if self.lower_bounds.shape != (5,):
            raise ValueError("lower_bounds must contain 5 values")
        if self.upper_bounds.shape != (5,):
            raise ValueError("upper_bounds must contain 5 values")
        if np.any(self.upper_bounds <= self.lower_bounds):
            raise ValueError("Each upper bound must be strictly greater than its lower bound")

    def normalize(self, objectives: Union[np.ndarray, list]) -> np.ndarray:
        """
        Normalizes raw objective vector into urgency values in [0.0, 1.0].
        """
        objs = np.asarray(objectives, dtype=np.float32)
        if objs.shape != (5,):
            raise ValueError(f"Expected exactly 5 objectives, got {objs.shape}")
        if not np.isfinite(objs).all():
            raise ValueError("objectives vector contains non-finite values")

        clipped = np.clip(objs, self.lower_bounds, self.upper_bounds)
        scaled = (clipped - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)

        urgency = scaled.copy()
        # Throughput inversion: higher throughput = lower urgency
        urgency[3] = 1.0 - scaled[3]

        return urgency.astype(np.float32)

    def get_ranges(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.lower_bounds.copy(), self.upper_bounds.copy()
