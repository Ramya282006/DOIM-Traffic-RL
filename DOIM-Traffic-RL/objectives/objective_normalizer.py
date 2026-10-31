import numpy as np


class ObjectiveNormalizer:

    """
    Stable normalization of the five traffic objectives.

    Objective order:

        0. Waiting
        1. Queue
        2. CO2
        3. Throughput
        4. Fairness

    Output convention:

        0 = low urgency
        1 = high urgency

    Therefore:

        Waiting  : higher raw value -> higher urgency
        Queue    : higher raw value -> higher urgency
        CO2      : higher raw value -> higher urgency
        Throughput: lower raw value -> higher urgency
        Fairness : higher raw value -> higher urgency
    """

    def __init__(
        self,
        lower_bounds=None,
        upper_bounds=None,
    ):

        # --------------------------------------------------
        # Default calibration ranges
        # --------------------------------------------------

        if lower_bounds is None:

            lower_bounds = np.array(
                [
                    0.0,       # waiting
                    0.0,       # queue
                    0.0,       # CO2
                    0.0,       # throughput
                    0.0,       # fairness
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

        self.lower_bounds = np.asarray(
            lower_bounds,
            dtype=np.float32,
        )

        self.upper_bounds = np.asarray(
            upper_bounds,
            dtype=np.float32,
        )

        if self.lower_bounds.shape != (5,):

            raise ValueError(
                "lower_bounds must contain 5 values"
            )

        if self.upper_bounds.shape != (5,):

            raise ValueError(
                "upper_bounds must contain 5 values"
            )

        if np.any(
            self.upper_bounds
            <= self.lower_bounds
        ):

            raise ValueError(
                "Each upper bound must be greater "
                "than its lower bound"
            )

    # ======================================================
    # Normalize objectives
    # ======================================================

    def normalize(
        self,
        objectives,
    ):

        objectives = np.asarray(
            objectives,
            dtype=np.float32,
        )

        if objectives.shape != (5,):

            raise ValueError(
                "Expected exactly 5 objectives"
            )

        # --------------------------------------------------
        # Clip to calibrated range
        # --------------------------------------------------

        clipped = np.clip(
            objectives,
            self.lower_bounds,
            self.upper_bounds,
        )

        # --------------------------------------------------
        # Min-max scaling
        # --------------------------------------------------

        scaled = (
            clipped - self.lower_bounds
        ) / (
            self.upper_bounds
            - self.lower_bounds
        )

        # --------------------------------------------------
        # Objective directions
        #
        # First 3 + fairness:
        # higher raw value = worse
        #
        # Throughput:
        # higher raw value = better
        # --------------------------------------------------

        urgency = scaled.copy()

        # Throughput inversion
        urgency[3] = 1.0 - scaled[3]

        return urgency.astype(
            np.float32
        )

    # ======================================================
    # Get calibration ranges
    # ======================================================

    def get_ranges(self):

        return (
            self.lower_bounds.copy(),
            self.upper_bounds.copy(),
        )