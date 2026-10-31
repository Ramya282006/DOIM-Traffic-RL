import numpy as np


class DOIMReward:

    """
    Adaptive multi-objective reward using DOIM weights.

    Objective order:

        0. Waiting
        1. Queue
        2. CO2
        3. Throughput
        4. Fairness

    Both the normalized objectives and DOIM weights
    are assumed to be in the same objective order.

    The normalizer represents every objective as
    an urgency/cost:

        0 = desirable
        1 = undesirable

    Therefore:

        reward = - weighted urgency
    """

    def __init__(self):

        self.num_objectives = 5

    # ======================================================
    # COMPUTE REWARD
    # ======================================================

    def compute(
        self,
        normalized_objectives,
        weights,
    ):

        urgency = np.asarray(
            normalized_objectives,
            dtype=np.float32,
        )

        weights = np.asarray(
            weights,
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Validate shapes
        # --------------------------------------------------

        if urgency.shape != (5,):

            raise ValueError(
                "normalized_objectives "
                "must have shape (5,)"
            )

        if weights.shape != (5,):

            raise ValueError(
                "weights must have shape (5,)"
            )

        # --------------------------------------------------
        # Validate finite values
        # --------------------------------------------------

        if not np.isfinite(
            urgency
        ).all():

            raise ValueError(
                "normalized_objectives "
                "contains non-finite values"
            )

        if not np.isfinite(
            weights
        ).all():

            raise ValueError(
                "weights contains "
                "non-finite values"
            )

        # --------------------------------------------------
        # Normalize weights defensively
        # --------------------------------------------------

        weight_sum = np.sum(
            weights
        )

        if weight_sum <= 0:

            raise ValueError(
                "Weight sum must be positive"
            )

        weights = (
            weights / weight_sum
        )

        # --------------------------------------------------
        # Adaptive weighted objective cost
        # --------------------------------------------------

        weighted_urgency = (
            weights * urgency
        )

        reward = -float(
            np.sum(
                weighted_urgency
            )
        )

        return reward

    # ======================================================
    # GET OBJECTIVE CONTRIBUTIONS
    # ======================================================

    def get_contributions(
        self,
        normalized_objectives,
        weights,
    ):

        urgency = np.asarray(
            normalized_objectives,
            dtype=np.float32,
        )

        weights = np.asarray(
            weights,
            dtype=np.float32,
        )

        weights = (
            weights
            / np.sum(weights)
        )

        return (
            weights * urgency
        ).astype(
            np.float32
        )