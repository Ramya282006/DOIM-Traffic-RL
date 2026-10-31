import numpy as np


class ObjectiveInteractionAnalyzer:

    """
    Computes a rolling objective-interaction matrix.

    Objective order:
        0 -> Waiting
        1 -> Queue
        2 -> CO2
        3 -> Throughput
        4 -> Fairness

    The interaction matrix is estimated from a rolling
    history of normalized objective values.
    """

    def __init__(
        self,
        num_objectives=5,
        window_size=30,
    ):

        self.num_objectives = num_objectives
        self.window_size = window_size

        self.history = []

    # ==================================================
    # Add objective observation
    # ==================================================

    def update(self, objectives):

        objectives = np.asarray(
            objectives,
            dtype=np.float32
        )

        if objectives.shape != (
            self.num_objectives,
        ):
            raise ValueError(
                f"Expected {self.num_objectives} "
                f"objectives, got {objectives.shape}"
            )

        self.history.append(
            objectives.copy()
        )

        # Keep only rolling window
        if len(self.history) > self.window_size:

            self.history.pop(0)

    # ==================================================
    # Compute interaction matrix
    # ==================================================

    def get_interaction_matrix(self):

        # Not enough history
        if len(self.history) < 2:

            return np.eye(
                self.num_objectives,
                dtype=np.float32
            )

        data = np.asarray(
            self.history,
            dtype=np.float32
        )

        # Correlation matrix
        correlation = np.corrcoef(
            data,
            rowvar=False
        )

        # Numerical safety
        correlation = np.nan_to_num(
            correlation,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Force symmetry
        correlation = (
            correlation
            + correlation.T
        ) / 2.0

        # Force diagonal
        np.fill_diagonal(
            correlation,
            1.0
        )

        return correlation.astype(
            np.float32
        )

    # ==================================================
    # Get flattened interaction vector
    # ==================================================

    def get_interaction_vector(self):

        matrix = self.get_interaction_matrix()

        # Extract upper triangular elements
        indices = np.triu_indices(
            self.num_objectives,
            k=1
        )

        return matrix[
            indices
        ].astype(
            np.float32
        )

    # ==================================================
    # Reset history
    # ==================================================

    def reset(self):

        self.history.clear()