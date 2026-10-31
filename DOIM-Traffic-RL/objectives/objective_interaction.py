import numpy as np


class ObjectiveInteractionAnalyzer:

    def __init__(self, window_size=30):

        self.window_size = window_size

        self.history = []

    # ======================================================
    # Add objective observation
    # ======================================================

    def update(self, objective_vector):

        objective_vector = np.asarray(
            objective_vector,
            dtype=np.float32,
        )

        if objective_vector.shape != (5,):

            raise ValueError(
                "Expected objective vector of shape (5,)"
            )

        self.history.append(objective_vector)

        # Keep only recent observations
        if len(self.history) > self.window_size:

            self.history.pop(0)

    # ======================================================
    # Compute interaction matrix
    # ======================================================

    def get_interaction_matrix(self):

        # Not enough observations
        if len(self.history) < 2:

            return np.eye(5, dtype=np.float32)

        data = np.asarray(
            self.history,
            dtype=np.float32,
        )

        # Correlation between objective trajectories
        correlation = np.corrcoef(
            data.T
        )

        # Numerical protection
        correlation = np.nan_to_num(
            correlation,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Ensure diagonal represents self-interaction
        np.fill_diagonal(
            correlation,
            1.0,
        )

        return correlation.astype(
            np.float32
        )

    # ======================================================
    # Flatten interaction matrix
    # ======================================================

    def get_interaction_vector(self):

        matrix = self.get_interaction_matrix()

        # Use upper triangular part without diagonal
        indices = np.triu_indices(
            5,
            k=1,
        )

        return matrix[indices]

    # ======================================================
    # Reset history
    # ======================================================

    def reset(self):

        self.history.clear()