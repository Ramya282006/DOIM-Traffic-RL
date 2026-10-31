import torch
import torch.nn as nn
import torch.nn.functional as F


class DOIM(nn.Module):
    """
    Dynamic Objective Interaction Manifold.

    Inputs
    ------
    traffic_embedding:
        Spatial-temporal traffic representation.
        Shape: [batch, traffic_dim]

    objective_vector:
        Current normalized objective state.
        Shape: [batch, num_objectives]

    interaction_matrix:
        Dynamic objective interaction matrix.
        Shape: [batch, num_objectives, num_objectives]

    Outputs
    -------
    manifold:
        Learned latent objective-interaction representation.
        Shape: [batch, manifold_dim]

    weights:
        Adaptive objective preference distribution.
        Shape: [batch, num_objectives]
    """

    def __init__(
        self,
        traffic_dim=64,
        num_objectives=5,
        interaction_dim=32,
        manifold_dim=64,
    ):

        super().__init__()

        self.num_objectives = num_objectives

        # --------------------------------------------------
        # Objective encoder
        # --------------------------------------------------

        self.objective_encoder = nn.Sequential(

            nn.Linear(
                num_objectives,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                32
            ),

            nn.ReLU(),
        )

        # --------------------------------------------------
        # Interaction encoder
        # --------------------------------------------------

        self.interaction_encoder = nn.Sequential(

            nn.Linear(
                num_objectives * num_objectives,
                interaction_dim
            ),

            nn.ReLU(),

            nn.Linear(
                interaction_dim,
                interaction_dim
            ),

            nn.ReLU(),
        )

        # --------------------------------------------------
        # Manifold projection
        # --------------------------------------------------

        combined_dim = (
            traffic_dim
            + 32
            + interaction_dim
        )

        self.manifold_encoder = nn.Sequential(

            nn.Linear(
                combined_dim,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                manifold_dim
            ),

            nn.LayerNorm(
                manifold_dim
            ),

            nn.Tanh(),
        )

        # --------------------------------------------------
        # Adaptive preference head
        # --------------------------------------------------

        self.preference_head = nn.Sequential(

            nn.Linear(
                manifold_dim,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                num_objectives
            ),
        )

    def forward(
        self,
        traffic_embedding,
        objective_vector,
        interaction_matrix,
    ):

        # ==================================================
        # Encode current objective state
        # ==================================================

        objective_embedding = (
            self.objective_encoder(
                objective_vector
            )
        )

        # ==================================================
        # Encode objective interactions
        # ==================================================

        interaction_flat = (
            interaction_matrix
            .reshape(
                interaction_matrix.shape[0],
                -1
            )
        )

        interaction_embedding = (
            self.interaction_encoder(
                interaction_flat
            )
        )

        # ==================================================
        # Combine information
        # ==================================================

        combined = torch.cat(
            [
                traffic_embedding,
                objective_embedding,
                interaction_embedding,
            ],
            dim=-1,
        )

        # ==================================================
        # Dynamic Objective Interaction Manifold
        # ==================================================

        manifold = (
            self.manifold_encoder(
                combined
            )
        )

        # ==================================================
        # Adaptive objective preferences
        # ==================================================

        logits = (
            self.preference_head(
                manifold
            )
        )

        weights = F.softmax(
            logits,
            dim=-1
        )

        return manifold, weights