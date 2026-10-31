from collections import deque

import numpy as np
import torch

from environment.state_extractor import (
    TrafficStateExtractor,
)

from objectives.objective_engine import (
    MultiObjectiveEngine,
)

from objectives.objective_normalizer import (
    ObjectiveNormalizer,
)

from analysis.objective_interaction import (
    ObjectiveInteractionAnalyzer,
)

from models.spatial_encoder import (
    SpatialTrafficEncoder,
    create_grid_edge_index,
)

from models.temporal_encoder import (
    TemporalTrafficEncoder,
)

from models.doim import DOIM


class DOIMProcessor:

    """
    Integrates:

        traffic state
            ↓
        GNN
            ↓
        GRU
            ↓
        objectives
            ↓
        normalization
            ↓
        interaction analysis
            ↓
        DOIM
            ↓
        adaptive weights
    """

    def __init__(
        self,
        tls_ids,
        history_length=10,
    ):

        self.tls_ids = list(tls_ids)

        # --------------------------------------------------
        # State extraction
        # --------------------------------------------------

        self.state_extractor = (
            TrafficStateExtractor(
                self.tls_ids
            )
        )

        # --------------------------------------------------
        # Objectives
        # --------------------------------------------------

        self.objective_engine = (
            MultiObjectiveEngine()
        )

        self.normalizer = (
            ObjectiveNormalizer()
        )

        self.interaction_analyzer = (
            ObjectiveInteractionAnalyzer(
                num_objectives=5,
                window_size=30,
            )
        )

        # --------------------------------------------------
        # Spatial encoder
        # --------------------------------------------------

        self.spatial_encoder = (
            SpatialTrafficEncoder(
                input_dim=5,
                hidden_dim=64,
                output_dim=64,
            )
        )

        # --------------------------------------------------
        # Temporal encoder
        # --------------------------------------------------

        self.temporal_encoder = (
            TemporalTrafficEncoder(
                spatial_dim=64,
                hidden_dim=64,
            )
        )

        # --------------------------------------------------
        # DOIM
        # --------------------------------------------------

        self.doim = DOIM(
            traffic_dim=64,
            num_objectives=5,
            interaction_dim=32,
            manifold_dim=64,
        )

        # --------------------------------------------------
        # Temporal history
        # --------------------------------------------------

        self.history = deque(
            maxlen=history_length
        )

        # --------------------------------------------------
        # Network graph
        # --------------------------------------------------

        self.edge_index = (
            create_grid_edge_index()
        )

        # --------------------------------------------------
        # Evaluation mode for now
        #
        # Training will be enabled later.
        # --------------------------------------------------

        self.spatial_encoder.eval()
        self.temporal_encoder.eval()
        self.doim.eval()

    # ======================================================
    # RESET INTERNAL STATE
    # ======================================================

    def reset(self):

        self.history.clear()

        self.objective_engine = (
            MultiObjectiveEngine()
        )

        self.interaction_analyzer = (
            ObjectiveInteractionAnalyzer(
                num_objectives=5,
                window_size=30,
            )
        )

    # ======================================================
    # PROCESS CURRENT TRAFFIC STATE
    # ======================================================

    def process(
        self,
        completed_vehicles,
    ):

        # --------------------------------------------------
        # Extract network state
        # --------------------------------------------------

        network_states = (
            self.state_extractor
            .get_network_state()
        )

        feature_matrix = (
            self.state_extractor
            .get_feature_matrix()
        )

        # --------------------------------------------------
        # Calculate objectives
        # --------------------------------------------------

        raw_objectives = (
            self.objective_engine
            .compute_objectives(
                network_states,
                completed_vehicles,
            )
        )

        normalized_objectives = (
            self.normalizer.normalize(
                raw_objectives
            )
        )

        # --------------------------------------------------
        # Interaction matrix
        # --------------------------------------------------

        self.interaction_analyzer.update(
            normalized_objectives
        )

        interaction_matrix = (
            self.interaction_analyzer
            .get_interaction_matrix()
        )

        # --------------------------------------------------
        # GNN
        # --------------------------------------------------

        x = torch.tensor(
            feature_matrix,
            dtype=torch.float32,
        )

        with torch.no_grad():

            spatial_embeddings = (
                self.spatial_encoder(
                    x,
                    self.edge_index,
                )
            )

        # --------------------------------------------------
        # Aggregate network representation
        # --------------------------------------------------

        network_embedding = (
            spatial_embeddings.mean(
                dim=0
            )
        )

        # --------------------------------------------------
        # Temporal history
        # --------------------------------------------------

        self.history.append(
            network_embedding
        )

        # --------------------------------------------------
        # GRU requires history
        # --------------------------------------------------

        sequence = torch.stack(
            list(self.history)
        ).unsqueeze(0)

        with torch.no_grad():

            temporal_embedding = (
                self.temporal_encoder(
                    sequence
                )
            )

        # --------------------------------------------------
        # DOIM input tensors
        # --------------------------------------------------

        objective_tensor = torch.tensor(
            normalized_objectives,
            dtype=torch.float32,
        ).unsqueeze(0)

        interaction_tensor = torch.tensor(
            interaction_matrix,
            dtype=torch.float32,
        ).unsqueeze(0)

        # --------------------------------------------------
        # DOIM
        # --------------------------------------------------

        with torch.no_grad():

            manifold, weights = (
                self.doim(
                    temporal_embedding,
                    objective_tensor,
                    interaction_tensor,
                )
            )

        # --------------------------------------------------
        # Return complete representation
        # --------------------------------------------------

        return {

            "feature_matrix":
                feature_matrix,

            "raw_objectives":
                raw_objectives,

            "normalized_objectives":
                normalized_objectives,

            "interaction_matrix":
                interaction_matrix,

            "traffic_embedding":
                temporal_embedding[
                    0
                ].cpu().numpy(),

            "manifold":
                manifold[
                    0
                ].cpu().numpy(),

            "weights":
                weights[
                    0
                ].cpu().numpy(),
        }