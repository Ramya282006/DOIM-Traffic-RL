import os
import sys
from collections import deque

import numpy as np
import torch
import traci

from environment.state_extractor import TrafficStateExtractor
from objectives.objective_engine import MultiObjectiveEngine
from objectives.objective_normalizer import ObjectiveNormalizer
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


# ============================================================
# SUMO CONFIGURATION
# ============================================================

SUMO_BINARY = "sumo"

SUMO_CONFIG = os.path.join(
    "scenarios",
    "directional_shift",
    "north_south_surge.sumocfg",
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("REAL SUMO → GNN → GRU → DOIM INTEGRATION TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Start SUMO
    # --------------------------------------------------------

    traci.start([
        SUMO_BINARY,
        "-c",
        SUMO_CONFIG,
        "--seed",
        "42",
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ])

    try:

        # ====================================================
        # Discover traffic lights
        # ====================================================

        tls_ids = sorted(
            traci.trafficlight.getIDList()
        )

        print("\nTraffic lights:")
        print(tls_ids)

        assert len(tls_ids) == 9

        # ====================================================
        # Create modules
        # ====================================================

        state_extractor = (
            TrafficStateExtractor(
                tls_ids
            )
        )

        objective_engine = (
            MultiObjectiveEngine()
        )

        normalizer = (
            ObjectiveNormalizer()
        )

        interaction_analyzer = (
            ObjectiveInteractionAnalyzer(
                num_objectives=5,
                window_size=30,
            )
        )

        spatial_encoder = (
            SpatialTrafficEncoder(
                input_dim=5,
                hidden_dim=64,
                output_dim=64,
            )
        )

        temporal_encoder = (
            TemporalTrafficEncoder(
                spatial_dim=64,
                hidden_dim=64,
                num_layers=1,
            )
        )

        doim = DOIM(
            traffic_dim=64,
            num_objectives=5,
            interaction_dim=32,
            manifold_dim=64,
        )

        spatial_encoder.eval()
        temporal_encoder.eval()
        doim.eval()

        edge_index = (
            create_grid_edge_index()
        )

        # ====================================================
        # Temporal buffer
        # ====================================================

        temporal_buffer = deque(
            maxlen=10
        )

        # ====================================================
        # Tracking
        # ====================================================

        completed_vehicles = 0

        integration_steps = 0

        # ====================================================
        # Simulation
        # ====================================================

        for step in range(120):

            traci.simulationStep()

            integration_steps += 1

            # ------------------------------------------------
            # Traffic state
            # ------------------------------------------------

            network_states = (
                state_extractor
                .get_network_state()
            )

            feature_matrix = (
                state_extractor
                .get_feature_matrix()
            )

            # ------------------------------------------------
            # Objective computation
            # ------------------------------------------------

            completed_vehicles = (
                traci.simulation
                .getArrivedNumber()
            )

            raw_objectives = (
                objective_engine
                .compute_objectives(
                    network_states,
                    completed_vehicles,
                )
            )

            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            normalized_objectives = (
                normalizer
                .normalize(
                    raw_objectives
                )
            )

            # ------------------------------------------------
            # Interaction analysis
            # ------------------------------------------------

            interaction_analyzer.update(
                normalized_objectives
            )

            interaction_matrix = (
                interaction_analyzer
                .get_interaction_matrix()
            )

            # ------------------------------------------------
            # GNN
            # ------------------------------------------------

            x = torch.tensor(
                feature_matrix,
                dtype=torch.float32,
            )

            with torch.no_grad():

                spatial_embeddings = (
                    spatial_encoder(
                        x,
                        edge_index,
                    )
                )

            # ------------------------------------------------
            # Spatial aggregation
            # ------------------------------------------------

            network_embedding = (
                spatial_embeddings
                .mean(dim=0)
            )

            # ------------------------------------------------
            # Temporal buffer
            # ------------------------------------------------

            temporal_buffer.append(
                network_embedding
            )

            # ------------------------------------------------
            # Need enough temporal history
            # ------------------------------------------------

            if len(temporal_buffer) < 2:

                continue

            sequence = torch.stack(
                list(temporal_buffer)
            )

            sequence = sequence.unsqueeze(
                0
            )

            # ------------------------------------------------
            # GRU
            # ------------------------------------------------

            with torch.no_grad():

                temporal_embedding = (
                    temporal_encoder(
                        sequence
                    )
                )

            # ------------------------------------------------
            # Prepare DOIM inputs
            # ------------------------------------------------

            objective_tensor = (
                torch.tensor(
                    normalized_objectives,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
            )

            interaction_tensor = (
                torch.tensor(
                    interaction_matrix,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
            )

            # ------------------------------------------------
            # DOIM
            # ------------------------------------------------

            with torch.no_grad():

                manifold, weights = (
                    doim(
                        temporal_embedding,
                        objective_tensor,
                        interaction_tensor,
                    )
                )

            # ------------------------------------------------
            # Periodic reporting
            # ------------------------------------------------

            if step % 20 == 0:

                print("\n" + "-" * 80)

                print(
                    f"Simulation time: "
                    f"{traci.simulation.getTime():.1f}s"
                )

                print(
                    "Feature matrix:",
                    feature_matrix.shape
                )

                print(
                    "Raw objectives:",
                    np.round(
                        raw_objectives,
                        2
                    )
                )

                print(
                    "Normalized objectives:",
                    np.round(
                        normalized_objectives,
                        3
                    )
                )

                print(
                    "Interaction matrix:"
                )

                print(
                    np.round(
                        interaction_matrix,
                        2
                    )
                )

                print(
                    "Temporal embedding:",
                    temporal_embedding.shape
                )

                print(
                    "DOIM manifold:",
                    manifold.shape
                )

                print(
                    "Adaptive weights:",
                    np.round(
                        weights[0]
                        .cpu()
                        .numpy(),
                        3
                    )
                )

                print(
                    "Weight sum:",
                    weights.sum().item()
                )

        # ====================================================
        # Validation
        # ====================================================

        assert integration_steps == 120

        assert temporal_embedding.shape == (
            1,
            64
        )

        assert manifold.shape == (
            1,
            64
        )

        assert weights.shape == (
            1,
            5
        )

        assert np.isfinite(
            normalized_objectives
        ).all()

        assert np.isfinite(
            interaction_matrix
        ).all()

        assert torch.isfinite(
            manifold
        ).all()

        assert torch.isfinite(
            weights
        ).all()

        assert abs(
            weights.sum().item() - 1.0
        ) < 1e-5

        print("\n" + "=" * 80)
        print(
            "REAL SUMO → GNN → GRU → DOIM"
        )
        print(
            "INTEGRATION TEST PASSED"
        )
        print("=" * 80)

    finally:

        traci.close()


if __name__ == "__main__":
    main()