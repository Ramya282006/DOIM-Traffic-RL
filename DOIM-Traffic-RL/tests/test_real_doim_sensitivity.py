import os
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
# CONFIGURATION
# ============================================================

SUMO_CONFIG = os.path.join(
    "scenarios",
    "directional_shift",
    "north_south_surge.sumocfg",
)

CHECKPOINTS = [
    30,
    60,
    90,
    120,
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("REAL TRAFFIC DOIM SENSITIVITY TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Start SUMO
    # --------------------------------------------------------

    traci.start([
        "sumo",
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
        # Modules
        # ====================================================

        tls_ids = sorted(
            traci.trafficlight.getIDList()
        )

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
        # Temporal history
        # ====================================================

        temporal_buffer = deque(
            maxlen=10
        )

        snapshots = {}

        # ====================================================
        # Simulation
        # ====================================================

        for step in range(121):

            traci.simulationStep()

            current_time = int(
                traci.simulation.getTime()
            )

            # ------------------------------------------------
            # State
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
            # Objectives
            # ------------------------------------------------

            completed = (
                traci.simulation
                .getArrivedNumber()
            )

            raw_objectives = (
                objective_engine
                .compute_objectives(
                    network_states,
                    completed,
                )
            )

            normalized_objectives = (
                normalizer.normalize(
                    raw_objectives
                )
            )

            # ------------------------------------------------
            # Interaction
            # ------------------------------------------------

            interaction_analyzer.update(
                normalized_objectives
            )

            interaction_matrix = (
                interaction_analyzer
                .get_interaction_matrix()
            )

            # ------------------------------------------------
            # Spatial encoder
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

            network_embedding = (
                spatial_embeddings
                .mean(dim=0)
            )

            temporal_buffer.append(
                network_embedding
            )

            if len(temporal_buffer) < 2:
                continue

            # ------------------------------------------------
            # Temporal encoder
            # ------------------------------------------------

            sequence = torch.stack(
                list(temporal_buffer)
            ).unsqueeze(0)

            with torch.no_grad():

                temporal_embedding = (
                    temporal_encoder(
                        sequence
                    )
                )

            # ------------------------------------------------
            # DOIM inputs
            # ------------------------------------------------

            objective_tensor = torch.tensor(
                normalized_objectives,
                dtype=torch.float32,
            ).unsqueeze(0)

            interaction_tensor = torch.tensor(
                interaction_matrix,
                dtype=torch.float32,
            ).unsqueeze(0)

            # ------------------------------------------------
            # DOIM
            # ------------------------------------------------

            with torch.no_grad():

                manifold, weights = doim(
                    temporal_embedding,
                    objective_tensor,
                    interaction_tensor,
                )

            # ------------------------------------------------
            # Save checkpoints
            # ------------------------------------------------

            if current_time in CHECKPOINTS:

                snapshots[current_time] = {
                    "raw": raw_objectives.copy(),
                    "normalized": normalized_objectives.copy(),
                    "interaction": interaction_matrix.copy(),
                    "embedding": temporal_embedding[
                        0
                    ].cpu().numpy().copy(),
                    "manifold": manifold[
                        0
                    ].cpu().numpy().copy(),
                    "weights": weights[
                        0
                    ].cpu().numpy().copy(),
                }

        # ====================================================
        # Display snapshots
        # ====================================================

        for time in CHECKPOINTS:

            if time not in snapshots:
                continue

            data = snapshots[time]

            print("\n" + "-" * 80)

            print(
                f"TRAFFIC STATE @ {time}s"
            )

            print("-" * 80)

            print(
                "Raw objectives:"
            )

            print(
                np.round(
                    data["raw"],
                    2
                )
            )

            print(
                "\nNormalized objectives:"
            )

            print(
                np.round(
                    data["normalized"],
                    3
                )
            )

            print(
                "\nAdaptive weights:"
            )

            print(
                np.round(
                    data["weights"],
                    4
                )
            )

            print(
                "Weight sum:",
                data["weights"].sum()
            )

    finally:

        traci.close()

    # ========================================================
    # Pairwise sensitivity
    # ========================================================

    print("\n")
    print("=" * 80)
    print("PAIRWISE DOIM SENSITIVITY")
    print("=" * 80)

    available = [
        t for t in CHECKPOINTS
        if t in snapshots
    ]

    for i in range(len(available)):

        for j in range(i + 1, len(available)):

            t1 = available[i]
            t2 = available[j]

            s1 = snapshots[t1]
            s2 = snapshots[t2]

            objective_difference = np.linalg.norm(
                s1["normalized"]
                - s2["normalized"]
            )

            manifold_difference = np.linalg.norm(
                s1["manifold"]
                - s2["manifold"]
            )

            weight_difference = np.linalg.norm(
                s1["weights"]
                - s2["weights"]
            )

            embedding_difference = np.linalg.norm(
                s1["embedding"]
                - s2["embedding"]
            )

            print(
                f"\n{t1}s ↔ {t2}s"
            )

            print(
                f"Objective difference : "
                f"{objective_difference:.4f}"
            )

            print(
                f"Traffic embedding Δ  : "
                f"{embedding_difference:.4f}"
            )

            print(
                f"DOIM manifold Δ       : "
                f"{manifold_difference:.4f}"
            )

            print(
                f"Weight vector Δ       : "
                f"{weight_difference:.4f}"
            )

    # ========================================================
    # Validation
    # ========================================================

    assert len(snapshots) >= 2

    for time, data in snapshots.items():

        assert data["embedding"].shape == (
            64,
        )

        assert data["manifold"].shape == (
            64,
        )

        assert data["weights"].shape == (
            5,
        )

        assert np.isfinite(
            data["embedding"]
        ).all()

        assert np.isfinite(
            data["manifold"]
        ).all()

        assert np.isfinite(
            data["weights"]
        ).all()

        assert abs(
            data["weights"].sum() - 1.0
        ) < 1e-5

    print("\n" + "=" * 80)
    print(
        "REAL DOIM SENSITIVITY TEST PASSED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()