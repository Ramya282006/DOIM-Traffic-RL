import torch

from models.spatial_encoder import (
    SpatialTrafficEncoder,
    create_grid_edge_index,
)


def main():

    print("=" * 80)
    print("SPATIAL TRAFFIC GNN TEST")
    print("=" * 80)

    # --------------------------------------------------
    # Create encoder
    # --------------------------------------------------

    model = SpatialTrafficEncoder(
        input_dim=5,
        hidden_dim=64,
        output_dim=64,
    )

    model.eval()

    # --------------------------------------------------
    # Create 3x3 graph
    # --------------------------------------------------

    edge_index = create_grid_edge_index()

    print("\nEdge index shape:")
    print(edge_index.shape)

    print("\nNumber of directed edges:")
    print(edge_index.shape[1])

    # --------------------------------------------------
    # Synthetic traffic state
    #
    # 9 intersections × 5 features
    # --------------------------------------------------

    x = torch.tensor(
        [
            [1.0, 2.0, 13.0, 3.0, 100.0],  # A0
            [2.0, 3.0, 12.0, 4.0, 200.0],  # A1
            [1.0, 1.0, 14.0, 2.0, 100.0],  # A2

            [3.0, 4.0, 10.0, 5.0, 300.0],  # B0
            [8.0, 10.0, 5.0, 12.0, 900.0], # B1
            [2.0, 3.0, 11.0, 4.0, 200.0],  # B2

            [1.0, 1.0, 14.0, 2.0, 100.0],  # C0
            [2.0, 3.0, 12.0, 3.0, 150.0],  # C1
            [1.0, 2.0, 13.0, 3.0, 100.0],  # C2
        ],
        dtype=torch.float32
    )

    print("\nInput shape:")
    print(x.shape)

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    with torch.no_grad():

        embeddings = model(
            x,
            edge_index
        )

    print("\nOutput embedding shape:")
    print(embeddings.shape)

    # --------------------------------------------------
    # Inspect B1 embedding
    # --------------------------------------------------

    print("\nB1 spatial embedding:")
    print(
        embeddings[4]
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    assert x.shape == (
        9,
        5
    )

    assert edge_index.shape == (
        2,
        24
    )

    assert embeddings.shape == (
        9,
        64
    )

    assert torch.isfinite(
        embeddings
    ).all()

    print("\n" + "=" * 80)

    print(
        "SPATIAL GNN TEST PASSED"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()