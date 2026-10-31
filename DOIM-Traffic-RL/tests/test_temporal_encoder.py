import torch

from models.temporal_encoder import (
    TemporalTrafficEncoder,
)


def main():

    print("=" * 80)
    print("TEMPORAL TRAFFIC GRU TEST")
    print("=" * 80)

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    batch_size = 2
    sequence_length = 10
    spatial_dim = 64

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = TemporalTrafficEncoder(
        spatial_dim=64,
        hidden_dim=64,
        num_layers=1,
    )

    model.eval()

    # --------------------------------------------------
    # Synthetic spatial sequence
    #
    # 2 scenarios
    # 10 timesteps
    # 64-dimensional spatial representation
    # --------------------------------------------------

    x = torch.randn(
        batch_size,
        sequence_length,
        spatial_dim
    )

    print("\nInput shape:")
    print(x.shape)

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    with torch.no_grad():

        temporal_embedding = model(x)

    print("\nTemporal embedding shape:")
    print(
        temporal_embedding.shape
    )

    # --------------------------------------------------
    # Validate
    # --------------------------------------------------

    assert temporal_embedding.shape == (
        batch_size,
        64
    )

    assert torch.isfinite(
        temporal_embedding
    ).all()

    # --------------------------------------------------
    # Test temporal sensitivity
    # --------------------------------------------------

    x_changed = x.clone()

    # Modify only the final timestep
    x_changed[:, -1, :] += 5.0

    with torch.no_grad():

        embedding_changed = model(
            x_changed
        )

    difference = torch.norm(
        temporal_embedding
        - embedding_changed
    ).item()

    print(
        "\nDifference after changing "
        "final timestep:"
    )

    print(
        f"{difference:.6f}"
    )

    assert difference > 0.0

    # --------------------------------------------------
    # Final
    # --------------------------------------------------

    print("\n" + "=" * 80)

    print(
        "TEMPORAL GRU TEST PASSED"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()