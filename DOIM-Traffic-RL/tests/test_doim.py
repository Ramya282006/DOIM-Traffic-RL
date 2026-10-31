import torch

from models.doim import DOIM


def main():

    print("=" * 80)
    print("DOIM MODULE TEST")
    print("=" * 80)

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    batch_size = 4

    traffic_dim = 64

    num_objectives = 5

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = DOIM(
        traffic_dim=traffic_dim,
        num_objectives=num_objectives,
        interaction_dim=32,
        manifold_dim=64,
    )

    print("\nModel created successfully.")

    # --------------------------------------------------
    # Synthetic traffic representation
    # --------------------------------------------------

    traffic_embedding = torch.randn(
        batch_size,
        traffic_dim
    )

    # --------------------------------------------------
    # Synthetic objective vectors
    # --------------------------------------------------

    objective_vector = torch.rand(
        batch_size,
        num_objectives
    )

    # --------------------------------------------------
    # Synthetic interaction matrices
    # --------------------------------------------------

    interaction_matrix = torch.eye(
        num_objectives
    ).unsqueeze(0).repeat(
        batch_size,
        1,
        1
    )

    # Add small random interactions
    noise = torch.randn(
        batch_size,
        num_objectives,
        num_objectives
    ) * 0.3

    noise = (
        noise + noise.transpose(1, 2)
    ) / 2

    interaction_matrix = (
        interaction_matrix + noise
    )

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    manifold, weights = model(
        traffic_embedding,
        objective_vector,
        interaction_matrix,
    )

    # --------------------------------------------------
    # Print shapes
    # --------------------------------------------------

    print(
        "\nTraffic embedding shape:",
        traffic_embedding.shape
    )

    print(
        "Objective vector shape:",
        objective_vector.shape
    )

    print(
        "Interaction matrix shape:",
        interaction_matrix.shape
    )

    print(
        "Manifold shape:",
        manifold.shape
    )

    print(
        "Adaptive weights shape:",
        weights.shape
    )

    # --------------------------------------------------
    # Print values
    # --------------------------------------------------

    print("\nAdaptive objective weights:")

    print(weights.detach())

    # --------------------------------------------------
    # Validate weights
    # --------------------------------------------------

    weight_sums = weights.sum(
        dim=1
    )

    print(
        "\nWeight sums:"
    )

    print(
        weight_sums.detach()
    )

    assert torch.all(
        weights >= 0
    )

    assert torch.allclose(
        weight_sums,
        torch.ones(batch_size),
        atol=1e-5
    )

    # --------------------------------------------------
    # Validate manifold
    # --------------------------------------------------

    assert manifold.shape == (
        batch_size,
        64
    )

    # --------------------------------------------------
    # Test gradient flow
    # --------------------------------------------------

    loss = (
        manifold.mean()
        + weights.mean()
    )

    loss.backward()

    gradients_exist = any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    assert gradients_exist

    print(
        "\nGradient flow: VERIFIED"
    )

    print("\n" + "=" * 80)
    print(
        "DOIM MODULE TEST PASSED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()