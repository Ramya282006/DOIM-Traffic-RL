import torch

from models.doim import DOIM


def main():

    print("=" * 80)
    print("DOIM SENSITIVITY TEST")
    print("=" * 80)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = DOIM(
        traffic_dim=64,
        num_objectives=5,
        interaction_dim=32,
        manifold_dim=64,
    )

    model.eval()

    # --------------------------------------------------
    # Four traffic regimes
    # --------------------------------------------------

    regimes = {

        "NORMAL": [
            0.10,  # waiting
            0.10,  # queue
            0.15,  # CO2
            0.90,  # throughput
            0.10,  # fairness disparity
        ],

        "SEVERE_CONGESTION": [
            0.90,
            0.95,
            0.80,
            0.15,
            0.85,
        ],

        "EMISSION_CRITICAL": [
            0.40,
            0.40,
            0.95,
            0.50,
            0.30,
        ],

        "THROUGHPUT_CRITICAL": [
            0.75,
            0.70,
            0.40,
            0.10,
            0.50,
        ],
    }

    # --------------------------------------------------
    # Use one representative interaction matrix
    # --------------------------------------------------

    interaction = torch.tensor(
        [
            [
                1.00, 0.80, 0.60, -0.40, 0.70
            ],
            [
                0.80, 1.00, 0.70, -0.50, 0.85
            ],
            [
                0.60, 0.70, 1.00, -0.30, 0.50
            ],
            [
                -0.40, -0.50, -0.30, 1.00, -0.20
            ],
            [
                0.70, 0.85, 0.50, -0.20, 1.00
            ],
        ],
        dtype=torch.float32
    )

    interaction = interaction.unsqueeze(0)

    # --------------------------------------------------
    # Same traffic embedding initially
    # --------------------------------------------------

    traffic_embedding = torch.zeros(
        1,
        64
    )

    # --------------------------------------------------
    # Evaluate regimes
    # --------------------------------------------------

    results = {}

    with torch.no_grad():

        for name, values in regimes.items():

            objective_vector = torch.tensor(
                [values],
                dtype=torch.float32
            )

            manifold, weights = model(
                traffic_embedding,
                objective_vector,
                interaction
            )

            results[name] = {
                "manifold": manifold,
                "weights": weights,
            }

            print("\n" + "-" * 80)

            print(name)

            print(
                "Objectives:",
                values
            )

            print(
                "Weights:",
                weights[0].numpy()
            )

            print(
                "Weight sum:",
                weights.sum().item()
            )

    # --------------------------------------------------
    # Compare latent representations
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("MANIFOLD DIFFERENCE ANALYSIS")
    print("=" * 80)

    names = list(results.keys())

    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            a = results[names[i]]["manifold"]

            b = results[names[j]]["manifold"]

            distance = torch.norm(
                a - b
            ).item()

            print(
                f"{names[i]:25s}"
                f" ↔ "
                f"{names[j]:25s}"
                f": {distance:.4f}"
            )

    # --------------------------------------------------
    # Final validation
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print(
        "SENSITIVITY TEST COMPLETED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()