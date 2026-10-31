import torch

from models.doim import DOIM


def main():

    print("=" * 80)
    print("DOIM OBJECTIVE-INTERACTION SENSITIVITY TEST")
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
    # Keep objective state CONSTANT
    # --------------------------------------------------

    objective_vector = torch.tensor(
        [
            [
                0.70,   # waiting
                0.75,   # queue
                0.60,   # CO2
                0.30,   # throughput
                0.70,   # fairness
            ]
        ],
        dtype=torch.float32
    )

    # --------------------------------------------------
    # Keep traffic representation CONSTANT
    # --------------------------------------------------

    traffic_embedding = torch.zeros(
        1,
        64
    )

    # ==================================================
    # Interaction regime 1
    # Strong congestion coupling
    # ==================================================

    interaction_1 = torch.tensor(
        [
            [
                1.00,  0.90,  0.80, -0.60,  0.85
            ],
            [
                0.90,  1.00,  0.85, -0.70,  0.90
            ],
            [
                0.80,  0.85,  1.00, -0.50,  0.70
            ],
            [
               -0.60, -0.70, -0.50,  1.00, -0.40
            ],
            [
                0.85,  0.90,  0.70, -0.40,  1.00
            ],
        ],
        dtype=torch.float32
    )

    # ==================================================
    # Interaction regime 2
    # Objective conflict
    # ==================================================

    interaction_2 = torch.tensor(
        [
            [
                1.00, -0.40,  0.20,  0.60, -0.30
            ],
            [
               -0.40,  1.00, -0.60,  0.70,  0.20
            ],
            [
                0.20, -0.60,  1.00,  0.50, -0.70
            ],
            [
                0.60,  0.70,  0.50,  1.00,  0.40
            ],
            [
               -0.30,  0.20, -0.70,  0.40,  1.00
            ],
        ],
        dtype=torch.float32
    )

    # ==================================================
    # Interaction regime 3
    # Weak interactions
    # ==================================================

    interaction_3 = torch.tensor(
        [
            [
                1.00, 0.05, -0.02, 0.10, 0.04
            ],
            [
                0.05, 1.00, 0.03, -0.06, 0.08
            ],
            [
               -0.02, 0.03, 1.00, 0.02, -0.04
            ],
            [
                0.10, -0.06, 0.02, 1.00, 0.05
            ],
            [
                0.04, 0.08, -0.04, 0.05, 1.00
            ],
        ],
        dtype=torch.float32
    )

    interactions = [
        (
            "STRONG_COUPLING",
            interaction_1
        ),
        (
            "OBJECTIVE_CONFLICT",
            interaction_2
        ),
        (
            "WEAK_INTERACTION",
            interaction_3
        ),
    ]

    results = {}

    # ==================================================
    # Evaluate
    # ==================================================

    with torch.no_grad():

        for name, interaction in interactions:

            manifold, weights = model(
                traffic_embedding,
                objective_vector,
                interaction.unsqueeze(0)
            )

            results[name] = {
                "manifold": manifold,
                "weights": weights,
            }

            print("\n" + "-" * 80)

            print(name)

            print(
                "Weights:",
                weights[0].numpy()
            )

            print(
                "Weight sum:",
                weights.sum().item()
            )

    # ==================================================
    # Compare manifold representations
    # ==================================================

    print("\n" + "=" * 80)
    print("INTERACTION-INDUCED MANIFOLD DIFFERENCES")
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

    # ==================================================
    # Final message
    # ==================================================

    print("\n" + "=" * 80)

    print(
        "INTERACTION SENSITIVITY TEST COMPLETED"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()