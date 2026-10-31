import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv


class SpatialTrafficEncoder(nn.Module):

    """
    Graph-based spatial encoder for the 3x3 traffic network.

    Input:
        [num_intersections, 5]

    Features:
        [queue, waiting, speed, density, CO2]

    Output:
        [num_intersections, hidden_dim]
    """

    def __init__(
        self,
        input_dim=5,
        hidden_dim=64,
        output_dim=64,
    ):

        super().__init__()

        self.gcn1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.gcn2 = GCNConv(
            hidden_dim,
            output_dim
        )

        self.activation = nn.ReLU()

    def forward(
        self,
        x,
        edge_index,
    ):

        # ----------------------------------------------
        # First message-passing layer
        # ----------------------------------------------

        x = self.gcn1(
            x,
            edge_index
        )

        x = self.activation(x)

        # ----------------------------------------------
        # Second message-passing layer
        # ----------------------------------------------

        x = self.gcn2(
            x,
            edge_index
        )

        return x


def create_grid_edge_index():

    """
    Creates the physical topology of the 3x3 SUMO grid.

    Node indexing:

        0  1  2
        3  4  5
        6  7  8

    Corresponding traffic-light IDs:

        A0 A1 A2
        B0 B1 B2
        C0 C1 C2
    """

    edges = []

    # ----------------------------------------------
    # Horizontal connections
    # ----------------------------------------------

    horizontal = [
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 5),
        (6, 7),
        (7, 8),
    ]

    # ----------------------------------------------
    # Vertical connections
    # ----------------------------------------------

    vertical = [
        (0, 3),
        (3, 6),
        (1, 4),
        (4, 7),
        (2, 5),
        (5, 8),
    ]

    # ----------------------------------------------
    # Make graph undirected
    # ----------------------------------------------

    for source, target in (
        horizontal + vertical
    ):

        edges.append(
            [source, target]
        )

        edges.append(
            [target, source]
        )

    return torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()