import torch
import torch.nn as nn


class TemporalTrafficEncoder(nn.Module):

    """
    Temporal encoder for spatial traffic representations.

    Input:
        [batch, sequence_length, spatial_dim]

    Output:
        [batch, temporal_dim]

    The final hidden state represents the recent
    temporal evolution of the traffic network.
    """

    def __init__(
        self,
        spatial_dim=64,
        hidden_dim=64,
        num_layers=1,
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=spatial_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):

        # x:
        # [batch, sequence, spatial_dim]

        output, hidden = self.gru(x)

        # Final GRU hidden state
        temporal_embedding = hidden[-1]

        return temporal_embedding


def aggregate_spatial_embeddings(
    spatial_embeddings
):

    """
    Aggregate node-level spatial embeddings
    into one network-level representation.

    Input:
        [num_intersections, spatial_dim]

    Output:
        [spatial_dim]
    """

    return spatial_embeddings.mean(
        dim=0
    )