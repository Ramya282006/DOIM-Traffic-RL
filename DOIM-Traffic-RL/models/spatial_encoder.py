import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv as PyGGCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class NativeGCNConv(nn.Module):
    """
    Kipf & Welling (ICLR 2017) Graph Convolutional Layer implemented in pure PyTorch.
    Computes:
        H^(l+1) = \tilde{D}^(-1/2) \tilde{A} \tilde{D}^(-1/2) H^(l) W^(l) + b
    where \tilde{A} = A + I_N is adjacency with added self-loops.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(-2)
        device = x.device

        # Adjacency matrix with self-loops
        adj = torch.eye(num_nodes, device=device, dtype=x.dtype)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            adj.index_put_((src, dst), torch.ones_like(src, dtype=x.dtype))
            adj.index_put_((dst, src), torch.ones_like(dst, dtype=x.dtype))

        # Degree matrix normalization \tilde{D}^(-1/2)
        deg = torch.sum(adj, dim=-1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)
        norm_adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(0)

        # Message passing and linear projection
        support = self.linear(x)
        return torch.matmul(norm_adj, support)


def get_gcn_layer(in_channels: int, out_channels: int):
    """
    Factory returning PyG GCNConv if installed, or native PyTorch GCNConv fallback.
    """
    if HAS_PYG:
        return PyGGCNConv(in_channels, out_channels)
    return NativeGCNConv(in_channels, out_channels)


class SpatialTrafficEncoder(nn.Module):
    """
    Graph-based spatial encoder for the traffic network.
    
    Input:
        [num_intersections, 5]
    Features:
        [queue, waiting, speed, density, CO2]
    Output:
        [num_intersections, output_dim]
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        output_dim: int = 64,
    ):
        super().__init__()

        self.gcn1 = get_gcn_layer(input_dim, hidden_dim)
        self.gcn2 = get_gcn_layer(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.gcn1(x, edge_index)
        x = self.activation(x)

        # Layer 2
        x = self.gcn2(x, edge_index)
        return x


def create_grid_edge_index() -> torch.Tensor:
    """
    Creates the physical topology of the 3x3 SUMO grid.
    
    Node indexing:
        0  1  2  (A0, A1, A2)
        3  4  5  (B0, B1, B2)
        6  7  8  (C0, C1, C2)
    """
    horizontal = [
        (0, 1), (1, 2),
        (3, 4), (4, 5),
        (6, 7), (7, 8),
    ]

    vertical = [
        (0, 3), (3, 6),
        (1, 4), (4, 7),
        (2, 5), (5, 8),
    ]

    edges = []
    for source, target in horizontal + vertical:
        edges.append([source, target])
        edges.append([target, source])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()