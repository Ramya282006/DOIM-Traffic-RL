"""
DOIM-Traffic-RL: Independent Reference Validation Tests for GCN Invariants and Equivariance.
Tests:
  1. Node permutation equivariance: GCN(P X, P A P^T) == P * GCN(X, A)
  2. Edge index boundary validation and fail-closed error handling
  3. Graph with empty edge set (self-loop fallback)
  4. Graph with isolated nodes (numerical stability without NaN/Inf)
  5. Node-level spatial pooling invariance across 2D and 3D shapes
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from models.spatial_encoder import NativeGCNConv, SpatialTrafficEncoder, create_grid_edge_index
from models.temporal_encoder import aggregate_spatial_embeddings


class TestGCNInvariants:
    """Verifies mathematical invariants of the Graph Convolutional Network implementation."""

    def test_node_permutation_equivariance(self):
        """
        Permutation Equivariance Theorem:
        For any node permutation π with permutation matrix P:
            GCN(P X, P A P^T) == P · GCN(X, A)
        """
        torch.manual_seed(42)
        num_nodes = 9
        in_dim = 5
        out_dim = 16

        conv = NativeGCNConv(in_dim, out_dim, bias=True)
        conv.eval()

        # Random node features
        x = torch.randn(num_nodes, in_dim)
        edge_index = create_grid_edge_index()

        # Output on original graph
        with torch.no_grad():
            y_orig = conv(x, edge_index)

        # Generate arbitrary permutation of node indices
        perm = torch.randperm(num_nodes)
        p_matrix = torch.eye(num_nodes)[perm]  # [N, N]

        # Permute node features: X' = P · X
        # Node perm[i] is placed at position i
        x_perm = p_matrix @ x

        # The inverse permutation maps original node u to its new index in x_perm
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(num_nodes)

        # Permute edge index so edge (u, v) becomes (inv_perm[u], inv_perm[v])
        edge_index_perm = inv_perm[edge_index]

        with torch.no_grad():
            y_perm = conv(x_perm, edge_index_perm)

        # Check: y_perm must equal P · y_orig
        y_orig_permuted = p_matrix @ y_orig
        assert torch.allclose(y_perm, y_orig_permuted, atol=1e-5), \
            f"Max absolute discrepancy: {torch.max(torch.abs(y_perm - y_orig_permuted)).item()}"

    def test_edge_index_out_of_bounds_raises_value_error(self):
        """Edge indices >= num_nodes or < 0 must raise informative ValueError."""
        num_nodes = 9
        conv = NativeGCNConv(5, 16)
        x = torch.randn(num_nodes, 5)

        # Case 1: node index 9 exceeds valid [0, 8] range
        invalid_edge_high = torch.tensor([[0, 9], [1, 2]], dtype=torch.long)
        with pytest.raises(ValueError, match="edge_index contains indices out of bounds"):
            conv(x, invalid_edge_high)

        # Case 2: negative node index
        invalid_edge_neg = torch.tensor([[-1, 3], [0, 2]], dtype=torch.long)
        with pytest.raises(ValueError, match="edge_index contains indices out of bounds"):
            conv(x, invalid_edge_neg)

    def test_empty_edge_index_self_loop_fallback(self):
        """Graph with no edges (only isolated nodes) operates correctly via self-loops."""
        num_nodes = 9
        conv = NativeGCNConv(5, 16)
        conv.eval()
        x = torch.randn(num_nodes, 5)
        empty_edges = torch.zeros((2, 0), dtype=torch.long)

        with torch.no_grad():
            out = conv(x, empty_edges)

        assert out.shape == (num_nodes, 16)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        # With empty edges, A_tilde = I_N, D_tilde = I_N, so GCN(X) = X W + b
        expected = conv.linear(x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_isolated_nodes_numerical_stability(self):
        """Isolated nodes connected only by self-loops maintain finite, non-zero embeddings."""
        num_nodes = 9
        conv = NativeGCNConv(5, 16)
        x = torch.randn(num_nodes, 5)

        # Only connect nodes 0-1 and 1-2, leaving nodes 3..8 completely isolated
        sparse_edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        out = conv(x, sparse_edges)

        assert out.shape == (num_nodes, 16)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_spatial_pooling_dimension_invariance(self):
        """
        aggregate_spatial_embeddings pools across node dimension for both 2D [N, D]
        and 3D [B, N, D] inputs.
        """
        # 2D case: [9, 64] -> [64]
        spatial_2d = torch.randn(9, 64)
        pooled_2d = aggregate_spatial_embeddings(spatial_2d)
        assert pooled_2d.shape == (64,)
        assert torch.allclose(pooled_2d, spatial_2d.mean(dim=0))

        # 3D batched case: [4, 9, 64] -> [4, 64]
        spatial_3d = torch.randn(4, 9, 64)
        pooled_3d = aggregate_spatial_embeddings(spatial_3d)
        assert pooled_3d.shape == (4, 64)
        assert torch.allclose(pooled_3d, spatial_3d.mean(dim=1))
