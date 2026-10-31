"""
DOIM-Traffic-RL: Unified Differentiable DOIM Architecture
Integrates Spatial GNN, Temporal GRU, Learned Objective Interaction Manifold,
Adaptive Preference Head, Actor-Critic Policy, and Auxiliary Transition Predictor.
All parameters receive gradients end-to-end.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spatial_encoder import get_gcn_layer, create_grid_edge_index


class DOIMNetwork(nn.Module):
    """
    End-to-end trainable Dynamic Objective Interaction Manifold (DOIM) Network.
    
    Data Flow:
      1. Spatial Traffic Features X [B, N, 5] -> Spatial GNN -> [B, N, 64] -> Mean Pool -> [B, 64]
      2. Temporal History [B, L, 64] -> Temporal GRU -> [B, 64]
      3. Normalized Objectives z [B, 5] -> Objective MLP -> [B, 32]
      4. Interaction Tensor M [B, 5, 5] -> Interaction MLP -> [B, 32]
      5. Fusion [traffic_temp(64), obj(32), int(32)] = 128 -> Manifold MLP -> Latent Manifold m [B, 64]
      6. m -> Preference Head -> Softmax(logits / tau) -> Adaptive Weights w [B, 5]
      7. [m, raw_state] -> Multi-Discrete Actor Policy (5 TLS binary logits)
      8. [m, raw_state] -> Critic Value Head V(s) [B, 1]
      9. [m, action] -> Auxiliary Predictor -> Predicted Next Objectives z_{t+1} [B, 5]
    """

    def __init__(
        self,
        num_intersections: int = 9,
        num_controllable_tls: int = 5,
        node_feature_dim: int = 5,
        num_objectives: int = 5,
        spatial_dim: int = 64,
        temporal_dim: int = 64,
        interaction_dim: int = 32,
        objective_dim: int = 32,
        manifold_dim: int = 64,
        raw_state_dim: int = 25, # 5 controllable TLS * 5 local features
        temperature: float = 1.0,
    ):
        super().__init__()

        self.num_intersections = num_intersections
        self.num_controllable_tls = num_controllable_tls
        self.num_objectives = num_objectives
        self.temperature = float(temperature)

        # ----------------------------------------------------------------------
        # 1. Spatial GNN Encoder (Kipf-Welling Message Passing)
        # ----------------------------------------------------------------------
        self.gcn1 = get_gcn_layer(node_feature_dim, spatial_dim)
        self.gcn2 = get_gcn_layer(spatial_dim, spatial_dim)
        self.spatial_activation = nn.ReLU()

        # ----------------------------------------------------------------------
        # 2. Temporal GRU Encoder
        # ----------------------------------------------------------------------
        self.gru = nn.GRU(
            input_size=spatial_dim,
            hidden_size=temporal_dim,
            num_layers=1,
            batch_first=True,
        )

        # ----------------------------------------------------------------------
        # 3. Objective & Interaction Latent Encoders
        # ----------------------------------------------------------------------
        self.objective_encoder = nn.Sequential(
            nn.Linear(num_objectives, objective_dim),
            nn.ReLU(),
            nn.Linear(objective_dim, objective_dim),
            nn.ReLU(),
        )

        self.interaction_encoder = nn.Sequential(
            nn.Linear(num_objectives * num_objectives, interaction_dim),
            nn.ReLU(),
            nn.Linear(interaction_dim, interaction_dim),
            nn.ReLU(),
        )

        # ----------------------------------------------------------------------
        # 4. Dynamic Objective Interaction Manifold Fusion Core
        # ----------------------------------------------------------------------
        fusion_dim = temporal_dim + objective_dim + interaction_dim
        self.manifold_core = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, manifold_dim),
            nn.LayerNorm(manifold_dim),
            nn.Tanh(),
        )

        # ----------------------------------------------------------------------
        # 5. Adaptive Objective Preference Head
        # ----------------------------------------------------------------------
        self.preference_head = nn.Sequential(
            nn.Linear(manifold_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_objectives),
        )

        # ----------------------------------------------------------------------
        # 6. Actor Policy Heads (Multi-Discrete for 5 controllable intersections)
        # ----------------------------------------------------------------------
        policy_input_dim = manifold_dim + raw_state_dim
        self.actor_backbone = nn.Sequential(
            nn.Linear(policy_input_dim, 64),
            nn.ReLU(),
        )
        self.action_heads = nn.ModuleList([
            nn.Linear(64, 2) for _ in range(num_controllable_tls)
        ])

        # ----------------------------------------------------------------------
        # 7. Critic Value Head
        # ----------------------------------------------------------------------
        self.critic_head = nn.Sequential(
            nn.Linear(policy_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # ----------------------------------------------------------------------
        # 8. Auxiliary Objective Transition Predictor (Counterfactual dynamics)
        # ----------------------------------------------------------------------
        aux_input_dim = manifold_dim + num_controllable_tls
        self.aux_predictor = nn.Sequential(
            nn.Linear(aux_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_objectives),
        )

        # Static grid edge index cache
        self.register_buffer("default_edge_index", create_grid_edge_index())

    def encode_spatial(
        self,
        node_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Processes node feature matrix with 2-layer GCN and pools across nodes.
        
        Args:
            node_features: [B, N, D_in] or [N, D_in]
            edge_index: [2, E] (uses default 3x3 grid graph if None)
            
        Returns:
            Pooled spatial embedding: [B, spatial_dim]
        """
        if edge_index is None:
            edge_index = self.default_edge_index

        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)

        batch_size, num_nodes, _ = node_features.shape
        embeddings = []

        for b in range(batch_size):
            x = node_features[b]
            h = self.spatial_activation(self.gcn1(x, edge_index))
            h = self.gcn2(h, edge_index)
            # Global mean pooling over intersections
            h_pooled = h.mean(dim=0)
            embeddings.append(h_pooled)

        return torch.stack(embeddings, dim=0)

    def encode_temporal(self, spatial_sequence: torch.Tensor) -> torch.Tensor:
        """
        Processes recent temporal sequence of pooled spatial embeddings via GRU.
        
        Args:
            spatial_sequence: [B, L, spatial_dim]
            
        Returns:
            Final GRU hidden state: [B, temporal_dim]
        """
        _, hidden = self.gru(spatial_sequence)
        return hidden[-1]

    def forward_manifold(
        self,
        spatial_history: torch.Tensor,
        objective_vector: torch.Tensor,
        interaction_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs the latent objective interaction manifold and adaptive weights.
        
        Returns:
            manifold: [B, manifold_dim]
            weights: [B, num_objectives] on simplex Δ^4
        """
        # Temporal context from history
        h_temporal = self.encode_temporal(spatial_history)

        # Objective representation
        h_obj = self.objective_encoder(objective_vector)

        # Interaction representation
        b = interaction_matrix.shape[0]
        h_int = self.interaction_encoder(interaction_matrix.reshape(b, -1))

        # Manifold core
        fused = torch.cat([h_temporal, h_obj, h_int], dim=-1)
        manifold = self.manifold_core(fused)

        # Adaptive preference distribution
        pref_logits = self.preference_head(manifold)
        weights = F.softmax(pref_logits / self.temperature, dim=-1)

        return manifold, weights

    def forward(
        self,
        spatial_history: torch.Tensor,
        objective_vector: torch.Tensor,
        interaction_matrix: torch.Tensor,
        raw_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass producing manifold, preference weights, action logits, and value.
        """
        manifold, weights = self.forward_manifold(
            spatial_history, objective_vector, interaction_matrix
        )

        # Condition policy and value on both manifold and raw traffic state
        policy_input = torch.cat([manifold, raw_state], dim=-1)
        actor_feats = self.actor_backbone(policy_input)

        # Multi-discrete action logits
        action_logits = [head(actor_feats) for head in self.action_heads]

        # Critic value
        value = self.critic_head(policy_input)

        return {
            "manifold": manifold,
            "weights": weights,
            "action_logits": action_logits,
            "value": value,
        }

    def predict_next_objectives(
        self,
        manifold: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Auxiliary forward pass: predicts next-step objective vector z_{t+1}.
        """
        aux_input = torch.cat([manifold, actions.float()], dim=-1)
        return self.aux_predictor(aux_input)
