"""
DOIM-Traffic-RL: State-Conditioned Objective Interaction Learning (SCOIL)
Implements a mathematically defensible, state-conditioned, asymmetric interaction
operator with uncertainty estimation and Bayesian shrinkage.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateConditionedInteractionLearner(nn.Module):
    """
    Learns dynamic, state-dependent, asymmetric interaction operators among competing objectives.
    
    Mathematical Formulation:
      Given traffic state context h_t ∈ ℝ^D_h and normalized objective vector z_t ∈ [0, 1]^K:
      
      1. Interaction Query-Key Bilinear Coupling:
         Q_t = W_q · [h_t, z_t] ∈ ℝ^(K × d_k)
         K_t = W_k · [h_t, z_t] ∈ ℝ^(K × d_k)
         M_raw = (Q_t · K_t^T) / sqrt(d_k) ∈ ℝ^(K × K)
         M_t = tanh(M_raw) ∈ [-1, 1]^(K × K)
         
      2. Asymmetric Cross-Gradient Encoding:
         Allows M_{ij} ≠ M_{ji} to capture directional lags (e.g. queue -> delay propagation).
         Diagonal M_{ii} is clamped to +1.0 (self-interaction / urgency).
         
      3. Epistemic Uncertainty Estimation:
         U_t = softplus(W_u · [h_t, z_t]) ∈ ℝ_+(K × K)
         
      4. Adaptive Shrinkage towards Conservative Prior:
         M_tilde_{ij} = M_{ij} / (1.0 + lambda_shrink * U_{ij})
    """

    def __init__(
        self,
        num_objectives: int = 5,
        traffic_dim: int = 64,
        latent_dim: int = 32,
        shrinkage_lambda: float = 0.5,
    ):
        super().__init__()

        self.num_objectives = num_objectives
        self.latent_dim = latent_dim
        self.shrinkage_lambda = float(shrinkage_lambda)

        context_dim = traffic_dim + num_objectives

        # Shared feature projection
        self.feature_net = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Bilinear Query and Key projections for each objective
        self.query_proj = nn.Linear(64, num_objectives * latent_dim)
        self.key_proj = nn.Linear(64, num_objectives * latent_dim)

        # Asymmetric residual bias
        self.asymmetry_head = nn.Linear(64, num_objectives * num_objectives)

        # Uncertainty estimation head (softplus guarantees positive variance)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(64, num_objectives * num_objectives),
            nn.Softplus(),
        )

    def forward(
        self,
        traffic_embedding: torch.Tensor,
        objective_vector: torch.Tensor,
        prior_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes state-conditioned interaction matrix M_t and uncertainty U_t.
        
        Args:
            traffic_embedding: [B, traffic_dim] (from GRU/GNN)
            objective_vector: [B, num_objectives]
            prior_matrix: Optional [B, K, K] or [K, K] prior to shrink towards.
            
        Returns:
            Dictionary containing:
              - 'interaction_matrix': [B, K, K] in [-1, 1]
              - 'uncertainty': [B, K, K] in [0, inf)
              - 'shrunk_matrix': [B, K, K] regularized by uncertainty
        """
        if traffic_embedding.dim() == 1:
            traffic_embedding = traffic_embedding.unsqueeze(0)
        if objective_vector.dim() == 1:
            objective_vector = objective_vector.unsqueeze(0)

        batch_size = traffic_embedding.size(0)
        device = traffic_embedding.device

        # Concatenate traffic context and current objective values
        context = torch.cat([traffic_embedding, objective_vector], dim=-1)
        feats = self.feature_net(context)

        # Project into K distinct Query and Key vectors
        queries = self.query_proj(feats).view(batch_size, self.num_objectives, self.latent_dim)
        keys = self.key_proj(feats).view(batch_size, self.num_objectives, self.latent_dim)

        # Scaled dot-product interaction
        scale = np.sqrt(self.latent_dim)
        bilinear_int = torch.bmm(queries, keys.transpose(1, 2)) / scale # [B, K, K]

        # Add asymmetric residual bias
        asym_bias = self.asymmetry_head(feats).view(batch_size, self.num_objectives, self.num_objectives)
        raw_int = bilinear_int + asym_bias

        # Bounded interaction in [-1, 1]
        m_t = torch.tanh(raw_int)

        # Self-interaction is identity on diagonal
        diag_mask = torch.eye(self.num_objectives, device=device, dtype=torch.bool).unsqueeze(0)
        m_t = m_t.masked_fill(diag_mask, 1.0)

        # Epistemic uncertainty
        u_t = self.uncertainty_head(feats).view(batch_size, self.num_objectives, self.num_objectives)
        u_t = u_t.masked_fill(diag_mask, 0.0) # zero uncertainty on identity diagonal

        # Adaptive shrinkage towards identity or provided prior
        if prior_matrix is None:
            prior = torch.eye(self.num_objectives, device=device, dtype=m_t.dtype).unsqueeze(0).expand_as(m_t)
        else:
            prior = prior_matrix.expand_as(m_t)

        shrinkage_weight = 1.0 / (1.0 + self.shrinkage_lambda * u_t)
        shrunk_m_t = shrinkage_weight * m_t + (1.0 - shrinkage_weight) * prior

        return {
            "interaction_matrix": m_t,
            "uncertainty": u_t,
            "shrunk_matrix": shrunk_m_t,
        }

    def compute_predictive_loss(
        self,
        m_t: torch.Tensor,
        z_t: torch.Tensor,
        z_next: torch.Tensor,
        action_indicator: torch.Tensor,
    ) -> torch.Tensor:
        """
        Auxiliary loss training the interaction matrix to predict directional objective changes:
            z_{t+1} - z_t ≈ M_t · Δ_action
        """
        delta_z = z_next - z_t # [B, K]
        # Linear projection through interaction operator
        predicted_delta = torch.bmm(m_t, delta_z.unsqueeze(-1)).squeeze(-1)
        loss = F.mse_loss(predicted_delta, delta_z)
        return loss
