"""
DOIM-Traffic-RL: State-Conditioned Objective Interaction Learning (SCOIL)
Implements a mathematically defensible, state-conditioned, asymmetric predictive
interaction operator with heteroscedastic uncertainty estimation and Bayesian shrinkage.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateConditionedInteractionLearner(nn.Module):
    """
    Learns dynamic, state-dependent, asymmetric predictive interaction operators among competing objectives.
    
    Mathematical Formulation:
      Given traffic context h_t ∈ ℝ^D_h and normalized objective urgency vector z_t ∈ [0, 1]^K:
      
      1. Interaction Query-Key Bilinear Coupling:
         Q_t = W_q · [h_t, z_t] ∈ ℝ^(K × d_k)
         K_t = W_k · [h_t, z_t] ∈ ℝ^(K × d_k)
         M_bilinear = (Q_t · K_t^T) / sqrt(d_k) ∈ ℝ^(K × K)
         
      2. Asymmetric Cross-Gradient Encoding:
         M_raw = M_bilinear + W_asym · [h_t, z_t]
         M_t = tanh(M_raw) ∈ [-1, 1]^(K × K)
         Allows M_{ij} ≠ M_{ji} to capture directional traffic lags (e.g. queue -> delay propagation).
         Diagonal M_{ii} is clamped to +1.0 (self-interaction / identity anchor).
         
      3. State-Conditioned Predictive Uncertainty Estimation:
         U_t = softplus(W_u · [h_t, z_t]) ∈ ℝ_+(K × K)
         Represents heteroscedastic predictive dispersion across objective trade-off dimensions.
         
      4. Adaptive Shrinkage towards Conservative Prior:
         w_shrink = 1.0 / (1.0 + lambda_shrink * U_t)
         M_tilde = w_shrink ⊙ M_t + (1 - w_shrink) ⊙ M_prior
         Where M_prior can be Identity (independent baseline) or an Empirical Prior Matrix.
    """

    def __init__(
        self,
        num_objectives: int = 5,
        traffic_dim: int = 64,
        latent_dim: int = 32,
        shrinkage_lambda: float = 0.1,
        num_controllable_tls: int = 5,
        shrinkage_mode: str = "identity",  # "identity", "empirical", "none"
        empirical_prior: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.num_objectives = num_objectives
        self.latent_dim = latent_dim
        self.shrinkage_lambda = float(shrinkage_lambda)
        self.num_controllable_tls = num_controllable_tls
        self.shrinkage_mode = str(shrinkage_mode)

        if empirical_prior is not None:
            self.register_buffer("empirical_prior", empirical_prior.float())
        else:
            self.empirical_prior = None

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

        # Nonnegative predictive uncertainty dispersion head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(64, num_objectives * num_objectives),
            nn.Softplus(),
        )

        # Primary action-impact projection (maps discrete actions to primary objective shift)
        self.action_impact = nn.Linear(num_controllable_tls, num_objectives, bias=False)

    def forward(
        self,
        traffic_embedding: torch.Tensor,
        objective_vector: torch.Tensor,
        prior_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes state-conditioned interaction matrix M_t and predictive uncertainty U_t.
        
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
        bilinear_int = torch.bmm(queries, keys.transpose(1, 2)) / scale  # [B, K, K]

        # Add asymmetric residual bias
        asym_bias = self.asymmetry_head(feats).view(batch_size, self.num_objectives, self.num_objectives)
        raw_int = bilinear_int + asym_bias

        # Bounded interaction in [-1, 1]
        m_t = torch.tanh(raw_int)

        # Self-interaction is identity on diagonal
        diag_mask = torch.eye(self.num_objectives, device=device, dtype=torch.bool).unsqueeze(0)
        m_t = m_t.masked_fill(diag_mask, 1.0)

        # Predictive uncertainty dispersion score
        u_t = self.uncertainty_head(feats).view(batch_size, self.num_objectives, self.num_objectives)
        u_t = u_t.masked_fill(diag_mask, 0.0)  # zero uncertainty on identity diagonal

        # Adaptive shrinkage towards prior
        if self.shrinkage_mode == "none":
            shrunk_m_t = m_t
        else:
            if prior_matrix is not None:
                prior = prior_matrix.to(device).expand_as(m_t)
            elif self.shrinkage_mode == "empirical" and self.empirical_prior is not None:
                prior = self.empirical_prior.to(device).unsqueeze(0).expand_as(m_t)
            else:
                prior = torch.eye(self.num_objectives, device=device, dtype=m_t.dtype).unsqueeze(0).expand_as(m_t)

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
        actions: torch.Tensor,
        z_next: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Auxiliary loss training the interaction matrix to model cross-objective predictive dynamics:
            Δz_primary = W_a · a_t
            Δz_cross = M_t · Δz_primary
            z_hat_{t+1} = clamp(z_t + Δz_cross, 0.0, 1.0)
            
        If uncertainty is provided, optimizes heteroscedastic negative log-likelihood:
            NLL = 0.5 * ( (z_hat - z_next)^2 / (sigma^2 + eps) + log(sigma^2 + eps) )
        Otherwise optimizes MSE.
        """
        delta_primary = self.action_impact(actions.float())  # [B, K]
        delta_cross = torch.bmm(m_t, delta_primary.unsqueeze(-1)).squeeze(-1)  # [B, K]
        z_pred_next = torch.clamp(z_t + delta_cross, 0.0, 1.0)

        if uncertainty is not None:
            # Average uncertainty per objective: [B, K]
            sigma_sq = uncertainty.mean(dim=-1) + 1e-4
            sq_err = (z_pred_next - z_next) ** 2
            nll = 0.5 * (sq_err / sigma_sq + torch.log(sigma_sq)).mean()
            return nll

        return F.mse_loss(z_pred_next, z_next)
