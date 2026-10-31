# Mathematical Theory of State-Conditioned Objective Interaction Learning (SCOIL)

**Document:** `research_docs/OBJECTIVE_INTERACTION_THEORY.md`  
**Classification:** Core Research Contribution  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems / IEEE T-PAMI  
**Date:** October 31, 2026  

---

## 1. Scientific Problem Formulation

In multi-objective reinforcement learning (MORL) for complex physical systems such as urban traffic networks, the system performance is evaluated across a vector of $K$ competing objectives:
$$\mathbf{J}(t) = [J_1(t), J_2(t), \dots, J_K(t)]^\top \in \mathbb{R}^K$$
In this repository, $K=5$: Waiting Time ($J_1$), Queue Length ($J_2$), $\text{CO}_2$ Emissions ($J_3$), Throughput ($J_4$), and Fairness Disparity ($J_5$).

The central research hypothesis investigates:
> *"Can explicitly learning state-dependent, asymmetric interactions among competing traffic objectives produce more robust, interpretable, and equitable traffic-signal policies than conventional fixed-weight and rolling correlation approaches under changing traffic conditions?"*

---

## 2. Why Rolling Pearson Correlation Fails

Prior iterations of the prototype estimated an empirical interaction matrix using a rolling $W$-step sample Pearson correlation coefficient:
$$\rho_{ij}(t) = \frac{\sum_{\tau=t-W+1}^t (z_{\tau,i} - \bar{z}_i)(z_{\tau,j} - \bar{z}_j)}{\sqrt{\sum_{\tau} (z_{\tau,i} - \bar{z}_i)^2 \sum_{\tau} (z_{\tau,j} - \bar{z}_j)^2}}$$

This methodology exhibits three fatal scientific limitations:

### 2.1 Confounding by Shared Non-Stationary Demand
When traffic demand increases along a corridor during a rush-hour surge:
- Inflow rates rise.
- Queue lengths increase ($\uparrow$).
- Total vehicle waiting times increase ($\uparrow$).
- Stop-and-go $\text{CO}_2$ emissions increase ($\uparrow$).

Because all three metrics increase simultaneously driven by the exogenous volume influx, their sample covariance is strongly positive: $\text{Cov}(J_{\text{wait}}, J_{\text{co2}}) > 0$. The resulting Pearson correlation $\rho \approx +0.85$ falsely suggests that wait time and $\text{CO}_2$ emissions are aligned in an intrinsic operational synergy. In reality, under fixed signal capacity, extending green time to flush queues produces high-speed cruising that reduces per-vehicle $\text{CO}_2$, whereas chopping green phases to cycle through all waiting approaches minimizes peak wait times at the cost of massive stop-and-go emission penalties. **Sample correlation measures common trend, not control-relevant trade-offs.**

### 2.2 Forced Symmetry Violations
Pearson correlation is mathematically symmetric by definition: $\rho_{ij} = \rho_{ji}$. In traffic networks, objective cross-dependencies are strictly **directional and asymmetric**:
- Halting a platoon directly and instantaneously impacts queue accumulation:
  $$\frac{\partial J_{\text{queue}}(t)}{\partial a_t} \text{ (immediate)}$$
- Clearing an intersection queue impacts network-wide average accumulated delay only after vehicles travel through downstream links:
  $$\frac{\partial J_{\text{wait}}(t+\tau)}{\partial J_{\text{queue}}(t)} \text{ with transport lag } \tau > 0$$
Forcing $M_{ij} = M_{ji}$ loses directional transport dynamics.

### 2.3 Non-Differentiability and Zero Policy Adaptation
Rolling Pearson correlation is computed via NumPy routines outside the computational graph. It cannot be differentiated with respect to neural representations or policy parameters, leaving the interaction estimate completely blind to the agent's actions.

---

## 3. Proposed Formulation: State-Conditioned Objective Interaction Operator (SCOIO)

Instead of passive statistical correlation, we introduce a learned, state-conditioned operator $\mathbf{M}_t = \mathcal{M}_{\theta}(s_t, z_t) \in \mathbb{R}^{K \times K}$.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               STATE-CONDITIONED OBJECTIVE INTERACTION OPERATOR (SCOIO)                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   Spatial-Temporal Traffic State h_t ∈ ℝ^64      Normalized Objectives z_t ∈ [0, 1]^K  │
│                       │                                         │                      │
│                       └───────────────────┬─────────────────────┘                      │
│                                           ▼                                            │
│                              [Shared Feature MLP]                                      │
│                                           ▼  feats ∈ ℝ^64                              │
│                      ┌────────────────────┼────────────────────┐                       │
│                      ▼                    ▼                    ▼                       │
│               [Query Proj W_q]     [Key Proj W_k]    [Asymmetry Head W_a]              │
│                      │                    │                    │                       │
│                      ▼                    ▼                    │                       │
│                   Q_t ∈ ℝ^(K×d)        K_t ∈ ℝ^(K×d)           │                       │
│                      │                    │                    │                       │
│                      └──────────┬─────────┘                    │                       │
│                                 ▼                              │                       │
│                       Bilinear (Q · K^T)/√d                    │                       │
│                                 │                              │                       │
│                                 └──────────────┬───────────────┘                       │
│                                                ▼                                       │
│                                   M_raw = Bilinear + Asymmetry                         │
│                                                ▼                                       │
│                                  M_t = tanh(M_raw) ∈ [-1, 1]^(K×K)                     │
│                                                ▼                                       │
│                                      [Uncertainty Head U_t]                            │
│                                                ▼                                       │
│                                M̃_t = Shrinkage(M_t, U_t, Prior)                        │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Bilinear Cross-Attention Formulation
For each objective $i \in \{1, \dots, K\}$, we compute a latent Query vector $q_i(t) \in \mathbb{R}^{d_k}$ and Key vector $k_i(t) \in \mathbb{R}^{d_k}$ conditioned on the spatial-temporal network state $h_t$ and current objective vector $z_t$:
$$q_i(t) = W_q^{(i)} [h_t, z_t] + b_q^{(i)}$$
$$k_j(t) = W_k^{(j)} [h_t, z_t] + b_k^{(j)}$$

The raw interaction intensity between objective $i$ and objective $j$ is given by:
$$M_{\text{raw}, ij}(t) = \frac{\langle q_i(t), k_j(t) \rangle}{\sqrt{d_k}} + A_{ij}(h_t, z_t)$$
where $A_{ij}(h_t, z_t)$ is an explicit asymmetric cross-gradient bias.

The bounded interaction operator is obtained via hyperbolic tangent activation:
$$M_{ij}(t) = \tanh\left( M_{\text{raw}, ij}(t) \right) \in [-1, 1]$$
with diagonal self-interaction clamped to unity: $M_{ii}(t) \equiv 1.0$.

### 3.2 Epistemic Uncertainty and Bayesian Shrinkage
Under distribution shifts (e.g., sudden transition from normal flow to a directional surge), the neural interaction estimates may suffer from high epistemic uncertainty. We parameterize the uncertainty tensor $\mathbf{U}_t \in \mathbb{R}_+^{K \times K}$:
$$\mathbf{U}_t = \text{Softplus}\left( W_u [h_t, z_t] + b_u \right)$$

When uncertainty $U_{ij}(t)$ is large, the interaction estimate is adaptively regularized towards a conservative prior $\mathbf{M}_0$ (such as the identity matrix $\mathbf{I}_K$) via a James-Stein type shrinkage estimator:
$$\tilde{M}_{ij}(t) = \beta_{ij}(t) M_{ij}(t) + (1 - \beta_{ij}(t)) M_{0, ij}$$
where the adaptive shrinkage weight is:
$$\beta_{ij}(t) = \frac{1}{1 + \lambda_{\text{shrink}} U_{ij}(t)} \in (0, 1]$$

---

## 4. Distinction: Statistical Correlation vs. Predictive vs. Decision-Relevant

| Dimension | Rolling Pearson Correlation | Proposed SCOIO Mechanism |
|---|---|---|
| **Mathematical Nature** | Bivariate sample covariance over time window. | State-conditioned neural operator $\mathcal{M}(s_t, z_t)$. |
| **Symmetry** | Strictly symmetric ($M_{ij} = M_{ji}$). | Asymmetric ($M_{ij} \neq M_{ji}$) capturing directional lags. |
| **State Dependence** | Independent of current spatial queue distribution. | Conditioned on spatial GNN & temporal GRU hidden state. |
| **Differentiability** | Non-differentiable (NumPy array). | Fully differentiable PyTorch module ($\nabla_{\Theta} M \neq 0$). |
| **Noise & Surge Robustness**| Distorted by common exogenous demand influx. | Uncertainty-aware Bayesian shrinkage against prior. |
| **Policy Coupling** | Disconnected from actor-critic backprop. | Jointly optimized via PPO and auxiliary transition loss. |

---

## 5. Honest Scientific Boundary (Causal Claims)

> **IMPORTANT SCIENTIFIC CAVEAT:**  
> While the proposed mechanism models *directional and state-conditioned objective interactions*, we explicitly refrain from claiming that the model discovers *real-world causal laws* from observational traffic data. In simulation, the interaction operator represents a **predictive response model** learned under the simulator's specific physics and routing dynamics. True causal identification would require formal instrumental variables, do-calculus interventions, or randomized phase perturbations with unconfoundedness guarantees. Within this paper, the mechanism is described strictly as a **State-Conditioned Objective Interaction Operator**.
