# Phase 28.2: DOIM & SCOIL Architectural Ablation Plan
**Project:** DOIM-PPO (*Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems*)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard:** Ablation Rigor & Component Contribution Verification  
**Auditing Panel:** Independent Research Validation Team  
**Date:** November 3, 2026  

---

## 1. Architectural Decomposition

DOIM-PPO integrates four core structural modules:
1. **Spatial Encoder (GCN):** Graph Convolutional Network capturing spatial dependencies across the 3x3 grid network topology.
2. **Temporal Encoder (GRU):** Gated Recurrent Unit capturing historical traffic evolution ($H=12$ steps).
3. **SCOIL Dynamic Interaction Manifold ($\mathcal{M}(s)$):** State-Conditioned Objective Interaction Learning module estimating cross-objective interaction matrix $\Psi(s) \in [-1, 1]^{5 \times 5}$.
4. **Adaptive Weight Modulator:** Computes dynamic scalarization weights:
   $$\mathbf{w}(s) = \text{Softmax}\left(\mathbf{w}_0 + \alpha \mathcal{M}(s) \mathbf{r}_{\text{norm}}(s)\right)$$

---

## 2. Root Cause Analysis: Pilot 1.77% Ablation Delta

In the pilot Option A, Full DOIM-PPO achieved $510.94\,\text{s}$ waiting time while Ablation No-SCOIL achieved $501.90\,\text{s}$ (a 1.77% delta).
- **Rollout Dynamics:** With batch rollout size $n_{\text{steps}} = 128$, a 1,000-step training run yields only:
  $$\lfloor 1000 / 128 \rfloor = 7\text{--}8 \text{ PPO updates}$$
- **Gradient Landscape:** During the first 8 Adam updates, gradient descent is dominated by policy entropy reduction and value baseline fitting. The policy parameters $\theta$ have not received enough updates to diverge based on dynamic objective interactions versus a static zero-manifold baseline.
- **Empirical Evidence of Manifold Learning:** Inspection of the SCOIL layer weights in `task-1701` confirms that non-zero off-diagonal couplings were actively learned (e.g., Delay-Emissions coupling $\Psi_{1,3} = -0.1997$). The manifold was structurally functional, but the training horizon was insufficient for downstream policy differentiation.

---

## 3. Tier 2 Confirmatory Ablation Suite

To decisively establish the scientific contribution of each architectural innovation in IEEE T-ITS, five controlled variants will be evaluated:

| Variant Name | Spatial GCN | Temporal GRU | SCOIL Interaction Manifold | Objective Scalarization | Parameter Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DOIM-PPO (Full)** | Yes | Yes | Learned Neural $\mathcal{M}(s)$ | Dynamic Adaptive | 48,938 |
| **Ablation 1: No-SCOIL** | Yes | Yes | Zero Matrix ($\mathcal{M} = \mathbf{0}$) | Fixed Equal Weights | 47,820 |
| **Ablation 2: Rolling Pearson**| Yes | Yes | Empirical Pearson Correlation | Dynamic Statistical | 47,820 |
| **Ablation 3: No-GCN** | No (MLP) | Yes | Learned Neural $\mathcal{M}(s)$ | Dynamic Adaptive | 38,410 |
| **Ablation 4: Static FW-PPO**| No (MLP) | No | None | Fixed Weights ($0.2$) | 32,840 |

---

## 4. Acceptance & Verification Criteria for IEEE T-ITS
The contribution of DOIM is confirmed if and only if:
1. **Dynamic Sensitivity:** $\mathcal{M}(s)$ produces statistically significant variation across traffic states ($p < 0.01$ ANOVA across low vs. high congestion).
2. **Pareto Dominance:** Full DOIM-PPO achieves higher 5D Hypervolume than No-SCOIL and Rolling Pearson ($p < 0.05$ across 5 seeds).
3. **Stability:** The learned interaction matrix satisfies positive semi-definiteness bounds $\lambda_{\min}(\mathcal{M}) \ge -1.0$.
