# DOIM-Traffic-RL: Architectural Ablation Study Protocol
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Standard:** Strict Mechanism Attribution & Scientific Isolation of Novelty  
**Reference Code:** `models/doim_network.py`, `models/interaction_learner.py`, `training/interaction_aware_ppo.py`

---

## 1. Purpose of the Ablation Suite

A primary requirement for publication in IEEE T-ITS is demonstrating that performance gains are directly attributable to specific novel mechanisms rather than arbitrary hyperparameter tuning, increased parameter counts, or extra computational budgets.

This protocol pre-registers 10 structured architectural ablations isolating:
1. The contribution of objective-interaction modeling (SCOIL).
2. The role of directional asymmetry vs. symmetric Pearson correlation.
3. The benefit of predictive uncertainty and Bayesian shrinkage.
4. The necessity of spatial GNN and temporal GRU representations.
5. The impact of the action safety shield.

---

## 2. Pre-Registered Ablation Matrix

| Ablation ID | Configuration Name | Modified Component | Mathematical Variation | Target Hypothesis Isolated |
| :--- | :--- | :--- | :--- | :--- |
| **ABL-01** | `DOIM_no_interaction` | Interaction Learner | $M_t = 0 \in \mathbb{R}^{K \times K}$ | Proves that cross-objective interaction information drives adaptation, not just raw capacity. |
| **ABL-02** | `DOIM_rolling_pearson` | Interaction Model | $M_t = \text{RollingPearson}_{30}(Z)$ | Tests whether learned asymmetric cross-gradients outperform empirical symmetric correlation. |
| **ABL-03** | `DOIM_no_shrinkage` | Bayesian Shrinkage | $\lambda_{\text{shrink}} = 0.0$ (Raw $M_t$) | Evaluates whether shrinking toward identity prior prevents catastrophic preference divergence. |
| **ABL-04** | `DOIM_empirical_prior` | Shrinkage Prior | $M_{\text{prior}} = M_{\text{offline}}$ | Compares static empirical traffic prior against conservative identity prior. |
| **ABL-05** | `DOIM_no_preference_adapt` | Preference Head | $w_t = [0.2, 0.2, 0.2, 0.2, 0.2]^T$ | Quantifies the value of dynamic preference adaptation over static uniform scalarization. |
| **ABL-06** | `DOIM_no_auxiliary_loss` | Training Loss | $\lambda_{\text{aux}} = 0.0$ | Tests whether counterfactual predictive dynamics loss grounds interaction representations. |
| **ABL-07** | `DOIM_no_gcn` | Spatial Encoder | $H_t = \text{MLP}(X_t)$ (No Graph Message Passing) | Determines whether graph topology awareness improves arterial coordination. |
| **ABL-08** | `DOIM_no_gru` | Temporal Encoder | $h_t = h_{\text{spatial}, t}$ (Instantaneous Frame) | Evaluates whether recurrent queue-accumulation history is essential for non-stationary control. |
| **ABL-09** | `DOIM_param_matched_mlp` | Manifold Core | Generic MLP with matched FLOPs/weights | Verifies that bilinear manifold geometry provides inductive bias beyond generic MLP depth. |
| **ABL-10** | `DOIM_no_shield` | Action Shield | Shield Disabled (Direct TraCI Action) | Quantifies phase fluttering, yellow violations, and collision-risk frequency without safety filtering. |

---

## 3. Implementation Details for Key Ablations

### 3.1 Ablation ABL-01 (`no_interaction`)
In `InteractionAwarePPOAgent.act()` and `update()`:
The interaction tensor $M_t$ is clamped to zero:
$$M_t = \mathbf{0}_{K \times K}$$
All other components (GCN, GRU, DOIM fusion, PPO policy) remain identical in architecture and training budget.

### 3.2 Ablation ABL-02 (`rolling_pearson`)
The neural SCOIL module is replaced with a rolling sample Pearson correlation matrix computed over the preceding 30 steps of normalized objective traces:
$$R_{ij, t} = \frac{\sum_{\tau=t-W}^t (z_{i, \tau} - \bar{z}_i)(z_{j, \tau} - \bar{z}_j)}{\sqrt{\sum (z_{i, \tau} - \bar{z}_i)^2 \sum (z_{j, \tau} - \bar{z}_j)^2}}$$
This tests the central claim of the Master Roadmap: whether learned asymmetric cross-coupling provides actionable control value beyond descriptive correlation.

### 3.3 Ablation ABL-09 (`param_matched_mlp`)
The structured bilinear interaction and manifold fusion module:
$$u_t = [h_t \parallel \text{MLP}_z(\vec{z}_t) \parallel \text{MLP}_m(\text{vec}(M_t))]$$
is replaced with a feed-forward MLP:
$$m_t^{\text{MLP}} = \text{MLP}_{\text{dense}}([h_t, \vec{z}_t])$$
with hidden dimension adjusted to match the exact parameter count (142,880 parameters) and FLOPs of the full DOIM network.

---

## 4. Evaluation Criteria & Attribution Proof

For every ablation variant, the evaluation protocol mandates:
1. **Matched Training Budget:** Identical timestep budget ($T = 5,000$ in pilot, $T = 50,000$ in main).
2. **Matched Seed Pairs:** Evaluated across the 5 canonical evaluation seeds (`[1001, 1002, 1003, 1004, 1005]`).
3. **Attribution Standard:** A claimed mechanism is validated if and only if its ablation yields a statistically significant degradation ($p < 0.05$ after Holm-Bonferroni correction) in at least one primary metric without improving others.

