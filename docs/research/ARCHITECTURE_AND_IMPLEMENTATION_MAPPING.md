# DOIM-Traffic-RL: Architecture and Implementation Traceability Mapping
## Complete Mathematical Formulation to Source Code Traceability

This document establishes an exact, one-to-one mapping between the formal mathematical formulation of the **Dynamic Objective Interaction Manifold (DOIM)** architecture, the **Interaction-Aware Multi-Objective Proximal Policy Optimization (IA-MOPPO)** algorithm, and the actual Python source implementation in the repository.

---

## 1. System Pipeline and Tensor Dimensionality Mapping

| Mathematical Object | Mathematical Symbol | Tensor Shape | PyTorch Module / Class | Code Reference |
| :--- | :---: | :---: | :--- | :--- |
| **Raw Node Features** | $\mathbf{X}_t \in \mathbb{R}^{N \times D_{\text{in}}}$ | `[B, 9, 5]` | `TrafficStateExtractor` / `DOIMTrafficEnv` | [`DOIMTrafficEnv`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/environment/sumo_env.py) |
| **Grid Adjacency / Graph Topology** | $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ | `[2, 24]` | `create_grid_edge_index()` | [`create_grid_edge_index`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/spatial_encoder.py) |
| **Spatial Graph Convolutions** | $\mathbf{H}_t^{(l)} = \sigma(\mathbf{\tilde{D}}^{-\frac{1}{2}}\mathbf{\tilde{A}}\mathbf{\tilde{D}}^{-\frac{1}{2}}\mathbf{H}_t^{(l-1)}\mathbf{\Theta}^{(l)})$ | `[B, 9, 64]` | `DOIMNetwork.gcn1`, `gcn2` | [`DOIMNetwork.encode_spatial`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/doim_network.py) |
| **Spatial Graph Readout / Pooling** | $\mathbf{s}_t = \frac{1}{N}\sum_{i=1}^N \mathbf{h}_{t,i}^{(2)}$ | `[B, 64]` | `DOIMNetwork.encode_spatial` | [`DOIMNetwork.encode_spatial`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/doim_network.py) |
| **Temporal Sequence History** | $\mathbf{S}_{t-L+1:t}$ (causal zero-padding) | `[B, 10, 64]` | `RolloutBuffer.node_histories` | [`RolloutBuffer`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/training/interaction_aware_ppo.py) |
| **Temporal Recurrent State** | $\mathbf{h}_t^{\text{temp}} = \text{GRU}(\mathbf{s}_t, \mathbf{h}_{t-1}^{\text{temp}})$ | `[B, 64]` | `DOIMNetwork.gru` | [`DOIMNetwork.encode_temporal`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/doim_network.py) |
| **Raw Objective Vector** | $\mathbf{o}_t \in \mathbb{R}^K$ | `[B, 5]` | `MultiObjectiveEngine.compute_raw_vector` | [`MultiObjectiveEngine`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/objectives/objective_engine.py) |
| **Authoritative Normalized Urgency** | $\mathbf{z}_t = \text{clamp}\left(\frac{\mathbf{o}_t - \mathbf{o}_{\min}}{\mathbf{o}_{\max} - \mathbf{o}_{\min}}, 0, 1\right)$ | `[B, 5]` | `ObjectiveNormalizer.normalize` | [`ObjectiveNormalizer`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/objectives/objective_normalizer.py) |
| **Projected Objective Features** | $\mathbf{e}_t^{\text{obj}} = \phi_{\text{obj}}(\mathbf{z}_t)$ | `[B, 32]` | `DOIMNetwork.obj_mlp` | [`DOIMNetwork.forward_manifold`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/doim_network.py) |
| **Bilinear Cross-Objective Queries** | $\mathbf{Q}_t = \mathbf{W}_q \psi([\mathbf{h}_t^{\text{temp}}, \mathbf{z}_t])$ | `[B, 5, 16]` | `SCOIL.query_proj` | [`StateConditionedInteractionLearner`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/interaction_learner.py) |
| **Bilinear Cross-Objective Keys** | $\mathbf{K}_t = \mathbf{W}_k \psi([\mathbf{h}_t^{\text{temp}}, \mathbf{z}_t])$ | `[B, 5, 16]` | `SCOIL.key_proj` | [`StateConditionedInteractionLearner`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/interaction_learner.py) |
| **Asymmetric Interaction Matrix** | $\mathbf{M}_t = \tanh\left(\frac{\mathbf{Q}_t\mathbf{K}_t^\top}{\sqrt{d}} + \mathbf{B}_t^{\text{asym}}\right)$ | `[B, 5, 5]` | `SCOIL.forward` | [`StateConditionedInteractionLearner`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/interaction_learner.py) |
| **Predictive Uncertainty Dispersion** | $\mathbf{U}_t \in \mathbb{R}_{+}^{5 \times 5}$ | `[B, 5, 5]` | `SCOIL.uncertainty_head` | [`StateConditionedInteractionLearner`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/interaction_learner.py) |
| **Adaptive Bayesian Shrinkage** | $\mathbf{\tilde{M}}_t = \frac{1}{1 + \lambda \mathbf{U}_t} \odot \mathbf{M}_t + \frac{\lambda \mathbf{U}_t}{1 + \lambda \mathbf{U}_t} \odot \mathbf{M}_{\text{prior}}$ | `[B, 5, 5]` | `SCOIL.forward` | [`StateConditionedInteractionLearner`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/interaction_learner.py) |
| **Flattened Interaction Context** | $\mathbf{e}_t^{\text{int}} = \phi_{\text{int}}(\text{vec}(\mathbf{\tilde{M}}_t))$ | `[B, 32]` | `DOIMNetwork.int_mlp` | [`DOIMNetwork.forward_manifold`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/doim_network.py) |
| **Latent DOIM Manifold** | $\mathbf{m}_t = \phi_{\text{fuse}}([\mathbf{h}_t^{\text{temp}}, \mathbf{e}_t^{\text{obj}}, \mathbf{e}_t^{\text{int}}])$ | `[B, 64]` | `DOIMNetwork.manifold_mlp` | [`DOIMNetwork.forward_manifold`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/doim_network.py) |
| **Adaptive Preference Weights** | $\mathbf{w}_t = \text{Softmax}(\mathbf{W}_w \mathbf{m}_t / \tau)$ | `[B, 5]` | `DOIMNetwork.preference_head` | [`DOIMNetwork.forward_manifold`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/doim_network.py) |
| **Multi-Discrete Actor Logits** | $\pi_\theta(\mathbf{a} \mid \mathbf{s}_t) = \prod_{j=1}^J \text{Categorical}(\mathbf{p}_{t,j})$ | 5 TLS $\times$ `[B, 2]` | `DOIMNetwork.action_heads` | [`DOIMNetwork.forward`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/doim_network.py) |
| **Critic State Value** | $V_\phi(\mathbf{s}_t) \in \mathbb{R}$ | `[B, 1]` | `DOIMNetwork.critic_head` | [`DOIMNetwork.forward`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/doim_network.py) |
| **Residual Transition Predictor** | $\mathbf{\hat{z}}_{t+1} = \text{clamp}(\mathbf{z}_t + \Delta_{\text{aux}}(\mathbf{m}_t, \mathbf{a}_t^{\text{exec}}, \mathbf{z}_t), 0, 1)$ | `[B, 5]` | `DOIMNetwork.aux_predictor` | [`DOIMNetwork.predict_next_objectives`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/models/doim_network.py) |
| **Safety Shield Link Mapping** | Request-to-Controlled-Link bitmask | Bitmask | `TrafficSafetyShield` | [`TrafficSafetyShield`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/environment/safety_shield.py) |
| **Safety Fault Invariant** | Fail-safe all-red hold on invalid state | Exception | `SafetyFaultError` | [`SafetyFaultError`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/environment/safety_shield.py) |

---

## 2. Loss Functions and Mathematical Alignment

### 2.1 Joint Optimization Objective
$$\mathcal{L}_{\text{total}}(\theta, \phi, \psi) = \mathcal{L}_{\text{clip}}(\theta) + c_v \mathcal{L}_{\text{vf}}(\phi) + c_{\text{aux}} \left( \mathcal{L}_{\text{aux}} + \mathcal{L}_{\text{scoil}} \right) + \mathcal{L}_{\text{pref}}(\theta) - c_{\text{ent}} \mathcal{H}(\pi_\theta)$$

Implemented in [`InteractionAwarePPOAgent.update`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/training/interaction_aware_ppo.py#L391-L400):
```python
loss = (
    policy_loss
    + self.vf_coef * value_loss
    + self.aux_lambda * (aux_loss + scoil_pred_loss)
    + pref_loss
    - self.ent_coef * entropy.mean()
)
```

### 2.2 PPO Clipped Surrogate Loss with Proposed Action Alignment
$$\mathcal{L}_{\text{clip}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$
where $r_t(\theta) = \frac{\pi_\theta(a_{\text{proposed}} \mid s_t)}{\pi_{\theta_{\text{old}}}(a_{\text{proposed}} \mid s_t)}$.

Implemented in [`InteractionAwarePPOAgent.update`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/training/interaction_aware_ppo.py#L350-L362):
```python
for k, logits in enumerate(logits_list):
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    new_log_probs += dist.log_prob(b_act[:, k])
    entropy += dist.entropy()

ratio = torch.exp(new_log_probs - b_old_lp)
surr1 = ratio * b_adv
surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * b_adv
policy_loss = -torch.min(surr1, surr2).mean()
```

### 2.3 Auxiliary Physical Transition Conditioning on Executed Actions
The auxiliary residual objective transition model and SCOIL cross-objective predictive loss are explicitly conditioned on $\mathbf{a}_t^{\text{exec}}$:
```python
pred_next_obj = self.network.predict_next_objectives(manifold, b_exec_act, b_obj)
aux_loss = F.mse_loss(pred_next_obj, b_next_obj)
scoil_pred_loss = self.interaction_learner.compute_predictive_loss(
    live_int_mat, b_obj, b_exec_act, b_next_obj, uncertainty=live_unc
)
```

### 2.4 Preference Regularization and Anti-Collapse Barrier
$$\mathcal{L}_{\text{pref}} = c_{\text{KL}} D_{\text{KL}}(\mathbf{u}_t \parallel \mathbf{w}_t) - c_{\text{ent}} \mathcal{H}(\mathbf{w}_t) + \lambda_{\text{smooth}} \mathbb{E}\left[\|\mathbf{w}_t - \mathbf{w}_{t-1}\|_2^2\right]$$

Implemented in [`InteractionAwarePPOAgent.update`](file:///d:/DOIM-Traffic-RL/DOIM-Traffic-RL/training/interaction_aware_ppo.py#L375-L390):
```python
u_t = (b_obj + 1e-4) / (b_obj + 1e-4).sum(dim=-1, keepdim=True)
kl_pref = F.kl_div(torch.log(weights + 1e-8), u_t, reduction="batchmean")
pref_entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
temporal_diff_sq = ((weights - b_prev_w) ** 2).sum(dim=-1)
smooth_loss = (b_mask * temporal_diff_sq).sum() / mask_sum
pref_loss = self.pref_kl_coef * kl_pref - self.pref_entropy_coef * pref_entropy + self.smooth_lambda * smooth_loss
```

---

## 3. End-to-End Autograd Tape Verification

The architecture recomputes all spatial GCN and temporal GRU layers dynamically during the PPO optimization loop (`InteractionAwarePPOAgent.update`). No cached or detached representations are passed to the loss. Consequently:
- Gradients from `policy_loss`, `value_loss`, `aux_loss`, and `pref_loss` flow backwards through `manifold_mlp`, `gru`, and `gcn1`/`gcn2`.
- Gradients from `scoil_pred_loss` train the bilinear projections $\mathbf{W}_q, \mathbf{W}_k$ and uncertainty head $\mathbf{U}_t$.
- All gradients are jointly clipped using `nn.utils.clip_grad_norm_` at threshold $\rho_{\text{clip}} = 0.5$.

