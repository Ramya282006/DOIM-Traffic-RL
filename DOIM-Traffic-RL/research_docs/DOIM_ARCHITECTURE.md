# Dynamic Objective Interaction Manifold (DOIM): Mathematical Architecture Specification

**Document:** `research_docs/DOIM_ARCHITECTURE.md`  
**Version:** 2.0 (Trainable Differentiable Implementation)  
**Standard:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Date:** October 31, 2026  

---

## 1. Overview and Core Motivation

In urban traffic signal control under non-stationary traffic demands, competing traffic objectives (waiting delay, queue length, carbon dioxide emissions, vehicle throughput, and corridor fairness) exhibit dynamic, time-varying interactions. Traditional multi-objective reinforcement learning (MORL) approaches typically rely on either:
1. Static scalarization weights fixed a priori, which cannot adapt to demand shifts.
2. Post-hoc empirical correlations (e.g. rolling Pearson correlation), which are purely descriptive, confounded by shared demand surges, and disconnected from gradient-based policy updates.

The **Dynamic Objective Interaction Manifold (DOIM)** architecture bridges this gap by formulating a fully differentiable neural manifold that learns a state-conditioned representation of objective interactions, predicts downstream objective trade-offs, and adaptively guides multi-objective policy optimization via end-to-end backpropagation.

---

## 2. Mathematical Definition of the Five Traffic Objectives

Let $\mathcal{N} = \{1, \dots, N\}$ denote the set of intersections ($N=9$ in the 3×3 grid), and let $\mathcal{V}_t$ denote the set of active vehicles in the network at discrete simulation time $t \in \mathbb{N}$ (time step $\Delta t = 1.0\text{ s}$).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE FIVE TRAFFIC OBJECTIVES                            │
├────┬─────────────┬──────────────────────────────────────┬─────────────┬───────────────┤
│ ID │ Name        │ Mathematical Formulation             │ Units       │ Direction     │
├────┼─────────────┼──────────────────────────────────────┼─────────────┼───────────────┤
│ 1  │ Waiting Time│ J_wait(t) = ∑_{v ∈ V_t} W_v(t)       │ veh·s       │ Minimize (↓)  │
│ 2  │ Queue Length│ J_queue(t) = ∑_{l ∈ L_in} N_halt(l,t)│ vehicles    │ Minimize (↓)  │
│ 3  │ CO2 Emission│ J_co2(t) = ∑_{v ∈ V_t} e_co2(v,t)·Δt │ milligrams  │ Minimize (↓)  │
│ 4  │ Throughput  │ J_thru(t) = N_completed(t)           │ vehicles    │ Maximize (↑)  │
│ 5  │ Fairness    │ J_fair(t) = 1.0 - Jain(D_t)          │ [0, 1]      │ Minimize (↓)  │
└────┴─────────────┴──────────────────────────────────────┴─────────────┴───────────────┘
```

### Detailed Objective Formulations:

1. **Total Waiting Time ($J_{\text{wait}}$):**
   $$J_{\text{wait}}(t) = \sum_{v \in \mathcal{V}_t} W_v(t)$$
   where $W_v(t)$ is the accumulated waiting time (seconds with speed $< 0.1\text{ m/s}$) of vehicle $v$.
   - **Physical Interpretation:** Aggregate cumulative delay experienced by travelers currently in the network.
   - **Conflicts:** Clearing one arterial corridor extends the wait times of cross-street vehicles.

2. **Total Queue Length ($J_{\text{queue}}$):**
   $$J_{\text{queue}}(t) = \sum_{l \in \mathcal{L}_{\text{in}}} N_{\text{halt}}(l, t)$$
   where $N_{\text{halt}}(l, t)$ is the count of stopped vehicles on incoming lane $l$.
   - **Physical Interpretation:** Physical queue occupancy; directly governs spillback and gridlock risks.
   - **Conflicts:** Flushing queues from an upstream intersection can overload downstream links, generating secondary bottlenecks.

3. **Instantaneous Carbon Dioxide Emissions ($J_{\text{co2}}$):**
   $$J_{\text{co2}}(t) = \sum_{v \in \mathcal{V}_t} e_{\text{co2}}(v, t) \cdot \Delta t$$
   where $e_{\text{co2}}(v, t)$ is the instantaneous vehicle emission rate calculated by the HBEFA3 emission model in SUMO ($\text{mg/s}$).
   - **Physical Interpretation:** Direct environmental footprint. Stop-and-go acceleration cycles emit 300–500% more $\text{CO}_2$ per second than steady cruising.
   - **Conflicts:** Frequent signal changes to minimize instantaneous queue lengths force vehicles into repetitive braking and acceleration, dramatically elevating total $\text{CO}_2$ emissions.

4. **Network Throughput ($J_{\text{thru}}$):**
   $$J_{\text{thru}}(t) = N_{\text{completed}}(t)$$
   where $N_{\text{completed}}(t)$ is the exact count of vehicles successfully reaching their destination during simulation step $t$.
   - **Physical Interpretation:** Network discharge capacity and trip completion velocity.
   - **Conflicts:** Maximizing throughput inherently prioritizes high-volume corridors, starving low-volume residential/cross streets.

5. **Corridor Delay Disparity / Jain Fairness ($J_{\text{fair}}$):**
   Let $\bar{d}_k(t)$ denote the average delay of vehicles entering approach corridor $k \in \{1, \dots, K\}$. Jain's Fairness Index $\mathcal{J}(t) \in [1/K, 1]$ is:
   $$\mathcal{J}(t) = \frac{\left( \sum_{k=1}^K \bar{d}_k(t) \right)^2}{K \sum_{k=1}^K \bar{d}_k(t)^2}$$
   The fairness objective penalty is defined as normalized inequity:
   $$J_{\text{fair}}(t) = 1.0 - \mathcal{J}(t) \in [0, 1 - 1/K]$$
   - **Physical Interpretation:** Uniformity of service quality across all entering traffic approaches.
   - **Conflicts:** Allocating green time to minor approaches with minimal demand to maintain equity reduces total system-wide vehicle throughput.

---

## 3. Objective Normalization and Urgency Mapping

To prevent numerical domination by large scale metrics (e.g. $\text{CO}_2 \approx 10^5\text{ mg}$ vs queue $\approx 10^1$), each objective $J_i(t)$ is mapped to a bounded urgency score $z_i(t) \in [0, 1]$ using calibrated lower and upper bounds $[L_i, U_i]$:

$$\tilde{z}_i(t) = \text{clip}\left( \frac{J_i(t) - L_i}{U_i - L_i}, 0.0, 1.0 \right)$$

For minimization objectives ($i \in \{1, 2, 3, 5\}$):
$$z_i(t) = \tilde{z}_i(t)$$

For the maximization objective (Throughput, $i = 4$):
$$z_4(t) = 1.0 - \tilde{z}_4(t)$$

Thus, $z_i(t) = 0$ represents optimal performance (zero urgency), while $z_i(t) = 1$ represents severe degradation (maximum urgency).

---

## 4. End-to-End Pipeline Architecture

```
[SUMO Traffic State]
        │  X_t ∈ ℝ^(9×5) (Queue, Waiting, Speed, Density, CO2 per node)
        ▼
[Spatial GNN Encoder (Kipf-Welling GCN)]
        │  H_t = GCN(X_t, E) ∈ ℝ^(9×64)
        │  h_t^spatial = MeanPool(H_t) ∈ ℝ^64
        ▼
[Temporal History Buffer]
        │  S_t = [h_(t-L+1)^spatial, ..., h_t^spatial] ∈ ℝ^(L×64)  (L = 10)
        ▼
[Temporal GRU Encoder]
        │  h_t^temporal = GRU(S_t) ∈ ℝ^64
        │
        ├── [Normalized Objectives z_t ∈ [0,1]^5] ──► [Objective MLP] ──► e_t^obj ∈ ℝ^32
        │
        └── [Interaction Matrix M_t ∈ ℝ^(5×5)]   ──► [Interaction MLP] ──► e_t^int ∈ ℝ^32
        │
        ▼
[Manifold Fusion Core]
        │  fused = [h_t^temporal, e_t^obj, e_t^int] ∈ ℝ^128
        │  m_t = LayerNorm(Tanh(MLP(fused))) ∈ ℝ^64
        │
        ├──► [Preference Head] ──► w_t = Softmax(MLP(m_t) / τ) ∈ Δ^4
        │
        ├──► [Actor Policy]    ──► π_θ(a_t | m_t, s_t^raw) (Multi-Discrete 5 TLS)
        │
        ├──► [Critic Value]    ──► V_ψ(m_t, s_t^raw) ∈ ℝ
        │
        └──► [Auxiliary Head]  ──► ẑ_(t+1) = MLP(m_t, a_t) ∈ ℝ^5 (Transition Predictor)
```

---

## 5. Tensor Dimension Reference

| Layer / Tensor | Symbol | Shape (Batched) | Description |
|---|---|---|---|
| Node Feature Matrix | $X_t$ | `[B, 9, 5]` | Raw features across 9 intersections |
| Grid Edge Index | $E$ | `[2, 24]` | Undirected physical topology edges |
| Node Embeddings | $H_t$ | `[B, 9, 64]` | Post-GCN spatial features per intersection |
| Pooled Spatial Vector | $h_t^{\text{spatial}}$ | `[B, 64]` | Global network spatial embedding |
| Temporal Sequence | $S_t$ | `[B, 10, 64]` | 10-step rolling window of spatial vectors |
| Temporal Hidden State | $h_t^{\text{temporal}}$ | `[B, 64]` | Final hidden state of GRU cell |
| Normalized Objectives | $z_t$ | `[B, 5]` | Bounded urgency scores in $[0, 1]^5$ |
| Objective Embedding | $e_t^{\text{obj}}$ | `[B, 32]` | Latent representation of objective state |
| Interaction Tensor | $M_t$ | `[B, 5, 5]` | State-conditioned objective interaction operator |
| Interaction Embedding | $e_t^{\text{int}}$ | `[B, 32]` | Latent representation of objective dependencies |
| Fused Latent Vector | $f_t$ | `[B, 128]` | Concatenated `[h_temporal, e_obj, e_int]` |
| **Latent Manifold** | $m_t$ | `[B, 64]` | Latent objective interaction manifold |
| **Adaptive Preferences**| $w_t$ | `[B, 5]` | Softmax probability simplex weights ($\sum w = 1$) |
| Raw Local State | $s_t^{\text{raw}}$ | `[B, 25]` | Active phases, queues, speeds for 5 controllable TLS |
| Action Logits | $logits_k$ | 5 × `[B, 2]` | Binary switch logits per controllable light |
| Critic Value | $V_t$ | `[B, 1]` | Scalar state value baseline |
| Predicted Objectives | $\hat{z}_{t+1}$ | `[B, 5]` | Auxiliary counterfactual objective forecast |

---

## 6. Optimization Objective & Unified Loss Formulation

The entire network is optimized jointly via a multi-task objective:

$$\mathcal{L}_{\text{total}}(\Theta) = \mathcal{L}_{\text{PPO}}(\theta) + c_1 \mathcal{L}_{\text{VF}}(\psi) + c_2 \mathcal{L}_{\text{aux}}(\phi) + c_3 \mathcal{L}_{\text{smooth}}(w) - c_4 \mathcal{H}[\pi_\theta]$$

Where:
1. **PPO Clipped Surrogate Loss:**
   $$\mathcal{L}_{\text{PPO}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
   where $r_t(\theta) = \frac{\pi_\theta(a_t \mid m_t, s_t^{\text{raw}})}{\pi_{\theta_{\text{old}}}(a_t \mid m_t, s_t^{\text{raw}})}$ and $\hat{A}_t$ is the Generalized Advantage Estimator (GAE).
2. **Value Function Regression Loss:**
   $$\mathcal{L}_{\text{VF}}(\psi) = \frac{1}{2} \hat{\mathbb{E}}_t \left[ \left( V_\psi(m_t, s_t^{\text{raw}}) - R_t^{\text{target}} \right)^2 \right]$$
3. **Auxiliary Transition Loss:**
   $$\mathcal{L}_{\text{aux}}(\phi) = \hat{\mathbb{E}}_t \left[ \left\| \hat{z}_{t+1}(m_t, a_t) - z_{t+1} \right\|_2^2 \right]$$
   *This ensures the manifold $m_t$ learns predictive representations of how actions impact future objective trade-offs.*
4. **Preference Smoothness Regularization:**
   $$\mathcal{L}_{\text{smooth}}(w) = \hat{\mathbb{E}}_t \left[ \left\| w_t - w_{t-1} \right\|_2^2 \right]$$
   *This penalizes high-frequency oscillations in objective weights, stabilizing the policy learning landscape.*
5. **Policy Entropy Bonus:**
   $$\mathcal{H}[\pi_\theta] = \hat{\mathbb{E}}_t \left[ - \sum_{k=1}^K \sum_{a \in \{0, 1\}} \pi_\theta(a \mid s_t) \log \pi_\theta(a \mid s_t) \right]$$

---

## 7. Resolution of Baseline Defects

| Defect ID | Prior Flawed State | New Verified Architecture |
|---|---|---|
| **DEF-01** | Modules placed in `eval()` mode under `torch.no_grad()`. | All modules are active `nn.Module` subclasses with active autograd graph during training. |
| **DEF-02** | Gradients disconnected from representation layers. | Backpropagation passes from PPO actor, critic, and auxiliary predictor directly through manifold core, GRU, and GNN. |
| **DEF-10** | Graph topology hardcoded with mismatch to controllable TLS. | GCN operates on 9 physical junctions, followed by explicit concatenation with raw controllable TLS state vectors. |
| **Simplex Guarantee** | Ad-hoc normalization with division by zero risk. | Numerically stable `Softmax(logits / \tau)` ensuring strictly $w_i > 0$ and $\sum_{i=1}^5 w_i = 1.0$. |
