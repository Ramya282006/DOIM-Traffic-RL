# DOIM-Traffic-RL: Formal Mathematical Problem Formulation
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Standard:** Rigorous Q1 Mathematical and Methodological Specification  
**Architecture:** GCN + GRU + SCOIL + DOIM + IA-MOPPO  

---

## 1. Multi-Objective POMDP Formulation

We formulate multi-intersection urban traffic signal control as a Multi-Objective Partially Observable Markov Decision Process (MO-POMDP) defined by the tuple:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \vec{\mathcal{R}}, \Omega, \mathcal{O}, \gamma \rangle$$

### 1.1 State and Observation Space
The global microscopic traffic state $s_t \in \mathcal{S}$ encompasses all vehicle positions, velocities, routes, and queue lengths across the road network $\mathcal{G} = (\mathcal{V}, \mathcal{E})$.

At each intersection $v_i \in \mathcal{V}$ ($|\mathcal{V}| = 9$), local loop detectors yield a feature vector $x_{i, t} \in \mathbb{R}^5$:
$$x_{i, t} = [q_{i, t}, w_{i, t}, v_{i, t}, \rho_{i, t}, p_{i, t}]^T$$
where $q_{i, t}$ is approach queue length, $w_{i, t}$ is accumulated waiting time, $v_{i, t}$ is mean approach speed, $\rho_{i, t}$ is vehicle density, and $p_{i, t}$ is the current phase indicator. The full network feature snapshot is $X_t \in \mathbb{R}^{|\mathcal{V}| \times 5}$.

The policy receives a localized observation vector $o_t \in \Omega$ comprising:
1. The dynamic latent manifold state $m_t \in \mathbb{R}^{64}$.
2. The current preference simplex vector $w_t \in \Delta^4$.
3. The local physical states of the 5 controllable traffic lights: $s_{raw, t} \in \mathbb{R}^{25}$.

### 1.2 Action Space
Traffic control is executed across $N_c = 5$ controllable signalized junctions:
$$\mathcal{A} = \{0, 1\}^{N_c}$$
For junction $j \in \{1, \dots, N_c\}$, action $a_{j, t} = 0$ indicates maintaining the current phase, while $a_{j, t} = 1$ requests a transition to the next nominal signal phase.

---

## 2. Spatial-Temporal Traffic Representation

### 2.1 Spatial Graph Convolution (GCN)
Spatial dependencies across neighboring intersections are captured using a 2-layer Graph Convolutional Network:
$$H^{(1)}_t = \text{ReLU}\left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X_t W^{(0)}_{G} \right)$$
$$H^{(2)}_t = \text{ReLU}\left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(1)}_t W^{(1)}_{G} \right)$$
where $\tilde{A} = A + I_N$ is the adjacency matrix with self-loops, and $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$. The node-level spatial embeddings $H^{(2)}_t \in \mathbb{R}^{N \times d_s}$ are aggregated via permutation-invariant mean pooling:
$$h_{spatial, t} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} H^{(2)}_{i, t} \in \mathbb{R}^{d_s}$$

### 2.2 Temporal GRU Encoding
To capture queue buildup and clearance dynamics across a rolling horizon of $L = 10$ steps, the spatial feature sequence $(h_{spatial, t-L+1}, \dots, h_{spatial, t})$ is processed by a Gated Recurrent Unit (GRU):
$$h_{t} = \text{GRU}(h_{spatial, t}, h_{t-1}) \in \mathbb{R}^{d_h}$$
where $h_t \in \mathbb{R}^{64}$ represents the temporal traffic context.

---

## 3. Authoritative Multi-Objective Formulation & Normalization

The system optimizes $K = 5$ distinct traffic performance dimensions:
$$\vec{z}_{raw, t} = [z_{\text{wait}}, z_{\text{queue}}, z_{\text{co2}}, z_{\text{tp}}, z_{\text{fair}}]^T$$

1. **Waiting Time ($z_{\text{wait}}$):** Total vehicle-seconds spent halting across all intersections ($\text{speed} < 0.1 \text{ m/s}$):
   $$z_{\text{wait}, t} = \sum_{i \in \mathcal{V}} w_{i, t} \quad [\text{veh}\cdot\text{s}, \text{ minimize}]$$
2. **Queue Length ($z_{\text{queue}}$):** Total number of halted vehicles on incoming approach lanes:
   $$z_{\text{queue}, t} = \sum_{i \in \mathcal{V}} q_{i, t} \quad [\text{vehicles}, \text{ minimize}]$$
3. **CO₂ Emissions ($z_{\text{co2}}$):** Step-integrated fuel combustion mass derived from the HBEFA3 emission model:
   $$z_{\text{co2}, t} = \sum_{v \in \mathcal{V}_{active}} \text{EM}_{co2}(v, t) \cdot \Delta t \quad [\text{mg}, \text{ minimize}]$$
4. **Throughput ($z_{\text{tp}}$):** Number of vehicles successfully completing their routes during step $t$:
   $$z_{\text{tp}, t} = N_{\text{arrived}}(t) \quad [\text{vehicles}, \text{ maximize}]$$
5. **Corridor Delay Fairness Disparity ($z_{\text{fair}}$):** Dimensionless inequity across arterial corridors:
   $$z_{\text{fair}, t} = 1.0 - J(\vec{d}_t) = 1.0 - \frac{\left( \sum_{c=1}^C d_{c, t} \right)^2}{C \sum_{c=1}^C d_{c, t}^2} \quad [\in [0, 1], \text{ minimize}]$$

### 3.1 Normalization into Standardized Urgency Space
Each raw objective is mapped into an urgency score $z_{k, t} \in [0.0, 1.0]$:
$$z_{k, t} = \begin{cases}
\frac{\text{clip}(z_{raw, k, t}, L_k, U_k) - L_k}{U_k - L_k}, & \text{for minimization } (k \in \{0, 1, 2, 4\}) \\
1.0 - \frac{\text{clip}(z_{raw, k, t}, L_k, U_k) - L_k}{U_k - L_k}, & \text{for maximization (throughput, } k=3)
\end{cases}$$
where $0.0$ signifies ideal performance (low urgency) and $1.0$ indicates critical congestion/inequity (high urgency).

---

## 4. State-Conditioned Objective Interaction Learning (SCOIL)

### 4.1 Bilinear Interaction Query-Key Coupling
Let $c_t = [h_t, \vec{z}_t] \in \mathbb{R}^{d_h + K}$ represent the joint context. SCOIL projects $c_t$ into $K$ separate queries and keys:
$$Q_t = \text{reshape}(W_q \phi(c_t), [K, d_k]), \quad K_t = \text{reshape}(W_k \phi(c_t), [K, d_k])$$
The raw bilinear coupling is:
$$M_{\text{bilinear}, t} = \frac{Q_t K_t^T}{\sqrt{d_k}} \in \mathbb{R}^{K \times K}$$

### 4.2 Asymmetric Coupling and Self-Interaction
To model directional temporal lags (e.g. queue buildup causes waiting time increases, but queue dissipation does not instantaneously clear accumulated delay):
$$M_{\text{raw}, t} = M_{\text{bilinear}, t} + \text{reshape}(W_{\text{asym}} \phi(c_t), [K, K])$$
$$M_t = \tanh(M_{\text{raw}, t}) \in [-1, 1]^{K \times K}$$
Self-interaction along the diagonal is anchored to unity:
$$M_{t, kk} = 1.0 \quad \forall k \in \{1, \dots, K\}$$

### 4.3 Predictive Uncertainty & Bayesian Shrinkage
SCOIL estimates predictive trade-off dispersion:
$$U_t = \text{Softplus}(W_u \phi(c_t)) \in \mathbb{R}_+^{K \times K}, \quad U_{t, kk} = 0.0$$
The interaction matrix is adaptively shrunk toward a conservative prior $M_{\text{prior}}$:
$$w_{\text{shrink}, t} = \frac{1}{1 + \lambda_{\text{shrink}} U_t}$$
$$\tilde{M}_t = w_{\text{shrink}, t} \odot M_t + (1 - w_{\text{shrink}, t}) \odot M_{\text{prior}}$$

### 4.4 Auxiliary Predictive Dynamics Loss
The interaction operator is trained via counterfactual objective transition prediction:
$$\Delta \vec{z}_{\text{primary}, t} = W_a a_t \in \mathbb{R}^K$$
$$\Delta \vec{z}_{\text{cross}, t} = \tilde{M}_t \Delta \vec{z}_{\text{primary}, t} \in \mathbb{R}^K$$
$$\hat{\vec{z}}_{t+1} = \text{clamp}(\vec{z}_t + \Delta \vec{z}_{\text{cross}, t}, 0.0, 1.0)$$

With heteroscedastic dispersion $\sigma_k^2 = \frac{1}{K} \sum_j U_{t, kj} + \epsilon$:
$$\mathcal{L}_{\text{scoil}} = \frac{1}{2} \sum_{k=1}^K \left( \frac{(\hat{z}_{t+1, k} - z_{t+1, k})^2}{\sigma_k^2} + \log \sigma_k^2 \right)$$

---

## 5. Dynamic Objective Interaction Manifold (DOIM)

### 5.1 Manifold Fusion Core
The latent manifold $m_t$ integrates the temporal traffic representation $h_t$, current normalized objective urgency $\vec{z}_t$, and vector of predicted cross-interactions $\text{vec}(\tilde{M}_t)$:
$$u_t = [h_t \in \mathbb{R}^{64} \parallel \text{MLP}_z(\vec{z}_t) \in \mathbb{R}^{32} \parallel \text{MLP}_m(\text{vec}(\tilde{M}_t)) \in \mathbb{R}^{32}]$$
$$m_t = \tanh\left( \text{LayerNorm}(W_m u_t + b_m) \right) \in \mathbb{R}^{64}$$

### 5.2 Dynamic Preference Simplex
Preference weights $w_t \in \Delta^4$ are generated via temperature-scaled softmax:
$$w_{t, k} = \frac{\exp(g_k(m_t) / \tau)}{\sum_{j=1}^K \exp(g_j(m_t) / \tau)}, \quad \sum_{k=1}^K w_{t, k} = 1.0, \quad w_{t, k} > 0$$

### 5.3 Scalarized Reward Function
The scalarized reinforcement learning reward is the negative preference-weighted urgency:
$$r_t = - \sum_{k=1}^K w_{t, k} \cdot z_{k, t} \in [-1.0, 0.0]$$

---

## 6. Interaction-Aware PPO (IA-MOPPO) Optimization

### 6.1 Policy & Value Objectives
The actor policy factorizes into independent categorical distributions across controllable signals:
$$\pi_\theta(a_t \mid o_t) = \prod_{j=1}^{N_c} \pi_{\theta, j}(a_{j, t} \mid m_t, s_{\text{raw}, t})$$
Clipped surrogate objective with Generalized Advantage Estimation (GAE):
$$\mathcal{L}_{\text{policy}}(\theta) = - \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
$$\mathcal{L}_{\text{value}}(\psi) = \mathbb{E}_t \left[ (V_\psi(m_t, s_{\text{raw}, t}) - R_t)^2 \right]$$

### 6.2 Anti-Collapse & Temporal Smoothness Regularization
To prevent preference collapse and high-frequency preference chatter:
1. **Urgency Alignment Barrier:** Kullback-Leibler divergence to physical urgency distribution $u_t = \frac{\vec{z}_t + \delta}{\sum (\vec{z}_t + \delta)}$:
   $$\mathcal{L}_{\text{KL}} = D_{\text{KL}}(w_t \parallel u_t)$$
2. **Entropy Barrier:** Penalizes degenerate single-objective collapse:
   $$\mathcal{H}(w_t) = - \sum_{k=1}^K w_{t, k} \log(w_{t, k} + \epsilon)$$
3. **Temporal Smoothness Loss:** Penalizes preference chattering between consecutive steps of the same episode:
   $$\mathcal{L}_{\text{smooth}} = \frac{\sum_{t=1}^B M_t \|w_t - w_{t-1}\|^2_2}{\sum_{t=1}^B M_t + \epsilon}$$
   where $M_t = 1.0 - \mathbb{I}(\text{reset at } t)$.

### 6.3 Joint Objective Function
All parameters are optimized end-to-end via joint backpropagation:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + c_v \mathcal{L}_{\text{value}} + \lambda_{\text{aux}} \mathcal{L}_{\text{scoil}} + c_{\text{kl}} \mathcal{L}_{\text{KL}} - c_{\text{ent}} \mathcal{H}(w_t) + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}} - c_{\pi} \mathcal{H}(\pi_\theta)$$

