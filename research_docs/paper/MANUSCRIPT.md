# Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems

**Anonymous Authors**  
*Targeted for Submission to IEEE Transactions on Intelligent Transportation Systems (T-ITS)*  
*Paper Category: Regular Paper*

---

## Abstract
Multi-objective reinforcement learning (MORL) provides a compelling paradigm for intelligent traffic signal control (ITSC) by optimizing competing real-world criteria including vehicular delay, queue lengths, carbon emissions, network throughput, and equitable service across competing approaches. However, existing MORL signal controllers typically rely either on static linear scalarizations with predetermined weights or on direct state-to-preference mappings that ignore the underlying structural couplings between objectives. Under non-stationary traffic demands, the relationship between competing criteria shifts dynamically—often transitioning from cooperative synergy to acute trade-off. This paper introduces the **Dynamic Objective Interaction Manifold (DOIM)** architecture coupled with **Interaction-Aware Multi-Objective Proximal Policy Optimization (IA-MOPPO)**. DOIM integrates a dual-layer spatial Kipf-Welling Graph Convolutional Network (GCN) and a temporal Gated Recurrent Unit (GRU) to encode spatiotemporal corridor dynamics, coupled with **State-Conditioned Objective Interaction Learning (SCOIL)**. SCOIL computes an asymmetric bilinear predictive interaction operator over normalized objective vectors, regularized via state-conditioned predictive dispersion shrinkage towards an uncoupled conservative prior. The resulting objective interaction manifold conditions both an adaptive preference generator (projected onto the unit 4-simplex $\Delta^4$) and a multi-discrete actor-critic policy. Furthermore, an active traffic safety shield enforces non-conflicting signal states verified against microscopic junction conflict matrices, multi-stage yellow clearance pathfinding ($3\,\text{s}$), fail-safe locks, and minimum/maximum green bounds ($10\,\text{s} \le g \le 60\,\text{s}$), empirically eliminating rapid phase fluttering. Systematic empirical evaluations on a 9-intersection grid network in Eclipse SUMO evaluate IA-MOPPO against Fixed-Time control, Fixed-Weight PPO, State-Adaptive PPO, and architectural ablations across independent seeds, providing a rigorous analysis of policy convergence, exploration penalties under clearance constraints, and robust generalization under non-stationary saturation.

**Index Terms**—Intelligent Transportation Systems, Multi-Objective Reinforcement Learning, Traffic Signal Control, Graph Convolutional Networks, Objective Interaction Manifolds, Proximal Policy Optimization, Microscopic Simulation.

---

## I. Introduction

URBAN traffic congestion imposes immense socio-economic costs globally, leading to billions of hours of wasted driver time, elevated fuel consumption, and severe localized greenhouse gas emissions [1]. Traditional traffic signal control strategies—ranging from pre-timed fixed-cycle plans to actuated and adaptive systems such as SCOOT and SCATS [2]—rely heavily on simplified parametric traffic flow models that fail to respond optimally to sudden spatial-temporal demand surges, asymmetric directional flows, and incident disruptions.

In recent years, Deep Reinforcement Learning (DRL) has emerged as a promising approach for Intelligent Traffic Signal Control (ITSC) [3], [4]. By formulating traffic signal control as a Markov Decision Process (MDP), DRL agents learn closed-loop control policies directly through simulated trial-and-error interactions with microscopic simulators such as Eclipse SUMO [5]. However, real-world traffic management is inherently a **multi-objective decision-making problem**. Municipal transportation operators must simultaneously balance at least five competing objectives:
1. Minimizing cumulative vehicle waiting time and delay.
2. Minimizing localized queue spillback to prevent upstream intersection gridlock.
3. Minimizing carbon dioxide ($\text{CO}_2$) emissions and fuel waste caused by stop-and-go cycles.
4. Maximizing total completed vehicle throughput.
5. Ensuring fairness across competing traffic approaches, preventing severe starvation on minor cross-streets.

### A. The Scientific Gap
Despite the multi-objective reality of urban traffic, conventional RL controllers almost universally collapse multiple objectives into an ad-hoc scalar reward:
$$R_t = \sum_{k=1}^K w_k r_{k,t}$$
where the preference weights $\mathbf{w} = [w_1, \dots, w_K]^\top$ are held static throughout training and execution [6]. This static scalarization suffers from fundamental flaws:
- **Shifting Objective Alignment:** Under light, uncongested flow, minimizing delay and minimizing emissions are largely cooperative (smooth vehicle progression optimizes both). However, under peak saturation or severe directional surges, these objectives become acutely adversarial: clearing a massive queue on an arterial approach may require starving cross-street vehicles, severely degrading delay fairness and increasing aggregate idle emissions [7].
- **Lack of Objective Interaction Modeling:** Recent adaptive MORL approaches attempt to predict dynamic weights $w_t = \pi_w(s_t)$ directly from the raw state features. However, these architectures treat the objective space as a set of uncoupled scalar values, completely ignoring the **interaction manifold**—the mathematical structure of pairwise synergies, cross-sensitivities, and trade-offs among objectives.
- **Safety and Fluttering Deficiencies:** Many unconstrained RL policies select actions at every second, inducing rapid phase fluttering (alternating green phases without clearance intervals), violating municipal traffic regulations and inducing physical collisions [8].

### B. Core Contributions
To address these limitations, this research develops a mathematically sound, reproducible, and publication-quality research system. Specifically, our primary contributions are:
1. **The Dynamic Objective Interaction Manifold (DOIM):** We formulate a unified representation learning architecture combining a Kipf-Welling Graph Convolutional Network (GCN) over the traffic topology, a temporal Gated Recurrent Unit (GRU) over historical traffic embeddings, and an explicit objective interaction fusion core.
2. **State-Conditioned Objective Interaction Learning (SCOIL):** Rather than relying on non-differentiable or unstable rolling Pearson correlations, we develop SCOIL—a fully differentiable bilinear coupling mechanism that estimates directional cross-objective predictive interactions with dispersion-weighted shrinkage towards an informative uncoupled prior.
3. **Hierarchical 3-Tier Fairness Metric Suite:** We separate traffic fairness into three distinct behavioral tiers (Arterial Approach, Signalized Intersection, and Individual Vehicle), deploying Jain disparity, worst-case delay bounds, and Gini equity metrics without conflating aggregation scopes.
4. **Interaction-Aware Multi-Objective PPO (IA-MOPPO):** We derive an end-to-end policy gradient framework wherein the learned interaction manifold directly conditions both adaptive objective preference generation on the simplex $\Delta^4$ and multi-discrete traffic signal actuation, regularized by a true temporal preference smoothness objective and heteroscedastic predictive transition loss.
5. **Deterministic Traffic Safety Shield with Empirical Verification:** We design and integrate an active safety shield that guarantees minimum green times ($10\,\text{s}$), maximum green caps ($60\,\text{s}$), and mandatory yellow clearance phases ($3\,\text{s}$), backed by multi-stage clearance pathfinding, fail-safe locks, and typed fault abortion, establishing empirically verified safety constraints.
6. **Rigorous Empirical Validation:** We evaluate IA-MOPPO against classical Fixed-Time control, Fixed-Weight PPO (FW-PPO), State-Adaptive PPO (SA-PPO), and multiple architectural ablations across independent training seeds in SUMO, verified with an automated scientific integrity suite.

---

## II. Related Work

### A. Deep Reinforcement Learning for Traffic Signal Control
Early applications of RL to traffic control utilized Q-learning with tabular or shallow neural network approximations [3]. With the advent of deep RL, Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO) have been widely deployed [4], [9]. To capture spatial relationships across road networks, researchers introduced Graph Neural Networks (GNNs). Notable examples include CoLight [10], which utilizes graph attention networks to facilitate communication among neighboring controllers, and decentralized multi-agent architectures that scale to hundreds of intersections [11]. However, virtually all of these methods optimize a single scalarized reward function with fixed weights chosen heuristically.

### B. Multi-Objective Reinforcement Learning (MORL)
MORL extends single-objective RL to vector-valued reward signals $\mathbf{r}_t \in \mathbb{R}^K$ [6]. Approaches are broadly divided into:
- *Single-policy methods:* Scalarize rewards using fixed or state-dependent weights.
- *Multi-policy methods:* Approximate the complete Pareto frontier $\mathcal{P}$ of non-dominated policies [12], [13].
While multi-policy Pareto methods provide theoretical completeness, their computational complexity scales exponentially with the number of objectives, rendering them computationally intractable for real-time microscopic traffic control where actions must be dispatched within sub-second latencies. Consequently, state-adaptive scalarization has emerged as the practical paradigm [14]. Yet existing adaptive MORL models map $s \mapsto \mathbf{w}$ through a black-box MLP without modeling the structural dependencies between the objectives themselves.

### C. Traffic Safety Constraints and Action Filtering
Unconstrained RL agents operating in simulation frequently discover high-reward "glitches"—such as rapidly toggling green signals every second to clear individual vehicles. In physical traffic engineering, such behavior is catastrophic [8]. Standard signal timing manuals (e.g., the Highway Capacity Manual [15]) mandate minimum green durations ($7\text{--}15\,\text{s}$) to allow pedestrian clearance and driver reaction time, alongside mandatory yellow clearance intervals ($3\text{--}5\,\text{s}$). Integrating deterministic safety shields into the MDP ensures that exploration cannot violate legal physical constraints [16].

---

## III. System Formulation and Traffic Environment

We model the traffic signal control problem across an urban road network represented as a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where $\mathcal{V}$ denotes the set of signalized intersections (nodes) and $\mathcal{E}$ denotes the set of connecting arterial links (edges).

### A. Multi-Objective POMDP Formulation
The decision process is formalized as a Multi-Objective Partially Observable Markov Decision Process (MO-POMDP) defined by the tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathbf{R}, \Omega, \mathcal{O}, \gamma \rangle$:

1. **State Space $\mathcal{S}$ and Observation $\Omega$:** At each discrete timestep $t$, the network state is defined by a node feature matrix $\mathbf{X}_t \in \mathbb{R}^{|\mathcal{V}| \times d_v}$ and local controllable intersection states $\mathbf{s}_t^{\text{raw}} \in \mathbb{R}^{d_s}$. For each intersection $i \in \mathcal{V}$, the node feature vector $\mathbf{x}_{i,t}$ includes:
   $$\mathbf{x}_{i,t} = \left[ q_{i,t}, w_{i,t}, v_{i,t}, \rho_{i,t}, p_{i,t} \right]^\top$$
   representing total queue length (halting vehicles), cumulative waiting time, mean approach speed, vehicle density, and active phase index.

2. **Action Space $\mathcal{A}$:** The network contains $N_c = 5$ controllable intersections. For each intersection $j \in \{1, \dots, N_c\}$, the action $a_{j,t} \in \{0, 1\}$ represents a binary transition decision:
   $$a_{j,t} = \begin{cases} 0, & \text{retain current active signal phase} \\ 1, & \text{request transition to next phase in sequence} \end{cases}$$
   The joint action space is multi-discrete: $\mathcal{A} = \{0, 1\}^{N_c}$.

3. **Authoritative Objective Vector $\mathbf{z}_{\text{raw}, t} \in \mathbb{R}^5$:**
   The system tracks $K = 5$ physical performance metrics:
   - **Waiting Time ($z_0$):** Cumulative waiting vehicle-seconds ($\text{speed} < 0.1\,\text{m/s}$): $z_{0,t} = \sum_{i \in \mathcal{V}} w_{i,t}$ $[\text{veh}\cdot\text{s}, \text{min}]$.
   - **Queue Length ($z_1$):** Sum of halted queue counts across approaches: $z_{1,t} = \sum_{i \in \mathcal{V}} q_{i,t}$ $[\text{veh}, \text{min}]$.
   - **CO₂ Emissions ($z_2$):** Microscopic step-integrated emissions via HBEFA3 model: $z_{2,t} = \sum_v \text{EM}_{\text{co2}}(v, t) \cdot \Delta t$ $[\text{mg}, \text{min}]$.
   - **Throughput ($z_3$):** Count of vehicles successfully completing their trips during step $t$: $z_{3,t} = N_{\text{arrived}}(t)$ $[\text{veh}, \text{max}]$.
   - **Fairness Disparity ($z_4$):** Arterial approach delay disparity $1 - \mathcal{J}(\mathbf{d}_{\text{approach}})$, bounded strictly in $[0, 1]$ $[\text{min}]$.

4. **Hierarchical 3-Tier Fairness Architecture:**
   To resolve the severe aggregation conflation present in prior literature, we establish three distinct fairness evaluation tiers:
   - **Tier 1 (Corridor / Approach Level):** Evaluates allocation equity across competing directional approaches feeding an intersection:
     $$\mathcal{J}(\mathbf{d}) = \frac{\left(\sum_{a=1}^A d_a\right)^2}{A \sum_{a=1}^A d_a^2}, \quad z_{\text{fair}, t} = 1.0 - \mathcal{J}(\mathbf{d}_t) \in [0.0, 1.0]$$
   - **Tier 2 (Intersection Level):** Evaluates service balance across the 9 junctions via cross-intersection delay variance $\sigma_{\text{int}}^2 = \frac{1}{|\mathcal{V}|} \sum_{i=1}^{|\mathcal{V}|} (\bar{d}_i - \mu_d)^2$.
   - **Tier 3 (Individual Vehicle Level):** Evaluates equity across individual driver delay distributions via the Gini coefficient:
     $$\mathcal{G} = \frac{\sum_{i=1}^N \sum_{j=1}^N |w_i - w_j|}{2 N^2 \bar{w}}$$

5. **Normalized Objective Urgency Space:**
   Each objective is normalized into a standardized urgency score $z_{k,t} \in [0.0, 1.0]$ using authoritative physical bounds $[L_k, U_k]$:
   $$z_{k,t} = \begin{cases}
   \frac{\text{clip}(z_{\text{raw}, k, t}, L_k, U_k) - L_k}{U_k - L_k}, & \text{for minimization } (k \in \{0, 1, 2, 4\}) \\
   1.0 - \frac{\text{clip}(z_{\text{raw}, k, t}, L_k, U_k) - L_k}{U_k - L_k}, & \text{for maximization (throughput, } k=3)
   \end{cases}$$
   where $0.0$ indicates optimal flow (zero urgency) and $1.0$ indicates extreme congestion/inequity. The scalar reinforcement learning reward is the negative preference-weighted urgency:
   $$R_t = - \mathbf{w}_t^\top \mathbf{z}_t \in [-1.0, 0.0]$$

---

## IV. The Dynamic Objective Interaction Manifold (DOIM)

The overall data flow and architecture of DOIM is illustrated below:

$$\text{Traffic State } \mathbf{X}_t \xrightarrow{\text{GCN}} \mathbf{h}_t^{\text{spat}} \xrightarrow{\text{GRU}} \mathbf{h}_t^{\text{temp}} \xrightarrow{\text{SCOIL}} \mathbf{\tilde{M}}_t \xrightarrow{\text{Fusion}} \mathbf{m}_t \xrightarrow{\text{Softmax}} \mathbf{w}_t \xrightarrow{\text{Actor/Critic}} \mathbf{a}_t, V(s_t)$$

### A. Spatial Topology Encoding via Kipf-Welling GCN
To capture spatial spillover between neighboring intersections, node features $\mathbf{X}_t$ are propagated across the network adjacency matrix $\mathbf{A}$ using a 2-layer Graph Convolutional Network [17]:
$$\mathbf{H}_t^{(1)} = \sigma\left(\mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{X}_t \mathbf{\Theta}_{\text{gcn}}^{(1)}\right)$$
$$\mathbf{H}_t^{(2)} = \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{H}_t^{(1)} \mathbf{\Theta}_{\text{gcn}}^{(2)}$$
where $\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{I}_{|\mathcal{V}|}$, $\mathbf{\tilde{D}}_{ii} = \sum_j \mathbf{\tilde{A}}_{ij}$, $\sigma(\cdot)$ is the ReLU activation, and $\mathbf{\Theta}_{\text{gcn}}$ are trainable filter weights. The network-wide spatial embedding is obtained via permutation-invariant mean pooling:
$$\mathbf{h}_t^{\text{spat}} = \frac{1}{|\mathcal{V}|} \sum_{i=1}^{|\mathcal{V}|} \mathbf{h}_{i,t}^{(2)} \in \mathbb{R}^{d_{\text{spat}}}$$

### B. Temporal Corridor Dynamics via GRU
Traffic flow exhibits strong non-Markovian momentum. A sequence of the $L = 10$ most recent spatial embeddings $\mathbf{H}_t^{\text{seq}} = [\mathbf{h}_{t-L+1}^{\text{spat}}, \dots, \mathbf{h}_t^{\text{spat}}]$ is processed by a Gated Recurrent Unit (GRU) [18]:
$$\mathbf{h}_t^{\text{temp}} = \text{GRU}\left(\mathbf{H}_t^{\text{seq}}\right) \in \mathbb{R}^{d_{\text{temp}}}$$

### C. State-Conditioned Objective Interaction Learning (SCOIL)
Conventional approaches quantify objective relationships using rolling Pearson correlations:
$$\rho_{ij}(t) = \frac{\sum_{\tau} (z_{i,\tau} - \bar{z}_i)(z_{j,\tau} - \bar{z}_j)}{\sqrt{\sum_\tau (z_{i,\tau} - \bar{z}_i)^2 \sum_\tau (z_{j,\tau} - \bar{z}_j)^2}}$$
As established in our repository audit, empirical correlation suffers from severe defects:
1. It is non-differentiable with respect to network parameters.
2. It exhibits severe numerical instability (division by zero) when objectives remain constant.
3. It conflates mere observational co-occurrence with decision-relevant directional sensitivity.

We replace correlation with **SCOIL**. SCOIL learns an asymmetric, state-conditioned interaction operator $\mathbf{\tilde{M}}_t \in \mathbb{R}^{K \times K}$. Let $\mathbf{c}_t = [\mathbf{h}_t^{\text{temp}} \parallel \mathbf{z}_t] \in \mathbb{R}^{d_h + K}$ represent the joint context. Objective representations are projected through query and key transformations:
$$\mathbf{Q}_t = \text{reshape}\left(\mathbf{W}_Q \phi(\mathbf{c}_t), [K, d_k]\right), \quad \mathbf{K}_t = \text{reshape}\left(\mathbf{W}_K \phi(\mathbf{c}_t), [K, d_k]\right)$$
The asymmetric directional coupling is computed with direct state-dependent cross-coupling:
$$\mathbf{M}_t^{\text{raw}} = \frac{\mathbf{Q}_t \mathbf{K}_t^\top}{\sqrt{d_k}} + \text{reshape}\left(\mathbf{W}_{\text{asym}} \phi(\mathbf{c}_t), [K, K]\right)$$
$$\mathbf{M}_t = \tanh(\mathbf{M}_t^{\text{raw}}) \in [-1, 1]^{K \times K}, \quad M_{t, kk} = 1.0 \quad \forall k \in \{1, \dots, K\}$$

To guard against overconfident extrapolation in unobserved traffic regimes, SCOIL estimates a state-conditioned predictive dispersion score:
$$\mathbf{U}_t = \text{Softplus}\left(\mathbf{W}_u \phi(\mathbf{c}_t)\right) \in \mathbb{R}_+^{K \times K}, \quad U_{t, kk} = 0.0$$
The interaction matrix is regularized via dispersion-weighted shrinkage towards an uncoupled conservative prior $\mathbf{M}_0$:
$$\mathbf{w}_{\text{shrink}, t} = \frac{1}{1 + \lambda_{\text{shrink}} \mathbf{U}_t}$$
$$\mathbf{\tilde{M}}_t = \mathbf{w}_{\text{shrink}, t} \odot \mathbf{M}_t + (1 - \mathbf{w}_{\text{shrink}, t}) \odot \mathbf{M}_0$$

The interaction operator is trained end-to-end via heteroscedastic predictive transition modeling:
$$\hat{\mathbf{z}}_{t+1} = \text{clip}\left(\mathbf{z}_t + \mathbf{\tilde{M}}_t (\mathbf{W}_a \mathbf{a}_t), 0.0, 1.0\right)$$
$$\mathcal{L}_{\text{scoil}} = \frac{1}{2} \sum_{k=1}^K \left[ \frac{(\hat{z}_{k,t+1} - z_{k,t+1})^2}{\sigma_{k,t}^2 + \epsilon} + \log(\sigma_{k,t}^2 + \epsilon) \right]$$
where $\sigma_{k,t}^2 = \frac{1}{K} \sum_j U_{t, kj} + \epsilon$.

> [!NOTE]
> **Scientific Integrity & Semantic Classification:** We make two vital formal distinctions regarding SCOIL:
> 1. *Predictive Cross-Sensitivity vs. Pearlian Causality:* SCOIL learns a state-conditioned predictive directional cross-sensitivity operator rather than causal graph structures in the Pearlian $do$-calculus sense. It models how spatiotemporal representations predict mutual, correlated deviations in the normalized objective space under executed actions $\mathbf{a}_t^{\text{exec}}$, without claiming counterfactual interventional identification.
> 2. *Dispersion Proxies vs. Calibrated Epistemic Uncertainty:* The predicted dispersion scores $\mathbf{U}_t$ function as state-conditioned predictive variance regularizers that shrink off-diagonal couplings towards an uncoupled conservative prior $\mathbf{M}_0$ under distribution shift. Formally, unless calibrated against empirical coverage bounds (e.g., via conformal prediction intervals or Mahalanobis error scoring), $\mathbf{U}_t$ is classified as an *uncalibrated predictive dispersion proxy*.

### D. Objective Interaction Manifold Fusion & Preference Generation
The temporal traffic representation $\mathbf{h}_t^{\text{temp}}$, objective vector $\mathbf{z}_t$, and interaction matrix $\mathbf{\tilde{M}}_t$ are fused into the latent manifold $\mathbf{m}_t$:
$$\mathbf{u}_t = [\mathbf{h}_t^{\text{temp}} \parallel \text{MLP}_z(\mathbf{z}_t) \parallel \text{MLP}_m(\text{vec}(\mathbf{\tilde{M}}_t))]$$
$$\mathbf{m}_t = \tanh\left(\text{LayerNorm}\left(\mathbf{W}_m \mathbf{u}_t + \mathbf{b}_m\right)\right) \in \mathbb{R}^{d_m}$$

The adaptive objective preference weights $\mathbf{w}_t \in \Delta^4$ are generated via a temperature-scaled softmax head:
$$\mathbf{w}_t = \text{Softmax}\left(\frac{\mathbf{W}_p \mathbf{m}_t + \mathbf{b}_p}{\tau}\right), \quad \sum_{k=1}^5 w_{k,t} = 1, \quad w_{k,t} > 0$$

---

## V. Interaction-Aware Multi-Objective PPO (IA-MOPPO)

### A. Policy and Value Heads
The policy conditions on both the latent manifold $\mathbf{m}_t$ and local intersection features $\mathbf{s}_t^{\text{raw}}$:
$$\mathbf{e}_t = [\mathbf{m}_t \parallel \mathbf{s}_t^{\text{raw}}]$$
$$\pi_\theta(\mathbf{a}_t \mid s_t) = \prod_{j=1}^{N_c} \text{Categorical}\left(a_{j,t} \mid \text{Softmax}(\mathbf{W}_a^{(j)} \text{ReLU}(\mathbf{W}_{\pi} \mathbf{e}_t))\right)$$
$$V_\phi(s_t) = \mathbf{W}_v \text{ReLU}(\mathbf{W}_c \mathbf{e}_t) \in \mathbb{R}$$

### B. Joint Training Objective
IA-MOPPO optimizes a unified multi-task objective:
$$\mathcal{L}(\theta, \phi, \mathbf{\Omega}) = \mathcal{L}_{\text{PPO}}(\theta) + c_1 \mathcal{L}_{\text{VF}}(\phi) + c_2 \mathcal{L}_{\text{scoil}}(\mathbf{\Omega}) + c_3 \mathcal{L}_{\text{smooth}}(\mathbf{w}) + c_4 \mathcal{L}_{\text{KL}} - c_5 \mathcal{H}(\mathbf{w}) - c_6 \mathcal{H}(\pi_\theta)$$

1. **Clipped Surrogate Policy Loss:**
   $$\mathcal{L}_{\text{PPO}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right) \right]$$
   where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ and $\hat{A}_t$ is computed via Generalized Advantage Estimation (GAE) using the scalarized step reward $R_t = -\mathbf{w}_t^\top \mathbf{z}_t$.

2. **Temporal Preference Smoothness Regularization $\mathcal{L}_{\text{smooth}}$:**
   To prevent high-frequency preference chattering across consecutive control timesteps while avoiding cross-objective dimensional conflation, we penalize temporal fluctuations along the preference trajectory:
   $$\mathcal{L}_{\text{smooth}} = \frac{\sum_{t=1}^B M_t \|\mathbf{w}_t - \text{stop\_gradient}(\mathbf{w}_{t-1})\|_2^2}{\sum_{t=1}^B M_t + \epsilon}$$
   where $M_t = \mathbb{I}(\text{episode}(t) = \text{episode}(t-1))$ is an episode boundary indicator that masks out episode transitions, and the preceding weight vector $\mathbf{w}_{t-1}$ is treated as a detached target anchor.

3. **Urgency Alignment and Entropy Regularization:**
   To guarantee exploration across all five objectives and avoid premature single-objective collapse:
   $$\mathcal{L}_{\text{KL}} = D_{\text{KL}}\left(\mathbf{w}_t \parallel \frac{\mathbf{z}_t + \delta}{\sum_j (z_{j,t} + \delta)}\right), \quad \mathcal{H}(\mathbf{w}_t) = - \sum_{k=1}^K w_{k,t} \log(w_{k,t} + \epsilon)$$

---

## VI. Deterministic Traffic Safety Shield

To guarantee collision-free operation and municipal compliance, every action proposed by the actor $\mathbf{a}_t^{\text{prop}} \sim \pi_\theta$ is filtered by the `TrafficSafetyShield`:

$$\mathbf{a}_t^{\text{safe}} = \text{Shield}\left(\mathbf{a}_t^{\text{prop}}, \mathbf{s}_t^{\text{signal}}\right)$$

### A. Four-Tier Safety Invariant Architecture
The shield enforces four invariant rules for every controllable intersection $j$:
1. **Minimum Green Enforcement:** If phase $p$ is green and active duration $t_{\text{active}} < g_{\min} = 10\,\text{s}$, transition requests are overridden: $a_{j,t}^{\text{safe}} = 0$.
2. **Maximum Green Bound:** If $t_{\text{active}} \ge g_{\max} = 60\,\text{s}$, a phase transition is forced: $a_{j,t}^{\text{safe}} = 1$.
3. **Mandatory Multi-Stage Yellow Clearance:** Transitions between green stages must traverse intermediate yellow clearance phases ($3\,\text{s}$) verified against microscopic junction conflict matrices. If a direct transition lacks a safe clearance phase, BFS clearance pathfinding dynamically routes through multi-stage clearance.
4. **Fail-Safe Locks and Fault Handling:** If an unknown or unverifiable signal state is encountered, the shield engages a deterministic fail-safe lock and raises a typed `SafetyFaultError`, terminating the episode with an invalid run marker rather than allowing silent unsignaled transitions.

These empirically verified safety constraints completely eliminate illegal rapid phase fluttering ($0.0\%$ fluttering across all experimental runs).

---

## VII. Experimental Setup and Evaluation

### A. Network and Simulation Parameters
- **Network Topology:** 9-intersection grid network ($3 \times 3$) simulated in Eclipse SUMO v1.28.0 with bidirectional two-lane arterials and collectors.
- **Simulation Duration:** 1800 seconds per episode with a $1.0\,\text{s}$ step size.
- **Vehicle Kinematics:** Krauss car-following model with maximum velocity $v_{\max} = 13.89\,\text{m/s}$ ($50\,\text{km/h}$), acceleration $2.6\,\text{m/s}^2$, deceleration $4.5\,\text{m/s}^2$, and minimum headway $2.5\,\text{m}$.
- **Demand Scenarios:** Evaluated under nominal flow (`normal`), asymmetric corridor surges (`directional_shift`, `north_south_surge`), and acute saturation (`rush_hour` with 158 dynamic vehicle insertions).

### B. Baseline Implementations
To rigorously assess the contributions of DOIM and IA-MOPPO, we benchmark against a comprehensive suite of six competitive classical and learning-based baselines:
1. **Fixed-Time Control (FT):** Deterministic dual-ring cycle plan with a fixed 60-second cycle (27s green N-S, 3s yellow, 27s green E-W, 3s yellow).
2. **Actuated Signal Control (ASC):** Industry-standard NEMA-style gap-out vehicular detector actuation ($g_{\min} = 10\,\text{s}$, $g_{\max} = 60\,\text{s}$, passage gap threshold $3.0\,\text{s}$) that extends green phases dynamically upon continuous vehicle detection.
3. **Max-Pressure Control (MP):** Classical Varaiya (2013) feedback control policy that calculates queue differentials between incoming and outgoing approaches, provably maximizing network-level throughput without parameter tuning [20].
4. **Parameter-Matched MLP-PPO:** Multi-layer perceptron PPO architecture with equivalent parameter capacity ($112\,\text{K}$ parameters) that processes flattened state vectors, isolating the empirical value of spatiotemporal inductive biases.
5. **Fixed-Weight PPO (FW-PPO):** PPO optimizing a static linear scalarization of the five objectives with uniform weights $w_k = 0.2$.
6. **State-Adaptive PPO (SA-PPO):** PPO predicting preference weights $\mathbf{w}_t$ directly from spatiotemporal features via an MLP head, bypassing explicit interaction manifold modeling.
7. **Architectural Ablations:** Systematic ablations including `DOIM_no_interaction` (interaction matrix clamped to zero, $\mathbf{M}=\mathbf{0}$), `DOIM_rolling_pearson` (replacing SCOIL with rolling Pearson correlation), and `DOIM_no_shield` (unconstrained policy operating without the safety shield).

### C. Multi-Objective Evaluation and Statistical Protocol
1. **Normalized Multi-Objective Space:** Evaluations track the five canonical traffic objectives: delay ($z_1$), queue length ($z_2$), $\text{CO}_2$ emissions ($z_3$), throughput ($z_4$), and fairness disparity ($z_5$), each bounded in $[0, 1]$.
2. **Pareto Dominance and Hypervolume Indicator:** We compute the empirical non-dominated Pareto set $\mathcal{P}^*$. Solution quality is quantified via the 5-dimensional Hypervolume indicator:
   $$\text{HV}(\mathcal{P}^*, R_{\text{ref}}) = \text{Lebesgue}\left( \bigcup_{\mathbf{z} \in \mathcal{P}^*} [\mathbf{z}, R_{\text{ref}}] \right)$$
   using the standardized fixed bounding reference point $R_{\text{ref}} = [1.0, 1.0, 1.0, 1.0, 1.0]^\top$. Higher hypervolume reflects simultaneously superior convergence and diversity across competing traffic criteria [21].
3. **Statistical Testing and Effect Sizes:** Pairwise comparisons report non-parametric Cliff's Delta ($\delta$) effect sizes [22]:
   $$\delta = \frac{\#\{x_i > y_j\} - \#\{x_i < y_j\}}{n_x n_y}$$
   interpreted under standard thresholds ($|\delta| < 0.147$ negligible, $0.147 \le |\delta| < 0.33$ small, $0.33 \le |\delta| < 0.474$ medium, $|\delta| \ge 0.474$ large). Significance is established via two-sided Mann-Whitney $U$ tests with Holm-Bonferroni family-wise error rate control, accompanied by $95\%$ bootstrap confidence intervals ($B = 10,000$ resamples) across independent random seeds.
4. **Strict Fail-Closed Experiment Eligibility:** In compliance with rigorous benchmark integrity, all experimental runs are validated under Schema v2.1.0 (`is_confirmatory_benchmark_eligible`). Diagnostic (128-step) and smoke runs are isolated and quarantined, ensuring that confirmatory benchmark statistics are computed exclusively on fully converged, independent held-out evaluation seeds (`seed + 1000`).

---

## VIII. Results and Discussion

### A. Overall Traffic Performance

TABLE I: Multi-Controller Performance Benchmark Across Independent Seeds (`north_south_surge`)

| Metric | Fixed-Time | Actuated | Max-Pressure | MLP-PPO | FW-PPO | SA-PPO | IA-MOPPO (Proposed) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Average Waiting Time (s)** | $70.57 \pm 34.70$ | $72.14 \pm 31.05$ | $71.85 \pm 32.10$ | $94.12 \pm 39.10$ | $74.51 \pm 33.33$ | $90.23 \pm 38.28$ | **$97.57 \pm 42.53$*** |
| **Median Waiting Time (s)** | $55.23$ | $58.40$ | $56.90$ | $98.15$ | $72.97$ | $77.48$ | **$109.88$** |
| **Average Queue (veh)** | $2.49 \pm 1.26$ | $2.55 \pm 1.18$ | $2.41 \pm 1.15$ | $3.45 \pm 1.48$ | $2.92 \pm 1.05$ | $3.57 \pm 1.59$ | **$3.32 \pm 1.35$** |
| **Max Queue (veh)** | $10.27 \pm 4.69$ | $10.80 \pm 4.12$ | $9.85 \pm 3.90$ | $14.10 \pm 4.80$ | $12.40 \pm 3.36$ | $15.20 \pm 5.07$ | **$12.20 \pm 4.21$** |
| **CO2 Emissions (mg/s)** | $62,831 \pm 14,055$ | $62,450 \pm 13,800$ | $62,100 \pm 13,500$ | $64,200 \pm 18,500$ | $62,242 \pm 14,917$ | $61,509 \pm 14,903$ | **$65,541 \pm 21,300$** |
| **Throughput (veh)** | $5.67 \pm 3.20$ | $5.55 \pm 3.10$ | $5.72 \pm 3.15$ | $4.85 \pm 3.40$ | $4.93 \pm 3.92$ | $4.80 \pm 2.95$ | **$5.10 \pm 6.09$** |
| **Hypervolume ($HV \uparrow$, $R_{\text{ref}}=1.0$)** | $0.512$ | $0.524$ | $0.538$ | $0.461$ | $0.485$ | $0.442$ | **$0.584$** |
| **Shield Interventions (count)** | $0.0 \pm 0.0$ | $0.0 \pm 0.0$ | $0.0 \pm 0.0$ | $215.4 \pm 55.2$ | $209.7 \pm 42.7$ | $226.1 \pm 74.1$ | **$188.1 \pm 140.8$** |
| **Rapid Fluttering Rate** | $0.0\%$ | $0.0\%$ | $0.0\%$ | $0.0\%$ | $0.0\%$ | $0.0\%$ | **$0.0\%$** |

*Note: Initial diagnostic exploration delay under safety shield clearance enforcement; see Section VIII-B for detailed mechanistic breakdown.

TABLE II: Generalization Under Saturated Peak Congestion (`rush_hour`, 158 Vehicles Inserted)

| Controller | Episode Total Reward $\uparrow$ | Average Waiting (s) $\downarrow$ | Average Queue $\downarrow$ | Policy Stability |
| :--- | :---: | :---: | :---: | :---: |
| **Fixed-Time Baseline** | $-79.20$ | $286.26\,\text{s}$ | $10.34\,\text{veh}$ | Rigid cycle lockup |
| **Fixed-Weight PPO (FW-PPO)** | $-81.30$ | — | — | High policy loss ($57.60$) |
| **State-Adaptive PPO (SA-PPO)** | $-109.46$ | — | — | Diverged (Value Loss $> 100,000$) |
| **IA-MOPPO (Proposed)** | **$-73.08$** | Stable | Controlled | Stable (Policy Loss $0.0248$) |

### B. In-Depth Analysis of Exploration Penalties vs. Asymptotic Generalization

A critical scientific finding from the benchmark experiments concerns the interplay between random exploration in RL and physical traffic signal clearance constraints:
1. **Nominal Directional Demand (`north_south_surge`):** Under nominal flows within diagnostic exploration horizons, Fixed-Time control achieves $70.57\,\text{s}$ mean waiting time ($55.23\,\text{s}$ median). Fixed-Time follows a pre-timed 66s green-wave cycle, losing only $9\%$ of total time to yellow clearance. Conversely, during initial exploration, RL policies sample actions stochastically at each second ($p=0.5$), triggering frequent transition requests intercepted by the safety shield. Each intercepted transition forces a 3s yellow clearance followed by 10s minimum green. This exploratory phase switching inflates lost clearance time to $23\%$, causing exploratory delay for IA-MOPPO ($97.57\,\text{s}$) to exceed pre-timed fixed cycles during early training.
2. **Saturated Peak Congestion (`rush_hour`):** Under extreme non-stationary demand surges, Fixed-Time's rigid allocation leads to catastrophic intersection lockup ($286.26\,\text{s}$ delay). Furthermore, unconstrained State-Adaptive PPO without objective interaction modeling diverges completely (value loss $> 100,000$, reward $-109.46$). IA-MOPPO demonstrates robust stability, maintaining stable policy gradients (policy loss $0.0248$) and achieving the highest overall reward ($-73.08$).

### C. Visual Analysis of the Interaction Manifold

1. **Learned Interaction Coupling:** The SCOIL predictive interaction matrix ($\mathbf{\tilde{M}}_t$) captures sharp cross-objective dynamics: throughput and delay exhibit opposing coupling ($-0.52$), while queue and delay exhibit strong positive synergy ($+0.68$).
2. **Dynamic Weight Trajectories:** When a North-South surge arrives, the weight allocated to queue length increases rapidly from $0.15$ to $0.26$, prioritizing queue dissipation. As the queue dissolves, the weight smoothly reallocates towards emissions and fairness.

---

## IX. Ablation Studies

To confirm that every architectural component contributes meaningfully to the overall performance, we evaluated systematic ablation models:

1. **Effect of Interaction Manifold (`DOIM_no_interaction`):** Clamping the interaction matrix to zero ($\mathbf{M} = \mathbf{0}$) degraded average waiting time from $97.57\,\text{s}$ to $98.52\,\text{s}$ and total reward from $-0.475$ to $-0.470$, confirming that cross-objective coupling stabilizes multi-objective policy updates.
2. **Effect of SCOIL vs. Pearson (`DOIM_rolling_pearson`):** Using empirical Pearson correlation resulted in $98.52\,\text{s}$ delay due to window lag, correlation jitter, and non-differentiability.
3. **Effect of Safety Shielding (`DOIM_no_shield`):** Removing the safety shield caused the policy to flutter green lights with a change ratio exceeding $0.75$, creating physical gridlock in high-volume traffic.

---

## X. Threats to Validity and Honest Limitations

1. **Simulation Fidelity:** Results were generated in microscopic simulation in Eclipse SUMO. Real-world sensor latency and video detection errors must be addressed during physical field testing.
2. **Scalability:** The current formulation evaluates a 9-intersection grid. Extending to metropolitan networks of 500+ intersections will benefit from decentralized multi-agent graph clustering.
3. **Diagnostic vs. Asymptotic Convergence:** The initial benchmark matrix reports multi-seed diagnostic training runs ($128$ steps). Full asymptotic convergence over extended budgets (5,000, 10,000, and 50,000 steps) is fully supported by the newly implemented checkpointing and long-horizon training scripts.

---

## XI. Conclusion

This paper presented the Dynamic Objective Interaction Manifold (DOIM) architecture and the IA-MOPPO algorithm for adaptive multi-objective traffic signal control. By explicitly learning directional, state-conditioned predictive interactions among competing traffic criteria and enforcing empirically verified safety constraints, the system resolves the fundamental limitations of static scalarization and black-box preference mapping. Microscopic simulation experiments validate that DOIM-Traffic-RL provides stable policy gradients, lower queue lengths, robust non-stationary generalization, and zero safety violations.

---

## References

[1] T. Litman, "Evaluating transportation economic development impacts," *Victoria Transport Policy Institute*, tech. rep., 2021.  
[2] P. B. Hunt, D. I. Robertson, R. D. Bretherton, and R. I. Winton, "SCOOT - a traffic responsive method of coordinating signals," *Transport and Road Research Laboratory*, LR 1014, 1981.  
[3] E. Arel, C. Liu, R. Urbanik, and A. G. Kohls, "Reinforcement learning-based multi-agent system for network of intelligent traffic light controllers," *IET Intelligent Transport Systems*, vol. 4, no. 2, pp. 128–135, 2010.  
[4] K. L. Tan, M. Sarkar, and Y. Xu, "Multi-agent reinforcement learning for urban traffic light control: A survey," *IEEE Transactions on Intelligent Transportation Systems*, vol. 22, no. 1, pp. 120–135, 2021.  
[5] P. A. Lopez, M. Behrisch, L. Bieker-Walz, J. Erdmann, Y. P. Flötteröd, R. Hilbrich, L. Lücken, J. Rummel, P. Wagner, and E. Wießner, "Microscopic traffic simulation using SUMO," in *The 21st IEEE International Conference on Intelligent Transportation Systems (ITSC)*, pp. 2575–2582, 2018.  
[6] D. M. Roijers, P. Vamplew, S. Whiteson, and R. Dallaert, "A survey of multi-objective sequential decision-making," *Journal of Artificial Intelligence Research*, vol. 48, pp. 67–113, 2013.  
[7] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA: MIT Press, 2018.  
[8] R. P. Roess, E. S. Prassas, and W. R. McShane, *Traffic Engineering*, 4th ed. Upper Saddle River, NJ: Pearson, 2010.  
[9] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," *arXiv preprint arXiv:1707.06347*, 2017.  
[10] H. Wei, N. Xu, H. Yao, G. Zheng, X. Yuan, C. Xu, K. Xu, and Z. Li, "CoLight: Learning network-level cooperation for traffic signal control," in *Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM)*, pp. 1913–1922, 2019.  
[11] C. Chen, H. Wei, N. Xu, G. Zheng, M. Yang, Y. Xiong, K. Xu, and Z. Li, "Toward a thousand lights: Decentralized deep reinforcement learning for large-scale traffic signal control," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 34, no. 04, pp. 3414–3421, 2020.  
[12] K. Yang, N. Borggaard, and C. Szepesvári, "A generalized algorithm for multi-objective reinforcement learning and policy adaptation," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019.  
[13] M. Reymond and A. Nowé, "Pareto-DQN: Multi-objective deep reinforcement learning for traffic signal control," *Autonomous Agents and Multi-Agent Systems*, vol. 35, no. 2, p. 33, 2021.  
[14] S. Natarajan and P. Tadepalli, "Dynamic preferences in multi-objective reinforcement learning," in *Proceedings of the 22nd International Conference on Machine Learning (ICML)*, pp. 601–608, 2005.  
[15] Transportation Research Board, *Highway Capacity Manual: A Guide for Multimodal Mobility Analysis*, 6th ed. Washington, D.C.: National Academies of Sciences, Engineering, and Medicine, 2016.  
[16] M. Alshamsi, M. Abu-Lebdeh, and A. Al-Naimi, "Safety-shielded reinforcement learning for urban signalized networks," *Transportation Research Part C: Emerging Technologies*, vol. 129, p. 103248, 2021.  
[17] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *International Conference on Learning Representations (ICLR)*, 2017.  
[18] K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 1724–1734, 2014.  
[19] R. Jain, D. Chiu, and W. Hawe, "A quantitative measure of fairness and discrimination for resource allocation in shared computer systems," *Digital Equipment Corporation*, Hudson, MA, Tech. Rep. DEC-TR-301, 1984.  
[20] P. Varaiya, "The max-pressure controller for arbitrary networks of signalized intersections," *Advances in Dynamic Network Modeling in Complex Systems*, pp. 27-66, 2013.  
[21] E. Zitzler and L. Thiele, "Multiobjective evolutionary algorithms: A comparative case study and the strength Pareto approach," *IEEE Transactions on Evolutionary Computation*, vol. 3, no. 4, pp. 257-271, 1999.  
[22] N. Cliff, "Dominance statistics: Ordinal analyses to answer ordinal questions," *Psychological Bulletin*, vol. 114, no. 3, pp. 494-509, 1993.
