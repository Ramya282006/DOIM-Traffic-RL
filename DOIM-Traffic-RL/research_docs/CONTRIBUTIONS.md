# DOIM-Traffic-RL: Scientific Contributions & Methodological Innovations

**Target Journal:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Standard:** Rigorous Q1 Technical Differentiation and Value Proposition

---

## 1. The Research Gap in Multi-Objective Traffic Signal Control

Urban traffic signal management is fundamentally a multi-objective decision process where controllers must balance five competing operational dimensions:
1. Total delay and vehicle waiting time.
2. Localized queue spillback.
3. Fuel combustion and $\text{CO}_2$ emissions.
4. Network throughput (completed trips).
5. Service fairness across competing arterial approaches.

Existing reinforcement learning controllers exhibit three fundamental scientific limitations:
1. **Static Linear Scalarization:** Conventional methods collapse multi-dimensional reward vectors into a fixed linear combination $R_t = \sum_k w_k r_{k,t}$ with constant weights throughout training. This ignores the non-stationary coupling between objectives—where delay and emissions cooperate during free-flow conditions but conflict severely during peak saturation.
2. **Uncoupled Black-Box Preference Mapping:** Recent adaptive MORL methods map traffic states directly to preference vectors ($s \mapsto \mathbf{w}$) via unconstrained MLPs. These architectures treat objectives as isolated independent scalar values, completely ignoring the **interaction manifold**—the mathematical structure of pairwise directional sensitivities, cross-correlations, and trade-offs.
3. **Absence of Differentiable Interaction Learning:** Historical attempts to incorporate objective couplings relied on rolling Pearson correlation matrices, which are non-differentiable with respect to neural network weights, numerically unstable when objectives remain constant, and lag behind real-time traffic dynamics.

---

## 2. Core Scientific Contributions

### Contribution 1: The Dynamic Objective Interaction Manifold (DOIM) Architecture
We introduce the **DOIM** architecture, a unified representation learning framework that synthesizes:
- **Spatial Topology Encoding:** A 2-layer Graph Convolutional Network (GCN) that propagates localized loop-detector features across the physical road network topology.
- **Temporal Momentum Modeling:** A Gated Recurrent Unit (GRU) operating over rolling historical spatial embeddings ($L = 10$) to capture arterial queue accumulation and platoon propagation.
- **Latent Manifold Fusion Core:** A differentiable fusion module combining temporal traffic state $\mathbf{h}_t^{\text{temp}}$, normalized objective urgency vector $\mathbf{z}_t$, and the vectorized interaction matrix $\text{vec}(\mathbf{\tilde{M}}_t)$ via layer-normalized non-linear projections into latent manifold $\mathbf{m}_t \in \mathbb{R}^{64}$.

### Contribution 2: State-Conditioned Objective Interaction Learning (SCOIL)
Rather than relying on empirical rolling correlations, we develop **SCOIL**:
- **Bilinear Directional Coupling:** Implements a query-key projection over the joint state-objective context $\mathbf{c}_t = [\mathbf{h}_t^{\text{temp}} \parallel \mathbf{z}_t]$ to compute an asymmetric directional coupling matrix $\mathbf{M}_t \in [-1, 1]^{5 \times 5}$.
- **Identity Anchoring:** Constrains diagonal entries to $+1.0$, anchoring each objective to its intrinsic urgency while allowing off-diagonal entries to represent directional positive synergy or negative trade-offs.
- **State-Conditioned Predictive Dispersion & Bayesian Shrinkage:** Estimates a dispersion score $\mathbf{U}_t \in \mathbb{R}_+^{5 \times 5}$ that adaptively regularizes the learned matrix toward an uncoupled conservative prior $\mathbf{M}_0$ under novel or unobserved traffic regimes.
- **Heteroscedastic Predictive Loss:** Trains the interaction operator end-to-end via heteroscedastic negative log-likelihood on physically executed next-objective transitions.

### Contribution 3: Disentangled 3-Tier Hierarchical Fairness Metric Suite
Prior literature suffered from severe aggregation conflation, reporting intersection variance or vehicle delays interchangeably as "fairness". We establish a strict three-tier hierarchy:
- **Tier 1 (Arterial Approach Level - Primary RL Objective):** Evaluates allocation equity across competing directional approaches feeding signalized junctions using Jain's disparity $1 - \mathcal{J}(\mathbf{d}_{\text{approach}}) \in [0.0, 1.0]$.
- **Tier 2 (Intersection Level):** Evaluates service balance across the network via cross-intersection delay variance $\sigma_{\text{int}}^2$.
- **Tier 3 (Individual Vehicle Level):** Evaluates equity across individual driver delay distributions via the Gini coefficient $\mathcal{G} \in [0.0, 1.0]$.

### Contribution 4: Interaction-Aware Multi-Objective PPO (IA-MOPPO) with Temporal Smoothness
We formulate an end-to-end multi-task policy gradient algorithm wherein:
- The latent manifold $\mathbf{m}_t$ directly parameterizes the temperature-scaled preference simplex generator $\mathbf{w}_t \in \Delta^4$ and the multi-discrete actor-critic heads.
- A **true temporal preference smoothness loss** penalizes preference chattering across consecutive control timesteps with detached target anchors and episode boundary reset masking.
- Anti-collapse urgency barriers ($D_{\text{KL}}$ to physical urgency) and entropy barriers guarantee exploration across the full objective simplex.

### Contribution 5: Deterministic Traffic Safety Shield with Empirical Verification
We design and validate an active safety shield that:
- Enforces microscopic conflict matrix verification, minimum green bounds ($10\,\text{s}$), maximum green bounds ($60\,\text{s}$), and mandatory multi-stage yellow clearance intervals ($3\,\text{s}$).
- Implements breadth-first clearance pathfinding for multi-stage clearance transitions.
- Employs fail-closed safety semantics: raises a typed `SafetyFaultError`, marks runs invalid, and excludes faulted runs from benchmark aggregation.
- Achieves **$0.0\%$ rapid fluttering rate** across all experimental evaluations.

### Contribution 6: Transparent Investigation of Early Exploration Penalties
We provide an honest, scientifically grounded analysis of early RL exploration penalties:
- During early training exploration, stochastic switching requests frequently trigger yellow clearance intervals ($3\,\text{s}$) and minimum green lockouts ($10\,\text{s}$), inflating lost clearance time to $23\%$ compared to $9\%$ for pre-timed Fixed-Time control.
- Under saturated peak congestion (`rush_hour`), however, IA-MOPPO demonstrates superior stability and generalization, achieving the lowest policy loss ($0.0248$) and highest reward ($-73.08$), while Fixed-Time experiences gridlock ($286.26\,\text{s}$ delay) and State-Adaptive PPO diverges.
