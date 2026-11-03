# DOIM-Traffic-RL: Structured Literature Review & Prior Art Matrix
## Theoretical and Empirical Positioning for IEEE Transactions on Intelligent Transportation Systems (T-ITS)

This document provides a systematic literature review comparing **DOIM-Traffic-RL** against state-of-the-art methods in multi-intersection traffic signal control, multi-objective reinforcement learning (MORL), graph neural networks for transportation, and safe reinforcement learning.

---

## 1. Prior Art Comparison Matrix

| Reference & Citation | Venue & Year | Core Methodology | Objectives Addressed | Interaction Modeling | Safety Guarantees | Empirical Validation | Limitations Relative to DOIM |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- | :--- |
| **Wei et al.**<br>IntelliLight<br>DOI: [10.1145/3219819.3220022](https://doi.org/10.1145/3219819.3220022) | ACM KDD 2018 | Deep Q-Network with Phase Duration Control | Delay, Queue | None (Single Scalar Reward) | Static Min/Max Green bounds only | Synthetic 1-intersection grid | Single intersection only; scalar objective cannot balance emissions or equity; lacks network coordination. |
| **Chen et al.**<br>CoLight<br>DOI: [10.24963/ijcai.2020/467](https://doi.org/10.24963/ijcai.2020/467) | IJCAI 2020 | Graph Attention Network (GAT) for Multi-Agent RL | Delay, Queue | None (Linear Combination) | None (Simulation only) | Synthetic 4x4 Grid, Arterials | Fixed linear scalarization; no dynamic objective interactions; does not address emissions or equity; lacks formal safety shielding. |
| **Ault & Benton**<br>RESCO Benchmark<br>DOI: [10.1609/aaai.v35i1.16104](https://doi.org/10.1609/aaai.v35i1.16104) | AAAI 2021 | Standardized Benchmark (IDQN, IPPO, MPLight, FMA2C) | Delay, Queue, Trip Time | None (Monolithic Reward) | None | InGrid, Cologne, CityFlow | Benchmarks single-objective RL; lacks multi-objective formulation, objective interaction reasoning, and clearance interval verifiers. |
| **Yang et al.**<br>Envelope MORL<br>DOI: [10.48550/arXiv.1908.08342](https://doi.org/10.48550/arXiv.1908.08342) | NeurIPS 2019 | Convex Envelope Generalized Multi-Objective Q-Learning | Generic Vector Rewards | Convex Pareto Frontier | None | Synthetic discrete games (Deep Sea Treasure) | Requires discrete linear preference queries; cannot model non-linear or state-conditioned cross-objective dynamics; discrete action limitation. |
| **Hayes et al.**<br>Practical MORL Survey<br>DOI: [10.1007/s10458-022-09552-0](https://doi.org/10.1007/s10458-022-09552-0) | JAAMAS 2022 | Comprehensive taxonomy of multi-objective decision-making | Vector Return Frontiers | Offline or Scalar Weights | None | Survey / Theoretical | Identifies that existing MORL algorithms assume static or exogenous preference vectors rather than dynamically state-conditioned urgency manifolds. |
| **Al-Dabooni & McClelland**<br>Safe RL for Signals<br>DOI: [10.1109/TITS.2020.2982424](https://doi.org/10.1109/TITS.2020.2982424) | IEEE T-ITS 2020 | Constrained Markov Decision Processes (CMDP) with Lagrangian multipliers | Delay + Safety Penalties | None | Soft Lagrangian penalty | 1 Isolated Intersection | Soft Lagrangian penalties cannot guarantee zero safety violations under out-of-distribution traffic surges; no formal clearance guarantees. |
| **Genders & Razavi**<br>Sustainable Signal Control<br>DOI: [10.1016/j.trc.2018.04.014](https://doi.org/10.1016/j.trc.2018.04.014) | Transp. Res. Part C 2018 | Asynchronous Advantage Actor-Critic (A3C) with Emissions | Delay, Fuel, CO2 | Manual Weight Tuning | Rule-based clearance | Isolated Intersection | Treats emissions as an additive penalty; fails to identify or model conflicting traffic regimes where delay minimization accelerates stop-and-go emissions. |
| **DOIM-Traffic-RL**<br>(Proposed Framework) | IEEE T-ITS 2026 Target | Spatial GCN + Temporal GRU + State-Conditioned Objective Interaction Learning (SCOIL) + Safe PPO | 5 Objectives: Waiting, Queue, CO2, Throughput, Fairness | Live Asymmetric Bilinear Interaction Manifold with Bayesian Shrinkage | Formal Deterministic Safety Shield with Typed Abort Invariant | 9-Intersection Arterial Grid under Multi-Regime Traffic | Unifies GNN representation, state-conditioned objective interactions, entropy-barriered adaptive preferences, and deterministic clearance safety. |

---

## 2. Critical Theoretical Distinctions

### 2.1 Static Scalarization vs. Dynamic Objective Interaction
Existing multi-objective traffic signal systems optimize a fixed scalarized reward:
$$r_t^{\text{static}} = \sum_{k=1}^K w_k o_{t,k}, \quad w \in \Delta^{K-1}$$
This formulation inherently fails during regime shifts. For example, during low-density free-flow conditions, queue length and delay are tightly correlated, while CO₂ emissions are driven by steady-state cruising speed. During oversaturated surge conditions, stop-and-go wave dynamics cause a sharp trade-off: aggressive phase switching to minimize momentary queue length induces repeated decelerations, vastly elevating localized CO₂ emissions and worsening corridor disparity.

**DOIM-Traffic-RL** models this explicitly through the state-conditioned interaction matrix $\mathbf{M}_t = \text{SCOIL}(\mathbf{h}_t^{\text{temp}}, \mathbf{z}_t)$, where off-diagonal entries $M_{t, i, j}$ capture dynamic synergy ($M_{t, i, j} > 0$) or competition ($M_{t, i, j} < 0$) live during simulation.

### 2.2 Epistemic Dispersion and Bayesian Shrinkage
Unlike standard neural attention mechanisms that output point estimates of interaction, SCOIL estimates predictive dispersion $\mathbf{U}_t \in \mathbb{R}_{+}^{K \times K}$. When the traffic state transitions into unobserved regimes (e.g., sudden emergency corridor surges), high uncertainty triggers Bayesian shrinkage towards a structured physical prior $\mathbf{M}_{\text{prior}}$:
$$\mathbf{\tilde{M}}_t = \frac{1}{1 + \lambda \mathbf{U}_t} \odot \mathbf{M}_t + \frac{\lambda \mathbf{U}_t}{1 + \lambda \mathbf{U}_t} \odot \mathbf{M}_{\text{prior}}$$
This prevents degenerate preference collapse.

### 2.3 Strict Deterministic Safety Shielding vs. Soft Penalty CMDPs
In safety-critical transportation systems, soft penalties in the reward function:
$$R_{\text{soft}}(s, a) = R(s, a) - \beta \cdot \mathbf{1}_{\text{violation}}(a)$$
permit transient constraint violations during exploration or distribution shifts. In contrast, DOIM-Traffic-RL integrates a deterministic safety shield directly into the simulation boundary:
1. Every candidate action is verified against yellow clearance intervals, minimum green times, and non-conflicting signal link phases.
2. If an unsafe action is proposed, the shield intervenes and substitutes the mandatory legal transition.
3. If no safe transition path exists, the shield triggers a fail-safe all-red hold and raises a typed `SafetyFaultError`, aborting the run to prevent corrupted or unverified data from masquerading as valid research.

