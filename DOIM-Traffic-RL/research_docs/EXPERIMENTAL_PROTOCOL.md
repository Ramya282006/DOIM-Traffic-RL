# Experimental Protocol and Scientific Methodology

## 1. Overview and Hypothesis

### 1.1 Central Research Question
> *"Can explicitly learning state-dependent interactions among competing traffic objectives produce more robust, interpretable, and equitable traffic-signal policies than conventional fixed-weight scalarization and direct state-adaptive approaches under dynamic and non-stationary traffic demands?"*

### 1.2 Formal Hypotheses
- **$H_1$ (Convergence & Performance Advantage):** Policies conditioned on the Dynamic Objective Interaction Manifold ($\mathcal{M}_t$) achieve statistically significant reductions in mean waiting time ($p < 0.05$, Cohen's $d > 0.5$) and queue length compared to fixed-weight baselines across independent stochastic seeds.
- **$H_2$ (Interaction Learning vs. Statistical Correlation):** State-Conditioned Objective Interaction Learning (SCOIL) with Bayesian shrinkage outperforms empirical rolling Pearson correlation under sudden traffic surges by capturing directional cross-gradients rather than mere co-occurrence.
- **$H_3$ (Safety Integrity):** A formal clearance shield enforcing deterministic minimum green times ($\ge 10\,\text{s}$), maximum green bounds ($\le 60\,\text{s}$), and yellow clearance intervals ($3\,\text{s}$) eliminates rapid phase fluttering ($< 1\%$ change ratio) without destabilizing RL convergence.
- **$H_4$ (Equitable Service):** Explicit vehicle-level delay disparity minimization ($Jain's \ge 0.85$) prevents localized starvation on minor approaches during directional surges.

---

## 2. Simulation Environment and Traffic Networks

### 2.1 Microscopic Simulator
- **Platform:** Eclipse SUMO (Simulation of Urban MObility) v1.28.0.
- **Interface:** TraCI (Traffic Control Interface) running on dedicated local TCP sockets.
- **Step Resolution:** $\Delta t = 1.0\,\text{second}$ per simulation step.
- **Collision & Teleport Policy:** Teleportation disabled (`--time-to-teleport -1`) to ensure strict physical conservation of vehicles.

### 2.2 Network Topology ($3 \times 3$ Grid Network)
- **Geometry:** 9 signalized intersections organized as a $3 \times 3$ grid with 300-meter link lengths.
- **Node Classification:**
  - 5 Controllable Multi-Phase Intersections: `A1, B0, B1, B2, C1` (arterial and central crossings).
  - 4 Boundary Fixed Intersections: `A0, A2, C0, C2`.
- **Lane Configuration:** 2 incoming and 2 outgoing lanes per approach link.
- **Signal Logic:** 4 distinct phases per intersection (Phase 0: N-S Green, Phase 1: N-S Yellow, Phase 2: E-W Green, Phase 3: E-W Yellow).

### 2.3 Traffic Demand Scenarios
1. **Directional Shift (`directional_shift`):**
   - High-volume North-South corridor surge ($1200\,\text{veh/hr/lane}$) transitioning to East-West cross-traffic surge ($1400\,\text{veh/hr/lane}$).
   - Tests dynamic adaptation to non-stationary spatial flow imbalances.
2. **Rush Hour Peak (`rush_hour`):**
   - Exponentially increasing demand peaking at $t = 600\,\text{s}$ across all perimeter inflows, followed by sustained saturation.
   - Tests queue spillback prevention and network throughput preservation.
3. **Balanced Normal (`normal`):**
   - Uniform stochastic Poisson arrival rates ($400\,\text{veh/hr/lane}$) on all arterial approaches.
   - Serves as the nominal baseline condition.

---

## 3. Evaluated Algorithms and Baselines

| Algorithm Identifier | Paradigm | Objective Weighting | Interaction Model | Representation | Safety Shield |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IA-MOPPO (Proposed)** | On-Policy Deep RL | Dynamic Simplex $\Delta^4$ | SCOIL Bilinear Manifold | Kipf GCN + Temporal GRU | Enforced |
| **Fixed_Time** | Classical Cyclic | Static Default | None | N/A | Implicit Cycle |
| **FW_PPO** | Multi-Objective RL | Constant $w_k = 0.2$ | None | Kipf GCN + Temporal GRU | Enforced |
| **SA_PPO** | Adaptive MORL | State-Dependent $w(s)$ | None (Direct MLP) | Kipf GCN + Temporal GRU | Enforced |
| **DOIM_no_interaction** | Ablation Study | Dynamic Simplex $\Delta^4$ | Zero Matrix $M = 0$ | Kipf GCN + Temporal GRU | Enforced |
| **DOIM_rolling_pearson**| Ablation Study | Dynamic Simplex $\Delta^4$ | Rolling Pearson $R_t$ | Kipf GCN + Temporal GRU | Enforced |
| **DOIM_no_shield** | Ablation Study | Dynamic Simplex $\Delta^4$ | SCOIL Bilinear Manifold | Kipf GCN + Temporal GRU | Disabled |

---

## 4. Multi-Objective Formulation

The system explicitly evaluates 5 physical traffic objectives:

1. **Cumulative Delay / Waiting Time ($z_1$, minimize):**
   $$z_1(t) = \sum_{v \in \mathcal{V}_t} w_v(t) \quad [\text{vehicle-seconds}]$$
2. **Queue Length ($z_2$, minimize):**
   $$z_2(t) = \sum_{l \in \mathcal{L}} \text{halting}(l, t) \quad [\text{vehicles}]$$
3. **CO2 Emissions ($z_3$, minimize):**
   $$z_3(t) = \sum_{v \in \mathcal{V}_t} \text{EM}_{CO_2}(v, t) \quad [\text{mg/s}]$$
4. **Throughput ($z_4$, maximize):**
   $$z_4(t) = N_{\text{arrived}}(t) \quad [\text{completed vehicles}]$$
5. **Delay Disparity / Fairness ($z_5$, minimize):**
   $$z_5(t) = \frac{\sum_{v} (w_v(t) - \bar{w}(t))^2}{|\mathcal{V}_t|} \quad [\text{variance of waiting times}]$$

---

## 5. Statistical Rigor and Reproducibility Protocol

### 5.1 Seeding Architecture
- **Training Seeds:** Fixed multi-seed set: `[42, 123, 456, 789, 2024]`.
- **Evaluation Seeds:** Disjoint evaluation seeds: `[1001, 1002, 1003, 1004, 1005]`.
- **Seed Injection:** Seed passed synchronously to `random`, `np.random`, `torch.manual_seed`, Gymnasium `env.reset(seed=s)`, and SUMO CLI (`--seed s`).
- **Validation:** Automated detection of cross-seed duplicate outputs (DEF-04 prevention).

### 5.2 Reporting Metrics
For every evaluation metric $X$:
- Sample Mean: $\bar{X} = \frac{1}{N}\sum_{i=1}^N X_i$
- Sample Standard Deviation: $s = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (X_i - \bar{X})^2}$
- Median: $\tilde{X}$
- 95% Confidence Interval: $\bar{X} \pm t_{N-1, 0.975} \cdot \frac{s}{\sqrt{N}}$
- Two-Sided Welch's $t$-test (unequal variance) against FW-PPO:
  $$t = \frac{\bar{X}_{\text{DOIM}} - \bar{X}_{\text{base}}}{\sqrt{\frac{s_{\text{DOIM}}^2}{N_1} + \frac{s_{\text{base}}^2}{N_2}}}$$
- Standardized Effect Size (Cohen's $d$).

### 5.3 Automated Verification Pipeline
Every completed run directory must pass:
1. `verify_finite_metrics`: Zero NaNs, Infs, or truncated records in `metrics_step.csv`.
2. `verify_json_csv_consistency`: Exact mathematical agreement between raw step data and `summary.json`.
3. `verify_simplex_weights`: $w_k \ge 0$ and $\sum_k w_k = 1.0 \pm 10^{-4}$.
4. SHA-256 Configuration Fingerprinting: Immutable record of all hyperparameters and network weights.
