# DOIM-PPO: Frozen Confirmatory Experimental Protocol
**Project:** DOIM-PPO (*Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems*)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard:** Open-Science Pre-Registration & Methodological Invariance  
**Date:** November 3, 2026  
**Status:** FROZEN — MANDATORY FOR ALL SUBSEQUENT EXPERIMENTAL RUNS  

---

## 1. Primary and Secondary Endpoints

### 1.1 Primary Confirmatory Endpoint
- **Network-Wide Accumulated Waiting Time Rate ($W_{\text{net}}$):**
  $$W_{\text{net}} = \frac{1}{T} \sum_{t=1}^T \sum_{j \in \mathcal{J}} \sum_{l \in \mathcal{L}_j^{\text{in}}} w_l(t) \quad [\text{veh}\cdot\text{s/step}]$$
  Minimization objective representing total vehicle delay accumulated across the network per simulation step.

### 1.2 Secondary Confirmatory Endpoints
1. **Network Throughput ($R_{\text{total}}$):**
   $$R_{\text{total}} = \sum_{t=1}^T r_{\text{arr}}(t) \quad [\text{completed vehicles}]$$
   Maximization objective measuring completed trips.
2. **Average Queue Length ($Q$):**
   $$Q = \frac{1}{T} \sum_{t=1}^T \sum_{l} q_l(t) \quad [\text{veh/step}]$$
   Minimization objective measuring halting vehicles ($v < 0.1\,\text{m/s}$).
3. **Average Carbon Dioxide Emissions ($E$):**
   $$E = \frac{1}{T} \sum_{t=1}^T e(t) \quad [\text{mg/s}]$$
   Minimization objective measuring HBEFA3 emission rate.
4. **Delay Disparity Index ($1 - J$):**
   $$\text{Disparity} = 1 - \frac{\left(\sum_{j=1}^M \bar{w}_j\right)^2}{M \sum_{j=1}^M \bar{w}_j^2} \in [0, 1]$$
   Minimization objective measuring spatial inequity across intersections.
5. **5D Normalized Hypervolume ($HV$):**
   Hypervolume computed in normalized minimization space against reference point $r = (1.2, 1.2, 1.2, 1.2, 1.2)$.
6. **Safety Shield Interventions ($I$):**
   Count of suppressed phase switch requests enforcing $10\,\text{s}$ minimum green and $3\,\text{s}$ yellow clearance.

---

## 2. Experimental Benchmark Matrix

### 2.1 Evaluated Algorithms (7)
1. **Fixed-Time:** Webster-optimized signal timing baseline.
2. **Actuated:** Vehicle gap-out detector-actuated baseline.
3. **Max-Pressure:** Queue-differential greedy pressure baseline.
4. **Fixed-Weight PPO (FW-PPO):** Static scalarization ($w_i = 0.2$).
5. **State-Adaptive PPO (SA-PPO):** Heuristic state-adaptive weighting.
6. **MLP-PPO:** Standard unaugmented PPO architecture.
7. **DOIM-PPO:** Proposed Spatio-Temporal GCN-GRU with SCOIL Dynamic Objective Interaction Manifold.

### 2.2 Benchmark Scenarios (3)
1. `grid3x3`: Uniform ambient demand with directional surge on the 3x3 research grid.
2. `arterial_corridor`: High-volume coordinated progression along E-W axis (2,600 veh/h) with cross-street starvation risk.
3. `incident_disruption`: Stochastic bottleneck mid-corridor (link B0B1) testing capacity loss and spillback recovery.

### 2.3 Independent Random Seeds
- **Training Seeds (RL Policies):** $s_{\text{train}} \in \{42, 43, 44, 45, 46\}$ ($N = 5$).
- **Held-Out Evaluation Seeds (All Controllers):** $s_{\text{eval}} \in \{1042, 1043, 1044, 1045, 1046\}$ ($N = 5$).
- **Strict Pairing Rule:** $s_{\text{eval}} = s_{\text{train}} + 1000$. All controllers evaluated under seed $s_{\text{eval}}$ encounter the identical SUMO vehicle generation sequence.

### 2.4 Training and Evaluation Horizons
- **Deep RL Training Budget:** $5,000$ simulation steps ($40$ PPO updates, $800$ gradient descent steps), verified by Option C feasibility pilot.
- **Deterministic Evaluation Horizon:** $1,800$ simulation seconds ($30$ simulation minutes).

---

## 3. Confirmatory Statistical Testing Protocol

1. **Primary Synthesis Model:** Linear Mixed-Effects Model (LMM):
   $$Y_{ijk} = \mu + \alpha_i (\text{Algorithm}) + \beta_j (\text{Scenario}) + (\alpha\beta)_{ij} + u_k (\text{Seed Random Effect}) + \epsilon_{ijk}$$
   with $df_{\text{resid}} = 72$.
2. **Pairwise Comparisons:** Paired Student's $t$-tests ($df = 4$) with Benjamini-Hochberg False Discovery Rate (FDR) controlled at $q = 0.05$.
3. **Non-Parametric Verification:** Multi-scenario pooled Wilcoxon signed-rank test ($N = 15$).
4. **Effect Size Measure:** Paired Cohen's $d_z = \bar{D} / s_D$ and Cliff's Delta.
5. **Practical Significance Criteria:** Improvement declared practically meaningful if:
   $$\Delta W_{\text{net}} \ge 5.0\% \quad \text{AND} \quad d_z \ge 0.50$$
6. **Data Integrity Rule:** Zero imputation. All failures logged transparently.
