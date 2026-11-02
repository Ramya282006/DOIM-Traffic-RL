# DOIM-Traffic-RL: Publication-Grade Experimental Protocol
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Standard:** Strict Q1 Scientific Methodology, Pre-Registration, and Reproducibility  
**Protocol Version:** `2.0-IEEE-TITS`  
**Reference Code:** `experiments/run_comprehensive_matrix.py`, `analysis/statistical_analyzer.py`

---

## 1. Overview and Scientific Hypotheses

### 1.1 Central Research Question
> *"Can an end-to-end differentiable, state-conditioned objective-interaction representation estimate time-varying trade-offs in urban traffic networks, enabling robust multi-objective signal control under non-stationary demand shifts while strictly enforcing microscopic safety and arterial corridor equity constraints?"*

### 1.2 Formal Testable Hypotheses
- **$H_1$ (Convergence & Non-Stationary Adaptation Advantage):** Under sudden directional demand shifts, policies conditioned on the Dynamic Objective Interaction Manifold ($\mathcal{M}_t$) achieve statistically significant reductions in mean waiting time ($p < 0.05$ after FDR correction, Cohen's $d > 0.8$) compared to fixed-weight and state-adaptive baselines.
- **$H_2$ (Interaction Learning vs. Statistical Correlation):** State-Conditioned Objective Interaction Learning (SCOIL) with Bayesian shrinkage outperforms empirical rolling Pearson correlation by estimating directional cross-coupling ($M_{ij} \neq M_{ji}$) rather than confounding shared demand trends.
- **$H_3$ (Microscopic Safety Integrity):** A formal action shield deriving conflict matrices and clearance graphs from junction geometry eliminates signal-induced movement conflicts without destabilizing reinforcement learning policy convergence.
- **$H_4$ (Multi-Tier Equity Preservation):** Incorporating corridor delay disparity ($1.0 - J(\vec{d}_{\text{approach}}) \in [0, 1]$) into adaptive preference weighting significantly curtails side-street starvation ($\ge 25\%$ reduction in 95th percentile delay) without degrading total network throughput.

---

## 2. Experimental Environment & Scenarios

### 2.1 Microscopic Simulation Setup
- **Simulator:** Eclipse SUMO (Simulation of Urban MObility) v1.28.0 via TraCI.
- **Simulation Step Length:** $\Delta t = 1.0\text{ s}$.
- **Teleportation:** Explicitly disabled (`--time-to-teleport -1`) to enforce physical conservation of all vehicles.
- **Network Topology:** 9-intersection $3 \times 3$ grid with 300-meter link lengths.
  - 5 Controllable TLS: `A1, B0, B1, B2, C1` (arterial corridor and central grid).
  - 4 Boundary Inflow TLS: `A0, A2, C0, C2`.

### 2.2 Pre-Registered Evaluation Scenarios
1. **Directional Shift (`directional_shift` / North-South Surge):**
   - High-volume North-South corridor surge ($2200\text{ veh/h}$) competing with baseline East-West flows ($800\text{ veh/h}$).
   - Tests dynamic preference re-weighting under acute spatial competition.
2. **Rush Hour Saturation (`rush_hour`):**
   - Concentrated peak demand ($3000\text{ veh/h}$) approaching network saturation capacity.
   - Tests queue spillback mitigation and throughput preservation under heavy load.
3. **Balanced Ambient (`normal`):**
   - Uniform stochastic Poisson arrival rates ($1200\text{ veh/h}$) across all perimeter entries.
   - Benchmarks nominal efficiency without extreme directional bias.
4. **Held-Out Generalization Scenarios:**
   - Unseen inflow intensities ($\pm 25\%$ demand scaling) and stochastic route seeds held out until final evaluation.

---

## 3. Evaluated Algorithm Suite & Baselines

| Algorithm Identifier | Objective Weighting | Interaction Model | Representation | Safety Shield | Training Instantiation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IA-MOPPO (Proposed)** | Dynamic Simplex $w_t \in \Delta^4$ | SCOIL Bilinear + Shrinkage | GCN + GRU + DOIM | Enforced | Fresh per training seed |
| **Fixed-Time** | Rule-Based Webster Cycle | None | Static Timings | Enforced (Cyclic) | Non-learning reference |
| **Fixed-Weight PPO (FW-PPO)** | Static Uniform ($w_k = 0.2$) | None | GCN + GRU | Enforced | Fresh per training seed |
| **State-Adaptive PPO (SA-PPO)** | Direct State MLP $w(s)$ | None | GCN + GRU | Enforced | Fresh per training seed |
| **DOIM_no_interaction** | Dynamic Simplex $w_t \in \Delta^4$ | Zero Matrix ($M = 0$) | GCN + GRU + DOIM | Enforced | Ablation baseline |
| **DOIM_rolling_pearson** | Dynamic Simplex $w_t \in \Delta^4$ | Rolling Pearson $R_t$ | GCN + GRU + DOIM | Enforced | Ablation baseline |
| **DOIM_no_shield** | Dynamic Simplex $w_t \in \Delta^4$ | SCOIL Bilinear + Shrinkage | GCN + GRU + DOIM | Disabled | Safety ablation |

---

## 4. Training & Evaluation Protocol

### 4.1 Staged Budget Qualification
- **Diagnostic / Smoke Stage (128 steps):** Verifies execution flow, TraCI socket communication, GNN autograd backward passes, and manifest generation.
- **Pilot Stage (5,000 steps):** Evaluates learning curve stability across 3 seeds (`[42, 123, 456]`), verifying gradient flow checksums and non-negative returns.
- **Main Mainstream Convergence (10,000 – 50,000 steps):** Executes across $\ge 5$ independent training seeds (`[42, 123, 456, 789, 2024]`).

### 4.2 Disjoint Seeding & Deterministic Evaluation
- **Independent Training Seeds:** `[42, 123, 456, 789, 2024]`. Every baseline model is trained from scratch with its own random weight initialization and stochastic trajectory.
- **Disjoint Evaluation Seeds:** `[1001, 1002, 1003, 1004, 1005]`.
- **Frozen Deterministic Evaluation:** Evaluation episodes are executed with frozen parameters (`torch.no_grad()`, `eval()` mode, deterministic greedy actions $\arg\max \pi_\theta$).
- **Common Horizon:** Strict evaluation horizon of $T_{\text{eval}} = 1800\text{ s}$ (30 minutes of microscopic simulation) across all algorithms.

---

## 5. Statistical Aggregation & Integrity Rules

1. **Independent Training Seed as Unit:** The primary statistical sample size $N$ is strictly the number of independent trained models ($N \ge 5$). Repeated evaluation episodes of the same model are averaged first to prevent pseudoreplication.
2. **Paired Comparisons:** Models trained on seed $s_i$ are evaluated on the exact same set of evaluation seeds $\{e_1, \dots, e_m\}$ as baselines, enabling paired Student's $t$-tests and Wilcoxon signed-rank tests.
3. **Multiple Testing Correction:** All cross-algorithm p-values are adjusted using the Holm-Bonferroni step-down procedure to guarantee Family-Wise Error Rate (FWER) control at $\alpha = 0.05$.
4. **Mandatory Reporting:** Every reported result must present Mean $\pm$ Standard Deviation, Median, 95% Student-t Confidence Interval, and Cohen's $d$ effect size.
5. **No Selective Reporting:** All training runs, whether converged or degraded, must be preserved in immutable run manifests. Failed runs with safety faults are explicitly documented in `rejected_runs`.
