# Phase 28.2: Frozen Tier 2 Confirmatory Benchmark Matrix
**Project:** DOIM-PPO (*Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems*)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard:** Unambiguous Frozen Experimental Design  
**Auditing Panel:** Independent Research Validation Team  
**Date:** November 3, 2026  

---

## 1. Experimental Design Evaluation

We evaluated two alternative 105-run confirmatory matrices:
- **Design Option A (Statistical Power Priority):** 7 algorithms $\times$ 3 scenarios $\times$ 5 seeds = **105 evaluation runs**. (4 RL algorithms $\times$ 3 scenarios $\times$ 5 seeds = 60 training runs).
- **Design Option B (Scenario Coverage Priority):** 7 algorithms $\times$ 5 scenarios $\times$ 3 seeds = **105 evaluation runs**. (4 RL algorithms $\times$ 5 scenarios $\times$ 3 seeds = 60 training runs).

### Decision:
To guarantee **IEEE T-ITS statistical publishability**, **Design Option A** is the scientifically superior choice. 
- With $N=5$ seeds, degrees of freedom $df=4$ allows valid Student's $t$ confidence intervals and hypothesis testing.
- Testing $N=3$ seeds leaves $df=2$, resulting in unacceptably wide confidence intervals that undermine statistical significance.
- The 3 selected scenarios in Design A cover the complete spectrum of traffic dynamics:
  1. `grid3x3` (baseline balanced grid network).
  2. `arterial_corridor` (high-speed coordinated arterial corridor).
  3. `incident_disruption` (stochastic lane blockage and incident disruption).

---

## 2. Complete Frozen 105-Run Confirmatory Matrix

### 2.1 Algorithm Roster (7 Algorithms)
1. **Fixed-Time** (Webster's optimal fixed-cycle baseline).
2. **Actuated** (Gap-out detector-actuated baseline).
3. **Max-Pressure** (Greedy queue-differential baseline).
4. **Fixed-Weight PPO** (Multi-objective PPO with static $w_i = 0.2$).
5. **State-Adaptive PPO** (Multi-objective PPO with heuristic weight adaptation).
6. **MLP-PPO** (Standard single-objective/flat PPO baseline).
7. **DOIM-PPO** (Our proposed spatio-temporal dynamic interaction manifold model).

### 2.2 Scenario & Seed Allocation Matrix

| Scenario Name | Traffic Regime | Training Seeds | Held-Out Evaluation Seeds | Training Steps | Eval Horizon |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `grid3x3` | Uniform Grid Flow | $\{42, 43, 44, 45, 46\}$ | $\{1042, 1043, 1044, 1045, 1046\}$ | $5,000$ | $1,800\,\text{s}$ |
| `arterial_corridor` | Coordinated Progression | $\{42, 43, 44, 45, 46\}$ | $\{1042, 1043, 1044, 1045, 1046\}$ | $5,000$ | $1,800\,\text{s}$ |
| `incident_disruption` | Capacity Loss & Recovery | $\{42, 43, 44, 45, 46\}$ | $\{1042, 1043, 1044, 1045, 1046\}$ | $5,000$ | $1,800\,\text{s}$ |

- **Total Confirmatory Runs:**
  - Non-RL Baselines (Fixed-Time, Actuated, Max-Pressure): $3 \text{ algos} \times 3 \text{ scenarios} \times 5 \text{ eval seeds} = 45$ evaluation runs.
  - RL Algorithms (FW-PPO, SA-PPO, MLP-PPO, DOIM-PPO): $4 \text{ algos} \times 3 \text{ scenarios} \times 5 \text{ train seeds} = 60$ training runs.
  - RL Evaluations: $4 \text{ algos} \times 3 \text{ scenarios} \times 5 \text{ eval seeds} = 60$ evaluation runs.
  - **Grand Total Executions:** $45 + 60 + 60 = \mathbf{165}$ executions (yielding **105 clean evaluation runs** for the comparative benchmark).

---

## 3. Strict Pre-Registration Protocol
1. **Zero Retries:** No run may be re-executed because of an unfavorable outcome.
2. **Failures Logged:** Any simulation crash or TraCI disconnect will be recorded in `failure_log.json`.
3. **Checksum Locking:** Checkpoints will be locked with SHA-256 hashes immediately upon training completion.
