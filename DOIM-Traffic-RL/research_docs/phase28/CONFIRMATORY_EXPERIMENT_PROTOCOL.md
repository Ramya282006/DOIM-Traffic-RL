# Phase 28: Preregistered Confirmatory Experimental Protocol
**Project:** DOIM-PPO (Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Protocol Status:** FROZEN — Version 1.0.0 (Pre-Execution Preregistration)  
**Date:** November 3, 2026  
**Commit:** `3561849b0b7a1333d2223a3c095dcb3676cf1700`  

---

## 1. Protocol Freezing Declaration

In accordance with Section 4 of the Phase 28 Master Directive and the Open Science Framework / IEEE reproducible research guidelines, this protocol formally **freezes** the experimental design, algorithms, traffic scenarios, random seeds, budgets, metrics, and statistical decision rules prior to inspecting confirmatory benchmark results.

Any subsequent deviation from this protocol must be documented with an explicit version increment and scientific justification.

---

## 2. Compared Algorithms & Architectures

The comparative benchmark evaluates seven distinct control architectures and two ablation baselines under identical evaluation environments:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BENCHMARK ALGORITHM SUITE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Fixed-Time (Classical Transyt baseline; cyclic 30s green / 3s yellow)    │
│ 2. Actuated (Time-gap vehicle actuation using loop detector induction)     │
│ 3. Max-Pressure (Greedy queue-differential balancing controller)            │
│ 4. Fixed-Weight PPO (Linear scalarization: w = [0.2, 0.2, 0.2, 0.2, 0.2])   │
│ 5. State-Adaptive PPO (Dynamic urgency weight heuristic w(s) ∝ z(s))        │
│ 6. Parameter-Matched MLP-PPO (Ablated spatial-temporal graph encoder)      │
│ 7. DOIM-PPO (Proposed: GCN + GRU + SCOIL + Preference Head + Shield)        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Ablation A: DOIM-PPO w/o Interaction Manifold (M = 0 static decouple)       │
│ Ablation B: DOIM-PPO w/ Rolling Pearson Correlation baseline                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Traffic Scenarios & Demand Profiles

Experiments are conducted on the calibrated SUMO 3x3 urban road network (`configs/network/grid3x3.net.xml`), encompassing 9 intersections (5 controllable traffic lights, 4 uncontrolled boundary nodes) and 24 arterial links:

| Scenario Identifier | Configuration Path | Demand Pattern | Duration | Primary Stress Condition |
| :--- | :--- | :--- | :---: | :--- |
| **Scenario 1: Directional Surge** | `scenarios/directional_shift/north_south_surge.sumocfg` | 1800 veh/h North-South surge; 600 veh/h East-West cross-traffic | 1,800s | Severe directional asymmetry; cross-street starvation risk |
| **Scenario 2: Rush Hour Congestion** | `scenarios/rush_hour/rush_hour.sumocfg` | 2400 veh/h grid-wide saturation demand | 1,800s | High network density; widespread queue spillback |
| **Scenario 3: Nominal Balanced** | `scenarios/normal/normal.sumocfg` | 1000 veh/h uniform Poisson distributed arrivals | 1,800s | Nominal balanced multi-objective baseline |
| **Scenario 4: Arterial Corridor** | `scenarios/arterial_corridor/arterial_corridor.sumocfg` | 2600 veh/h high-speed E-W arterial corridor | 1,800s | Arterial coordination trade-off against cross streets |
| **Scenario 5: Incident Disruption** | `scenarios/incident_disruption/incident_disruption.sumocfg` | Mid-episode lane capacity bottleneck on link B0B1 | 1,800s | Localized capacity drop; resilience and recovery |

---

## 4. Seeding & Evaluation Independence

To prevent overfitting to evaluation seeds and eliminate pseudo-random evaluation artifacts:
1. **Independent Training Seeds:**
   $$\mathcal{S}_{\text{train}} = \{42, 123, 456, 789, 2024\}$$
2. **Held-Out Evaluation Seeds:**
   $$\mathcal{S}_{\text{eval}} = \{s + 1000 \mid s \in \mathcal{S}_{\text{train}}\} = \{1042, 1123, 1456, 1789, 3024\}$$
   - Training rollouts and evaluation episodes operate on mutually disjoint random traffic arrival sequences.
3. **Statistical Unit:** Strictly the independent training seed. Repeated evaluation episodes within a trained checkpoint are averaged prior to computing cross-seed statistics.

---

## 5. Experiment Budget Tiers & Stopping Rules

| Tier Name | Training Steps / Seed | Checkpoint Interval | Target Use | Compute Approval Gate |
| :--- | :---: | :---: | :--- | :---: |
| **Smoke Tier** | 128 | 128 | Fast pipeline & environment sanity check | Automatic |
| **Pilot Tier** | 5,000 | 2,500 | Fast comparative baseline verification | Requires User Sign-Off |
| **Intermediate Tier**| 10,000 | 2,500 | Medium convergence benchmark | Requires User Sign-Off |
| **Main Convergence** | 50,000 | 10,000 | Full asymptotic publication benchmark | Requires User Sign-Off |

### Non-Discretionary Stopping Rule:
- Once an experiment is initiated, it must run to its designated step budget.
- Early stopping based on intermediate evaluation scores is strictly prohibited.
- Selective re-running of outlier or underperforming seeds is strictly prohibited.

---

## 6. Primary and Secondary Outcome Measures

### 6.1 Primary Outcomes (Reported in Physical Units)
1. **Average Delay ($w_{\text{avg}}$):** Mean waiting time in vehicle-seconds ($\text{veh}\cdot\text{s}$). [Minimization]
2. **Average Queue ($q_{\text{avg}}$):** Mean halting vehicle count across approaches. [Minimization]
3. **CO2 Emissions ($e_{\text{CO}_2}$):** Integrated microscopic mass in grams ($\text{g}$). [Minimization]
4. **Network Throughput ($R$):** Extrapolated completion rate in vehicles per hour ($\text{veh/hr}$). [Maximization]
5. **Fairness Disparity ($D_{\text{fair}}$):** Primary equity metric $1.0 - J(\text{delays}) \in [0, 1]$. [Minimization]

### 6.2 Secondary Multi-Objective & Safety Outcomes
1. **Three-Tier Equity Decomposition:**
   - Approach-level disparity and Gini coefficient ($F_{\text{approach}}$).
   - Junction-level disparity and variance ($F_{\text{junction}}$).
   - Vehicle-level 95th percentile delay and disparity ($F_{\text{veh}}$).
2. **2D Hypervolume Indicator ($\text{HV}$):** Evaluated against reference point $(1.0, 1.0)$ in normalized minimization space.
3. **Safety Shield Interventions ($I_{\text{shield}}$):** Count of actions modified by runtime safety shield.
4. **Phase Conflict Violations:** Must be strictly **0** in all compliant runs.

---

## 7. Statistical Decision Rules & Multiple Comparisons

1. **Significance Testing:** Two-sided Mann-Whitney U test (non-parametric) for pairwise algorithmic comparisons.
2. **Effect Size:** Cliff's delta $\delta \in [-1, 1]$ computed with metric-direction alignment:
   - For cost metrics (delay, queue, emissions, fairness disparity): $\delta > 0 \iff x < y$ (candidate superior).
   - For benefit metrics (throughput): $\delta > 0 \iff x > y$ (candidate superior).
3. **Confidence Intervals:** 95% Student-t confidence intervals computed across independent training seeds.
4. **Multiple Comparison Correction:** Holm-Bonferroni step-down correction applied across all hypothesis tests within each scenario family at family-wise error rate $\alpha = 0.05$.
5. **Practical Significance Criterion:** A result is claimed as practically superior only if both:
   - Adjusted $p$-value $p_{\text{adj}} < 0.05$.
   - Effect size magnitude $|\delta| \ge 0.33$ (medium effect) or $|\delta| \ge 0.474$ (large effect).

---

## 8. Data Exclusion & Missing Data Policy

- **Exclusion Criteria:** An experiment run is excluded from confirmatory analysis if and only if:
  - Step count in `metrics_step.csv` does not match the planned budget ($N_{\text{steps}} \neq N_{\text{budget}}$).
  - A fatal simulation crash or safety fault occurred (`safety_fault == True`).
  - The model checkpoint SHA-256 hash cannot be verified.
- **Missing Data Policy:** No synthetic imputation. If a seed run fails due to hardware crash, the failure is logged and the reduced sample size is reported honestly with adjusted degrees of freedom.

