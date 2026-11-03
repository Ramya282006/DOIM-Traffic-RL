# PHASE 34.4: METRIC INTEGRITY FORENSIC AUDIT & RECONCILIATION

**Mission ID:** `DOIM-PPO-PHASE34-INDEPENDENT-FORENSIC-AUDIT`  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Date:** 2026-11-04  
**Audit Standard:** Definitive Mathematical Definitions & Exact Recalculation from Raw Timestep Logs

---

## 1. Executive Summary & Resolution of Ambiguities

During Phases 28 through 33, reviewers noted inconsistencies and ambiguity in performance metrics, specifically:
1. Conflation between Network Waiting Time ($W_{\text{net}} \approx 510.94\,\text{s}$) and Per-Vehicle Delay ($\bar{d}_{\text{veh}} \approx 7.90\,\text{s/veh}$).
2. Inconsistent definitions of the Fairness metric (Jain's Fairness Index vs Delay Disparity).
3. Ambiguity regarding instantaneous step throughput vs cumulative episode throughput.
4. Accounting of Safety Shield interventions across baseline algorithms.

Phase 34 re-audited all 29 verified runs directly from the raw simulation logs (`metrics_step.csv`). Every metric has been given an explicit mathematical definition, physical units, optimization direction, and verified numerical values.

---

## 2. Definitive Mathematical Metric Suite

### 2.1 Primary Efficiency Metrics

#### 1. Instantaneous Network Waiting Time ($W_{\text{net}}(t)$) & Time-Averaged Network Delay ($W_{\text{net}}$)
- **Mathematical Definition:**
  $$W_{\text{net}}(t) = \sum_{v \in \mathcal{V}(t)} w_v(t) \cdot \mathbf{1}_{\{v_v(t) < 0.1\,\text{m/s}\}}, \quad W_{\text{net}} = \frac{1}{T} \sum_{t=1}^T W_{\text{net}}(t)$$
- **Physical Interpretation:** Cumulative seconds spent halting by all vehicles present in the entire network at time $t$. Time-averaged across $T = 1,800$ simulation seconds.
- **Unit:** Seconds ($\text{s}$).
- **Direction:** Minimization (lower is better).
- **Verified Values (grid3x3, Seed 1042):**
  - DOIM-PPO: **510.94 s**
  - MLP-PPO: **721.11 s**
  - FW-PPO: **812.44 s**
  - SA-PPO: **949.39 s**
  - Actuated: **1,454.51 s**
  - Fixed-Time: **366.93 s**
  - Max-Pressure: **158.53 s**

#### 2. Estimated Average Per-Vehicle Delay ($\bar{d}_{\text{veh}}$)
- **Mathematical Definition (via Little's Law / Steady-State Normalization):**
  $$\bar{d}_{\text{veh}} = \frac{W_{\text{net}}}{\bar{N}_{\text{veh}}}$$
  where $\bar{N}_{\text{veh}}$ is the time-averaged active vehicle population ($\bar{N}_{\text{veh}} \approx 64.64$ vehicles on `grid3x3`).
- **Physical Interpretation:** Average seconds of delay experienced by an individual vehicle per second of residence time in the network.
- **Unit:** Seconds per vehicle ($\text{s/veh}$).
- **Direction:** Minimization (lower is better).
- **Verified Values (grid3x3, Seed 1042):**
  - DOIM-PPO: **7.90 s/veh**
  - MLP-PPO: **11.16 s/veh**
  - FW-PPO: **12.57 s/veh**
  - SA-PPO: **14.69 s/veh**
  - Actuated: **22.50 s/veh**
  - Fixed-Time: **5.68 s/veh**
  - Max-Pressure: **2.45 s/veh**

#### 3. Average Queue Length ($\bar{Q}$)
- **Mathematical Definition:**
  $$\bar{Q} = \frac{1}{T} \sum_{t=1}^T \sum_{l \in \mathcal{L}} q_l(t)$$
- **Physical Interpretation:** Time-averaged total number of queued vehicles (speed $< 0.1\,\text{m/s}$) across all approaches.
- **Unit:** Vehicles ($\text{veh}$).
- **Direction:** Minimization (lower is better).
- **Verified Values (grid3x3, Seed 1042):**
  - DOIM-PPO: **7.26 veh** (Peak: 20.0)
  - MLP-PPO: **9.93 veh** (Peak: 25.0)
  - FW-PPO: **11.36 veh** (Peak: 30.0)
  - SA-PPO: **12.90 veh** (Peak: 32.0)
  - Actuated: **19.86 veh** (Peak: 43.0)
  - Fixed-Time: **6.01 veh** (Peak: 14.0)
  - Max-Pressure: **2.84 veh** (Peak: 11.0)

#### 4. Cumulative Vehicle Throughput ($\mathcal{C}_{\text{th}}$)
- **Mathematical Definition:**
  $$\mathcal{C}_{\text{th}} = \sum_{t=1}^T n_{\text{exited}}(t)$$
- **Physical Interpretation:** Total number of vehicles that completed their trips and exited the network boundary during the 1,800-second evaluation horizon.
- **Unit:** Vehicles ($\text{veh}$).
- **Direction:** Maximization (higher is better).
- **Verified Values (grid3x3, Seed 1042):**
  - DOIM-PPO: **641 veh**
  - MLP-PPO: **638 veh**
  - FW-PPO: **637 veh**
  - SA-PPO: **637 veh**
  - Actuated: **631 veh**
  - Fixed-Time: **643 veh**
  - Max-Pressure: **648 veh**

---

### 2.2 Equity & Multi-Tier Fairness Metrics

#### 5. Delay Disparity ($\delta_{\text{fair}}$) & Jain's Fairness Index ($J(x)$)
- **Mathematical Definition:**
  $$J(x) = \frac{\left( \sum_{i=1}^K x_i \right)^2}{K \sum_{i=1}^K x_i^2} \in \left[ \frac{1}{K}, 1.0 \right], \quad \delta_{\text{fair}} = 1.0 - J(x) \in [0.0, 1.0]$$
  where $x_i$ represents the aggregate accumulated delay on approach/corridor $i \in \{1, \dots, K\}$.
- **Standardization:**
  - In code and RL reward functions (`raw_fairness` in `metrics_step.csv`), the metric tracked is **Delay Disparity** $\delta_{\text{fair}}$ (lower is better, $0.0 = \text{perfect equality}$).
  - In presentation tables, both $\delta_{\text{fair}}$ and the classic **Jain's Fairness Index** $J(x) = 1.0 - \delta_{\text{fair}}$ (higher is better, $1.0 = \text{perfect equality}$) are explicitly reported.
- **Verified Values (grid3x3, Seed 1042):**
  - DOIM-PPO: Disparity = **0.5020** | Jain's Index = **0.4980**
  - MLP-PPO: Disparity = **0.5036** | Jain's Index = **0.4964**
  - FW-PPO: Disparity = **0.4854** | Jain's Index = **0.5146**
  - SA-PPO: Disparity = **0.4721** | Jain's Index = **0.5279**
  - Fixed-Time: Disparity = **0.4194** | Jain's Index = **0.5806**
  - Max-Pressure: Disparity = **0.3868** | Jain's Index = **0.6132**

---

### 2.3 Environmental & Safety Metrics

#### 6. Carbon Dioxide Emissions ($\text{CO}_2$)
- **Mathematical Definition:** Instantaneous SUMO HBEFA3 emission rate time-averaged over the episode.
- **Physical Interpretation:** Average network emission rate in milligrams per second ($\text{mg/s}$) and total episode release in kilograms ($\text{kg}$).
- **Verified Values (grid3x3, Seed 1042):**
  - DOIM-PPO: **113,768.5 mg/s** (Total: **204.78 kg**)
  - MLP-PPO: **119,450.2 mg/s** (Total: **215.01 kg**)
  - FW-PPO: **124,110.8 mg/s** (Total: **223.40 kg**)
  - Fixed-Time: **106,838.8 mg/s** (Total: **192.31 kg**)
  - Max-Pressure: **102,635.2 mg/s** (Total: **184.74 kg**)

#### 7. Safety Shield Interventions ($I_{\text{shield}}$)
- **Mathematical Definition:**
  $$I_{\text{shield}} = \sum_{t=1}^T \sum_{j=1}^M \mathbf{1}_{\{a_j^{\text{req}}(t) \ne a_j^{\text{exec}}(t)\}}$$
- **Physical Interpretation:** Total instances where an algorithm's proposed phase transition was intercepted and overridden by the deterministic safety shield to enforce either:
  1. Minimum green duration constraint ($g_{\min} \ge 10\,\text{s}$).
  2. Yellow clearance interval constraint ($y_{\text{clear}} \ge 3\,\text{s}$).
- **Verified Values (grid3x3, Seed 1042):**
  - **Fixed-Time:** **0** (Fixed cycle inherently satisfies all minimum green and clearance constraints)
  - **Actuated:** **0** (Internal gap-extension logic is pre-configured with safety timers)
  - **DOIM-PPO:** **4,249**
  - **MLP-PPO:** **2,127**
  - **FW-PPO:** **2,393**
  - **SA-PPO:** **1,142**
  - **Max-Pressure:** **6,240** (Max-Pressure attempts to switch phases virtually every step, resulting in continuous intervention)

---

## 3. Disruption of Arterial Green Waves: The Arterial Corridor Trade-off

A critical traffic engineering insight revealed by the recalculation across scenarios:
On `arterial_corridor` (Seed 1042):
- **Fixed-Time:** $W_{\text{net}} = 1,764.26\,\text{s}$, Throughput = $1,479\,\text{veh}$
- **Actuated:** $W_{\text{net}} = 5,196.26\,\text{s}$, Throughput = $1,353\,\text{veh}$
- **Max-Pressure:** $W_{\text{net}} = 7,705.01\,\text{s}$, Throughput = $1,332\,\text{veh}$

Under an arterial corridor, Fixed-Time control provides coordinated progression ("green waves") along the primary corridor. Decentralized greedy controllers like Max-Pressure and Actuated break progression whenever minor side-street queues request green, cascading into catastrophic network delay ($> 300\%$ increase in waiting time). This demonstrates that local queue-greedy algorithms do not generalize to linear coordinated arterial systems without explicit coordination mechanisms.

---

## 4. Artifact Manifest

- Recalculated dataset: `results/phase34/recalculated_metrics.csv`
- Script: `DOIM-Traffic-RL/analysis/phase34_metric_integrity.py`

