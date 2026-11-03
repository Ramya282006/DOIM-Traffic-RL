# Phase 28.2: Metric Definition & Aggregation Audit
**Project:** DOIM-PPO (*Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems*)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard:** Rigorous Mathematical Formulation & Code Traceability  
**Auditing Panel:** Independent Research Validation Team  
**Date:** November 3, 2026  

---

## 1. Overview & Forensic Audit Scope

In traffic reinforcement learning literature, confusing **system-wide accumulated waiting time** with **per-vehicle delay** is a common source of reviewer confusion. This document establishes the exact mathematical definitions, units, SUMO TraCI API calls, and code locations for all five traffic objectives in DOIM-PPO.

---

## 2. Objective Formulations & Code Implementations

### Objective 1: Waiting Time and Delay ($f_1$)
- **SUMO TraCI API:** `traci.lane.getWaitingTime(lane_id)`
- **Mathematical Definition (Instantaneous System Waiting Time):**
  $$w(t) = \sum_{j \in \mathcal{J}} \sum_{l \in \mathcal{L}_j^{\text{in}}} w_l(t) \quad [\text{veh}\cdot\text{s}]$$
  where $w_l(t)$ is the accumulated waiting time of all vehicles on lane $l$ having speed $< 0.1\,\text{m/s}$.
- **Episode Summary Metric (Network-Wide Mean Accumulated Waiting Time):**
  $$W_{\text{net}} = \frac{1}{T} \sum_{t=1}^T w(t) \quad [\text{veh}\cdot\text{s/step}]$$
  *Code Location:* `DOIM-Traffic-RL/objectives/waiting_time.py` -> `compute_total_waiting`  
  *Reported Values in Pilot:* DOIM-PPO: $510.94$, Fixed-Time: $366.93$, Max-Pressure: $158.53$.
- **Per-Vehicle Trip Delay (Secondary Traffic Metric):**
  $$\bar{d}_{\text{veh}} = \frac{\sum_{t=1}^T w(t)}{\sum_{t=1}^T N_{\text{veh}}(t)} \quad [\text{s/veh}]$$
  *Values in Pilot:* DOIM-PPO: $\approx 7.90\,\text{s}$, Fixed-Time: $\approx 8.05\,\text{s}$, Max-Pressure: $\approx 6.88\,\text{s}$.

### Objective 2: Queue Length ($f_2$)
- **SUMO TraCI API:** `traci.lane.getLastStepHaltingNumber(lane_id)`
- **Mathematical Definition:**
  $$q(t) = \sum_{j \in \mathcal{J}} \sum_{l \in \mathcal{L}_j^{\text{in}}} n_l^{\text{halt}}(t) \quad [\text{veh}]$$
- **Episode Summary Metric:**
  $$Q = \frac{1}{T} \sum_{t=1}^T q(t) \quad [\text{veh/step}]$$
  *Code Location:* `DOIM-Traffic-RL/objectives/queue_length.py` -> `compute_queue_length`  
  *Values in Pilot:* DOIM-PPO: $7.26\,\text{veh}$, Fixed-Time: $6.01\,\text{veh}$, Max-Pressure: $2.84\,\text{veh}$.

### Objective 3: Carbon Dioxide Emissions ($f_3$)
- **SUMO TraCI API:** `traci.lane.getCO2Emission(lane_id)`
- **Mathematical Definition:**
  $$e(t) = \sum_{j \in \mathcal{J}} \sum_{l \in \mathcal{L}_j^{\text{in}}} \text{CO2}_l(t) \quad [\text{mg/s}]$$
- **Episode Summary Metrics:**
  - Mean Emission Rate: $E = \frac{1}{T} \sum_{t=1}^T e(t)$ [mg/s]
  - Total Cumulative Mass: $M_{\text{CO}_2} = \sum_{t=1}^T e(t) \times 10^{-6}$ [kg]  
  *Values in Pilot:* DOIM-PPO: $113,768.5\,\text{mg/s}$ ($204.78\,\text{kg}$ total in 1,800s).

### Objective 4: Network Throughput ($f_4$)
- **SUMO TraCI API:** `traci.simulation.getArrivedNumber()`
- **Mathematical Definition:**
  $$r_{\text{arr}}(t) = \text{count of vehicles completing trips at step } t \quad [\text{veh}]$$
- **Episode Summary Metric (Total Completed Trips):**
  $$R_{\text{total}} = \sum_{t=1}^T r_{\text{arr}}(t) \quad [\text{completed vehicles}]$$
  *Values in Pilot:* DOIM-PPO: $641\,\text{veh}$, Fixed-Time: $643\,\text{veh}$, Max-Pressure: $648\,\text{veh}$.

### Objective 5: Delay Fairness ($f_5$)
- **Mathematical Formulation (Jain's Fairness Disparity Index):**
  $$J(\mathbf{w}) = \frac{\left(\sum_{j=1}^{|\mathcal{J}|} \bar{w}_j\right)^2}{|\mathcal{J}| \sum_{j=1}^{|\mathcal{J}|} \bar{w}_j^2} \in [0, 1], \quad \text{Disparity} = 1 - J(\mathbf{w})$$
  where $\bar{w}_j$ is the average delay at junction $j$. A disparity of $0$ indicates perfect equitable distribution across junctions.
  *Code Location:* `DOIM-Traffic-RL/objectives/fairness.py` -> `CorridorFairnessCalculator`  
  *Values in Pilot:* DOIM-PPO: $0.5020$, Fixed-Time: $0.4194$, Max-Pressure: $0.3868$.

---

## 3. Discrepancy Reconciliation Summary Table

| Metric Label | Unit | Mathematical Formula | DOIM-PPO Pilot Value | Fixed-Time Pilot Value | Max-Pressure Pilot Value |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **System Waiting Time ($W_{\text{net}}$)** | veh·s/step | $\frac{1}{T}\sum_t \sum_l w_l(t)$ | **510.9428** | 366.9322 | 158.5339 |
| **Per-Vehicle Delay ($\bar{d}_{\text{veh}}$)** | s/veh | $W_{\text{net}} / \bar{N}_{\text{veh}}$ | **7.9041** | 8.0573 | 6.8814 |
| **Mean Queue ($Q$)** | veh/step | $\frac{1}{T}\sum_t \sum_l q_l(t)$ | **7.2567** | 6.0056 | 2.8372 |
| **CO₂ Emission Rate ($E$)** | mg/s | $\frac{1}{T}\sum_t e(t)$ | **113,768.5** | 112,207.5 | 106,838.8 |
| **Total Throughput ($R_{\text{total}}$)** | veh | $\sum_t r_{\text{arr}}(t)$ | **641.0** | 643.0 | 648.0 |
| **Fairness Disparity ($1 - J$)** | $[0, 1]$ | $1 - \frac{(\sum w)^2}{N \sum w^2}$ | **0.5020** | 0.4194 | 0.3868 |
| **Safety Interventions ($I$)** | count | $\sum_t \mathbf{1}_{\text{shield}}$ | **4,249** | 0 | 6,240 |

---

## 4. Manuscript Standardization Directives
1. In the IEEE T-ITS manuscript, **both** System Waiting Time (veh·s/step) and Per-Vehicle Delay (s/veh) will be explicitly reported in Table I.
2. The exact definitions above will be formally stated in Section III.B (*Multi-Objective Formulation*).
