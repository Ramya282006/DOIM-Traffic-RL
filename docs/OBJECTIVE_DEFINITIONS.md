# DOIM-Traffic-RL: Authoritative Objective Definitions & Metric Validation
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Standard:** Rigorous Q1 Traffic Engineering & Multi-Objective Measurement Standards  
**Schema Version:** `2.0-IEEE-TITS`  
**Reference Code:** `objectives/objective_config.py`, `objectives/objective_engine.py`, `objectives/fairness.py`

---

## 1. Authoritative Objective Matrix

| Objective | Symbol | Physical Unit | Optimization Direction | SUMO TraCI Source | Aggregation Scope | Calibrated Bounds $[L_k, U_k]$ | Physical Range |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Waiting Time** | $z_{\text{wait}}$ | $\text{veh}\cdot\text{s}$ (vehicle-seconds) | Minimize | `traci.lane.getWaitingTime(lane_id)` | Sum across all approach lanes of 9 junctions | $[0.0, 600.0]$ | $[0.0, 72000.0]$ |
| **Total Queue Length** | $z_{\text{queue}}$ | Halting vehicles | Minimize | `traci.lane.getLastStepHaltingNumber(lane_id)` | Sum across all approach lanes of 9 junctions | $[0.0, 20.0]$ | $[0.0, 1000.0]$ |
| **Integrated $\text{CO}_2$** | $z_{\text{co2}}$ | $\text{mg}$ (milligrams) | Minimize | `traci.vehicle.getCO2Emission(vid)` (HBEFA3) | Integrated over step ($\Delta t = 1.0\text{s}$) across all network vehicles | $[0.0, 120000.0]$ | $[0.0, 1.0 \times 10^8]$ |
| **Step Throughput** | $z_{\text{tp}}$ | Completed vehicles | Maximize (Inverted Urgency) | `traci.simulation.getArrivedNumber()` | Non-negative arrival count at current time step | $[0.0, 20.0]$ | $[0.0, 500.0]$ |
| **Corridor Fairness Disparity** | $z_{\text{fair}}$ | Dimensionless index | Minimize | $1.0 - \text{JainIndex}(\vec{d}_{\text{approach}})$ | Disparity across 4 competing arterial corridors | $[0.0, 1.0]$ | $[0.0, 1.0]$ |

---

## 2. In-Depth Objective Specifications

### 2.1 Total Waiting Time ($z_{\text{wait}}$)
- **Mathematical Definition:**
  $$z_{\text{wait}, t} = \sum_{i \in \mathcal{V}} \sum_{l \in \mathcal{L}_i^{in}} w_{l, t}$$
  where $w_{l, t}$ represents the accumulated waiting time of all vehicles on incoming lane $l$ currently halted (speed $< 0.1\text{ m/s}$).
- **SUMO TraCI Semantics:** Retrieved via `traci.lane.getWaitingTime(lane_id)`. Reset automatically when a vehicle accelerates beyond $0.1\text{ m/s}$.
- **Unit Conversion:** Directly measured in vehicle-seconds ($\text{veh}\cdot\text{s}$).
- **Normalization:**
  $$\bar{z}_{\text{wait}, t} = \frac{\text{clamp}(z_{\text{wait}, t}, 0.0, 600.0)}{600.0} \in [0.0, 1.0]$$

### 2.2 Total Queue Length ($z_{\text{queue}}$)
- **Mathematical Definition:**
  $$z_{\text{queue}, t} = \sum_{i \in \mathcal{V}} \sum_{l \in \mathcal{L}_i^{in}} q_{l, t}$$
  where $q_{l, t}$ is the integer count of vehicles on lane $l$ with speed $< 0.1\text{ m/s}$.
- **SUMO TraCI Semantics:** Retrieved via `traci.lane.getLastStepHaltingNumber(lane_id)`.
- **Unit Conversion:** Measured in discrete halting vehicles.
- **Normalization:**
  $$\bar{z}_{\text{queue}, t} = \frac{\text{clamp}(z_{\text{queue}, t}, 0.0, 20.0)}{20.0} \in [0.0, 1.0]$$

### 2.3 Integrated $\text{CO}_2$ Emissions ($z_{\text{co2}}$)
- **Mathematical Definition:**
  $$z_{\text{co2}, t} = \sum_{v \in \mathcal{V}_{\text{active}}} \dot{m}_{\text{co2}}(v, t) \cdot \Delta t$$
  where $\dot{m}_{\text{co2}}(v, t)$ is the instantaneous mass emission rate ($\text{mg/s}$) from the HBEFA3 emission model.
- **SUMO TraCI Semantics:** Retrieved via `traci.vehicle.getCO2Emission(vid)` aggregated per intersection.
- **Unit Conversion:** Integrated over step duration ($\Delta t = 1.0\text{ s}$) to yield mass in milligrams ($\text{mg}$). For episode summaries, converted to kilograms ($\text{kg} = 10^{-6}\text{ mg}$).
- **Normalization:**
  $$\bar{z}_{\text{co2}, t} = \frac{\text{clamp}(z_{\text{co2}, t}, 0.0, 120000.0)}{120000.0} \in [0.0, 1.0]$$

### 2.4 Step Throughput ($z_{\text{tp}}$)
- **Mathematical Definition:**
  $$z_{\text{tp}, t} = N_{\text{arrived}}(t)$$
- **SUMO TraCI Semantics:** Retrieved via `traci.simulation.getArrivedNumber()`, measuring vehicles that reached their destination edge in the past time step.
- **Unit Conversion:** Discrete completed vehicles per step. In publication tables, also reported as vehicles per hour ($\text{veh/h} = \frac{3600}{T} \sum_t z_{\text{tp}, t}$).
- **Throughput Inversion (Urgency Formulation):** Because reinforcement learning scalarization minimizes objective urgency:
  $$\bar{z}_{\text{tp}, t} = 1.0 - \frac{\text{clamp}(z_{\text{tp}, t}, 0.0, 20.0)}{20.0} \in [0.0, 1.0]$$
  When throughput is high ($20\text{ veh}$), urgency is $0.0$ (optimal). When throughput is $0$, urgency is $1.0$ (critical).

### 2.5 Corridor Delay Fairness Disparity ($z_{\text{fair}}$)
- **Mathematical Definition:**
  $$J(\vec{d}) = \frac{\left( \sum_{c=1}^C d_c \right)^2}{C \sum_{c=1}^C d_c^2} \in \left[ \frac{1}{C}, 1.0 \right]$$
  $$z_{\text{fair}, t} = 1.0 - J(\vec{d}_t) \in [0.0, 1.0 - 1/C] \subset [0.0, 1.0]$$
  where $\vec{d}_t = [d_{N}, d_{S}, d_{E}, d_{W}]$ is the mean waiting time across competing arterial corridors.
- **Direction:** Minimization (lower disparity represents more equitable service). $0.0$ represents absolute equity; values approaching $1.0$ indicate severe starvation of cross streets.
- **Normalization:** Because $z_{\text{fair}} \in [0.0, 1.0]$, it is directly normalized with upper bound $1.0$:
  $$\bar{z}_{\text{fair}, t} = \text{clamp}(z_{\text{fair}, t}, 0.0, 1.0)$$

---

## 3. Hierarchical Fairness Multi-Tier Distinction

To eliminate the conflation of network-aggregate variance with vehicle-level equity, the system strictly separates fairness into three hierarchical tiers:

### Tier 1: Approach / Corridor-Level Fairness (Primary Research Objective)
- Evaluates equity across competing arterial traffic flows (e.g., North-South surge vs. East-West cross streets).
- Primary Metric: Approach Delay Disparity ($1.0 - J(\vec{d}_{\text{approach}})$).
- Complementary Metrics: Worst-Approach Delay ($\max_c d_c$), Max-to-Min Disparity Ratio ($\frac{\max_c d_c}{\min_c d_c + \epsilon}$).

### Tier 2: Intersection-Level Fairness
- Evaluates equity across the 9 network junctions (identifying spatial bottlenecks).
- Primary Metric: Intersection Delay Disparity ($1.0 - J(\vec{d}_{\text{intersection}})$).
- Complementary Metrics: Worst-Intersection Delay, Variance of Intersection Waiting Times ($\sigma^2_{\text{tls}}$ in $s^2$).

### Tier 3: Vehicle-Level Fairness
- Evaluates equity across the microscopic population of individual active vehicles.
- Primary Metric: Microscopic Vehicle Delay Disparity ($1.0 - J(\vec{w}_{\text{vehicles}})$).
- Complementary Metrics: Maximum Individual Delay ($\max_v w_v$), 95th Percentile Delay ($P_{95}(w)$), Population Gini Coefficient ($G \in [0, 1]$).

---

## 4. Boundary & Edge Case Handling

1. **Zero-Delay / Empty Network:** When no vehicles are halting in the network ($\sum d_c = 0$), service is mathematically equitable:
   $$J(\vec{0}) = 1.0, \quad z_{\text{fair}} = 0.0$$
2. **Teleporting Vehicles:** `time_to_teleport` is set to `-1` in all SUMO configurations, preventing artificial disappearance of gridlocked vehicles.
3. **Single Vehicle Inflow:** When only one approach is active ($K=1$), disparity is defined as $0.0$ to prevent penalizing the system for external demand asymmetry.

