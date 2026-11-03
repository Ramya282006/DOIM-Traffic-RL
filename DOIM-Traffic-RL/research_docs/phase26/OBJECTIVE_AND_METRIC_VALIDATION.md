# OBJECTIVE AND METRIC VALIDATION REPORT — PHASE 26
**Project:** Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Date:** November 2026  
**Author:** Traffic Systems Researcher & Experimental Statistician

---

## 1. Executive Summary

In multi-objective urban traffic signal control, controllers must balance competing operational, environmental, and equity objectives. This report formally defines, validates, and verifies the five traffic performance objectives used throughout DOIM-Traffic-RL:
1. Average Vehicle Delay ($O_1$, seconds)
2. Cumulative Queue Length ($O_2$, vehicles)
3. Carbon Dioxide Emissions ($O_3$, g/s, SUMO HBEFA3 model)
4. Network Throughput ($O_4$, discharged vehicles)
5. Multi-Tier Traffic Equity and Fairness ($O_5$, three distinct mathematical tiers)

We document physical units, SUMO TraCI extraction APIs, sign conventions, normalization procedures preventing scale dominance, and mathematical differentiation among the fairness tiers.

---

## 2. Objective Formulations and Physical Units

### 2.1 Objective 1: Average Vehicle Delay ($O_1$)
- **Physical Meaning:** Time lost by vehicles traveling below free-flow speed due to signal deceleration, red-light queues, and intersection conflicts.
- **SUMO TraCI API:** `traci.vehicle.getWaitingTime(veh_id)` and `traci.lane.getWaitingTime(lane_id)`.
- **Definition:** SUMO defines waiting time as the consecutive duration (in seconds) during which a vehicle's speed remains strictly below $0.1\,\text{m/s}$.
- **Mathematical Formula:**
  $$O_{1, t} = \frac{1}{N_{\text{active}, t}} \sum_{i=1}^{N_{\text{active}, t}} w_{i, t}^{\text{wait}}$$
  where $N_{\text{active}, t}$ is the number of active vehicles in the network at simulation second $t$, and $w_{i, t}^{\text{wait}}$ is the waiting time of vehicle $i$.
- **Unit:** Seconds ($\text{s}$). Cost objective (negated in reward: $r_{1, t} = -O_{1, t}$).

### 2.2 Objective 2: Queue Length ($O_2$)
- **Physical Meaning:** Spatial accumulation of standing vehicles waiting for right-of-way on incoming intersection approaches.
- **SUMO TraCI API:** `traci.lane.getLastStepHaltingNumber(lane_id)`.
- **Definition:** Count of vehicles on approach lanes with speed $< 0.1\,\text{m/s}$.
- **Mathematical Formula:**
  $$O_{2, t} = \sum_{l \in \mathcal{L}_{\text{in}}} n_{l, t}^{\text{halt}}$$
  where $\mathcal{L}_{\text{in}}$ is the set of all incoming approach lanes across the 9 network junctions.
- **Unit:** Vehicles ($\text{veh}$). Cost objective (negated in reward: $r_{2, t} = -O_{2, t}$).

### 2.3 Objective 3: CO₂ Emissions ($O_3$)
- **Physical Meaning:** Greenhouse gas emissions generated through vehicle idling, deceleration, and stop-and-go acceleration cycles.
- **SUMO TraCI API:** `traci.vehicle.getCO2Emission(veh_id)` / `traci.lane.getCO2Emission(lane_id)`.
- **Model:** SUMO HBEFA3 (Handbook Emission Factors for Road Transport, version 3) micro-emission model, accounting for instantaneous speed, acceleration, and vehicle fleet category.
- **Unit Standardization:** SUMO reports emissions natively in milligrams per second ($\text{mg/s}$). To prevent numerical instability and scale explosion, the environment standardizes $O_{3, t}$ to grams per second ($\text{g/s}$):
  $$O_{3, t} = \frac{1}{1000} \sum_{i=1}^{N_{\text{active}, t}} e_{i, t}^{\text{CO2}}$$
- **Unit:** Grams per second ($\text{g/s}$). Cost objective (negated in reward: $r_{3, t} = -O_{3, t}$).

### 2.4 Objective 4: Network Throughput ($O_4$)
- **Physical Meaning:** Productivity of the road network, measured as the number of vehicles successfully traversing and departing the network boundaries.
- **SUMO TraCI API:** `traci.simulation.getArrivedNumber()` aggregated over decision interval $\Delta t = 5\,\text{s}$.
- **Mathematical Formula:**
  $$O_{4, t} = N_{\text{arrived}, [t, t+\Delta t]}$$
  Expressed as hourly rate: $\text{Throughput} = \frac{3600}{\Delta t} O_{4, t} \;\text{veh/h}$.
- **Unit:** Vehicles per step ($\text{veh}/\Delta t$) or Vehicles per hour ($\text{veh/h}$). Productivity objective (positive reward: $r_{4, t} = +O_{4, t}$).

---

## 3. Multi-Tier Traffic Equity and Fairness Formulations ($O_5$)

A recurring critique in intelligent transportation reviews is treating fairness as a vague, scalar heuristic. To achieve genuine scientific rigor, DOIM-Traffic-RL defines and validates three mathematically distinct tiers of traffic equity:

```
               +------------------------------------------------+
               |             MULTI-TIER FAIRNESS                |
               +------------------------------------------------+
                                       |
        +------------------------------+------------------------------+
        |                              |                              |
        v                              v                              v
+---------------+              +---------------+              +---------------+
|    TIER 1     |              |    TIER 2     |              |    TIER 3     |
| Approach-Level|              | Junction-Level|              | Vehicle-Level |
|  Disparity    |              | Delay Variance|              |Gini Coeff (G) |
|   (F_app)     |              |   (F_junc)    |              |   (F_gini)    |
+---------------+              +---------------+              +---------------+
| Max-Min lane  |              | Spatial       |              | Individual    |
| delay delta   |              | variance      |              | driver delay  |
| across phases |              | across 9 nodes|              | inequality    |
+---------------+              +---------------+              +---------------+
```

### 3.1 Tier 1: Approach-Level Disparity ($F_{\text{approach}}$)
- **Scope:** Measures immediate equity between conflicting movements at a single intersection (e.g., major arterial green time vs. minor cross-street starvation).
- **Mathematical Formula:**
  $$F_{\text{approach}, t} = \max_{l \in \mathcal{L}_{\text{in}}} \bar{w}_{l, t} - \min_{l \in \mathcal{L}_{\text{in}}} \bar{w}_{l, t}$$
  where $\bar{w}_{l, t}$ is the average waiting time on approach lane $l$.
- **Behavioral Target:** Prevents minor cross-streets from experiencing extreme wait times while the arterial flows freely.
- **Range:** $[0, \infty)$, where 0 indicates perfectly balanced approach wait times.

### 3.2 Tier 2: Junction-Level Delay Variance ($F_{\text{junction}}$)
- **Scope:** Measures spatial equity across the entire 3×3 network grid, preventing boundary intersections from being sacrificed to clear interior bottleneck nodes.
- **Mathematical Formula:**
  $$F_{\text{junction}, t} = \frac{1}{|J|} \sum_{j \in J} \left( D_{j, t} - \bar{D}_t \right)^2$$
  where $D_{j, t}$ is the average vehicle delay at junction $j \in J$ ($|J| = 9$), and $\bar{D}_t = \frac{1}{|J|}\sum_{j} D_{j, t}$.
- **Behavioral Target:** Enforces balanced network-wide progression rather than localized over-optimization.
- **Range:** $[0, \infty)$, where 0 indicates uniform average delay across all 9 intersections.

### 3.3 Tier 3: Vehicle-Level Gini Coefficient ($G$)
- **Scope:** Evaluates individual user fairness across all traveling drivers, matching econometric inequality metrics.
- **Mathematical Formula:**
  $$G_t = \frac{\sum_{i=1}^{N_t} \sum_{k=1}^{N_t} |w_{i, t} - w_{k, t}|}{2 N_t^2 \bar{w}_t}$$
  where $w_{i, t}$ is the cumulative delay of vehicle $i$, and $\bar{w}_t$ is the population mean delay.
- **Behavioral Target:** Guarantees that individual trips have low inequality; avoids situations where 90% of drivers experience zero delay while 10% experience catastrophic delay.
- **Range:** $[0, 1]$, where 0 indicates perfect equality (all drivers experience identical delay) and 1 indicates extreme inequality (one driver absorbs all network delay).

### 3.4 Mathematical Independence Proof and Verification
The three tiers are mathematically non-collinear:
- A network can have zero junction-level variance ($F_{\text{junction}} = 0$) while having extreme approach disparity ($F_{\text{approach}} \gg 0$) if every junction equally starves its cross-street.
- A network can have zero approach disparity at each junction while having high vehicle-level Gini ($G \to 1$) if travel times across routes differ substantially.
- Unit test `tests/test_fairness_definitions.py` programmatically proves this mathematical independence using synthetic counterexample distributions.

---

## 4. Multi-Objective Normalization and Scale Balance

Because raw traffic metrics vary by orders of magnitude (e.g., CO₂ emissions $\sim 10^3\,\text{g/s}$, delay $\sim 10^1\,\text{s}$, throughput $\sim 10^0\,\text{veh/step}$), unnormalized scalarization causes gradient explosion in high-magnitude objectives and complete starvation of lower-magnitude ones.

### 4.1 Running Moment Normalization
DOIM-Traffic-RL employs Welford's algorithm to maintain running estimates of the mean $\mu_k$ and standard deviation $\sigma_k$ for each objective $k \in \{1, \dots, 5\}$ across training rollouts:

$$\tilde{O}_{k, t} = \frac{O_{k, t} - \mu_{k, t}}{\sigma_{k, t} + \epsilon_{\text{norm}}}$$

### 4.2 Standardized Vector Reward
The normalized scalar reward passed to the value and policy heads is:
$$r_{k, t} = \begin{cases} -\tilde{O}_{k, t}, & k \in \{1, 2, 3, 5\} \text{ (Delay, Queue, CO₂, Fairness costs)} \\ +\tilde{O}_{k, t}, & k = 4 \text{ (Throughput productivity)} \end{cases}$$

This ensures:
$$\mathbb{E}[\tilde{O}_{k, t}] \approx 0, \quad \text{Var}(\tilde{O}_{k, t}) \approx 1 \quad \forall k \in \{1, \dots, 5\}$$
guaranteeing balanced gradient propagation across all objective critics.

---

## 5. Verification Test Matrix

| Objective / Metric | Verification Test | Assertion Checked | Outcome |
|:---|:---|:---|:---:|
| **$O_1$ Delay** | `test_metric_correctness.py` | Non-negative, zero on empty network, monotonic with stopped vehicles. | **PASSED** |
| **$O_2$ Queue** | `test_metric_correctness.py` | Matches lane-by-lane halting count; non-negative. | **PASSED** |
| **$O_3$ CO₂ Emissions** | `test_metric_correctness.py` | Converted from mg to grams; matches HBEFA3 instantaneous rates. | **PASSED** |
| **$O_4$ Throughput** | `test_metric_correctness.py` | Tracks departed vehicles monotonically; matches cumulative count. | **PASSED** |
| **$O_5$ Fairness Tiers** | `test_fairness_definitions.py` | Independence of $F_{\text{app}}$, $F_{\text{junc}}$, and Gini; proper bounded ranges. | **PASSED** |
| **Reward Normalization** | `test_ia_moppo_integration.py` | Running moments update online; rewards remain numerically bounded in $[-5, 5]$. | **PASSED** |

All traffic objectives and performance metrics are rigorously validated and mathematically verified.
