# PHASE 36: FROZEN CONFIRMATORY EXPERIMENTAL PROTOCOL

**Mission ID:** `DOIM-PPO-PHASE36-FROZEN-PROTOCOL`  
**Project:** Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems  
**Canonical Algorithm Identity:** DOIM-PPO (Internal implementation: `IA_MOPPO`)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Status:** FROZEN & RATIFIED FOR EXECUTION  

---

## 1. Protocol Objective & Governance

The objective of Phase 36 is to execute the confirmatory benchmark matrix under strict, pre-registered, and frozen experimental conditions. This protocol eliminates degrees of researcher freedom, ensures bitwise reproducibility, enforces deterministic evaluation seeds, and mandates absolute transparency regarding completed versus unexecuted experimental cells.

### Governance Principles:
1. **Zero Fabrication:** Points are awarded only for runs with complete physical records on disk (`manifest.json`, `summary.json`, `metrics_step.csv`).
2. **Deterministic Evaluation:** All stochastic traffic generation in SUMO is controlled by fixed random number generator seeds.
3. **Safety Compliance:** The deterministic safety shield is strictly active across all RL agents, enforcing minimum green ($10\,\text{s}$) and yellow clearance ($4\,\text{s}$) intervals.
4. **Canonical Identity:** The proposed method is exclusively designated **DOIM-PPO** in all publications and reports.

---

## 2. Experimental Factorial Design

The full confirmatory matrix comprises a $7 \times 3 \times 5$ factorial design ($105$ cells):

### 2.1 Evaluated Algorithms ($K = 7$)
- **Transportation Engineering Baselines (3):**
  1. `Fixed-Time`: Webster-calibrated pre-timed signal plan with fixed cycle and split.
  2. `Actuated`: Gap-out vehicle-actuated controller with minimum/maximum green bounds.
  3. `Max-Pressure`: Decentralized queue-differential greedy pressure controller.
- **Deep Reinforcement Learning Baselines & Proposed Method (4):**
  4. `FW-PPO`: Fixed-Weight PPO with static uniform objective scalarization.
  5. `SA-PPO`: State-Adaptive PPO with state-conditioned linear objective weights.
  6. `MLP-PPO`: Parameter-matched un-modeled deep multi-objective PPO.
  7. `DOIM-PPO` (Proposed): Dynamic Objective Interaction Manifold with SCOIL second-order curvature modeling.

### 2.2 Traffic Network Scenarios ($M = 3$)
1. **`grid3x3` (Directional Shift / North-South Surge):**
   - 9 signalized intersections in a $3 \times 3$ grid.
   - Non-stationary demand with a peak surge along the north-south arterial axis.
   - Config: `scenarios/directional_shift/north_south_surge.sumocfg`.
2. **`arterial_corridor` (Coordination & Progression):**
   - 5 coordinated consecutive intersections along a major arterial corridor.
   - Evaluates platoon progression, green-wave preservation, and cross-street service.
   - Config: `scenarios/arterial_corridor/arterial_corridor.sumocfg`.
3. **`incident_disruption` (Safety & Resilience):**
   - Urban network subjected to dynamic lane blockages, vehicle breakdowns, and spillback.
   - Evaluates adaptive rerouting response, bottleneck relief, and robustness.
   - Config: `scenarios/incident_disruption/incident_disruption.sumocfg`.

### 2.3 Evaluation Seeds ($N = 5$)
Five frozen pseudorandom seeds:
$$\mathcal{S}_{\text{eval}} = \{1042, 1043, 1044, 1045, 1046\}$$
For RL agents trained with training seed $s_{\text{train}} \in \{42, 43, 44, 45, 46\}$, the evaluation seed is matched via $s_{\text{eval}} = s_{\text{train}} + 1000$.

### 2.4 Simulation Duration & Evaluation Horizon
- Episode Horizon: $T = 1800$ simulation seconds ($30$ minutes of simulated traffic).
- Decision Interval: $\Delta t = 1.0$ second (or 5.0 seconds macro-step where phase switching occurs).

---

## 3. Five-Dimensional Performance Vector

At every evaluation step, the environment logs the 5 core objective components:
1. $f_1$: Network Waiting Time ($W_{\text{net}}$, seconds)
2. $f_2$: Mean Halting Queue Length ($Q_{\text{net}}$, vehicles)
3. $f_3$: Network Intersection Pressure ($\Pi_{\text{net}}$, vehicles)
4. $f_4$: CO2 Emission Rate ($E_{\text{rate}}$, mg/s) and Cumulative Release ($E_{\text{cum}}$, kg)
5. $f_5$: Delay Disparity ($\delta_{\text{fair}} \in [0, 1]$) and Jain's Index ($J = 1 - \delta_{\text{fair}}$)

Additionally, safety metrics are recorded:
- Safety Shield Interventions ($I_{\text{shield}}$, total count and per-step rate)
- Collision / Red-Light Violation Count (Target = 0)
- Network Cumulative Completed Throughput ($Y_{\text{cum}}$, vehicles)

---

## 4. Execution Sequence & Prioritization

Given local CPU compute constraints, experiments are executed in prioritized tiers:
1. **Tier 1 (Multi-Seed RL Benchmarking on `grid3x3`):**
   Evaluate trained `DOIM-PPO`, `MLP-PPO`, `FW-PPO`, and `SA-PPO` models across seeds 1043, 1044, 1045, and 1046.
2. **Tier 2 (Generalization Matrix on `arterial_corridor`):**
   Execute multi-seed baselines (Fixed-Time, Actuated, Max-Pressure) and evaluate RL models across seeds 1042–1046.
3. **Tier 3 (Robustness Matrix on `incident_disruption`):**
   Execute baselines and RL models under incident disruption.
4. **Honest Accounting:** Any cells not completed within available execution rounds are documented transparently as unexecuted.

---

## 5. Verification Sign-Off

- **Random Seeds Frozen:** Yes ($\{1042, 1043, 1044, 1045, 1046\}$)
- **Evaluation Duration Frozen:** Yes ($1,800\,\text{s}$)
- **Deterministic Shield Frozen:** Yes (Active on all runs)
- **Zero Fabrication Commitment:** Formally ratified

