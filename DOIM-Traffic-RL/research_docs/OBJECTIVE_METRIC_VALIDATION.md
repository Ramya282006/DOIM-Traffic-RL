# Multi-Objective Metric and Fairness Validation
## DOIM-Traffic-RL | Rigorous Objective Definitions, Normalization Bounds, and Population Semantics

**Status:** Scientifically Formulated & Validated  
**Scope:** Traffic Objective Normalization, Inversion Mechanics, Hierarchical Fairness Formulation  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  

---

## 1. Multi-Objective State Representation

In multi-objective traffic signal control, combining heterogeneous physical quantities (seconds, vehicles, milligrams per second, counts, and dimensionless equity indices) into a coherent reinforcement learning vector requires rigorous normalization and direction alignment.

In DOIM-Traffic-RL, every objective $\zeta_k$ is mapped into a normalized urgency scalar $z_k \in [0.0, 1.0]$, where:
- $z_k = 0.0$: Ideal operational state (zero penalty / lowest urgency).
- $z_k = 1.0$: Critical operational breakdown (maximum penalty / highest urgency).

The scalarized reward is computed as:
$$R_t = -\sum_{k=1}^5 w_{k, t} z_{k, t}, \quad \mathbf{w}_t \in \Delta^4$$
ensuring that $R_t \in [-1.0, 0.0]$ is strictly maximized when all objective urgencies are minimized.

---

## 2. Comprehensive Objective Specification Matrix

| Index ($k$) | Objective Name | Physical Metric ($\zeta_k$) | Measurement Definition | Units | Optimization Direction | Calibration Range $[\zeta^{\min}, \zeta^{\max}]$ | Normalization Formula ($z_k$) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **$0$** | **Delay / Waiting Time** | Mean vehicle waiting time | Cumulative waiting time of vehicles with speed $< 0.1\,\text{m/s}$ | $\text{s}$ | Minimize ($\downarrow$) | $[0.0, 600.0]\,\text{s}$ | $z_0 = \text{clip}\left(\frac{\zeta_0 - 0}{600}, 0, 1\right)$ |
| **$1$** | **Queue Length** | Average halting queue | Mean number of halting vehicles per approach lane across all controlled junctions | $\text{veh/lane}$ | Minimize ($\downarrow$) | $[0.0, 20.0]\,\text{veh}$ | $z_1 = \text{clip}\left(\frac{\zeta_1 - 0}{20}, 0, 1\right)$ |
| **$2$** | **Carbon Emissions** | Total $\text{CO}_2$ emissions | Total instantaneous network $\text{CO}_2$ emissions via HBEFA3 microscopic fuel model | $\text{mg/s}$ | Minimize ($\downarrow$) | $[0.0, 120000.0]\,\text{mg/s}$ | $z_2 = \text{clip}\left(\frac{\zeta_2 - 0}{120000}, 0, 1\right)$ |
| **$3$** | **Network Throughput** | Completed vehicles | Number of vehicles successfully completing their trips during the step | $\text{veh/s}$ | **Maximize ($\uparrow$)** | $[0.0, 20.0]\,\text{veh}$ | **$z_3 = 1.0 - \text{clip}\left(\frac{\zeta_3 - 0}{20}, 0, 1\right)$** |
| **$4$** | **Fairness Disparity** | Delay disparity index | Deviation from perfect equity ($1.0 - \text{Jain's Index}$) across competing flows | Dimensionless | Minimize ($\downarrow$) | $[0.0, 1.0]$ | $z_4 = \text{clip}(\zeta_4, 0, 1)$ |

### 2.1 The Mathematical Necessity of Throughput Inversion

Unlike delay, queue, emissions, and disparity (where lower raw values are desirable), throughput is an **output maximization** goal. If raw throughput were directly penalized with a negative sign, higher throughput would paradoxically decrease reward.

By applying the linear inversion:
$$z_3 = 1.0 - \text{clip}\left(\frac{\zeta_3 - \zeta_3^{\min}}{\zeta_3^{\max} - \zeta_3^{\min}}, 0, 1\right)$$
- When throughput reaches network capacity ($\zeta_3 = 20\,\text{veh}$), urgency $z_3 = 0.0$ (zero penalty).
- When throughput drops to zero ($\zeta_3 = 0\,\text{veh}$), urgency $z_3 = 1.0$ (maximum penalty).
Thus, minimizing urgency $z_3$ strictly maximizes physical throughput.

---

## 3. Hierarchical Fairness Formulation

A frequent scientific defect in intelligent transportation literature is treating "fairness" as a monolithic scalar without specifying the statistical population over which equity is evaluated. In DOIM-Traffic-RL, we formalize and validate three distinct hierarchical levels:

### 3.1 Level 1: Vehicle-Level Delay Disparity ($1 - J_{\text{veh}}$)

Evaluated across individual active drivers in the network. For individual vehicle accumulated waiting times $\mathbf{d} = [d_1, \dots, d_{|\mathcal{V}_{\text{active}}|}]^\top$:
$$J_{\text{veh}}(\mathbf{d}) = \frac{\left(\sum_{v=1}^{|\mathcal{V}_{\text{active}}|} d_v\right)^2}{|\mathcal{V}_{\text{active}}| \sum_{v=1}^{|\mathcal{V}_{\text{active}}|} d_v^2} \in \left[\frac{1}{|\mathcal{V}_{\text{active}}|}, 1.0\right]$$
$$\text{Disparity}_{\text{veh}} = 1.0 - J_{\text{veh}}(\mathbf{d}) \in [0.0, 1.0]$$
- **Desirable State ($0.0$):** All drivers experience similar delays regardless of route.
- **Undesirable State ($1.0$):** Severe starvation, where arterial drivers experience 0s delay while minor cross-street drivers are trapped for minutes.

### 3.2 Level 2: Intersection-Level Delay Disparity ($1 - J_{\text{int}}$)

Evaluated across aggregate delays experienced at each of the $N_{\text{tls}} = 5$ controllable signalized intersections $\mathbf{W} = [W_1, \dots, W_{N_{\text{tls}}}]^\top$:
$$J_{\text{int}}(\mathbf{W}) = \frac{\left(\sum_{j=1}^{N_{\text{tls}}} W_j\right)^2}{N_{\text{tls}} \sum_{j=1}^{N_{\text{tls}}} W_j^2}$$
$$\text{Disparity}_{\text{int}} = 1.0 - J_{\text{int}}(\mathbf{W})$$

> [!NOTE]
> **Spatial Aggregation Masking Effect:**
> As demonstrated in `tests/test_fairness_and_objectives.py`, two intersections can each have identical average delays of $50\,\text{s}$ (yielding $\text{Disparity}_{\text{int}} = 0.0$), yet individual vehicle delays inside one intersection might vary wildly between $10\,\text{s}$ and $90\,\text{s}$. Reporting vehicle-level disparity alongside intersection disparity is therefore essential to prevent deceptive equity claims.

### 3.3 Level 3: Active Population vs. Completed-Trip Cohort Fairness

A subtle experimental pitfall occurs when measuring fairness exclusively on vehicles active in the network at timestamp $t$:
1. If arterial vehicles clear rapidly through green waves, they exit the network and are purged from `traci.vehicle.getIDList()`.
2. The remaining network population consists disproportionately of stopped or newly inserted vehicles.
3. If an agent starves a cross-street approach completely, those vehicles remain active with growing delay, while hundreds of satisfied arterial vehicles disappear from the active metric.

To resolve this population artifact:
- **Online MOMDP Reward:** Uses current active vehicle disparity $z_4 = 1 - J_{\text{veh}}$ to provide instantaneous gradient feedback.
- **Offline Post-Hoc Evaluation:** Records trip metrics for all completed vehicles via `traci.simulation.getArrivedIDList()`, computing full-trip cohort fairness over the entire simulation horizon.

---

## 4. Empirical Test Verification

The unit test suite `tests/test_fairness_and_objectives.py` validates:
1. Bounds compliance: All 5 normalized objectives satisfy $z_k \in [0.0, 1.0]$ under all tested extremes.
2. Inversion monotonicity: Increased throughput strictly decreases urgency $z_3$.
3. Jain index limits: $J = 1.0$ under uniform delay, $J = 1/K$ under single-driver starvation.
4. Reward sign alignment: Congested traffic states produce lower rewards than smooth traffic states ($R_{\text{good}} > R_{\text{bad}}$), with $R \le 0.0$ everywhere.
