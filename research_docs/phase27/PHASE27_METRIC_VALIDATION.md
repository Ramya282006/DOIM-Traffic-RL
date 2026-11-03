# Phase 27: Multi-Objective Traffic Metric Validation & Analytical Verification
**Project:** DOIM-PPO (Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Document Status:** Complete Empirical and Mathematical Audit  

---

## 1. Overview and Problem Formulation

Real-time urban traffic control inherently involves competing socio-technical objectives. In DOIM-PPO, five canonical traffic objectives are tracked, normalized, and optimized:
1. **Total Delay ($z_1 \in [0, 1]$):** Cumulative vehicle delay in vehicle-seconds.
2. **Total Queue Length ($z_2 \in [0, 1]$):** Spatial congestion measured as halting vehicle count.
3. **CO2 Emissions ($z_3 \in [0, 1]$):** Microscopic HBEFA3 pollutant mass integral.
4. **Network Throughput ($z_4 \in [0, 1]$):** Completed vehicle trip completion rate.
5. **Traffic Equity / Fairness ($z_5 \in [0, 1]$):** Multi-tier delay disparity across network entities.

To eliminate scalarization bias and ensure Pareto-optimal policy exploration, each raw objective $r_k(t)$ is dynamically normalized into an urgency score $z_k(t) \in [0, 1]$.

---

## 2. Objective Mathematical Formulations & Analytical Benchmarks

### 2.1 Total Delay ($w_{\text{tot}}$)
- **Definition:** The instantaneous sum of accumulated waiting times across all active vehicles in the network:
  $$w_{\text{tot}}(t) = \sum_{v \in \mathcal{V}(t)} w_v(t)$$
- **Unit:** Vehicle-seconds ($\text{veh}\cdot\text{s}$).
- **Optimization Direction:** Minimization (lower is better).
- **Analytical Benchmark:**
  - Active vehicle waiting times: $[12.5, 3.5, 4.0, 0.0, 10.0]\text{ s}$.
  - Analytical sum: $12.5 + 3.5 + 4.0 + 0.0 + 10.0 = 30.0\text{ veh}\cdot\text{s}$.
  - Verified in `test_objective_reference_values.py`: error $< 10^{-6}$.

### 2.2 Total Queue Length ($q_{\text{tot}}$)
- **Definition:** Aggregate number of vehicles halting (speed $< 0.1\text{ m/s}$) on incoming approach lanes:
  $$q_{\text{tot}}(t) = \sum_{l \in \mathcal{L}_{\text{in}}} n_{\text{halt}, l}(t)$$
- **Unit:** Vehicles ($\text{veh}$).
- **Optimization Direction:** Minimization (lower is better).
- **Analytical Benchmark:**
  - Halting counts: $[3, 5, 2, 0, 8]$.
  - Analytical sum: $3 + 5 + 2 + 0 + 8 = 18\text{ veh}$.
  - Verified in `test_objective_reference_values.py`: error $< 10^{-6}$.

### 2.3 Carbon Dioxide Emissions ($e_{\text{CO}_2}$)
- **Definition:** Microscopic instantaneous emission integration using SUMO's HBEFA3 vehicle pollutant model:
  $$e_{\text{CO}_2}(t) = \left( \sum_{v \in \mathcal{V}(t)} \max(0, \dot{e}_v(t)) \right) \cdot \Delta t$$
- **Unit:** Milligrams ($\text{mg}$) converted to Grams ($\text{g}$) via $1.0\text{ g} = 1000.0\text{ mg}$.
- **Optimization Direction:** Minimization (lower is better).
- **Analytical Benchmark:**
  - Vehicle rates: $[100.0, 200.0, 300.0]\text{ mg/s}$, step $\Delta t = 1.0\text{ s}$.
  - Total mass emitted: $600.0\text{ mg} = 0.60\text{ g}$.
  - At $\Delta t = 2.5\text{ s}$: $1500.0\text{ mg} = 1.50\text{ g}$.
  - Physically impossible negative rates (regenerative artifacts) clamped to $0.0$.
  - Verified in `test_objective_reference_values.py`: error $< 10^{-6}$.

### 2.4 Throughput & Completion Rate ($R_{\text{hourly}}$)
- **Definition:** Cumulative vehicle arrivals completed within the simulation horizon, extrapolated to an hourly rate:
  $$R_{\text{hourly}}(t) = \frac{\sum_{\tau=1}^t N_{\text{arrived}}(\tau)}{t} \times 3600.0$$
- **Unit:** Vehicles per hour ($\text{veh/hr}$).
- **Optimization Direction:** Maximization (higher is better).
- **Analytical Benchmark:**
  - $20\text{ vehicles}$ completed in $180.0\text{ s}$:
  - Hourly rate: $\frac{20}{180} \times 3600.0 = 400.0\text{ veh/hr}$.
  - Verified in `test_objective_reference_values.py`: error $< 10^{-6}$.

---

## 3. Rigorous Three-Tier Equity Framework

Traffic fairness cannot be adequately captured by a single aggregate metric. DOIM-PPO implements a mathematically rigorous three-tier hierarchical equity framework:

```
┌─────────────────────────────────────────────────────────────┐
│                 THREE-TIER FAIRNESS SYSTEM                  │
├─────────────────────────────────────────────────────────────┤
│ Tier 1: Approach / Corridor Equity (4 arterial directions)  │
│ Tier 2: Junction / Intersection Equity (9 network junctions)│
│ Tier 3: Vehicle-Level Individual Equity (all active trips)  │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 Mathematical Metrics

1. **Jain's Fairness Index:**
   $$J(x) = \frac{\left( \sum_{i=1}^K x_i \right)^2}{K \sum_{i=1}^K x_i^2}$$
   - **Boundedness:** $J(x) \in [1/K, 1.0]$.
   - **Scale Invariance:** For any scalar $c > 0$, $J(c \cdot x) = J(x)$.
   - **Homogeneity:** $J([x, x, \dots, x]) = 1.0$.

2. **Delay Disparity (Primary Minimization Metric):**
   $$D(x) = 1.0 - J(x) \in \left[ 0, 1 - \frac{1}{K} \right]$$
   - $D = 0.0$: Perfect equity across corridors.
   - $D \to 1.0$: Severe corridor starvation.

3. **Gini Inequality Coefficient:**
   $$G(x) = \frac{2 \sum_{i=1}^n i \cdot x_{(i)} - (n + 1) \sum_{i=1}^n x_{(i)}}{n \sum_{i=1}^n x_{(i)}}$$
   - $x_{(i)}$ denotes sorted ascending delays.
   - **Boundedness:** $G(x) \in [0.0, 1.0]$.
   - **Scale Invariance:** $G(c \cdot x) = G(x)$ for all $c > 0$.

4. **Delay Variance ($\sigma^2$):**
   $$\sigma^2(x) = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2$$
   - Unit: $\text{s}^2$ (seconds squared).

5. **Worst-Approach Delay ($W_{\text{worst}}$):**
   $$W_{\text{worst}} = \max_{i \in \mathcal{K}} x_i$$
   - Unit: $\text{s}$ (seconds).

6. **Max-to-Min Disparity Ratio ($R_{\text{disp}}$):**
   $$R_{\text{disp}} = \frac{\max(x)}{\min(x) + \epsilon}$$

### 3.2 Hand-Calculated Closed-Form Benchmarks

| Metric | Input Vector $x$ | Formula Step | Closed-Form Value | Code Output | Abs Error | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Jain Index | $[15, 15, 15, 15]$ | $\frac{60^2}{4 \times 900} = \frac{3600}{3600}$ | **$1.000000$** | $1.000000$ | $0.0$ | **PASS** |
| Delay Disparity | $[15, 15, 15, 15]$ | $1.0 - 1.0$ | **$0.000000$** | $0.000000$ | $0.0$ | **PASS** |
| Jain Index | $[0, 0, 0, 10]$ | $\frac{10^2}{4 \times 100} = \frac{100}{400}$ | **$0.250000$** | $0.250000$ | $0.0$ | **PASS** |
| Delay Disparity | $[0, 0, 0, 10]$ | $1.0 - 0.25$ | **$0.750000$** | $0.750000$ | $0.0$ | **PASS** |
| Jain Index | $[10, 20, 30, 40]$ | $\frac{100^2}{4 \times 3000} = \frac{10000}{12000}$ | **$0.833333$** | $0.833333$ | $< 10^{-6}$ | **PASS** |
| Delay Disparity | $[10, 20, 30, 40]$ | $1.0 - 5/6 = 1/6$ | **$0.166667$** | $0.166667$ | $< 10^{-6}$ | **PASS** |
| Gini Coeff | $[10, 20, 30, 40]$ | $\frac{2(300) - 5(100)}{4(100)} = \frac{100}{400}$ | **$0.250000$** | $0.250000$ | $0.0$ | **PASS** |
| Worst Delay | $[10, 50, 20, 5]$ | $\max(10, 50, 20, 5)$ | **$50.000000$** | $50.000000$ | $0.0$ | **PASS** |
| Max/Min Ratio | $[10, 50, 20, 5]$ | $\frac{50.0}{5.0}$ | **$10.000000$** | $10.000000$ | $0.0$ | **PASS** |

---

## 4. Effect Size Metric-Direction Consistency

In statistical effect size reporting (e.g., Cliff's delta $\delta \in [-1, 1]$), the sign convention must unambiguously reflect algorithmic superiority:
- For **benefit metrics** (throughput, reward, speed): $x > y \implies \delta > 0$ (treatment superior).
- For **cost metrics** (delay, queue, CO2, fairness disparity): $x < y \implies \delta > 0$ (treatment superior because cost is lower).

In Phase 27, `StatisticalAnalyzer.cliffs_delta` was enhanced with `higher_is_better: bool = True` and an automated helper `compute_metric_effect_size(x, y, metric_name)`:

```python
cost_metrics = {
    "delay", "total_delay", "mean_delay", "queue", "queue_length",
    "mean_queue", "co2", "co2_emissions", "emissions", "waiting_time",
    "fairness_disparity", "gini",
}
higher_is_better = metric_name.lower().strip() not in cost_metrics
return StatisticalAnalyzer.cliffs_delta(x, y, higher_is_better=higher_is_better)
```

This guarantees that in all manuscript tables and statistical analyses, a **positive effect size consistently indicates that DOIM-PPO outperforms the comparative baseline**.

---

## 5. Conclusion

The DOIM-PPO multi-objective metric implementation is grounded in rigorous mathematical theory, adheres to closed-form analytical benchmarks, and enforces metric-direction consistency across statistical reporting.
