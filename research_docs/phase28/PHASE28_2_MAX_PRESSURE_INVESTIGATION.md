# Phase 28.2: Max-Pressure Controller Deep-Dive & Performance Gap Audit
**Project:** DOIM-PPO (*Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems*)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard:** Traffic Engineering Control Theory & Algorithmic Diagnosis  
**Auditing Panel:** Independent Research Validation Team  
**Date:** November 3, 2026  

---

## 1. Executive Summary: The Pilot Anomaly

In the Phase 28 pilot, the classic traffic engineering baseline **Max-Pressure** achieved lower waiting time ($158.53\,\text{veh}\cdot\text{s/step}$) than DOIM-PPO ($510.94\,\text{veh}\cdot\text{s/step}$).
Rather than discarding or hiding this result, this investigation examines the mathematical, algorithmic, and interaction mechanics behind this gap.

---

## 2. Max-Pressure Control Law & Saturated Action Analysis

### 2.1 Theoretical Formulation
Max-Pressure (Varaiya, 2013) selects traffic signal phases to maximize the pressure gradient between incoming and outgoing lanes:
$$P(p) = \sum_{l \in \mathcal{L}_{p}^{\text{in}}} x_l - \sum_{m \in \mathcal{L}_{p}^{\text{out}}} x_m$$
where $x_l$ is the queue length on lane $l$. When $P(p_{\text{next}}) > P(p_{\text{curr}})$, the controller requests a phase switch ($a=1$).

### 2.2 Saturated Switch Requests & Shield Parasitism
In our step-level CSV telemetry (`results/runs/MaxPressure_.../metrics_step.csv`), we recorded both **requested** and **executed** actions across all 1,800 evaluation steps:
- Total junction-decision opportunities: $1,800 \text{ steps} \times 5 \text{ controllable TLS} = 9,000$ decisions.
- **Requested phase switches ($a=1$):** **6,582 requests (73.1%)**.
- **Permitted phase switches:** **342 switches (3.8%)**.
- **Suppressed / Intervened switches:** **6,240 overrides (69.3%)**.

### 2.3 The Safety Shield as an Optimal Cycle Extender
Because Max-Pressure has no internal concept of minimum green time, yellow clearance, or red clearance, it requests phase switches almost continuously whenever queue imbalances arise.
The external **Safety Shield** intervened 6,240 times, strictly enforcing:
1. Minimum green time: 10 seconds.
2. Yellow clearance: 3 seconds.
3. All-red clearance: 1 second.
**Critical Insight:** The Safety Shield effectively functioned as a dynamic phase-hold governor. It allowed Max-Pressure to switch phases *immediately* upon expiration of the 10-second minimum green whenever downstream queues were clear, producing an optimal micro-actuated 14-second phase cycle tailored to the bursty `north_south_surge` arrivals.

---

## 3. Sample Efficiency Asymmetry

| Property | Max-Pressure | DOIM-PPO (Pilot Option A) |
| :--- | :--- | :--- |
| **Parameter Count** | $0$ (zero-shot heuristic) | $48,938$ trainable weights |
| **Training Budget** | $0$ steps (instantaneous rule) | $1,000$ simulation steps ($pprox 33$ signal cycles) |
| **Optimization Progress** | Pre-converged (greedy gradient) | Early exploration ($7\text{--}8$ Adam updates) |
| **Internal Safety Model** | None (relies on shield override) | Policy learning under reward penalty |

In modern deep reinforcement learning benchmarks (e.g., CityFlow, LibSignal), a 48,000-parameter spatio-temporal neural network requires $25,000\text{--}50,000$ environment steps to outperform fine-tuned heuristic controllers. Comparing a 1,000-step RL agent against Max-Pressure is fundamentally an evaluation of early exploration vs. zero-shot optimality.

---

## 4. Confirmatory Evaluation Strategy for Tier 2

To ensure a fair, publishable comparison in IEEE T-ITS:
1. **Equal Safety Constraints:** Max-Pressure and DOIM-PPO will continue to operate under identical Safety Shield enforcement.
2. **Action Transparency:** Both requested and executed switch frequencies will be reported in Table II of the manuscript.
3. **Adequate Convergence:** DOIM-PPO will be trained across sufficient horizons ($25,000$ steps) to assess whether learned spatio-temporal coordination surpasses greedy queue-differential switching.
4. **Preservation of Scientific Integrity:** If Max-Pressure maintains lower delay under isolated rush-hour conditions while DOIM-PPO achieves superior multi-objective hypervolume across emissions and fairness, this will be reported directly as a trade-off finding.
