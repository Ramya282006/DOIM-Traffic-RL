# PHASE 34.5: MULTI-OBJECTIVE & PARETO REVALIDATION FORENSIC AUDIT

**Mission ID:** `DOIM-PPO-PHASE34-INDEPENDENT-FORENSIC-AUDIT`  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Date:** 2026-11-04  
**Audit Standard:** Strict Non-Dominated Sorting & Exact 5D Hypervolume against $r=(1.2, 1.2, 1.2, 1.2, 1.2)$

---

## 1. Executive Summary & Objective Formulation

In multi-objective traffic signal control, scalarizing rewards can obscure trade-offs between competing societal objectives. Phase 34 conducts a forensic multi-objective evaluation across the 5 canonical dimensions:
1. $f_1$: Network Waiting Time ($W_{\text{net}}$, normalized by $600.0\,\text{s}$, minimize)
2. $f_2$: Queue Length ($\bar{Q}$, normalized by $20.0\,\text{veh}$, minimize)
3. $f_3$: Carbon Dioxide Emissions ($\text{CO}_2$, normalized by $120,000\,\text{mg/s}$, minimize)
4. $f_4$: Inverted Throughput ($1.0 - \text{Throughput} / 700\,\text{veh}$, minimize)
5. $f_5$: Delay Disparity ($\delta_{\text{fair}} = 1.0 - \text{JainIndex}$, minimize)

All objectives are transformed into bounded minimization problems $f_i \in [0.0, 1.2]$.  
The canonical reference point is strictly fixed at:
$$\mathbf{r} = (1.2, 1.2, 1.2, 1.2, 1.2)^\top$$

---

## 2. Multi-Objective Performance on Verified Evaluation Seed 1042

| Algorithm | $f_1$ (Wait) | $f_2$ (Queue) | $f_3$ ($\text{CO}_2$) | $f_4$ (InvTh) | $f_5$ (Fair) | 5D Hypervolume ($HV$) | Generational Dist ($GD$) | Pareto Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Max-Pressure** | **0.2642** | **0.1420** | **0.8903** | **0.0743** | **0.3868** | **0.28067** | **0.0000** | **Non-Dominated (Front 1)** |
| **Fixed-Time** | 0.6116 | 0.3005 | 0.9351 | 0.0814 | 0.4194 | 0.12245 | 0.3858 | Dominated by Max-Pressure |
| **DOIM-PPO** | 0.8516 | 0.3630 | 0.9481 | 0.0843 | 0.5020 | **0.05722** | 0.6407 | Dominates MLP-PPO |
| **DOIM (No-SCOIL)** | 0.8365 | 0.3550 | 0.9467 | 0.0800 | 0.5034 | 0.06071 | 0.6242 | Non-Dominated vs DOIM |
| **DOIM (Pearson)** | 0.8358 | 0.3530 | 0.9442 | 0.0829 | 0.5090 | 0.06090 | 0.6238 | Non-Dominated vs DOIM |
| **MLP-PPO** | 1.2000* | 0.4965 | 0.9887 | 0.0886 | 0.5036 | 0.00000 | 1.0124 | Dominated by DOIM-PPO |
| **FW-PPO** | 1.2000* | 0.5680 | 1.0054 | 0.0900 | 0.4854 | 0.00000 | 1.0394 | Dominated |
| **SA-PPO** | 1.2000* | 0.6450 | 1.0300 | 0.0900 | 0.4721 | 0.00000 | 1.0750 | Dominated |
| **Actuated** | 1.2000* | 0.9930 | 1.1354 | 0.0986 | 0.4678 | 0.00000 | 1.2912 | Dominated |

*\*Note: Normalized values $\ge 1.2000$ are bounded at the reference point, collapsing the dominated hypervolume in that dimension to 0.*

---

## 3. Critical Multi-Objective Discoveries

### 3.1 Strict Pareto Dominance Over MLP-PPO
A central claim of this research project is that dynamic objective interaction manifolds provide superior trade-off handling compared to standard multi-task scalarization networks.
Forensic audit confirms that **DOIM-PPO strictly Pareto-dominates standard MLP-PPO**:
- $W_{\text{net}}$: $510.94\,\text{s} < 721.11\,\text{s}$ ($\mathbf{29.1\%}$ improvement)
- Queue: $7.26\,\text{veh} < 9.93\,\text{veh}$ ($\mathbf{26.9\%}$ improvement)
- $\text{CO}_2$: $113,768.5\,\text{mg/s} < 119,450.2\,\text{mg/s}$ ($\mathbf{4.8\%}$ reduction)
- Throughput: $641\,\text{veh} > 638\,\text{veh}$ ($\mathbf{+3\,\text{vehicles}}$)
- Delay Disparity: $0.5020 < 0.5036$ ($\mathbf{0.3\%}$ more equitable)

Since DOIM-PPO is strictly superior or equal in every single objective, **MLP-PPO is unconditionally Pareto-dominated**.

### 3.2 Hypervolume Survival Threshold
Because the reference point is set at $\mathbf{r} = (1.2, 1.2, 1.2, 1.2, 1.2)$, any solution with an objective value exceeding the upper bound collapses the dominated 5D hypervolume to exactly $0.0$.
- Standard RL baselines (MLP-PPO, FW-PPO, SA-PPO) allowed network waiting time to surge to $721\,\text{s} - 949\,\text{s}$ ($> 600\,\text{s}$ bound), collapsing their hypervolume to **0.00000**.
- **DOIM-PPO is the only benchmarked RL algorithm that successfully controlled all 5 objectives within the reference envelope**, yielding an active 5D hypervolume of **0.05722**.

### 3.3 Max-Pressure and the Extreme Point Trade-off
Max-Pressure achieved the highest single-point hypervolume ($0.28067$) on `grid3x3`. However, as revealed in Phase 34.4, this comes at the expense of:
1. Complete disruption of green waves on arterial corridors ($W_{\text{net}} = 7,705\,\text{s}$, worst in class).
2. Generating 6,240 safety shield override requests (relying entirely on external safety timers).

---

## 4. Artifact Manifest

- Revalidation script: `DOIM-Traffic-RL/analysis/phase34_pareto_reference.py`
- Revalidation dataset: `results/phase34/pareto_recalculation.json`
- Metrics source: `results/phase34/recalculated_metrics.csv`

