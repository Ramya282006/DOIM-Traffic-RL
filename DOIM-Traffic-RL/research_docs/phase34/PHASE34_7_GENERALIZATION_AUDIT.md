# PHASE 34.7: GENERALIZATION & DISTRIBUTION SHIFT FORENSIC AUDIT

**Mission ID:** `DOIM-PPO-PHASE34-INDEPENDENT-FORENSIC-AUDIT`  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Date:** 2026-11-04  
**Audit Standard:** Strict Forensic Cross-Scenario Evaluation & Honest Accounting of Evidence Gaps

---

## 1. Executive Summary & Verification Status

Generalization across diverse road network topologies, demand patterns, and non-stationary disruptions is a hallmark of robust traffic signal control. The repository defines three test scenarios:
1. `grid3x3` (North-South Surge): 9-intersection mesh network under heavy directional flow.
2. `arterial_corridor`: 5-intersection linear coordinated corridor with dominant east-west progression.
3. `incident_disruption`: Network undergoing capacity reduction due to localized incident bottlenecks.

### Forensic Finding on Execution Status:
- **`grid3x3`:** 19 completed evaluation runs ($\ge 1,800$ steps) physically exist on disk (Fixed-Time $\times 5$, Actuated $\times 5$, Max-Pressure $\times 5$, and 4 RL algorithms for Seed 1042).
- **`arterial_corridor`:** 10 completed baseline evaluation runs physically exist (Fixed-Time $\times 5$, Actuated $\times 4$, Max-Pressure $\times 1$). **Zero RL evaluation runs exist on disk**.
- **`incident_disruption`:** **Zero completed evaluation runs exist on disk**. (A previously launched Fixed-Time run terminated after 80 steps).

### Scientific Consequence:
Prior claims that DOIM-PPO was empirically proven to outperform all baselines across all three scenarios are **unsupported by physical experimental artifacts**. RL cross-scenario transfer remains an open confirmatory gap that must be honestly acknowledged in the paper.

---

## 2. Cross-Network Topological Shift & Baseline Performance

| Scenario | Algorithm | $N$ | Mean $W_{\text{net}}$ (s) | Std $W_{\text{net}}$ (s) | Mean Queue (veh) | Mean Throughput (veh) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`grid3x3`** | Fixed-Time | 5 | 372.69 | $\pm 19.41$ | 6.05 | 642.4 |
| `grid3x3` | Actuated | 5 | 1,531.87 | $\pm 72.11$ | 20.90 | 626.0 |
| `grid3x3` | Max-Pressure | 5 | 181.88 | $\pm 45.73$ | 3.17 | 644.2 |
| `grid3x3` | DOIM-PPO | 1 | 510.94 | — | 7.26 | 641.0 |
| `grid3x3` | MLP-PPO | 1 | 721.11 | — | 9.93 | 638.0 |
| `grid3x3` | FW-PPO | 1 | 812.44 | — | 11.36 | 637.0 |
| `grid3x3` | SA-PPO | 1 | 949.39 | — | 12.90 | 637.0 |
| **`arterial`**| Fixed-Time | 5 | **1,951.06** | $\pm 306.83$ | **24.93** | **1,460.0** |
| `arterial` | Actuated | 4 | 5,699.66 | $\pm 437.05$ | 69.63 | 1,348.5 |
| `arterial` | Max-Pressure | 1 | 7,705.01 | — | 88.44 | 1,332.0 |
| `arterial` | DOIM-PPO | 0 | *MISSING* | — | *MISSING* | *MISSING* |
| `arterial` | Other RL | 0 | *MISSING* | — | *MISSING* | *MISSING* |
| **`incident`**| All Algos | 0 | *MISSING* | — | *MISSING* | *MISSING* |

---

## 3. Traffic Engineering Analysis: Arterial Green Wave Disruption

The empirical contrast between `grid3x3` and `arterial_corridor` illuminates a classic traffic engineering trade-off:

### 3.1 On Mesh Networks (`grid3x3`)
In a 2D grid network with balanced distributed turns, queue differentials dominate network performance. Greedy decentralized algorithms like Max-Pressure excel by aggressively switching green phases to clear local queue spikes, achieving the lowest network waiting time ($181.88\,\text{s}$).

### 3.2 On Linear Corridors (`arterial_corridor`)
On a high-volume linear corridor, overall network efficiency depends on maintaining **coordinated progression ("green waves")** along the arterial.
- **Fixed-Time Control:** By maintaining fixed offsets and cycle times, vehicles travel through consecutive intersections without stopping, resulting in $W_{\text{net}} = 1,951.06\,\text{s}$ and the highest throughput ($1,460\,\text{veh}$).
- **Actuated Control:** Local gap-out sensors on cross-streets repeatedly interrupt arterial green phases. This fragments progression platoons, increasing delay by **+192.1%** ($5,699.66\,\text{s}$).
- **Max-Pressure Control:** Because Max-Pressure greedily serves whichever approach has the highest queue differential, cross-street queues frequently usurp green priority from arterial platoons. The green wave is entirely destroyed, causing queue spillback and catastrophic network delay: **$7,705.01\,\text{s}$ (+294.9% increase)** and the lowest throughput ($1,332\,\text{veh}$).

> **Scientific Insight:** A fundamental limitation of decentralized queue-based controllers is their inability to maintain progression bandwidth on coordinated arterials without explicit inter-intersection communication or green-wave constraints.

---

## 4. Remediation Plan & Future Confirmatory Recommendations

1. **Transparent Reporting in IEEE T-ITS Submission:**  
   The paper must explicitly delimit empirical RL verification to `grid3x3`, highlighting that arterial corridor and incident disruption benchmarks represent theoretical generalization challenges rather than claimed solved benchmarks.
2. **Execution Priority for Gap Closure:**  
   To complete the cross-scenario matrix in future campaigns, priority should be given to executing DOIM-PPO and MLP-PPO on `arterial_corridor` (Seed 1042) to establish whether the multi-objective manifold architecture learns progression coordination without manual offsets.

---

## 5. Artifact Manifest

- Generalization dataset: `results/phase34/generalization_recalculation.json`
- Script: `DOIM-Traffic-RL/analysis/phase34_generalization_audit.py`

