# Phase 28: Generalization & Distribution Shift Validation Report
**Project:** DOIM-PPO (Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Lead Auditor:** Intelligent Transportation Systems Researcher & Generalization Auditor  
**Date:** November 3, 2026  

---

## 1. Scientific Generalization Framework

A critical weakness of deep reinforcement learning for traffic signal control is poor transferability under non-stationary traffic demand and topological variations. An overfitted policy may excel on nominal Poisson arrivals but fail catastrophically during traffic surges or lane-blocking incidents.

To evaluate genuine generalization, DOIM-PPO defines four rigorous tiers of experimental evaluation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FOUR-TIER GENERALIZATION MATRIX                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier A: In-Distribution (Same demand pattern & same random arrival seed)    │
│ Tier B: Held-Out Demand (Same demand pattern, independent arrival seed)    │
│ Tier C: Demand Distribution Shift (Cross-scenario zero-shot evaluation)    │
│ Tier D: Road Network Topology Shift (Unseen intersection geometry)          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Held-Out Demand Evaluation (Tier B)

In DOIM-PPO, every evaluation episode is conducted on held-out pseudo-random arrival seeds disjoint from the training seeds:
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{eval}} = \emptyset, \quad \text{where } S_{\text{eval}} = S_{\text{train}} + 1000$$

### Verified Implementation in Codebase:
- TraCI is initialized with `--seed <eval_seed>` during evaluation rollouts.
- In `tests/test_deterministic_evaluation.py`:
  - Verified that identical evaluation seeds produce bitwise identical vehicle departure sequences.
  - Verified that differing evaluation seeds produce distinct vehicle arrival times and turning ratios.
- **Audit Result:** Prevents memorization of individual vehicle arrival timestamps.

---

## 3. Demand Distribution Shift (Tier C)

The repository provides five pre-configured SUMO demand scenarios representing acute real-world distribution shifts:

| Shift Scenario | Configuration | Baseline In-Distribution | Shift Stress Factor |
| :--- | :--- | :--- | :--- |
| **Directional Surge** | `north_south_surge.sumocfg` | Balanced nominal (1000 veh/h) | $3.0\times$ volume on N-S axis; severe asymmetric queuing |
| **Rush Hour Congestion** | `rush_hour.sumocfg` | Balanced nominal (1000 veh/h) | $2.4\times$ grid-wide volume; high risk of gridlock |
| **Arterial Corridor** | `arterial_corridor.sumocfg` | Balanced nominal (1000 veh/h) | $2.6\times$ volume on E-W axis; cross-street starvation |
| **Incident Disruption** | `incident_disruption.sumocfg` | Balanced nominal (1000 veh/h) | Mid-simulation lane capacity closure on link B0B1 |

### Historical Empirical Shift Evidence (Directional Surge vs. Rush Hour):
Comparing the performance of `Fixed_Time` and `IA_MOPPO` across the two historical scenarios:

| Metric | Scenario: `north_south_surge` (Surge) | Scenario: `rush_hour` (Heavy Congestion) | Performance Delta Under Shift |
| :--- | :---: | :---: | :---: |
| **Fixed-Time Delay (s)** | $55.13 \pm 5.33$ | $182.00$ (Single seed s42) | $+230.1\%$ delay escalation |
| **IA-MOPPO Delay (s)** | $82.19 \pm 45.69$ | $397.00$ (Single seed s42) | $+383.0\%$ delay escalation |
| **Fixed-Time Queue (veh)** | $1.92 \pm 0.25$ | $9.00$ | $+368.8\%$ queue growth |
| **IA-MOPPO Queue (veh)** | $2.85 \pm 1.11$ | $12.00$ | $+321.1\%$ queue growth |

**Finding:** As traffic density escalates to saturation in `rush_hour`, network queue lengths triple across all controllers, demonstrating realistic non-linear delay propagation.

---

## 4. Road Network Topology Shift (Tier D)

### 4.1 Architectural Invariant Verification
DOIM-PPO's spatial encoder (`SpatialTrafficEncoder`) is built upon the Kipf & Welling Graph Convolutional Network. In Phase 27 (`tests/test_gcn_invariants.py`), we proved:
1. **Permutation Equivariance:**
   $$\text{GCN}(P X, \sigma(E)) = P \cdot \text{GCN}(X, E)$$
   Holds to machine precision ($\le 1.192 \times 10^{-7}$). Reordering node indexing does not alter network representations.
2. **Dynamic Topology Extraction:**
   `TrafficSafetyShield` and `SignalSafetyVerifier` parse junction connections dynamically from arbitrary SUMO `net.xml` files without hardcoded link numbers.

### 4.2 Honest Audit of Existing Empirical Evidence
- **Available Topologies in Repository:**
  1. `configs/network/grid3x3.net.xml`: 3x3 urban grid (9 intersections, 5 controllable TLS).
  2. `environment/network/intersection.net.xml`: Single isolated 4-approach intersection.
- **Audited Reality:**
  - Zero historical runs were executed on `intersection.net.xml` or larger metropolitan networks.
  - **Reviewer Caveat:** *Parsing arbitrary `net.xml` files proves software flexibility, but does NOT constitute empirical evidence of policy transferability to unseen road networks.*
- **Scope Boundary:** Zero-shot policy transfer across diverse physical topologies without retraining is marked as an explicit research limitation in `REMAINING_EVIDENCE_GAPS.md`.

