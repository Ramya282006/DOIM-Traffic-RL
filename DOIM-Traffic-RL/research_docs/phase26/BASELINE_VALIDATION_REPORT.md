# BASELINE VALIDATION REPORT — PHASE 26
**Project:** Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Date:** November 2026  
**Author:** Principal Research Scientist & Benchmark Lead

---

## 1. Executive Summary

To establish scientifically rigorous and peer-defensible conclusions for IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS), the proposed **Interaction-Aware Multi-Objective PPO (IA-MOPPO)** framework must be benchmarked against a complete spectrum of competitive baselines. These baselines span classical transportation engineering controllers, state-of-the-art heuristic traffic algorithms, standard deep reinforcement learning architectures, and systematic architectural ablations.

This report documents the implementation, theoretical formulation, operational standardization, and empirical validation of all seven comparative baselines.

---

## 2. Baseline Architecture and Formulation Taxonomy

```
+---------------------------------------------------------------------------------------------------+
|                                 CONTROLLER BENCHMARK TAXONOMY                                     |
+---------------------------------------------------------------------------------------------------+
| CLASSICAL TRANSPORTATION CONTROLLERS:                                                             |
|   1. Fixed-Time (Webster Equisaturation Cycle & Splits)                                           |
|   2. Actuated (NEMA TS 2 Dual-Ring Standard with Passage Timers & Gap-Out)                         |
|   3. Max-Pressure (Varaiya 2013 Queue-Differential Network Stabilization)                         |
+---------------------------------------------------------------------------------------------------+
| REINFORCEMENT LEARNING ARCHITECTURES & ABLATIONS:                                                 |
|   4. MLP-PPO (Standard Feedforward RL, Scalarized Objectives, No GCN/GRU)                         |
|   5. Fixed-Weight PPO [FW-PPO] (GCN-GRU Spatio-Temporal, Fixed Uniform Weights w = [0.2]^5)       |
|   6. State-Adaptive PPO [SA-PPO] (GCN-GRU Spatio-Temporal, State MLP Weights, No DOIM/SCOIL)       |
|   7. IA-MOPPO [Proposed] (GCN-GRU Spatio-Temporal, DOIM Manifold, SCOIL Interaction-Aware)        |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Detailed Baseline Specifications

### 3.1 Baseline 1: Fixed-Time Controller (`baselines/fixed_time/`)
- **Theory:** Grounded in Webster's equisaturation method (Webster, 1958). The cycle length $C_0$ is computed from critical lane flow ratios:
  $$C_0 = \frac{1.5 L + 5}{1 - Y}$$
  where $L$ is total lost time per cycle and $Y = \sum_{i} y_i$ is the sum of critical movement saturation ratios ($y_i = q_i / s_i$).
- **Green Split Allocation:** Effective green time $g_i$ is distributed proportionally to critical movement demand:
  $$g_i = (C_0 - L) \cdot \frac{y_i}{Y}$$
- **Role in Study:** Serves as the fundamental classical reference benchmark representing non-adaptive urban signal infrastructure.

### 3.2 Baseline 2: Actuated Controller (`baselines/actuated/`)
- **Theory:** Genuine NEMA TS 2 dual-ring standard with detector actuation, vehicle extensions, and passage timers.
- **Operational Logic:**
  - *Minimum Green ($g_{\min} = 10\,\text{s}$):* Displays green for at least $g_{\min}$ regardless of detector calls.
  - *Vehicle Extension ($t_{\text{ext}} = 3.0\,\text{s}$):* Each vehicle detection resets the passage timer to $t_{\text{ext}}$, extending the active green phase.
  - *Gap-Out Expiration:* If time headway between consecutive vehicles exceeds $t_{\text{ext}}$, the passage timer expires ("gaps out"), and right-of-way yields to conflicting approach calls.
  - *Maximum Green ($g_{\max} = 60\,\text{s}$):* Enforces phase termination if continuous vehicle presence prevents gap-out.
- **Audit Verification:** Replaced previous simplified threshold logic with true NEMA passage timers in Phase 16, resolving AUDIT-01.

### 3.3 Baseline 3: Max-Pressure Controller (`baselines/max_pressure/`)
- **Theory:** Based on the network stabilization formulation of Varaiya (2013). At each junction $j$, the controller computes the pressure $P(m)$ for each admissible phase movement $m$:
  $$P(m) = \sum_{l \in \mathcal{L}_{\text{in}}(m)} q_l - \sum_{k \in \mathcal{L}_{\text{out}}(m)} q_k$$
  where $q_l$ is the vehicle queue length on upstream incoming lane $l$, and $q_k$ is the queue length on downstream receiving lane $k$.
- **Decision Rule:** The controller activates the phase $m^*$ that maximizes aggregate network pressure:
  $$m^* = \arg\max_{m \in \mathcal{M}} P(m)$$
  subject to safety shield minimum green constraints.
- **Role in Study:** Strongest theoretical non-RL baseline; provably guarantees network queue stability under admissible demand capacities.

### 3.4 Baseline 4: MLP-PPO (`algorithms/mlp_ppo.py`)
- **Architecture:** Multi-Layer Perceptron (3 fully connected hidden layers: $256 \to 128 \to 64$, ReLU activations) processing flat network state vectors.
- **Ablation Hypothesis:** Evaluates the necessity of spatial graph convolutions (GCN) and temporal recurrent memory (GRU).
- **Optimization:** Standard clipped PPO with scalarized multi-objective reward.

### 3.5 Baseline 5: Fixed-Weight PPO (FW-PPO) (`algorithms/fw_moppo.py`)
- **Architecture:** Full spatial Kipf-Welling GCN and temporal GRU encoder, identical to the proposed model.
- **Weight Strategy:** Uniform fixed objective weighting:
  $$\mathbf{w}_{\text{fixed}} = [0.2, 0.2, 0.2, 0.2, 0.2]^T$$
- **Ablation Hypothesis:** Directly isolates the contribution of dynamic, state-conditioned objective weight adaptation against static equal weighting.

### 3.6 Baseline 6: State-Adaptive PPO (SA-PPO) (`algorithms/sa_moppo.py`)
- **Architecture:** Full GCN-GRU encoder with a standard state-conditioned MLP preference head generating weights $\mathbf{w}_t = \sigma(\mathbf{W}_w \mathbf{h}_t + \mathbf{b}_w)$.
- **Ablation Hypothesis:** Isolates the specific contribution of the Dynamic Objective Interaction Manifold $\mathcal{M}_{\text{int}}$ and the SCOIL cross-sensitivity operator. SA-PPO adapts weights to state but remains blind to pairwise objective couplings and dispersion uncertainty.

### 3.7 Proposed Method: Interaction-Aware MOPPO (IA-MOPPO) (`algorithms/ia_moppo.py`)
- **Architecture:** GCN-GRU spatio-temporal encoder, DOIM interaction manifold parameterizing $\mathcal{M}_{\text{int}}$, SCOIL directional sensitivity estimator with epistemic dispersion regularization, and starvation-free preference learning with entropy barriers.

---

## 4. Benchmark Standardization and Parity Protocol

To prevent experimental bias, all seven controllers adhere to strict standardization protocols:

| Benchmark Dimension | Standardization Rule | Implementation Verification |
|:---|:---|:---:|
| **Traffic Network** | Identical 3×3 grid SUMO network (`3x3_grid.net.xml`). | `test_baselines_parity.py` |
| **Demand Scenarios** | Evaluated on identical route flows and insertion rates. | `test_scenario_specific_statistics.py` |
| **Control Frequency** | Identical decision step $\Delta t = 5.0\,\text{s}$ across all controllers. | `test_baselines_parity.py` |
| **Observation Space** | Identical 8-dimensional per-node traffic state features. | `test_baselines_parity.py` |
| **Action Space** | Identical 4-phase signal configurations per junction. | `test_baselines_parity.py` |
| **Safety Shield** | Identical conflict matrix, $g_{\min} = 10\,\text{s}$, $g_{\max} = 60\,\text{s}$, $y=3\,\text{s}, r=1\,\text{s}$. | `test_safety_shield.py` |
| **Random Seeds** | Synchronized SUMO traffic seed sequences across all comparisons. | `test_deterministic_evaluation.py` |

---

## 5. Baseline Parity Verification Results

Automated parity tests in `tests/test_baselines_parity.py` confirm:
1. Every baseline instantiates successfully within the unified environment wrapper.
2. The safety shield intercepts and regulates phase changes identically across all classical, heuristic, and RL controllers.
3. Telemetry and metric logging (delay, queue, emissions, throughput, fairness) use identical physical sensor routines.
4. No controller receives privileged traffic information unavailable to the others.

The baseline suite is rigorous, fair, and fully aligned with IEEE T-ITS experimental standards.
