# EXPERIMENTAL PROTOCOL AND BENCHMARK DESIGN — PHASE 26
**Project:** Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Date:** November 2026  
**Author:** Experimental Lead & Software Architect

---

## 1. Executive Summary

Empirical credibility in deep reinforcement learning for traffic signal control requires an uncompromising experimental protocol. Many published studies in the literature suffer from single-seed evaluations, unverified baselines, evaluation on memorized training routes, or opaque computational budgets.

This document establishes the official **Confirmatory Experimental Protocol** for DOIM-Traffic-RL. It defines the complete 350-run experimental matrix, multi-seed training regimes, held-out evaluation seeds, realistic computational budgeting across three execution tiers, and fail-closed data provenance rules.

---

## 2. The Confirmatory Experimental Matrix

The experimental matrix is designed to provide full factorial coverage across algorithms, demand distributions, and stochastic initializations:

$$\mathcal{M}_{\text{exp}} = \mathcal{A}_{\text{controllers}} \times \mathcal{S}_{\text{scenarios}} \times \mathcal{R}_{\text{seeds}}$$

```
+---------------------------------------------------------------------------------------------------+
|                                  THE 350-RUN FACTORIAL MATRIX                                     |
+---------------------------------------------------------------------------------------------------+
| CONTROLLERS (7):                                                                                  |
|   1. Fixed-Time (Webster Equisaturation)                                                          |
|   2. Actuated (NEMA TS 2 Dual-Ring Standard)                                                      |
|   3. Max-Pressure (Varaiya 2013 Stabilization)                                                    |
|   4. MLP-PPO (Feedforward Baseline, No GCN/GRU)                                                   |
|   5. Fixed-Weight PPO [FW-PPO] (GCN-GRU, Uniform w = [0.2]^5)                                     |
|   6. State-Adaptive PPO [SA-PPO] (GCN-GRU, State MLP Preference Head)                             |
|   7. IA-MOPPO [Proposed] (GCN-GRU, DOIM Manifold, SCOIL Interaction-Aware)                        |
+---------------------------------------------------------------------------------------------------+
| DEMAND SCENARIOS (5):                                                                             |
|   1. Low Demand (800 veh/h, under-saturated free-flow conditions)                                 |
|   2. Balanced Off-Peak (1,600 veh/h, normal urban operating conditions)                           |
|   3. Rush Hour (3,000 veh/h, oversaturated network gridlock conditions)                          |
|   4. Arterial Corridor (2,600 veh/h East-West progression vs. 700 veh/h cross-streets)           |
|   5. Incident Disruption (Surge volume + localized link blockage on B0B1)                         |
+---------------------------------------------------------------------------------------------------+
| TRAINING SEEDS (10):                                                                              |
|   {42, 123, 456, 789, 999, 101, 202, 303, 404, 505}                                              |
+---------------------------------------------------------------------------------------------------+
| TOTAL CONFIGURATIONS: 7 Algorithms x 5 Scenarios x 10 Seeds = 350 Experimental Runs               |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Statistical Unit of Analysis and Generalization Protocol

### 3.1 Primary Statistical Unit
In accordance with best statistical practices in reinforcement learning (Henderson et al., 2018; Agarwal et al., 2021):
- The **primary unit of statistical replication** is an independently trained model run initiated with a distinct random seed ($N \ge 5$, target $N = 10$).
- Aggregating steps within a single training run as independent samples is strictly prohibited, as it artificially deflates standard errors through autocorrelation.

### 3.2 Strict Separation of Evaluation Trajectories
To verify true policy generalization rather than route memorization:
- Training occurs under vehicle trips generated with training seeds $\mathcal{R}_{\text{train}} = \{42, 123, \dots, 505\}$.
- Periodic evaluation and final benchmarking are executed using disjoint held-out evaluation seeds $\mathcal{R}_{\text{eval}} = \{1042, 1123, \dots, 1505\}$.
- Vehicle departure times, origins, and turning movements in evaluation runs are entirely unseen by the policy during training.

---

## 4. Computational Budget and Execution Tier Architecture

### 4.1 Wall-Clock Computational Profiling
Empirical hardware profiling on the test environment established:
- **Neural Network Forward Latency:** $2.19\,\text{ms/step}$ ($457\,\text{steps/s}$).
- **SUMO Micro-Simulation Latency:** $39.88\,\text{ms/step}$ ($25.1\,\text{steps/s}$).
- **Total Decision Step Latency:** $42.07\,\text{ms/step}$ ($\approx 23.8\,\text{steps/s}$).

### 4.2 Computational Budget Analysis for Full Matrix
- Per-run duration at asymptotic budget ($50,000$ steps):
  $$t_{\text{run}} = 50,000 \times 0.04207\,\text{s} = 2,103.5\,\text{s} \approx 35.06\,\text{minutes}$$
- Full 350-run matrix requirement:
  $$T_{\text{total}} = 350 \times 35.06\,\text{min} \approx 12,271\,\text{minutes} \approx 204.5\,\text{hours} \approx 8.52\,\text{days of continuous CPU compute}$$

### 4.3 Three-Tier Execution Strategy
To maintain scientific integrity without exceeding local compute resources, DOIM-Traffic-RL organizes execution into three explicit tiers:

```
+---------------------------------------------------------------------------------------------------+
|                                     THREE-TIER EXECUTION PROTOCOL                                 |
+---------------------------------------------------------------------------------------------------+
| TIER 1: Smoke & Algorithmic Correctness                                                           |
|   - Scope: 7 algorithms x 1 scenario x 1 seed @ 500 steps.                                        |
|   - Purpose: Verifies gradient flow, shield intercepts, and TraCI stability in seconds.           |
|   - Status: COMPLETED & 100% VERIFIED via automated pytest suite.                                  |
+---------------------------------------------------------------------------------------------------+
| TIER 2: Core Confirmatory Matrix                                                                  |
|   - Scope: 7 algorithms x 3 core scenarios x 5 seeds @ 10,000 steps (105 runs).                   |
|   - Purpose: Provides formal non-parametric significance testing and Pareto dominance proof.      |
|   - Compute Time: ~61 hours total compute.                                                        |
+---------------------------------------------------------------------------------------------------+
| TIER 3: Asymptotic Benchmark Standard                                                             |
|   - Scope: 7 algorithms x 5 scenarios x 10 seeds @ 50,000 steps (350 runs).                       |
|   - Purpose: Full publication standard for massive parallel cluster execution.                    |
|   - Compute Time: ~204.5 hours total compute.                                                     |
+---------------------------------------------------------------------------------------------------+
```

---

## 5. Fail-Closed Data Governance and Manifest Schema

To prevent accidental mixing of diagnostic runs with asymptotic confirmatory runs, the statistical analysis engine (`analysis/statistical_analyzer.py`) enforces strict fail-closed data governance:

### 5.1 Canonical 12-Field Manifest Schema
Every valid experimental run must produce a manifest containing:
1. `experiment_id`: Unique UUID for the experiment campaign.
2. `run_id`: Unique identifier for the specific run.
3. `algorithm`: Exact algorithm name from the 7 certified controllers.
4. `scenario`: Standard scenario identifier.
5. `train_seed`: Integer seed used for policy weight initialization.
6. `eval_seed`: Integer seed used for evaluation trip generation.
7. `step_count`: Total simulation steps completed.
8. `duration_seconds`: Wall-clock execution time.
9. `timestamp`: ISO-8601 creation timestamp.
10. `git_commit`: Exact Git commit SHA under which the run was executed.
11. `metrics_summary`: Complete dictionary of all 5 objective metrics.
12. `status`: Execution state (`COMPLETED`, `FAILED`, `ABORTED`).

### 5.2 Fail-Closed Admission Gate
The function `is_confirmatory_benchmark_eligible()` rejects any run where:
- `step_count < MIN_CONFIRMATORY_STEPS` (default threshold 5,000 steps).
- `status != "COMPLETED"`.
- Any of the 12 schema fields are absent.
- Any objective metric exhibits null, NaN, or zero-variance triviality.

This guarantee ensures that preliminary diagnostic smoke runs are never represented as converged confirmatory benchmarks.
