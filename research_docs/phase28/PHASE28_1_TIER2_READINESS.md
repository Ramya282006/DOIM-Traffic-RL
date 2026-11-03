# Phase 28.1: Confirmatory Tier 2 Benchmark Readiness Assessment
**Project:** DOIM-PPO (Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Audit Standard:** Pre-Flight Readiness Review for 105-Run Confirmatory Campaign  
**Date:** November 3, 2026  

---

## 1. Readiness Audit Checklist

| Item | Requirement | Verification Method | Status | Audit Findings |
| :---: | :--- | :--- | :---: | :--- |
| **1** | All 7 Algorithms Execute | Live Execution | **PASSED** | Fixed-Time, Actuated, Max-Pressure, FW-PPO, SA-PPO, MLP-PPO, DOIM-PPO execute cleanly with zero errors. |
| **2** | All Scenarios Load in SUMO | Live TraCI Test | **PASSED** | Fixed broken relative paths in `arterial_corridor` and `incident_disruption`. All 5 scenarios verified. |
| **3** | Seed Hierarchy Separated | Manifest Audit | **PASSED** | $S_{\text{eval}} = S_{\text{train}} + 1000$ strictly enforced across all runner scripts. |
| **4** | Parameter Propagation | Code Audit | **PASSED** | Fixed `training_timesteps` propagation in `FW_PPO`, `SA_PPO`, and `IA_MOPPO` runner manifests. |
| **5** | Deterministic Reproducibility | Multi-Run Hash | **PASSED** | Proven bitwise reproducibility (SHA-256 identical checkpoints and 0 cell deltas). |
| **6** | Safety Shield Interlocks | Conflict Audit | **PASSED** | 0 conflicting greens across 25,200 simulation steps; 100% premature switch suppressions. |
| **7** | Unit Test Suite Passing | Pytest Automation| **PASSED** | 192 passed, 1 skipped (optional stable-baselines3), 0 failed in 26.85 seconds. |

---

## 2. Realistic Compute Requirements for Tier 2 (105 Runs)

### 2.1 Experimental Matrix Scope
- **Algorithms (7):** Fixed-Time, Actuated, Max-Pressure, FW-PPO, SA-PPO, MLP-PPO, DOIM-PPO.
- **Scenarios (3):** Directional Shift (`north_south_surge`), Rush Hour (`rush_hour`), Nominal Grid (`normal`).
- **Independent Seeds (5):** 42, 123, 456, 789, 2024 (Evaluation seeds: 1042, 1123, 1456, 1789, 3024).
- **Training Budget:** 5,000 steps per learning run.
- **Evaluation Budget:** 1,800 simulated seconds per evaluation run.
- **Total Executions:** $7 \times 3 \times 5 = 105 \text{ runs}$ (plus paired evaluation rollouts).

### 2.2 Realistic Runtime Projections
Based on actual empirical throughput measured in Phase 28 (Fixed-Time: 28.6 steps/s; DOIM-PPO: 11.0 steps/s; Evaluation: 31.2 steps/s):

| Component | Per-Run Duration | Total for 105 Runs (Sequential) | 4-Worker Parallel Runtime |
| :--- | :---: | :---: | :---: |
| **Heuristics (45 runs)** | ~1.5 - 2.5 min | ~1.8 CPU hours | ~0.5 hours |
| **PPO Baselines (45 runs)** | ~8.5 min | ~6.4 CPU hours | ~1.6 hours |
| **Proposed DOIM-PPO (15 runs)**| ~8.5 min | ~2.1 CPU hours | ~0.5 hours |
| **Process Overhead & Checkpointing**| ~15s per run | ~0.5 CPU hours | ~0.15 hours |
| **Safety Shield & Dynamic Port Binding**| ~10s per run | ~0.3 CPU hours | ~0.1 hours |
| **Failure Recovery Buffer (15%)**| --- | ~1.6 CPU hours | ~0.4 hours |
| **TOTAL TIER 2 RUNTIME** | --- | **~12.7 CPU hours** | **~3.25 to 3.75 hours** |

---

## 3. Final Audit Decision & Recommendation

### Decision: **A. READY FOR TIER 2**
*(All critical integrity, provenance, and execution checks have now passed).*

**Summary of Supporting Evidence:**
1. The three controller interface bugs discovered in the initial pilot were corrected and validated.
2. The broken scenario net-file paths were repaired and verified across all five scenarios.
3. Accurate training timestep provenance propagation was implemented across all baseline runners.
4. Bitwise reproducibility of both training weights and deterministic rollouts was independently proven.
5. All 192 regression and integrity tests pass with zero failures.

**Scorecard Notice:** The current research validation score remains preserved at **66/100** as a provisional historical score. It will only be updated upon the verified empirical completion of the authorized Tier 2 campaign.
