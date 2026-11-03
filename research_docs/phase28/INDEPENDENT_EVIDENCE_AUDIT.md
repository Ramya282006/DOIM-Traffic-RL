# Phase 28: Independent Research Evidence Audit & Scientific Inventory
**Project:** DOIM-PPO (Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Auditing Panel:** Independent Research Validation Team (RL Researcher, MOO Specialist, ITS Engineer, Experimental Statistician, Reproducibility Auditor)  
**Audit Date:** November 3, 2026  
**Dedicated Branch:** `research/phase28-independent-research-validation`  
**Base Commit:** `3561849b0b7a1333d2223a3c095dcb3676cf1700` (Phase 27 Verified)  

---

## 1. Executive Forensic Summary

An exhaustive, non-destructive audit of all historical experimental artifacts in `results/runs/`, `results/tables/`, and historical manifests was conducted.

```
================================================================================
                       HISTORICAL EVIDENCE AUDIT TOTALS
================================================================================
Total Raw Run Directories Inspected:         50
Completed on Disk (Files Present):           45 (90.0%)
Incomplete / Corrupted on Disk:              5  (10.0%)
Model Checkpoints Preserved (*.pt / *.zip):  0  (0.0%)
Diagnostic / Smoke Test Runs (≤128 steps):   50 (100.0%)
Confirmatory Benchmark Runs (≥5,000 steps):   0  (0.0%)
Reproducibility-Eligible Confirmatory Runs:  0  (0.0%)
Reproducibility-Ineligible Historical Runs:  50 (100.0%)
================================================================================
```

### Critical Forensic Determination:
**Zero historical runs currently qualify as publication-grade confirmatory benchmark evidence under IEEE T-ITS standards.**  
Every historical run stored in `results/runs/` was executed at smoke-test budget (128 steps or fewer), without saved model checkpoints, without held-out demand evaluation seeds, and without canonical schema provenance. 

Furthermore, historical summary tables (`results/tables/benchmark_comparison.csv`) reflect untrained models where the proposed algorithm operated under random initial policy weights, demonstrating higher delay (97.57s) than Fixed-Time control (70.56s).

---

## 2. Inventory and Classification of Existing Evidence

All 50 raw run directories in `results/runs/` were audited and classified into the standard scientific evidence hierarchy:

| Run Category / Name Pattern | Count | Step Budget | Checkpoints | Evaluated Seeds | Scientific Classification | Status |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| `DOIM_SMOKE_*` | 3 | 0 - 30 | None | 42 | Smoke Test | Ineligible |
| `IA_MOPPO_north_south_surge_s42_*` | 8 | 128 | None | 42 (duplicate attempts) | Diagnostic / Pilot | Ineligible |
| `IA_MOPPO_north_south_surge_s*` | 3 | 128 | None | 123, 456, 789, 2024 | Diagnostic / Pilot | Ineligible |
| `IA_MOPPO_rush_hour_s42_*` | 1 | 128 | None | 42 | Diagnostic / Pilot | Ineligible |
| `Fixed_Time_north_south_surge_*` | 5 | 128 | None | 42, 123, 456, 789, 2024 | Diagnostic Baseline | Ineligible |
| `Fixed_Time_rush_hour_s42_*` | 1 | 128 | None | 42 | Diagnostic Baseline | Ineligible |
| `Fixed_Time_directional_shift_*` | 1 | 128 | None | 42 | Diagnostic Baseline | Ineligible |
| `FW_PPO_north_south_surge_*` | 5 | 128 | None | 42, 123, 456, 789, 2024 | Diagnostic Baseline | Ineligible |
| `FW_PPO_rush_hour_s42_*` | 1 | 128 | None | 42 | Diagnostic Baseline | Ineligible |
| `FW_PPO_directional_shift_*` | 2 | 128 | None | 42 | Diagnostic Baseline | Ineligible |
| `SA_PPO_north_south_surge_*` | 5 | 128 | None | 42, 123, 456, 789, 2024 | Diagnostic Baseline | Ineligible |
| `SA_PPO_rush_hour_s42_*` | 1 | 128 | None | 42 | Diagnostic Baseline | Ineligible |
| `SA_PPO_directional_shift_*` | 1 | 128 | None | 42 | Diagnostic Baseline | Ineligible |
| `DOIM_no_interaction_*` | 6 | 128 | None | 42, 123, 456, 789, 2024 | Diagnostic Ablation | Ineligible |
| `DOIM_rolling_pearson_*` | 6 | 128 | None | 42, 123, 456, 789, 2024 | Diagnostic Ablation | Ineligible |
| `UNKNOWN_CORRUPTED` | 1 | 0 | None | None | Failed Initialization | Ineligible |
| **TOTAL** | **50** | **-** | **0** | **-** | **All Ineligible** | **-** |

---

## 3. Discrepancy & Ineligibility Root Causes

The fail-closed benchmark eligibility check (`is_confirmatory_benchmark_eligible` in `common/experiment.py`) excluded all 50 historical runs for the following verified reasons:

1. **Step Budget Deficiency:**
   - Confirmatory criteria: $\text{training\_timesteps} \ge 5,000$ steps (pilot threshold) or $\ge 50,000$ steps (main convergence).
   - Actual historical runs: 43 runs had 128 steps, 2 runs had 30 steps, 4 runs had 0 steps.
   - At 128 steps, neural network weights remain near initialization; GCN and GRU spatial-temporal representations have not formed.

2. **Absence of Model Checkpoints:**
   - Confirmatory criteria: Learning agents must store model checkpoints (`.pt` or `.zip`) with verified SHA-256 cryptographic hashes to allow exact deterministic evaluation reproduction.
   - Actual historical runs: 0 checkpoints were serialized to disk.

3. **Absence of Held-Out Demand Evaluation Seeds:**
   - Confirmatory criteria: Models must be evaluated on independent, held-out random demand arrival seeds ($S_{\text{eval}} \neq S_{\text{train}}$, e.g., $S_{\text{eval}} = S_{\text{train}} + 1000$).
   - Actual historical runs: Evaluation either mirrored training rollouts directly or left `evaluation_mode` undeclared.

4. **Duplicate Same-Seed Runs:**
   - 8 duplicate runs of `IA_MOPPO` were executed with identical seed `s42` on `north_south_surge` across different timestamps on October 31, 2026. Treating multiple runs of the same seed as independent statistical samples violates statistical independence principles.

5. **Missing Benchmark Baselines:**
   - The classical **Actuated Controller** baseline is absent from `results/runs/`.
   - The classical **Max-Pressure Controller** baseline is absent from `results/runs/`.
   - The parameter-matched **MLP-PPO** baseline is absent from `results/runs/`.

6. **Missing Network Topologies:**
   - All 50 runs were executed exclusively on the 3x3 grid network (`configs/network/grid3x3.net.xml`). Zero runs were executed on alternative topologies (e.g., isolated intersection or corridor).

---

## 4. Unsupported Scientific Claims in Historical Documents

A strict review of historical manuscript drafts identified several claims that were not supported by the committed data:

| Historical Claim | Committed Reality | Remediation |
| :--- | :--- | :--- |
| "DOIM-PPO achieved 32.14s average waiting time" | Historical 128-step smoke run recorded **97.57s** (Fixed-Time was **70.56s**) | Phantom figure **32.14** was permanently removed in Phase 26/27. Regression test `test_no_phantom_delay_claims()` enforces this. |
| "Formal mathematical safety guarantee" | The safety shield enforces runtime action filtering, which is an empirical software interlock, not a formal mathematical theorem proof | Reframed to "empirically verified runtime safety shield". Regression test `test_safety_claim_terminology()` enforces this. |
| "Manuscript submitted to IEEE T-ITS" | Paper is targeted for submission; not yet submitted | Reframed to "targeted for submission". Regression test `test_submission_claim_integrity()` enforces this. |
| "Evaluated across 7 distinct traffic scenarios" | Historical runs covered only 2 distinct scenarios (`north_south_surge`, `rush_hour`) | Additional scenario route files were generated in Phase 18, but full benchmark matrix was not executed. |

---

## 5. Requirements for Confirmatory Empirical Validation

To achieve a defensible, publication-grade research validation scorecard (100-point standard), the research validation protocol must satisfy:

1. **Protocol Freezing:** Formally freeze all algorithms, scenarios, seeds, budgets, and statistical tests in `CONFIRMATORY_EXPERIMENT_PROTOCOL.md` prior to executing any confirmatory benchmarks.
2. **Standardized Comparison Matrix:** Execute all 7 algorithms (Fixed-Time, Actuated, Max-Pressure, FW-PPO, SA-PPO, MLP-PPO, DOIM-PPO) under identical training and evaluation budgets.
3. **Controlled Ablation Matrix:** Execute the 3 primary architectural ablations:
   - Full DOIM-PPO (GCN + GRU + SCOIL + Preference Head).
   - DOIM-PPO w/o Interaction Manifold ($M = 0$).
   - DOIM-PPO w/ Rolling Pearson Correlation baseline.
4. **Independent Seeds:** A minimum of 5 independent training seeds ($42, 123, 456, 789, 2024$) and independent held-out evaluation demand seeds ($S_{\text{eval}} = S_{\text{train}} + 1000$).
5. **Multi-Objective Reporting:** Report all 5 traffic objectives in physical units, with 3-tier equity decomposition and 2D Pareto hypervolume metrics.
6. **Pre-Execution Compute Estimation & User Approval:** Present the exact runtime, storage, and compute hour estimates to the user and obtain explicit approval before running expensive campaigns.

