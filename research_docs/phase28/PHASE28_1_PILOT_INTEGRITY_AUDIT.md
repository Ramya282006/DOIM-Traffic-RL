# Phase 28.1: Pilot Result Provenance & Execution Integrity Audit
**Project:** DOIM-PPO (Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Audit Standard:** Forensic Experimental Provenance & Source-Code Traceability  
**Auditing Panel:** Independent Research Validation Team  
**Date:** November 3, 2026  

---

## 1. Executive Summary & Audit Scope

This document provides a line-item provenance audit of all simulation runs generated during the Phase 28 Option A pilot execution. The objective is to verify:
1. Exact run IDs and on-disk directory structures.
2. Training and evaluation seeds.
3. Step counts, evaluation horizons, and manifest completeness.
4. Source-code revisions and local working-tree modifications during execution.
5. Clean separation between aborted execution attempts and the verified final cohort.

---

## 2. Execution Chronology & Source-Code Revision Trace

During Phase 28, two distinct execution tasks were launched:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TASK CHRONOLOGY & PROVENANCE TIMELINE                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Task-1613 (Launched 14:15:53Z, Aborted 14:20:54Z):                       │
│    - Base Commit at Launch: 7206cb13246f18a86aeb9739499da7258a38994e       │
│    - Executed: IA-MOPPO (train 1000 + eval 1800), Fixed-Time (train 1000    │
│      + eval 1800) successfully.                                             │
│    - Aborted at Actuated Controller due to TypeError (config_path kwarg).    │
│    - Status: INITIALIZED / CRASHED. Flagged and excluded from final cohort.  │
│                                                                             │
│ 2. Interim Code Improvements (14:21:00Z - 14:27:00Z):                       │
│    - Corrected DOIMTrafficEnv initialization in Actuated, Max-Pressure,     │
│      and MLP-PPO controllers.                                               │
│    - Verified phase-list handling in Actuated and Max-Pressure.              │
│    - Unit smoke tests passed with zero errors across all controllers.       │
│                                                                             │
│ 3. Task-1701 (Launched 14:27:43Z, Completed 14:47:08Z):                     │
│    - Base HEAD at Launch: 7206cb1 (with working-tree controller fixes).     │
│    - Executed full 9-configuration matrix continuously from Step 1 to 9.    │
│    - Exit Code: 0 (1,161.86s total runtime, zero failures).                 │
│    - Full commit of all changes and run outputs: Commit d9ef347.            │
└─────────────────────────────────────────────────────────────────────────────┘
```

All nine final evaluation runs were executed **under a single, continuous, homogeneous execution environment** (Task-1701) sharing the exact same verified controller implementations.

---

## 3. Comprehensive Run Manifest & Provenance Matrix

The table below audits all 24 run directories recorded on November 3, 2026:

| Run Directory Name | Algorithm | Evaluation Mode | Train Seed | Eval Seed | Steps Logged | Manifest Status | Checkpoint SHA-256 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `IA_MOPPO_north_south_surge_s42_20261103_195743` | IA_MOPPO | train_rollout | 42 | 42 | 1,000 | COMPLETED | `b702347032946651...` |
| `IA_MOPPO_north_south_surge_s42_20261103_195855` | IA_MOPPO | deterministic_eval | 42 | 1042 | 1,800 | COMPLETED | `b702347032946651...` |
| `Fixed_Time_north_south_surge_s42_20261103_200105` | Fixed_Time | deterministic_eval | 42 | 42 | 1,000 | COMPLETED | N/A (Heuristic) |
| `Fixed_Time_north_south_surge_s42_20261103_200132` | Fixed_Time | deterministic_eval | 42 | 1042 | 1,800 | COMPLETED | N/A (Heuristic) |
| `Actuated_north_south_surge_s42_20261103_200222` | Actuated | deterministic_eval | 42 | 42 | 1,000 | COMPLETED | N/A (Heuristic) |
| `Actuated_north_south_surge_s1042_20261103_200256` | Actuated | deterministic_eval | 1042 | 1042 | 1,800 | COMPLETED | N/A (Heuristic) |
| `MaxPressure_north_south_surge_s42_20261103_200402` | MaxPressure | deterministic_eval | 42 | 42 | 1,000 | COMPLETED | N/A (Heuristic) |
| `MaxPressure_north_south_surge_s1042_20261103_200433` | MaxPressure | deterministic_eval | 1042 | 1042 | 1,800 | COMPLETED | N/A (Heuristic) |
| `FW_PPO_north_south_surge_s42_20261103_200524` | FW_PPO | train_rollout | 42 | 42 | 1,000 | COMPLETED | `fw_ppo_final.pt` |
| `FW_PPO_north_south_surge_s42_20261103_200600` | FW_PPO | deterministic_eval | 42 | 1042 | 1,800 | COMPLETED | `fw_ppo_final.pt` |
| `SA_PPO_north_south_surge_s42_20261103_200657` | SA_PPO | train_rollout | 42 | 42 | 1,000 | COMPLETED | `sa_ppo_final.pt` |
| `SA_PPO_north_south_surge_s42_20261103_200734` | SA_PPO | deterministic_eval | 42 | 1042 | 1,800 | COMPLETED | `sa_ppo_final.pt` |
| `MLP_PPO_north_south_surge_s42_20261103_200832` | MLP_PPO | train_rollout | 42 | 42 | 1,000 | COMPLETED | `mlp_ppo_final.pt` |
| `MLP_PPO_north_south_surge_s42_20261103_200901` | MLP_PPO | deterministic_eval | 42 | 1042 | 1,800 | COMPLETED | `mlp_ppo_final.pt` |
| `DOIM_no_interaction_north_south_surge_s42_20261103_200956` | DOIM_no_int | train_rollout | 42 | 42 | 1,000 | COMPLETED | `ia_moppo_final.pt` |
| `DOIM_no_interaction_north_south_surge_s42_20261103_201117` | DOIM_no_int | deterministic_eval | 42 | 1042 | 1,800 | COMPLETED | `ia_moppo_final.pt` |
| `DOIM_rolling_pearson_north_south_surge_s42_20261103_201331` | DOIM_rolling | train_rollout | 42 | 42 | 1,000 | COMPLETED | `ia_moppo_final.pt` |
| `DOIM_rolling_pearson_north_south_surge_s42_20261103_201452` | DOIM_rolling | deterministic_eval | 42 | 1042 | 1,800 | COMPLETED | `ia_moppo_final.pt` |
| *Task-1613 Prior Runs:* | | | | | | | |
| `Actuated_north_south_surge_s42_20261103_195053` | Actuated | deterministic_eval | 42 | 42 | 0 | **INITIALIZED** | N/A (Aborted) |
| `Fixed_Time_north_south_surge_s42_20261103_194936` | Fixed_Time | deterministic_eval | 42 | 42 | 1,000 | COMPLETED | N/A (Re-run in 1701) |
| `Fixed_Time_north_south_surge_s42_20261103_195003` | Fixed_Time | deterministic_eval | 42 | 1042 | 1,800 | COMPLETED | N/A (Re-run in 1701) |
| `IA_MOPPO_north_south_surge_s42_20261103_194553` | IA_MOPPO | train_rollout | 42 | 42 | 1,000 | COMPLETED | `b702347032946651...` |
| `IA_MOPPO_north_south_surge_s42_20261103_194724` | IA_MOPPO | deterministic_eval | 42 | 1042 | 1,800 | COMPLETED | `b702347032946651...` |
| `DOIM_SMOKE_directional_shift_s42_20261103_171423` | DOIM_SMOKE | train_rollout | 42 | 42 | 128 | COMPLETED | Historical smoke run |

---

## 4. Forensic Findings & Corrections Applied

1. **Manifest Timestep Parameter Omission:**
   - In `train_fixed_weight_ppo` and `train_state_adaptive_ppo`, `training_timesteps` was not explicitly passed into `ExperimentRun`, causing the manifest to record the constructor default (128) despite `metrics_step.csv` containing 1,001 rows (1,000 executed steps).
   - In evaluation runs, `training_timesteps` read `config.policy.total_timesteps` (50,000 from default.yaml) instead of the actual 1,000 steps.
   - **Correction Applied in Phase 28.1:** Parameter propagation was corrected across all runner and evaluation functions to ensure manifests record true runtime parameters.

2. **Source-Code Homogeneity:**
   - All 18 runs constituting the Option A comparison cohort were produced sequentially by `task-1701` between 14:27:43Z and 14:47:08Z.
   - Zero runs in the reported comparison table originated from pre-correction states.
