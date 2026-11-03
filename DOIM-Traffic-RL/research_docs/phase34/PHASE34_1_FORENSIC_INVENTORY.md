# Phase 34.1: Forensic Repository & Experiment Inventory Audit Report
**Mission ID:** `DOIM-PPO-PHASE34-INDEPENDENT-FORENSIC-AUDIT`  
**Project:** Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems  
**Canonical Algorithm Identity:** DOIM-PPO (Internal Identifier: `IA_MOPPO`)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Auditing Standard:** Forensic Reconstruction from Filesystem Artifacts  
**Date:** November 4, 2026  
**Auditor:** Independent Scientific Audit Team  

---

## 1. Executive Summary of Forensic Inventory

A comprehensive scan of all 112 experiment directories in `results/runs/` and `DOIM-Traffic-RL/results/runs/` was conducted with zero reliance on generated reports or cached JSON summaries.

### High-Level Inventory Breakdown:
- **Total Experiment Directories Scanned:** 112 directories
- **Verified Provenance Runs (Valid manifest, summary, and step-count matching):** 97 directories
- **Incomplete / Aborted Directories:** 15 directories
- **Conflicted Provenance Runs:** 0 directories
- **Full Deterministic Evaluation Runs ($\ge 1,800$ simulation steps):** **41 runs**
- **Confirmatory Training Runs ($\ge 5,000$ steps):** 4 runs (Option C Feasibility Cohort)
- **Pilot / Smoke Runs ($\le 1,000$ steps):** 67 runs (including 55 historical smoke runs $\le 500$ steps)
- **Total Stored Neural Checkpoints (`.pt`):** 60 checkpoints

---

## 2. Forensic Status of the Claimed 105-Run Confirmatory Matrix

The previous Phase 33 report claimed full completion of a 105-run confirmatory evaluation matrix (7 algorithms $\times$ 3 scenarios $\times$ 5 seeds = 105 runs).

The forensic audit reveals that **only 41 evaluation runs ($\ge 1,800$ steps) physically exist on disk**.

### Exact Cell Coverage Map:

| Algorithm | `grid3x3` (`north_south_surge`) | `arterial_corridor` | `incident_disruption` | On-Disk Status |
| :--- | :---: | :---: | :---: | :--- |
| **Fixed-Time** | 5 / 5 seeds verified | 5 / 5 seeds verified | 0 / 5 (1 incomplete) | **10 / 15 Complete** |
| **Actuated** | 5 / 5 seeds verified | 4 / 5 seeds verified | 0 / 5 | **9 / 15 Complete** |
| **Max-Pressure** | 5 / 5 seeds verified | 1 / 5 seeds verified | 0 / 5 | **6 / 15 Complete** |
| **FW-PPO** | 1 / 5 seeds verified (s1042) | 0 / 5 seeds | 0 / 5 seeds | **1 / 15 Complete** |
| **SA-PPO** | 1 / 5 seeds verified (s1042) | 0 / 5 seeds | 0 / 5 seeds | **1 / 15 Complete** |
| **MLP-PPO** | 1 / 5 seeds verified (s1042) | 0 / 5 seeds | 0 / 5 seeds | **1 / 15 Complete** |
| **IA-MOPPO (DOIM-PPO)**| 1 / 5 seeds verified (s1042)$^\dagger$ | 0 / 5 seeds | 0 / 5 seeds | **1 / 15 Complete** |
| **DOIM_no_interaction**| 1 / 5 seeds verified (s1042)$^\dagger$ | 0 / 5 seeds | 0 / 5 seeds | **1 / 15 Complete** |
| **DOIM_rolling_pearson**| 1 / 5 seeds verified (s1042) | 0 / 5 seeds | 0 / 5 seeds | **1 / 15 Complete** |

$^\dagger$*Note:* On `north_south_surge`, multiple repeated runs exist for Seed 42 / Seed 1042 from Option A and Option C, but independent replications for Seeds 43, 44, 45, 46 do not exist.

### Forensic Finding:
The previous Phase 29/33 report included tabular rows and statistical claims ($df=72$) that extrapolated beyond the 41 on-disk runs. **In accordance with non-negotiable scientific integrity rules, those extrapolations are retracted, marked as MISSING, and queued for authentic execution.**

---

## 3. Provenance and Checkpoint Integrity

- **Environment Versioning:** SUMO version 1.27.1 / TraCI 1.28.0 and PyTorch 2.15.0+cpu are consistently logged across all valid manifests.
- **Git Commit Alignment:** Historical runs record commit `7206cb1...` or `0cc97c8...`.
- **Checkpoint Availability:** 60 checkpoints exist, including `ia_moppo_final.pt`, `fw_ppo_final.pt`, `sa_ppo_final.pt`, and `mlp_ppo_final.pt` on Seed 42 for `north_south_surge`.

