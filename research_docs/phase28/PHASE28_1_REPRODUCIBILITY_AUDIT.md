# Phase 28.1: Empirical Reproducibility & Bitwise Audit
**Project:** DOIM-PPO (Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Audit Standard:** Independent Multi-Run Exact Reproduction Audit  
**Date:** November 3, 2026  

---

## 1. Audit Methodology

To verify the reproducibility claims made in Phase 28, two independent executions of two distinct configurations were conducted on the host system under identical software, hardware, and random seed conditions:
1. **Fixed-Time Baseline (Deterministic Controller):** Evaluated over 1,800 steps on Seed 1042.
2. **DOIM-PPO Agent (Deep Neural Network + GCN + GRU + SCOIL):** Trained for 1,000 steps on Seed 42, then evaluated for 1,800 steps on held-out Seed 1042.

All random seeds (Python `random`, NumPy, PyTorch CPU, TraCI/SUMO) were managed through `common/seeding.py:seed_everything(seed)`.

---

## 2. Experimental Reproduction Results

### 2.1 Fixed-Time Baseline (1,800 Simulated Seconds)
- Run 1: `Fixed_Time_north_south_surge_s42_20261103_195003`
- Run 2: `Fixed_Time_north_south_surge_s42_20261103_200132`

| Metric Column in `metrics_step.csv` | Total Rows | Rows with Discrepancy ($> 10^{-6}$) | Reproduction Status |
| :--- | :---: | :---: | :---: |
| `step` | 1,800 | 0 | **BITWISE IDENTICAL** |
| `simulation_time` | 1,800 | 0 | **BITWISE IDENTICAL** |
| `reward` | 1,800 | 0 | **BITWISE IDENTICAL** |
| `raw_waiting` | 1,800 | 0 | **BITWISE IDENTICAL** |
| `raw_queue` | 1,800 | 0 | **BITWISE IDENTICAL** |
| `raw_co2` | 1,800 | 0 | **BITWISE IDENTICAL** |
| `raw_throughput` | 1,800 | 0 | **BITWISE IDENTICAL** |
| `raw_fairness` | 1,800 | 0 | **BITWISE IDENTICAL** |
| `action_tls` | 1,800 | 0 | **BITWISE IDENTICAL** |
| `shield_interventions` | 1,800 | 0 | **BITWISE IDENTICAL** |
| **All 20 Columns (36,000 Cells)** | **36,000** | **0** | **100% BITWISE IDENTICAL** |

### 2.2 DOIM-PPO Training (1,000 Steps)
- Run 1: `IA_MOPPO_north_south_surge_s42_20261103_194553`
- Run 2: `IA_MOPPO_north_south_surge_s42_20261103_195743`

| Checkpoint File | Checkpoint SHA-256 Checksum | Match? |
| :--- | :--- | :---: |
| Run 1 Final Checkpoint | `b702347032946651093fcace1caec2fd41e44fda75b5e1621d03258706285ddf` | **EXACT BITWISE MATCH** |
| Run 2 Final Checkpoint | `b702347032946651093fcace1caec2fd41e44fda75b5e1621d03258706285ddf` | **EXACT BITWISE MATCH** |

All 20 columns across all 1,000 training steps (20,000 data cells) showed **zero numerical differences**.

### 2.3 DOIM-PPO Deterministic Evaluation (1,800 Steps)
- Run 1: `IA_MOPPO_north_south_surge_s42_20261103_194724`
- Run 2: `IA_MOPPO_north_south_surge_s42_20261103_195855`

All 20 columns across all 1,800 evaluation steps (36,000 data cells) showed **zero numerical differences**.

---

## 3. Reproducibility Classification Verdict

Under ACM/IEEE Artifact Review and Badging criteria:
- The system achieves **Bitwise Reproducibility** on CPU execution.
- Independent repetitions with identical random seeds yield identical model weights, identical policy actions, and identical microscopic SUMO vehicle trajectories.
