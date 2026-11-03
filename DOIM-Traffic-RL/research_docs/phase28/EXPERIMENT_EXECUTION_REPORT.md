# Phase 28: Experiment Execution & Compute Estimation Report
**Project:** DOIM-PPO (Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Date:** November 3, 2026  
**Hardware Profile:** 12th Gen Intel(R) Core(TM) i5-1235U (10 Cores, 12 Logical Processors, 16 GB RAM), Windows 11  

---

## 1. Empirical Hardware Throughput Benchmarking

To ensure rigorous, non-speculative compute estimations, an empirical hardware benchmark was executed on the host system measuring exact simulation, inference, backpropagation, and TraCI communication latencies:

```
================================================================================
                    EMPIRICAL BENCHMARK MEASUREMENTS
================================================================================
Fixed-Time Baseline Controller:    28.6 simulation steps / second
DOIM-PPO Agent (Train + Step):     11.0 simulation steps / second
Evaluation Rollout (Pure Rollout): 31.2 simulation steps / second
Single TraCI Step Latency:         ~32 ms (SUMO 1.27.1 on 3x3 Grid Network)
DOIM Neural Forward + Backprop:    ~58 ms (PyTorch GCN + GRU + SCOIL + Loss)
================================================================================
```

---

## 2. Per-Run Execution Time & Storage Projections

Each experimental run comprises:
- **Training Phase:** $N_{\text{train}}$ steps with policy gradient updates and safety filtering.
- **Evaluation Phase:** $N_{\text{eval}} = 1,800$ simulated seconds on held-out demand seed ($S_{\text{eval}} = S_{\text{train}} + 1000$).

| Tier & Step Budget | Training Time | Evaluation Time (1,800s) | Total Time per Run | Storage per Run |
| :--- | :---: | :---: | :---: | :---: |
| **Smoke Tier (128 steps)** | ~11.6s | ~57.7s | **~1.15 minutes** | ~120 KB |
| **Pilot Tier (5,000 steps)** | ~454.5s (7.6 min) | ~57.7s (1.0 min) | **~8.54 minutes** | ~2.1 MB |
| **Intermediate Tier (10,000 steps)** | ~909.1s (15.2 min) | ~57.7s (1.0 min) | **~16.11 minutes** | ~3.8 MB |
| **Main Convergence (50,000 steps)** | ~4,545.5s (75.8 min) | ~57.7s (1.0 min) | **~76.72 minutes** | ~9.5 MB |

---

## 3. Large-Scale Campaign Compute Estimates

### 3.1 Tier 2 Confirmatory Benchmark (105 Runs)
- **Matrix:** 7 Algorithms $\times$ 3 Scenarios $\times$ 5 Independent Seeds = **105 runs**
- **Algorithms:** Fixed-Time, Actuated, Max-Pressure, FW-PPO, SA-PPO, MLP-PPO, DOIM-PPO
- **Scenarios:** Directional Surge, Rush Hour, Nominal Balanced
- **Seeds:** 42, 123, 456, 789, 2024 (Evaluation seeds: 1042, 1123, 1456, 1789, 3024)

| Configuration | Sequential CPU Time | 4-Worker Parallel Time | Total Storage |
| :--- | :---: | :---: | :---: |
| **At 5,000 steps (Pilot Tier)** | **14.94 CPU hours** | **~3.74 hours** | ~220 MB |
| **At 10,000 steps (Intermediate)** | **28.19 CPU hours** | **~7.05 hours** | ~400 MB |
| **At 50,000 steps (Full Convergence)** | **134.26 CPU hours** | **~33.56 hours (~1.4 days)** | ~1.0 GB |

### 3.2 Tier 3 Multi-Scenario Campaign (350 Runs)
- **Matrix:** 7 Algorithms $\times$ 5 Scenarios $\times$ 10 Independent Seeds = **350 runs**
- **Scenarios:** Directional Surge, Rush Hour, Nominal, Arterial Corridor, Incident Disruption

| Configuration | Sequential CPU Time | 4-Worker Parallel Time | Total Storage |
| :--- | :---: | :---: | :---: |
| **At 5,000 steps (Pilot Tier)** | **49.82 CPU hours** | **~12.45 hours** | ~735 MB |
| **At 50,000 steps (Full Convergence)** | **447.53 CPU hours** | **~111.88 hours (~4.7 days)** | ~3.3 GB |

---

## 4. Parallelism, Determinism & Failure Recovery Overhead

1. **Available Host Parallelism:**
   - Intel Core i5-1235U with 10 physical cores / 12 threads.
   - Recommended maximum concurrent SUMO instances: **4 workers** to prevent CPU thermal throttling and TraCI port collisions.
2. **Deterministic Port Allocation:**
   - Parallel workers must use dynamic ephemeral TraCI ports (`--remote-port 0`) or deterministic port hashing ($P = 60000 + \text{worker\_id}$) to eliminate port conflict race conditions.
3. **Failure Recovery Overhead:**
   - Model checkpoints saved every 2,500 steps allow resuming interrupted runs without restarting from step 0.
   - Run manifests record process PID and start timestamps; failed runs are flagged as `INCOMPLETE` without corrupting completed runs.

---

## 5. Explicit User Approval Request & Decision Points

Under Section 2 and Section 5 of the Master Directive:
> *"Do not launch expensive experiment campaigns without presenting the estimated runtime and obtaining explicit user approval."*

The research validation team presents three rigorous execution options for user consideration:

- **Option A (Fast Targeted Pilot Matrix — Recommended for Immediate Local Execution):**
  - Scope: 7 algorithms on 1 primary scenario (Directional Surge) $\times$ 1 seed (42) at 1,000 steps + 1,800s evaluation.
  - Estimated Runtime: **~18 minutes total**.
  - Provides immediate, authentic empirical evidence comparing all 7 algorithms under identical conditions without long-running compute blocks.
- **Option B (Full Tier 2 Confirmatory Benchmark — Multi-Hour Run):**
  - Scope: 105 runs at 5,000 steps across 3 scenarios $\times$ 5 seeds.
  - Estimated Runtime: **~3.7 hours (with 4 workers) or ~14.9 hours (sequential)**.
  - Requires dedicated uninterrupted execution.
- **Option C (Cluster-Scale Tier 3 Campaign — Multi-Day Run):**
  - Scope: 350 runs at 50,000 steps.
  - Estimated Runtime: **~4.7 days (parallel) or ~18.6 days (sequential)**.
  - Best suited for high-performance computing clusters or overnight batch scheduling.
