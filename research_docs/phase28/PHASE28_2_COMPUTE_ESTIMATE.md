# Phase 28.2: Compute Budget, Runtime & Storage Requirements
**Project:** DOIM-PPO (*Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems*)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard:** Engineering Resource Planning & Empirical Benchmarking  
**Auditing Panel:** Independent Research Validation Team  
**Date:** November 3, 2026  

---

## 1. Empirical Timing Benchmarks (From Task-1701 On-Disk Data)

Wall-clock execution times measured on the active host CPU (8-core x86_64, Windows, Python 3.12, SUMO 1.27.1):

| Algorithm | Training Rate (s / 1,000 steps) | Evaluation Rate (s / 1,800 steps) |
| :--- | :---: | :---: |
| **Fixed-Time** | $0$ s (no training) | $45.2$ s |
| **Actuated** | $0$ s (no training) | $52.6$ s |
| **Max-Pressure** | $0$ s (no training) | $48.1$ s |
| **MLP-PPO** | $29.5$ s | $55.0$ s |
| **FW-PPO** | $35.8$ s | $62.3$ s |
| **SA-PPO** | $42.1$ s | $68.4$ s |
| **DOIM-PPO** | $71.1$ s | $129.9$ s |

---

## 2. Campaign Runtime Projections

### Option A: Standard Confirmatory Budget (5,000 Training Steps, 1,800 Eval Steps)
- **RL Training (60 runs):**
  - DOIM-PPO ($15 \times 5 \times 71.1\,\text{s}$): $5,332.5\,\text{s} \approx 1.48\,\text{hours}$
  - SA-PPO ($15 \times 5 \times 42.1\,\text{s}$): $3,157.5\,\text{s} \approx 0.88\,\text{hours}$
  - FW-PPO ($15 \times 5 \times 35.8\,\text{s}$): $2,685.0\,\text{s} \approx 0.75\,\text{hours}$
  - MLP-PPO ($15 \times 5 \times 29.5\,\text{s}$): $2,212.5\,\text{s} \approx 0.61\,\text{hours}$
  - *Subtotal RL Training:* **3.72 hours**
- **Evaluation (105 runs):**
  - Non-RL Baselines ($45 \text{ runs} \times \approx 48\,\text{s}$): $2,160\,\text{s} \approx 0.60\,\text{hours}$
  - RL Baselines ($45 \text{ runs} \times \approx 62\,\text{s}$): $2,790\,\text{s} \approx 0.78\,\text{hours}$
  - DOIM-PPO ($15 \text{ runs} \times 129.9\,\text{s}$): $1,948.5\,\text{s} \approx 0.54\,\text{hours}$
  - *Subtotal Evaluation:* **1.92 hours**
- **Grand Total Estimated Runtime:** **$\approx 5.64$ hours**.

---

## 3. Storage Footprint & System Resources

- **Telemetry Footprint:**
  - Each 1,800-step evaluation generates $\approx 420\,\text{KB}$ of CSV and JSON telemetry.
  - Each 5,000-step training run generates $\approx 1.1\,\text{MB}$ of logs + $1.8\,\text{MB}$ PyTorch checkpoint = $\approx 2.9\,\text{MB}$.
  - Total Storage Required for 165 runs: **$\approx 220\,\text{MB}$**.
  - Current Available Disk Space: **$> 100\,\text{GB}$** (Sufficient).
- **RAM Footprint:** SUMO + PyTorch consumes $\approx 1.8\,\text{GB}$ peak RAM (Host has 16+ GB, Sufficient).
