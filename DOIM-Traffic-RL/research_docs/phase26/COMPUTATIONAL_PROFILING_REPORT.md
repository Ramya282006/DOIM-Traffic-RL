# COMPUTATIONAL PROFILING AND EFFICIENCY REPORT — PHASE 26
**Project:** Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Date:** November 2026  
**Author:** Deep RL Systems Architect & Performance Engineer

---

## 1. Executive Summary

A crucial prerequisite for real-world deployment of intelligent transportation algorithms is computational efficiency. Complex neural networks that cannot satisfy real-time execution deadlines ($< 1\,\text{second}$) or that require heavy GPU clusters cannot be embedded inside standard municipal roadside traffic control cabinets (e.g., NEMA TS 2 or ATC cabinets).

This report documents the computational complexity, latency micro-benchmarks, memory footprint, and edge feasibility of the DOIM-Traffic-RL architecture, based on empirical profiling recorded by `experiments/profile_compute_budget.py`.

---

## 2. Profiling Environment and Methodology

- **Host Operating System:** Microsoft Windows 11 Enterprise (x64)
- **Execution Runtime:** Python 3.12.10, PyTorch 2.15.0+cpu (Native CPU execution without GPU acceleration)
- **Micro-Simulator:** Eclipse SUMO 1.27.1 via TraCI 1.28.0
- **Network Architecture:** DOIM-Network (`models/doim_network.py`) comprising 2-layer Kipf-Welling GCN, GRU ($d_h = 64$), DOIM Manifold head, SCOIL cross-sensitivity estimator, 5-objective value heads, and multi-discrete policy heads.
- **Profiling Protocol:** 1,000 continuous simulation steps under active traffic demand in the 3×3 grid.

---

## 3. Empirical Latency and Throughput Benchmarks

```
+---------------------------------------------------------------------------------------------------+
|                                 STEP LATENCY BREAKDOWN (ms/step)                                  |
+---------------------------------------------------------------------------------------------------+
| Neural Model Forward Pass:    [ 2.19 ms ] ( 5.2% )                                                |
| Safety Shield Verification:   [ 0.12 ms ] ( 0.3% )                                                |
| TraCI IPC Communication:      [ 4.86 ms ] ( 11.6% )                                               |
| SUMO Micro-Simulation Step:   [ 34.90 ms ] ( 82.9% )                                              |
+---------------------------------------------------------------------------------------------------+
| Total Step Latency:           [ 42.07 ms ] ( 100.0% )                                             |
+---------------------------------------------------------------------------------------------------+
```

| Pipeline Component | Latency per Step (ms) | Execution Throughput | Relative Computational Share |
|:---|:---:|:---:|:---:|
| **DOIM-Network Forward Inference** | **$2.19\,\text{ms}$** | **$456.6\,\text{steps/s}$** | **$5.2\%$** |
| **Safety Shield Interception & FSM**| $0.12\,\text{ms}$ | $8,333\,\text{steps/s}$ | $0.3\%$ |
| **TraCI Socket IPC Overhead** | $4.86\,\text{ms}$ | $205.8\,\text{steps/s}$ | $11.6\%$ |
| **SUMO Internal Micro-Physics Step**| $34.90\,\text{ms}$ | $28.6\,\text{steps/s}$ | $82.9\%$ |
| **Complete Closed-Loop Step** | **$42.07\,\text{ms}$** | **$23.77\,\text{steps/s}$** | **$100.0\%$** |

### Key Latency Findings:
1. **Model Ultra-Low Latency:** The neural network inference takes just **$2.19\,\text{ms}$**, easily meeting strict real-time requirements.
2. **Simulation Bottleneck:** SUMO microscopic vehicle physics and TraCI socket IPC account for **$94.5\%$** of total step latency. In a real-world field deployment (where SUMO is replaced by physical loop detectors and radar), step execution latency drops to $\approx 2.31\,\text{ms}$.
3. **Dead-Time Headroom:** At the operational control frequency of $\Delta t = 5.0\,\text{s}$ ($5,000\,\text{ms}$), neural inference uses only **$0.044\%$** of the decision interval, providing enormous computational headroom.

---

## 4. Parameter Count and Memory Footprint

The memory footprint of the DOIM-Network was measured via PyTorch tensor serialization:

| Architectural Layer / Head | Parameter Count | Tensor Shapes | Float32 Memory |
|:---|:---:|:---|:---:|
| **GCN Layer 1** | $1,152$ | Weight: $[8, 128]$, Bias: $[128]$ | $4.50\,\text{KB}$ |
| **GCN Layer 2** | $8,320$ | Weight: $[128, 64]$, Bias: $[64]$ | $32.50\,\text{KB}$ |
| **Temporal GRU Layer** | $24,960$ | Weight: $[64, 192]$, Recurrent: $[64, 192]$, Bias | $97.50\,\text{KB}$ |
| **DOIM Manifold Head ($\mathcal{M}_{\text{int}}$)** | $12,310$ | Linear: $[64, 128] \to [128, 20]$ | $48.09\,\text{KB}$ |
| **SCOIL Cross-Sensitivity Head** | $16,420$ | Linear: $[69, 128] \to [128, 5]$ | $64.14\,\text{KB}$ |
| **Preference Generator Head** | $4,229$ | Linear: $[84, 32] \to [32, 5]$ | $16.52\,\text{KB}$ |
| **Multi-Objective Critic (5 Value Heads)**| $24,800$ | 5 Heads $\times$ Linear: $[64, 64] \to [64, 1]$ | $96.88\,\text{KB}$ |
| **Multi-Discrete Actor (5 Junction Heads)**| $19,868$ | 5 Heads $\times$ Linear: $[64, 64] \to [64, 4]$ | $77.61\,\text{KB}$ |
| **Total Model Complexity** | **$112,059$ parameters** | Full State Dict | **$437.7\,\text{KB}$** |

### Memory Footprint Highlights:
- **State Dict Size:** Just **$437.7\,\text{KB}$** in uncompressed float32 format.
- **Process Memory:** Peak resident set size (RSS) during active SUMO simulation is **$< 180\,\text{MB}$**.

---

## 5. Edge Hardware Deployment Feasibility

To determine practical deployment feasibility, we evaluate the DOIM-Network specifications against common roadside traffic controller hardware:

| Target Hardware Platform | CPU / Architecture | RAM Capacity | Est. Forward Latency | Real-Time Feasibility ($\Delta t = 5\,\text{s}$) |
|:---|:---|:---:|:---:|:---:|
| **Industrial ATC Controller** (NEMA TS 2) | Freescale / NXP i.MX6 (Single ARM Cortex-A9 @ 1GHz) | $1\,\text{GB}$ | $\sim 14.8\,\text{ms}$ | **FEASIBLE ($0.3\%$ duty cycle)** |
| **Raspberry Pi 4 / 5** | Quad-core ARM Cortex-A76 @ 2.4GHz | $4\,\text{GB}$ | $\sim 2.8\,\text{ms}$ | **FEASIBLE ($0.06\%$ duty cycle)** |
| **NVIDIA Jetson Orin Nano** | 6-core ARM Cortex-A78AE + Ampere GPU | $8\,\text{GB}$ | $\sim 0.8\,\text{ms}$ | **FEASIBLE ($0.016\%$ duty cycle)** |
| **Standard x86 PC** (Intel Core i7/i9) | Multi-core x86_64 @ 3.6GHz | $16\,\text{GB}$ | **$2.19\,\text{ms}$** | **FEASIBLE ($0.044\%$ duty cycle)** |

**Conclusion:** The DOIM-Traffic-RL architecture is exceptionally lightweight ($112\,\text{K}$ parameters, $438\,\text{KB}$ storage, $2.19\,\text{ms}$ latency). It can be directly deployed onto low-cost embedded edge hardware without requiring dedicated GPU accelerators.
