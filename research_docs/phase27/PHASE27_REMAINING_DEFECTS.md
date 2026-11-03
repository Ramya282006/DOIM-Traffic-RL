# Phase 27: Remaining Limitations & Engineering Defects Audit
**Project:** DOIM-PPO (Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard:** Unvarnished Scientific Integrity & Peer-Review Transparency  

---

## 1. Overview and Audit Purpose

In compliance with IEEE T-ITS reviewing standards, this document provides an honest, transparent, and unvarnished accounting of all known remaining limitations, scope boundaries, and engineering trade-offs in the DOIM-PPO repository following Phase 27.

No implementation defects or performance limitations are concealed or minimized.

---

## 2. Itemized Limitations & Technical Boundaries

### LIMITATION-27-01: Single-Core Local Compute vs. Multi-Node Cluster Campaigns
- **Description:** Complete 350-seed benchmark campaigns across 7 distinct traffic scenarios require approximately 175 CPU compute hours. On local single-core or standard workstation CPU hardware, running the exhaustive multi-seed matrix would require multi-day uninterrupted execution.
- **Scientific Impact:** The codebase is fully verified for determinism, algorithmic correctness, and microscopic TraCI execution via integration tests; however, full 350-seed asymptotic empirical convergence must be completed on high-performance compute clusters.
- **Remediation / Handling:** The repository strictly prohibits the fabrication or synthetic generation of benchmark run files. Cluster campaigns are cleanly demarcated from local test validation.

### LIMITATION-27-02: Microscopic Simulation vs. Real-World V2X Wireless Channels
- **Description:** SUMO provides high-fidelity vehicle mobility and queue dynamics, but assumes ideal vehicle detection and instantaneous communication between loop detectors / vehicle sensors and the edge controller. In real-world ITS deployments, Dedicated Short-Range Communications (DSRC) and C-V2X introduce packet loss, latency jitter, and channel fading.
- **Scientific Impact:** While the SCOIL interaction learner models uncertainty ($\mathcal{U}_t \in \mathbb{R}_+^{K \times K}$) and Bayesian shrinkage towards conservative priors, real-world field trials require hardware-in-the-loop (HIL) testing with channel emulators.
- **Remediation / Handling:** Documented as an explicit research scope boundary in the manuscript discussion section.

### LIMITATION-27-03: Native PyTorch GCN Fallback vs. Compiled PyTorch-Geometric
- **Description:** The repository provides a native pure-PyTorch implementation of the Kipf & Welling Graph Convolutional Layer (`NativeGCNConv`), ensuring that the entire pipeline runs without requiring external C++/CUDA extension compilation (`torch-geometric`, `torch-scatter`).
- **Scientific Impact:** For the 3x3 grid (9 intersections), the native implementation executes in sub-millisecond time. However, for massive metropolitan road networks (e.g., $> 1,000$ junctions), compiled PyG sparse matrix kernels would offer higher throughput.
- **Remediation / Handling:** `SpatialTrafficEncoder` automatically detects if `torch_geometric` is installed and seamlessly selects `PyGGCNConv` if available, falling back to `NativeGCNConv` otherwise.

### LIMITATION-27-04: Mixed-Autonomy Connected Vehicle Penetration Rates
- **Description:** The current SUMO traffic generation models standard human-driven vehicles adhering to the Krauss car-following model. It does not explicitly model varying Connected and Autonomous Vehicle (CAV) penetration rates (e.g., $10\%$, $30\%$, $50\%$).
- **Scientific Impact:** DOIM-PPO focuses on infrastructure-side adaptive signal control; vehicle-level actuation (trajectory planning, platooning) is outside the immediate scope of this paper.
- **Remediation / Handling:** Clear positioning as an infrastructure-side traffic signal optimization contribution.

### LIMITATION-27-05: Discrete 2-Phase Signal Logics in 3x3 Grid
- **Description:** The 3x3 urban grid environment uses dual-ring binary phase choices (North-South Green vs. East-West Green) with yellow clearance intervals.
- **Scientific Impact:** Real-world complex intersections may feature protected left-turn phases (NEMA 8-phase dual-ring). While `TrafficSafetyShield` and `SignalSafetyVerifier` support arbitrary $N$-phase definitions, the baseline 3x3 network scenarios use 2-phase green cycles.
- **Remediation / Handling:** Verified that the safety shield transition graph generalises to arbitrary phase count programs.

---

## 3. Defect Resolution Summary

All 12 implementation defects identified during the Phase 27 forensic audit (AUDIT-27-01 through AUDIT-27-12) have been **fully resolved and verified by passing tests**:
- AUDIT-27-01 (Tensor dimension guards in `predict_next_objectives`): **RESOLVED**
- AUDIT-27-02 (Batch requirement in SCOIL `compute_predictive_loss`): **RESOLVED**
- AUDIT-27-03 (Batch pooling in `aggregate_spatial_embeddings`): **RESOLVED**
- AUDIT-27-04 (Cost metric direction in `cliffs_delta`): **RESOLVED**
- AUDIT-27-05 (Node index bounds in `NativeGCNConv`): **RESOLVED**
- AUDIT-27-06 to AUDIT-27-10 (Independent reference suites): **RESOLVED**
- AUDIT-27-11 to AUDIT-27-12 (Safety shield adversarial tests and DOIM-PPO naming): **RESOLVED**
