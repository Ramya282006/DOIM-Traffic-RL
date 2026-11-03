# LIMITATIONS AND SCIENTIFIC SCOPE
**Project:** Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Date:** November 2026

---

## 1. Introduction

In accordance with the highest scientific ethics and IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS) peer-review standards, this document articulates the explicit boundaries, theoretical assumptions, and empirical limitations of the DOIM-Traffic-RL research project. Recognizing and discussing limitations transparently strengthens scientific credibility and guides future research.

---

## 2. Theoretical Limitations

### 2.1 Associative Cross-Sensitivity vs. Pearlian Causality
- **Limitation:** State-Conditioned Objective Interaction Learning (SCOIL) predicts directional cross-sensitivities under the policy's state-action distribution. It does **not** constitute a Structural Causal Model (SCM) in the Pearlian sense.
- **Scientific Boundary:** Purely observational traffic data lacks exogenous instrumental variables or interventional do-calculus operators ($\text{do}(X = x)$). Therefore, SCOIL captures predictive cross-objective coupling rather than invariant counterfactual causal mechanisms.

### 2.2 Differential Geometry on the Preference Simplex
- **Limitation:** While $\text{int}(\Delta^{K-1})$ is endowed with the Fisher-Rao Riemannian metric, policy optimization in IA-MOPPO is executed via standard backpropagation on unconstrained latent preference logits $\mathbf{v}_t \in \mathbb{R}^K$, mapped through softmax.
- **Scientific Boundary:** The framework uses softmax as the canonical exponential family link function rather than computing explicit Riemannian natural gradient updates with Christoffel symbols or spherical retractions.

---

## 3. Empirical and Simulation Limitations

### 3.1 Simulation-to-Real (Sim-to-Real) Gap
- **Limitation:** All empirical benchmarks are conducted using microscopic simulation in Eclipse SUMO 1.27.1.
- **Physical Factors Not Fully Modeled:**
  - Real-world sensor dropout, inductive loop failure, and computer vision occlusion.
  - Variable driver reaction times, distracted driving, and illegal red-light running behavior.
  - Inclement weather impacts on tire-pavement friction and braking distances.
- **Mitigation:** The active safety shield enforces conservative yellow ($3\,\text{s}$) and red ($1\,\text{s}$) clearance intervals, and minimum green ($10\,\text{s}$) to absorb human driver reaction delays.

### 3.2 Network Scale and Topology
- **Limitation:** The primary benchmark network consists of a 3×3 urban grid (9 intersections, 5 controllable signalized junctions).
- **Scientific Boundary:** While a 3×3 grid provides sufficient complexity to exhibit gridlock cascades, green-wave coordination, and arterial cross-street trade-offs, scaling to metropolitan networks with hundreds of intersections will require hierarchical multi-agent reinforcement learning (MARL) decompositions.

### 3.3 Computational Resource Constraints for Asymptotic Execution
- **Limitation:** Profiled execution on native CPU hardware establishes that a full 350-run factorial matrix ($10\text{ seeds} \times 7\text{ algorithms} \times 5\text{ scenarios} \times 50,000\text{ steps}$) requires **$204.5\,\text{hours}$** ($\approx 8.52$ days) of continuous compute.
- **Mitigation:** The repository defines an explicit three-tier execution protocol. Tier 1 integration and diagnostic suites are completed and verified; Tier 2/3 asymptotic matrices are formally prepared and guarded by fail-closed filters (`is_confirmatory_benchmark_eligible`).

---

## 4. Operational Safety Assumptions

- **Driver Compliance Assumption:** The traffic safety shield guarantees mathematical conflict freedom and minimum green dwell times under the assumption that motorists obey traffic signal displays. It does not control autonomous vehicle trajectories directly and cannot physically prevent collisions caused by vehicles running red lights.
- **Fail-Safe Mechanism:** In the event of TraCI IPC communication failure or hardware exception, the controller defaults to a deterministic Fixed-Time or Actuated fail-safe cycle.

---

## 5. Summary

These limitations delineate the boundary between the verified scientific contributions of this work and future extensions. They in no way diminish the demonstrated validity of the DOIM interaction manifold, the SCOIL cross-sensitivity estimator, the safety shield, or the empirical findings within the defined operational scope.
