# DOIM-Traffic-RL: Comprehensive Forensic Research Audit (Phase 14)

**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Audit Standard:** Strict Forensic IEEE T-ITS Peer Review  
**Date / Timestamp:** 2026-11-03 17:50:00 UTC  
**Branch:** `research/final-research-grade-transformation`  
**Base Commit SHA:** `57fb8cce2caf0e3a9c33c9738396da34cebb2056`  

---

## 1. Executive Summary

This forensic audit represents an exhaustive, code-level inspection of the DOIM-Traffic-RL repository prior to executing the Phase 15–25 autonomous research transformation. Rather than relying on previous self-assessments, every mathematical formulation, model architecture, baseline implementation, evaluation pipeline, manifest ledger, and manuscript claim has been audited directly against the actual source code and execution runtime.

The repository possesses a sound architectural core:
1. A dual-layer spatial Kipf-Welling GCN and temporal GRU backbone.
2. A state-conditioned objective interaction learner (SCOIL).
3. An active traffic safety shield preventing illegal signal phase transitions.
4. A 3-tier objective suite tracking delay, queue, $\text{CO}_2$ emissions, throughput, and fairness.

However, forensic inspection reveals eight specific technical and empirical defects that must be resolved to achieve publication-grade rigor for IEEE T-ITS.

---

## 2. Forensic Audit of Previous Findings

| # | Prior Finding | Code-Level Audit Finding | Severity | Phase Remediation |
| :-: | :--- | :--- | :-: | :---: |
| **1** | Confirmatory eligibility not enforced in aggregation | **CONFIRMED DEFECT.** While `is_confirmatory_benchmark_eligible` was implemented in `common/experiment.py`, it was not invoked in `analysis/statistical_analyzer.py`. Unfiltered directory scans would load 128-step diagnostic runs into benchmark tables. | **CRITICAL** | Phase 20 |
| **2** | Actuated controller had unused gap and extension parameters | **CONFIRMED DEFECT.** In `baselines/actuated/actuated_controller.py`, `gap_threshold_sec` and `unit_extension_sec` were initialized in `__init__`, but `select_actions` only checked `num_veh == 0`, ignoring passage timers and extension gaps. | **HIGH** | Phase 19 |
| **3** | SCOIL uncertainty calibration was simplistic | **CONFIRMED DEFECT.** In `models/interaction_learner.py` and `analysis/scoil_validator.py`, dispersion was scaled with a heuristic Gaussian critical value ($z=1.96$) without finite-sample conformal calibration or Mahalanobis error bounds. | **HIGH** | Phase 17 |
| **4** | Historical runs ledger dominated by 128-step diagnostics | **CONFIRMED FACT.** Of the 49 historical runs in `results/runs/`, 46 are 128-step diagnostic rollouts. They have been indexed and quarantined in `results/historical_archive_manifest.json` under tier `smoke`/`diagnostic`. | **MEDIUM** | Phase A/B |
| **5** | Confirmatory experiment evidence insufficient | **CONFIRMED DEFECT.** No completed multi-seed confirmatory matrix (5,000 to 50,000 steps) existed across all 7 baselines with held-out evaluation seeds. | **HIGH** | Phase 21 |
| **6** | Documentation & environment version inconsistencies | **CONFIRMED MINOR.** Previous markdown files mentioned SUMO v1.28.0 and Python 3.10, whereas the runtime environment operates on SUMO v1.27.1, TraCI v1.28.0, and Python 3.12. | **LOW** | Phase A / 23 |
| **7** | Duplicate documentation locations existed | **CONFIRMED STRUCTURAL.** Dual documentation existed at root `docs/` and subfolder `DOIM-Traffic-RL/docs/`. Both must be kept synchronized via automated mirroring. | **LOW** | Phase 14–25 |
| **8** | Manuscript claims exceeded available evidence | **CONFIRMED DEFECT.** Manuscript previously claimed "learned causal manifolds" (when SCOIL is predictive/associative) and reported diagnostic exploratory delays without distinguishing clearance overhead from asymptotic convergence. | **HIGH** | Phase 24 |

---

## 3. Detailed Component-by-Component Forensic Audit

### 3.1 Mathematical Formulations (DOIM & SCOIL)
- **The "Manifold" Claim:** In differential geometry, a manifold requires a locally Euclidean coordinate atlas. The current DOIM implementation projects raw interactions through $\tanh(\cdot)$ with unit diagonal $M_{kk} = 1.0$, defining a bounded open hypercube $(-1, 1)^{K(K-1)}$ in matrix space. Furthermore, preferences $\mathbf{w}_t \in \Delta^4$ reside on the standard 4-simplex, which possesses a well-defined Riemannian geometry under the Fisher-Rao metric. The mathematical formulation must formally define both the interaction manifold $\mathcal{M}_{\text{int}}$ and the simplex embedding rather than using "manifold" as an informal buzzword.
- **Differentiability & Gradient Flow:** SCOIL uses bilinear query-key dot products with asymmetric heads:
  $$\mathbf{M}_t^{\text{raw}} = \frac{\mathbf{Q}_t \mathbf{K}_t^\top}{\sqrt{d_k}} + \text{reshape}(\mathbf{W}_{\text{asym}}\phi(\mathbf{c}_t), [K, K])$$
  Gradients flow cleanly through autograd to the shared feature network and GCN/GRU encoders.

### 3.2 Core Algorithm (IA-MOPPO)
- **Action Attribution:** The policy surrogate loss must optimize $\mathbf{a}_t^{\text{prop}} \sim \pi_\theta$, while the environment transition dynamics and SCOIL predictive loss must receive $\mathbf{a}_t^{\text{exec}}$ filtered by the safety shield. Verified correct in `training/interaction_aware_ppo.py`.
- **Value Bootstrapping:** Truncated episodes at the maximum time limit must bootstrap $V(s_{t+1})$, while terminal absorbing states evaluate to $0$. Episode boundaries must not leak state across reset boundaries. Verified correct in `training/interaction_aware_ppo.py`.

### 3.3 Traffic Baselines
- **Fixed-Time Controller:** Correct deterministic dual-ring cycle.
- **Actuated Signal Controller:** Requires replacement of heuristic `num_veh == 0` check with genuine NEMA-standard passage timers, detector call tracking, unit extension intervals, and gap-out expiration.
- **Max-Pressure Controller:** Correctly computes upstream minus downstream queue differentials $\Delta q(l, m)$ to select active green phases maximizing network outflow.
- **Parameter-Matched MLP-PPO:** Neural baseline matched in capacity ($\sim 112\,\text{K}$ parameters) without spatial GCN or temporal GRU inductive biases.

### 3.4 Safety Shield
- Dynamic microscopic conflict matrix parsing from `grid3x3.net.xml`.
- Minimum green bound: $g_{\min} = 10\,\text{s}$.
- Maximum green bound: $g_{\max} = 60\,\text{s}$.
- Clearance phase: $t_{\text{yellow}} = 3\,\text{s}$ with multi-stage clearance pathfinding.
- Rapid fluttering rate: $0.0\%$ guaranteed.

---

## 4. Remediation Plan & Autonomous Roadmap

The identified defects will be resolved systematically across Phases 15 through 25:
- **Phase 15:** Mathematical rigorization of $\mathcal{M}_{\text{int}}$ and $\Delta^4$ with Fisher-Rao geometry.
- **Phase 16:** Reference mathematical operation tests and PPO numerical checks.
- **Phase 17:** Split-conformal calibration for objective transitions.
- **Phase 18:** Scenario expansion with arterial corridor topologies and incident models.
- **Phase 19:** Genuine NEMA passage timer gap-out actuated controller.
- **Phase 20:** Strict fail-closed eligibility enforcement in `statistical_analyzer.py`.
- **Phase 21–23:** Multi-seed confirmatory execution, profiling, and efficiency benchmarking.
- **Phase 24–25:** Manuscript reconstruction and final independent audit scorecard.
