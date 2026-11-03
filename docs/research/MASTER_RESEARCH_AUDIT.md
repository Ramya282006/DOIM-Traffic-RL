# DOIM-Traffic-RL: Master Research Audit & Defect Register
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Evaluation Standard:** Strict Q1 Scientific, Mathematical, and Empirical Verification  
**Date:** November 2026  
**Audited Repository Commit:** `406bc1f86a78e04393e2ef4e67fb5dc28d2aba3a`  
**Master Audit Specification:** `DOIM_Traffic_RL_Research_Upgrade_Roadmap (1).docx`

---

## 1. Executive Summary

This master audit provides a comprehensive, line-by-line inspection of the `DOIM-Traffic-RL` repository. The project implements a multi-intersection traffic signal control system in SUMO with five competing objectives: total waiting time, total queue length, CO₂ emissions, throughput, and corridor delay fairness.

While foundational components for Spatial GNN, Temporal GRU, Dynamic Objective Interaction Manifold (DOIM), and State-Conditioned Objective Interaction Learning (SCOIL) exist, our audit identified critical scientific, mathematical, and empirical gaps that must be remediated to meet the standards of IEEE T-ITS.

---

## 2. System Architecture & Internal Dependency Flow

```
+--------------------------------------------------------------------------------------------------+
|                                    SUMO Simulation Environment                                   |
| - 3x3 Road Network (9 Intersections: A0-A2, B0-B2, C0-C2; 5 Controllable TLS: A1, B0, B1, B2, C1) |
+--------------------------------------------------------------------------------------------------+
                                        |
       +--------------------------------+--------------------------------+
       | Raw Node Features X_t [9, 5]                                    | Raw Detector Telemetry
       v                                                                 v
+-----------------------------+                           +----------------------------------------+
| 2-Layer Spatial GNN (GCN)   |                           | Unified Multi-Objective Engine         |
| X_spatial [9, 64]           |                           | 1. Waiting Time (veh·s)                |
+-----------------------------+                           | 2. Queue Length (vehicles)             |
       | Mean Pooling                                     | 3. Integrated CO2 (mg)                 |
       v                                                  | 4. Step Throughput (completed veh)     |
+-----------------------------+                           | 5. Corridor Fairness Disparity [0, 1]  |
| Temporal GRU Encoder        |                           +----------------------------------------+
| h_t ∈ ℝ^64 (traffic context)|                                          | Raw Vector z_raw ∈ ℝ^5
+-----------------------------+                                          v
       |                                                  +----------------------------------------+
       +--------------------+                             | Authoritative Objective Normalizer     |
                            |                             | Urgency vector z_t ∈ [0, 1]^5          |
                            v                             +----------------------------------------+
              +-----------------------------------+                      |
              | SCOIL Interaction Learner         |<---------------------+
              | - Bilinear Query-Key (Q·K^T)      |
              | - Asymmetric Residual Head        |
              | - Predictive Uncertainty U_t      |
              | - Adaptive Shrinkage to Prior     |
              | -> Interaction Matrix M_t [5, 5]  |
              +-----------------------------------+
                                |
                                v
+--------------------------------------------------------------------------------------------------+
| Dynamic Objective Interaction Manifold (DOIM) Fusion Core                                         |
| m_t = Tanh( LayerNorm( W_m [ h_t (64) || z_t (32) || vec(M_t) (32) ] ) ) ∈ ℝ^64                  |
+--------------------------------------------------------------------------------------------------+
        |                                                 |
        v                                                 v
+-------------------------------+             +----------------------------------------------------+
| Adaptive Preference Head      |             | Multi-Discrete Actor-Critic Policy                 |
| w_t = Softmax(g(m_t) / tau)   |             | Actor: [m_t, raw_state] -> Logits per TLS [5, 2]   |
| ∈ Δ^4 (5-objective simplex)   |             | Critic: [m_t, raw_state] -> Baseline Value V(s)    |
+-------------------------------+             +----------------------------------------------------+
        |                                                         |
        +-----------------------+                                 | Candidate Raw Actions a_t ∈ {0, 1}^5
                                |                                 v
                                |             +----------------------------------------------------+
                                |             | Traffic Signal Safety Shield & Verifier            |
                                |             | - Min/Max green timing verification                |
                                |             | - Mandatory yellow / all-red clearance intervals   |
                                |             | - Permissive vs priority conflict verification     |
                                |             | - Reachable all-red fail-safe pathfinding          |
                                |             | - Explicit typed safety fault & abort if no path   |
                                |             +----------------------------------------------------+
                                |                                 | Verified Safe Actions
                                v                                 v
+--------------------------------------------------------------------------------------------------+
| Dynamic Reward Scalarization: r_t = - ∑ w_{t, k} · z_{t, k}                                      |
| Live Simulation Step Execution in SUMO -> Next State Observation & Verification Log              |
+--------------------------------------------------------------------------------------------------+
```

---

## 3. Prioritized Issue Register

| Issue ID | Module & Function | Severity | Technical Defect | Scientific Consequence | Required Remediation | Verification Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ISSUE-01** | `training/interaction_aware_ppo.py`: `update()` | **Critical** | Dimension error in preference smoothness: `torch.mean((weights[:, 1:] - weights[:, :-1])**2)` penalizes adjacent objective dimensions instead of temporal changes across time. | Distorts the learned preference simplex by arbitrarily forcing adjacent objective weights to be equal, while failing to penalize temporal weight chatter. | Implement temporal smoothness $L_{smooth} = \text{mean}(\|w_t - w_{t-1}\|^2_2)$ masking episode boundaries with detached historical anchors. | Unit test with known numerical sequences confirming objective dimension invariance and temporal oscillation penalty. |
| **ISSUE-02** | `configs/default.yaml` & `objectives/objective_normalizer.py` | **Critical** | Config bounds list `upper: [600, 20, 120000, 20, 3000.0]`, mixing old variance bounds ($s^2$) with new normalized Jain disparity in $[0, 1]$. | Crushes normalized fairness urgency by a factor of 3,000, reducing fairness weight influence to zero and inducing policy collapse. | Unify into an authoritative typed `ObjectiveSpecification` schema with exact unit, direction, and calibrated range $[0, 1]$ for disparity. | Schema validation test enforcing bound parity across YAML, engine, normalizer, and reward modules. |
| **ISSUE-03** | `environment/safety_shield.py`: `filter_action()` | **Critical** | Option C (no verified safe transition exists) sets `cannot_safely_progress` but leaves the current phase active without raising an explicit fault or aborting the episode. | The controller silently stays in a potentially unsafe state, violating safety integrity and masquerading invalid runs as successful. | Raise a typed `SafetyFaultError`, abort the affected simulation episode, record full signal state, and mark run invalid for benchmark tables. | Test verifying immediate episode abortion and typed exception when conflicting transitions lack reachable all-red paths. |
| **ISSUE-04** | `models/interaction_learner.py`: `forward()` | **Major** | Softplus output head is termed "Epistemic Uncertainty", but a deterministic single feed-forward network cannot estimate epistemic model uncertainty. | Scientifically ungrounded claims in manuscript; reviewers will reject uncalibrated "epistemic" uncertainty assertions. | Rebrand output as nonnegative interaction uncertainty/dispersion score and implement heteroscedastic predictive variance estimation. | Calibration and prediction error correlation test on held-out demand perturbations. |
| **ISSUE-05** | `objectives/fairness.py` & `objectives/objective_engine.py` | **Major** | Conflation of intersection-level disparity with vehicle-level and corridor-level fairness. | Misleading fairness claims in manuscript where network-aggregate variance was described as vehicle equity. | Explicitly separate vehicle-level disparity ($1 - J_{veh}$), corridor-level disparity, and intersection disparity; report all complementary metrics. | Hand-computed test cases verifying exact numerical Jain index, disparity, Gini, and variance calculations. |
| **ISSUE-06** | `common/experiment.py` & `experiments/run_comprehensive_matrix.py` | **Major** | Run manifests lack strict separation between smoke tests, training rollouts, and deterministic evaluation benchmarks. | Potential risk of mixing exploratory or stochastic runs into final benchmark comparison tables. | Enforce strict manifest classification: only validated `deterministic_eval` runs with immutable provenance enter primary benchmark tables. | Test verifying manifest rejection of incomplete, mismatched, or non-deterministic benchmark rows. |
| **ISSUE-07** | `baselines/fixed_time/`, `baselines/ppo/`, `training/` | **Major** | Inconsistent evaluation horizons across baseline algorithms and lack of matched budget qualification. | Baseline comparisons become methodologically unfair if evaluation steps or reset semantics differ. | Enforce strict common evaluation horizon (`eval_horizon = 900` or `1800`), matched seeds, and independent model instantiation. | Test verifying identical horizon propagation and independent checkpoint hashes across all 4 algorithms. |
| **ISSUE-08** | `models/spatial_encoder.py` | **Minor** | Fixed 3x3 grid edge index assumed implicitly; potential indexing vulnerability if network topology expands. | Limits generalization claims to the single 3x3 synthetic grid without explicit topology parameterization. | Formalize dynamic edge index construction from arbitrary SUMO network XML connections. | Unit test verifying correct adjacency matrix extraction from SUMO `.net.xml`. |
| **ISSUE-09** | `research_docs/paper/MANUSCRIPT.md` | **Major** | Manuscript equations diverge from implemented code (e.g., preference regularization, uncertainty definitions, objective bounds). | Immediate desk rejection at IEEE T-ITS for discrepancies between mathematical formulation and experimental code. | Rewrite manuscript from scratch using verified mathematical formulations, actual empirical results, and strict IEEE formatting. | Automated test checking symbol and equation consistency between code docstrings and manuscript text. |
| **ISSUE-10** | `analysis/statistical_analyzer.py` | **Major** | Aggregation across repeated episodes of the same model without verifying independent training seed replication. | Pseudoreplication invalidates statistical significance tests (e.g. claiming $N=50$ when only 5 seeds were trained). | Use independent training seeds as primary statistical unit; apply paired Wilcoxon/t-tests, 95% CIs, Cohen's d, and FDR corrections. | Statistical tests verifying independent seed grouping and proper degree-of-freedom calculations. |

---

## 4. Verification & Audit Sign-Off

All findings in this register are derived directly from the active codebase at commit `406bc1f86a78e04393e2ef4e67fb5dc28d2aba3a`. Remediation must be executed strictly through executable code updates, regression tests, and verifiable experimental artifacts.

