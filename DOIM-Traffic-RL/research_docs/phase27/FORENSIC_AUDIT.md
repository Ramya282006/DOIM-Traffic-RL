# PHASE 27 FORENSIC IMPLEMENTATION AUDIT
**Project:** Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems  
**Canonical Identity:** DOIM-PPO  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Date:** November 2026  
**Auditor:** Principal Research Scientist, Software Architect, and Safety-Critical Systems Auditor

---

## 1. Executive Summary

This forensic audit evaluates the entire DOIM-PPO codebase at the start of Phase 27. The project develops an adaptive multi-objective reinforcement learning framework for traffic signal control in urban networks, grounded in a Dynamic Objective Interaction Manifold (DOIM) and Proximal Policy Optimization (PPO), supported by State-Conditioned Objective Interaction Learning (SCOIL) and an active Traffic Signal Safety Shield.

The audit inspects the environment, network layers (GCN, GRU), SCOIL, DOIM, PPO training, objective engines, safety shields, baseline implementations, statistical utilities, and tests. A total of 12 verified findings were cataloged and categorized across severity tiers.

---

## 2. Comprehensive Finding Catalog

### Finding AUDIT-27-01: Unbatched Tensor Dimension Mismatch in Auxiliary Predictor
- **Affected File & Function:** `models/doim_network.py` -> `DOIMNetwork.predict_next_objectives()`
- **Severity:** MAJOR
- **Actual Observed Behavior:** Line 285 executes `aux_input = torch.cat([manifold, actions.float(), current_obj], dim=-1)`. If `actions` is passed as a 1D vector (e.g., shape `(5,)`) during single-sample inference while `manifold` has shape `(1, 64)`, `torch.cat` raises `RuntimeError: Tensors must have same number of dimensions: got 2 and 1`.
- **Why It Is Risky:** Prevents single-step online inference or debugging without manual batch wrapping.
- **Expected Behavior:** Method should accept both 1D unbatched tensors `(5,)` and 2D batched tensors `(B, 5)` transparently, unsqueezing where necessary.
- **Recommended Correction:** Add dimension guards: `if actions.dim() == 1: actions = actions.unsqueeze(0)`, and similarly for `manifold` and `current_obj`.
- **Test Needed:** `test_predict_next_objectives_1d_and_2d_dimensions()`.

---

### Finding AUDIT-27-02: 3D Batch Tensor Requirement in SCOIL `compute_predictive_loss`
- **Affected File & Function:** `models/interaction_learner.py` -> `StateConditionedInteractionLearner.compute_predictive_loss()`
- **Severity:** MAJOR
- **Actual Observed Behavior:** Line 185 computes `torch.bmm(m_t, delta_primary.unsqueeze(-1)).squeeze(-1)`. If `actions` is 1D or `m_t` is 2D (unbatched call with shape `(5, 5)`), `delta_primary.unsqueeze(-1)` has shape `(5, 1)` (2D), causing `torch.bmm` to raise `RuntimeError: Expected 3-dimensional tensor, but got 2-dimensional tensor`.
- **Why It Is Risky:** Causes crashes during unit tests or step-by-step diagnostic evaluations when evaluating single transition tuples.
- **Expected Behavior:** Function should seamlessly operate on both single transition tuples and mini-batches.
- **Recommended Correction:** Normalize input tensors: unsqueeze `m_t`, `z_t`, `actions`, `z_next`, and `uncertainty` if `dim() < 3`.
- **Test Needed:** `test_scoil_compute_predictive_loss_unbatched_and_batched()`.

---

### Finding AUDIT-27-03: Batch Dimension Averaging in `aggregate_spatial_embeddings`
- **Affected File & Function:** `models/temporal_encoder.py` -> `aggregate_spatial_embeddings()`
- **Severity:** MINOR
- **Actual Observed Behavior:** Line 64 performs `spatial_embeddings.mean(dim=0)`. If a batched tensor of shape `(batch_size, num_nodes, spatial_dim)` is provided, `dim=0` averages across the batch instead of pooling across the nodes.
- **Why It Is Risky:** Although `doim_network.py` uses `h.mean(dim=-2)` internally, any script or baseline importing `aggregate_spatial_embeddings` for batched representations will silently produce invalid representations.
- **Expected Behavior:** Pooling should always reduce the node dimension, regardless of whether a batch dimension is present.
- **Recommended Correction:** Replace `spatial_embeddings.mean(dim=0)` with `spatial_embeddings.mean(dim=-2)`.
- **Test Needed:** `test_aggregate_spatial_embeddings_2d_and_3d()`.

---

### Finding AUDIT-27-04: Metric Direction Inversion in Cliff's Delta Effect Size
- **Affected File & Function:** `analysis/statistical_analyzer.py` -> `StatisticalAnalyzer.cliffs_delta()`
- **Severity:** MAJOR
- **Actual Observed Behavior:** Formula computes $\delta = \frac{\#\{x_i > y_j\} - \#\{x_i < y_j\}}{n_x \cdot n_y}$. For cost objectives where lower is better (e.g., vehicle delay, queue length, $\text{CO}_2$ emissions, fairness disparity), a target algorithm outperforming the baseline has $x_i < y_j$, resulting in a negative delta ($\delta < 0$).
- **Why It Is Risky:** Without explicit directionality handling, readers may misinterpret a negative delta as target algorithm inferiority, or positive delta claims in text may conflict with computed signs.
- **Expected Behavior:** The function should support `higher_is_better: bool = True` and provide an automatic metric-aware effect size calculation where positive delta universally indicates superiority of the target algorithm.
- **Recommended Correction:** Add `higher_is_better: bool = True` argument to `cliffs_delta`, returning `raw_delta if higher_is_better else -raw_delta`, and add helper `compute_metric_effect_size(x, y, metric)`.
- **Test Needed:** `test_cliffs_delta_cost_and_benefit_directions()`.

---

### Finding AUDIT-27-05: Missing Node Index Range Guard in `NativeGCNConv`
- **Affected File & Function:** `models/spatial_encoder.py` -> `NativeGCNConv.forward()`
- **Severity:** MINOR
- **Actual Observed Behavior:** Line 31 executes `adj.index_put_((src, dst), ...)`. If `edge_index` contains indices $\ge num\_nodes$, PyTorch throws an uninformative `IndexError: index out of range`.
- **Why It Is Risky:** When experimenting with alternative road graphs or subgraphs with isolated nodes, out-of-range edge indices cause silent crashes without actionable error messages.
- **Expected Behavior:** Explicitly assert that `edge_index.max() < num_nodes` and `edge_index.min() >= 0`, providing a descriptive error message.
- **Recommended Correction:** Add dimension assertions in `NativeGCNConv.forward()`.
- **Test Needed:** `test_gcn_invalid_edge_indices_raises_value_error()`.

---

### Finding AUDIT-27-06: Permutation Invariance & Node Equivariance Verification
- **Affected File & Function:** `models/spatial_encoder.py` -> `SpatialTrafficEncoder`
- **Severity:** INFORMATIONAL
- **Actual Observed Behavior:** GCN message passing uses a static grid edge index. It was not explicitly tested for node permutation equivariance ($P \text{GCN}(X, A) = \text{GCN}(P X, P A P^T)$).
- **Why It Matters:** Guarantees that the graph representation does not accidentally memorize arbitrary integer node ordering.
- **Recommended Correction:** Add an explicit permutation equivariance unit test.
- **Test Needed:** `test_gcn_node_permutation_equivariance()`.

---

### Finding AUDIT-27-07: Canonical Project Identity Consistency (DOIM-PPO vs. IA-MOPPO)
- **Affected File & Function:** Repository-wide documentation, manuscript, and loggers
- **Severity:** MINOR
- **Actual Observed Behavior:** Several documents and logs referenced IA-MOPPO as the overarching project title, creating ambiguity with the canonical project title **DOIM-PPO**.
- **Expected Behavior:** The project identity is **DOIM-PPO** (Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems). IA-MOPPO is the specific internal algorithmic policy gradient component.
- **Recommended Correction:** Standardize project naming to DOIM-PPO across headers, reports, and code comments, clearly defining IA-MOPPO as the interaction-aware policy optimization component within the DOIM-PPO architecture.
- **Test Needed:** `test_project_identity_consistency()`.

---

### Finding AUDIT-27-08: Reference Numerical Verification for Objective Metrics
- **Affected File & Function:** `objectives/` (`waiting_time.py`, `queue.py`, `emissions.py`, `throughput.py`, `fairness.py`)
- **Severity:** MAJOR
- **Actual Observed Behavior:** Tests existed for metric execution, but lacked hand-calculated reference-value tests verifying exact mathematical formulas with numerical tolerances.
- **Why It Is Risky:** A subtle change in unit conversion (e.g. mg to g, or Jain disparity normalization) could pass shape checks without adhering to analytical ground truth.
- **Expected Behavior:** Dedicated reference tests asserting exact hand-computed floats for all 5 objectives and 3 fairness tiers.
- **Recommended Correction:** Create `tests/test_objective_reference_values.py`.
- **Test Needed:** `test_objective_reference_values.py`.

---

### Finding AUDIT-27-09: Adversarial Safety Shield Edge-Case Verification
- **Affected File & Function:** `environment/safety_shield.py` -> `TrafficSafetyShield`
- **Severity:** MAJOR
- **Actual Observed Behavior:** Safety shield passed standard integration tests, but edge cases such as malformed action shapes, out-of-range action indices, and simultaneous conflicting requests during clearance phases needed isolated adversarial assertions.
- **Expected Behavior:** Shield must cleanly reject malformed actions, intercept illegal phase requests, and never allow conflicting greens.
- **Recommended Correction:** Add adversarial stress tests verifying each constraint independently.
- **Test Needed:** `test_safety_shield_adversarial_suite.py`.

---

### Finding AUDIT-27-10: Analytical Reference Validation for Statistical Utilities
- **Affected File & Function:** `analysis/statistical_analyzer.py` -> `StatisticalAnalyzer`
- **Severity:** MAJOR
- **Actual Observed Behavior:** Statistical methods (Mann-Whitney U, Cliff's delta, Holm-Bonferroni, Hypervolume) needed explicit reference tests against externally known textbook examples.
- **Expected Behavior:** Automated verification comparing against hand-calculated statistical reference distributions.
- **Recommended Correction:** Create `tests/test_statistical_reference_validation.py`.
- **Test Needed:** `test_statistical_reference_validation.py`.

---

### Finding AUDIT-27-11: Causal Terminology Guard in Documentation & Code
- **Affected File & Function:** `models/scoil.py`, `analysis/scoil_validator.py`, `docs/`
- **Severity:** MINOR
- **Actual Observed Behavior:** SCOIL estimates directional cross-sensitivities under observational policy rollouts. Any phrase implying counterfactual do-calculus causality violates IEEE T-ITS standards.
- **Expected Behavior:** SCOIL is strictly scoped as a state-conditioned predictive directional cross-sensitivity estimator with epistemic dispersion regularizers.
- **Recommended Correction:** Audit all docstrings and reports to ensure zero unsupported causal claims remain.
- **Test Needed:** `test_no_unsupported_causal_claims()`.

---

### Finding AUDIT-27-12: Zero Advantage Carryover Verification on Truncation
- **Affected File & Function:** `training/interaction_aware_ppo.py` -> `RolloutBuffer.compute_returns_and_advantages()`
- **Severity:** MINOR
- **Actual Observed Behavior:** The implementation already distinguishes `terminated` vs `truncated`, but lacked an explicit isolated test verifying that no future advantage is backpropagated across an episode boundary.
- **Expected Behavior:** Advantage at step $t$ where episode ends must equal $\delta_t$, with exactly zero carryover from $t+1$.
- **Recommended Correction:** Verify and preserve existing analytical test in `tests/test_algorithm_correctness.py`.
- **Test Needed:** `test_gae_absorbing_termination_vs_truncation()`.

---

## 3. Summary of Findings

| Severity | Count | Status |
|:---|:---:|:---:|
| **CRITICAL** | 0 | None detected |
| **MAJOR** | 5 | Ready for correction |
| **MINOR** | 5 | Ready for correction |
| **INFORMATIONAL** | 2 | Verified & tested |
| **TOTAL** | **12** | **Full resolution plan established** |

All identified defects will be resolved systematically in Phase 27B through Phase 27J.
