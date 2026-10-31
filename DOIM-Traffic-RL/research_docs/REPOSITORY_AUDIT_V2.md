# DOIM-Traffic-RL: Comprehensive Repository Audit & Defect Register (Phase 0)

**Date:** October 31, 2026  
**Auditor:** Senior Research Scientist & IEEE T-ITS Reviewer  
**Repository State:** Branch `ramya` @ commit `68e953a` (Recovery snapshot: `recovery-pre-correction`)  
**Audit Standard:** Strict IEEE Transactions on Intelligent Transportation Systems (T-ITS) Scientific Rigor  

---

## 1. Executive Summary of Audit

The `DOIM-Traffic-RL` repository has established a substantial foundation: it contains the Dynamic Objective Interaction Manifold (DOIM), State-Conditioned Objective Interaction Learning (SCOIL), spatial GCN, temporal GRU, safety shielding, multiple baselines, and structured experiment logging.

However, a strict line-by-line inspection reveals critical architectural and scientific disconnects that must be resolved before the research can be considered mathematically sound and defensible:
1. **Reward Disconnect & Reward-Hacking Vulnerability (P0):** The environment calculates step rewards using the legacy `DOIMProcessor`, while `IA-MOPPO` computes a separate set of weights. Furthermore, unconstrained dynamic scalarization allows the preference network to manipulate its own optimization signal (e.g., zeroing out difficult objectives), leading to degenerate preference collapse.
2. **Frozen GNN Representation During Training (P0):** Spatial graph embeddings are computed under `torch.no_grad()` during rollout and stored as detached tensors. During PPO updates, `encode_spatial` is never invoked, meaning the Kipf-Welling GCN parameters receive zero gradient feedback during actual training.
3. **SCOIL Formulation & Evidential Uncertainty (P1):** The interaction matrix $M(s, z)$ lacks a clear non-causal mathematical definition, and the uncertainty head uses a naive single-layer Softplus that does not reflect true epistemic uncertainty. The auxiliary next-objective transition predictor lacks proper temporal alignment and action conditioning.
4. **Fairness Metric Inconsistency (P1):** Jain's index was calculated across 9 intersection aggregated delays and arbitrarily multiplied by 1000, while the paper claims vehicle-level fairness.
5. **SUMO Static Timer Preemption (P1):** SUMO's internal 42-second static phase timers can trigger transitions independent of the safety shield when the agent selects the "keep" action ($a=0$).
6. **Statistical Unit Integrity (P1):** The statistical analyzer must strictly treat distinct trained models across independent seeds as the statistical units, avoiding episode-level pooling.
7. **Scientific Humility & Manuscript Claims (P1):** Unsupported assertions of submission to IEEE and causal identification must be systematically excised.

---

## 2. Prioritized Defect Register

### DEF-A01 [Severity: P0] — Disconnected Reward and Degenerate Preference Risk
- **Affected Files:** `training/interaction_aware_ppo.py:406-415`, `environment/sumo_env.py:324-339`, `environment/doim_processor.py`.
- **Exact Technical Problem:**
  In `interaction_aware_ppo.py`, `agent.act()` outputs adaptive weights $\mathbf{w}_t^{\text{agent}}$, but `env.step(action)` returns a scalar reward computed using `self.latest_doim_result["weights"]` from the legacy `DOIMProcessor`. The agent updates its critic and policy on a reward computed by a completely different model.
  Furthermore, if the agent optimizes $R_t = \sum_k w_k(s_t) \cdot (-z_{k,t})$ end-to-end, the gradient with respect to $\mathbf{w}$ will trivially push the weights to $1$ for the objective with the lowest cost (smallest negative value) and $0$ for difficult objectives (e.g. queue and waiting time), causing **trivial preference collapse**.
- **Scientific Consequence:**
  Invalid optimization dynamics: the policy is conditioned on weights that do not govern the return, and naive scalarization invites severe reward hacking.
- **Required Correction:**
  1. Have `sumo_env.py` expose the raw objective vector $\mathbf{z}_t \in \mathbb{R}^K$.
  2. Define the multi-objective optimization problem mathematically. The preference head must be regularized against a balanced prior distribution (e.g., symmetric Dirichlet prior / cross-entropy anchor) to prevent degenerate preference collapse to the simplex boundaries.
  3. The policy optimization and preference generation must be clearly distinguished: the policy maximizes the scalarized return given current preferences $\mathbf{w}_t$, while the preference head adjusts weights based on regional urgency and objective trade-offs without collapsing.
- **Verification Test:**
  `test_preference_non_collapse`: Verify that under prolonged high-delay traffic, the preference vector does not collapse to an indicator vector (entropy $H(\mathbf{w}_t) > H_{\min}$).

---

### DEF-A02 [Severity: P0] — Detached GNN Representation in Rollout Buffer
- **Affected Files:** `training/interaction_aware_ppo.py:383-388, 428-433, 215-263`.
- **Exact Technical Problem:**
  During rollout, node features are extracted and passed through `with torch.no_grad(): agent.network.encode_spatial(...)`. The resulting pooled vector `new_spatial` is concatenated into `spatial_history` and stored in `RolloutBuffer` as a detached CPU tensor. During `agent.update()`, `spatial_hists` is fed directly into `encode_temporal(b_hist)` without ever invoking `encode_spatial`.
- **Scientific Consequence:**
  The spatial GCN layers (`gcn1`, `gcn2`) receive **zero gradients** during actual PPO training. The claimed spatial graph representation learning is entirely frozen.
- **Required Correction:**
  1. The `RolloutBuffer` must store the raw node feature history sequences $\mathbf{X}_{t-L+1:t} \in \mathbb{R}^{L \times N \times 5}$.
  2. During mini-batch updates in `agent.update()`, `self.network.encode_spatial` must be called on each step of the history sequence within the active autograd graph before passing to `encode_temporal`.
- **Verification Test:**
  `test_gnn_actual_training_gradient`: Check `param.grad is not None` and `torch.norm(param.grad) > 0` specifically for `network.gcn1.weight` and `network.gcn2.weight` after a live PPO update.

---

### DEF-A03 [Severity: P1] — SCOIL Formulation, Predictive Transition, and Evidential Uncertainty
- **Affected Files:** `models/interaction_learner.py:68-73, 125-137`, `models/doim_network.py:253-263`.
- **Exact Technical Problem:**
  1. The uncertainty head is a single linear layer with Softplus, which cannot estimate epistemic uncertainty without ensembling or evidential modeling.
  2. The auxiliary objective transition predictor in `doim_network.py` predicts $\mathbf{\hat{z}}_{t+1} = \text{MLP}([\mathbf{m}_t, \mathbf{a}_t])$, failing to condition on the current objective vector $\mathbf{z}_t$ and the interaction matrix $\mathbf{M}_t$.
  3. The paper claims "causal interaction learning", which is scientifically invalid without interventional calculus or DAG identifiability.
- **Scientific Consequence:**
  Unsubstantiated claims of causal discovery and Bayesian uncertainty; auxiliary loss fails to provide a physically grounded transition model.
- **Required Correction:**
  1. Re-frame $\mathbf{M}(s, z)$ as a **state-conditioned predictive cross-coupling operator** (directional sensitivity $\partial z_i / \partial z_j$), removing all causal claims.
  2. Formulate evidential variance estimation or Monte Carlo dropout to provide justified uncertainty for Bayesian shrinkage.
  3. Formulate the transition predictor as $\mathbf{\hat{z}}_{t+1} = \mathbf{z}_t + \Delta \mathbf{z}(s_t, \mathbf{z}_t, \mathbf{a}_t, \mathbf{M}_t)$, enforcing proper residual action-conditioned dynamics.
- **Verification Test:**
  `test_action_sensitive_transition_prediction`: Verify that changing action $\mathbf{a}_t$ alters $\mathbf{\hat{z}}_{t+1}$ with non-zero gradient.

---

### DEF-A04 [Severity: P1] — Fairness Metric Entity & Scale Inconsistency
- **Affected Files:** `objectives/objective_engine.py:73-76`, `objectives/fairness.py`, `research_docs/paper/MANUSCRIPT.md`.
- **Exact Technical Problem:**
  `objective_engine.py` computes Jain's fairness index across 9 intersection-aggregated delays, then multiplies by 1000 for ad-hoc scaling. The manuscript describes this as "vehicle-level fairness".
- **Scientific Consequence:**
  Intersection-level fairness conflated with vehicle-level equity; artificial scaling corrupts objective normalization bounds.
- **Required Correction:**
  1. Distinctly implement:
     - **Vehicle-level delay disparity:** $J_{\text{veh}} = \frac{(\sum_v w_v)^2}{|\mathcal{V}|\sum_v w_v^2}$ using vehicle waiting times from TraCI.
     - **Corridor-level approach fairness:** Jain's index across incoming arterial approaches.
  2. Retain disparity as $1.0 - J \in [0, 1]$ without multiplying by 1000.
  3. Accurately update documentation and manuscript.
- **Verification Test:**
  `test_vehicle_vs_intersection_fairness`: Verify numerical correctness and distinguish vehicle-level from approach-level fairness.

---

### DEF-A05 [Severity: P1] — SUMO Static Phase Timer Preemption
- **Affected Files:** `environment/sumo_env.py:280-293`, `environment/safety_shield.py`.
- **Exact Technical Problem:**
  When the agent selects action 0 (keep phase), no TraCI phase command is issued. SUMO's static XML program (`duration="42"`) continues running its internal clock. At step 42, SUMO automatically advances to yellow regardless of the RL agent's intended green extension.
- **Scientific Consequence:**
  The agent does not possess true variable-length green control; SUMO static timers preempt RL policies.
- **Required Correction:**
  Call `traci.trafficlight.setPhaseDuration(tid, 999999)` upon green phase activation to disable SUMO's default timer and grant exclusive control to the Safety Shield.
- **Verification Test:**
  `test_phase_duration_override`: Verify that green phases can be held past 42 seconds up to the $60\,\text{s}$ maximum green threshold without automatic SUMO preemption.

---

### DEF-A06 [Severity: P1] — Statistical Unit Independence in Evaluation
- **Affected Files:** `analysis/statistical_analyzer.py`.
- **Exact Technical Problem:**
  The statistical analyzer groups runs by algorithm name, risking the pooling of repeated runs from the same seed or single-run multiple evaluations as independent samples.
- **Scientific Consequence:**
  Violates statistical testing assumptions, artificially inflating degrees of freedom and invalidating p-values.
- **Required Correction:**
  Enforce strict grouping by independent training seeds ($N = N_{\text{seeds}}$). Compute paired Welch's t-test and Wilcoxon signed-rank tests only across models evaluated on identical seed pairings.
- **Verification Test:**
  `test_statistical_unit_independence`: Ensure analyzer rejects duplicate seed entries and validates true independent sample size.

---

### DEF-A07 [Severity: P1] — Unsupported Manuscript Claims
- **Affected Files:** `research_docs/paper/MANUSCRIPT.md`, `research_docs/FINAL_IMPLEMENTATION_REPORT.md`.
- **Exact Technical Problem:**
  Drafts claimed "Submitted to IEEE T-ITS", "Guaranteed Q1 publication", and "Causal objective interactions".
- **Scientific Consequence:**
  Violation of scientific integrity and non-negotiable instructions.
- **Required Correction:**
  Remove all submission assertions, label manuscript as "Working Research Draft", replace causal claims with predictive cross-coupling, and present honest empirical findings.
- **Verification Test:**
  Textual audit confirming zero instances of unverified claims.

---

## 3. Implementation Plan by Phase

- **Phase 1:** Fix the End-to-End Learning Architecture (Mathematical MOMDP formulation, anti-collapse preference regularization, full GNN autograd flow, training integration tests).
- **Phase 2:** Scientifically Strengthen SCOIL (Predictive cross-coupling, residual action-conditioned dynamics, evidential uncertainty, controlled matrix tests).
- **Phase 3:** Correct Objective Definitions and Metrics (True vehicle-level fairness, unscaled disparity, verified throughput tracking).
- **Phase 4:** Traffic Signal Safety Verification (TraCI phase duration override, automated transition verifier).
- **Phase 5:** Staged Experimental Validation Pipeline (Multi-seed runs, multiple traffic demand scenarios, controlled ablations).
- **Phase 6:** Statistical Validation (Independent training seed units, paired tests, effect sizes).
- **Phase 7:** Reproducibility & Software Quality (Testing suite, clean configs, no machine-specific paths).
- **Phase 8:** Regenerate Research Documentation & IEEE Manuscript Draft.
- **Phase 9:** Final Acceptance Criteria & Delivery.
