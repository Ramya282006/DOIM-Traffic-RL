# DOIM-Traffic-RL: Research Defect Register

**Standard:** IEEE Transactions on Intelligent Transportation Systems (T-ITS) Q1 Code Integrity  
**Status:** All Critical (P0) and High (P1) Defects Identified, Classified, and Assigned Remediation Status.

---

## Defect Inventory

### DEF-01: Objective Configuration Fallback Bypass in `DOIMProcessor`
- **Severity:** P0 (Critical Integrity)
- **Module:** `DOIM-Traffic-RL/environment/doim_processor.py`
- **Description:** `DOIMProcessor` constructed `MultiObjectiveEngine()` and `ObjectiveNormalizer()` with empty parameter lists, discarding custom configuration dictionaries and falling back to unparameterized defaults during initialization and upon calling `.reset()`.
- **Root Cause:** Lack of dependency injection for `objective_config` in `DOIMProcessor` constructor.
- **Blast Radius:** Environment objective calculations could diverge from YAML configuration.
- **Status:** REMEDIATED. Added parameter `objective_config` to `DOIMProcessor` and `DOIMTrafficEnv`.

---

### DEF-02: Fairness Normalization Dynamic Range Squashing
- **Severity:** P0 (Critical Numerical)
- **Module:** `DOIM-Traffic-RL/configs/default.yaml`
- **Description:** Upper bound for fairness was set to `3000.0` in `normalization_bounds.upper`, whereas Jain delay disparity mathematically ranges within $[0.0, 1.0]$. This squashed normalized fairness values to $[0.0, 0.00033]$.
- **Root Cause:** Historical copy-paste error from an absolute waiting time bound.
- **Blast Radius:** The policy was blind to fairness urgency, receiving near-zero gradients.
- **Status:** REMEDIATED. Upper bound corrected to `1.0`.

---

### DEF-03: Multi-Tier Fairness Scope Conflation
- **Severity:** P0 (Scientific Validity)
- **Module:** `DOIM-Traffic-RL/objectives/fairness.py`
- **Description:** Per-intersection delay aggregates, approach delays, and vehicle waiting times were combined under ambiguous method names, conflating distinct spatial aggregation tiers.
- **Root Cause:** Incomplete specification of fairness population in earlier iterations.
- **Blast Radius:** Misleading fairness comparisons in statistical evaluations.
- **Status:** REMEDIATED. Clear three-tier hierarchy established: Tier 1 (Approach), Tier 2 (Intersection), Tier 3 (Vehicle).

---

### DEF-04: Bypassed Uncertainty Parameter in SCOIL Training Loss
- **Severity:** P0 (Mathematical Correctness)
- **Module:** `DOIM-Traffic-RL/training/interaction_aware_ppo.py`
- **Description:** `compute_predictive_loss()` supported an uncertainty parameter for heteroscedastic negative log-likelihood (NLL), but the PPO update loop called it without this argument, silently executing fallback MSE loss.
- **Root Cause:** Omission of `uncertainty=scoil_out["uncertainty"]` in rollout training loop.
- **Blast Radius:** The uncertainty prediction head received zero direct supervisory gradient.
- **Status:** REMEDIATED. Passed `uncertainty` tensor into `compute_predictive_loss()`.

---

### DEF-05: Conflated Requested vs. Executed Actions in Transition Dynamics
- **Severity:** P1 (Physical Consistency)
- **Module:** `DOIM-Traffic-RL/environment/sumo_env.py` & `training/interaction_aware_ppo.py`
- **Description:** When the safety shield intercepted an action (e.g. $a_{\text{prop}}=1 \to a_{\text{exec}}=0$), the transition model was supervised with $a_{\text{prop}}$, falsely training the dynamics network that requesting a phase transition leads to the state that resulted from maintaining the phase.
- **Root Cause:** Single action tensor used for both policy advantage estimation and environmental transition modeling.
- **Blast Radius:** Distorted auxiliary transition dynamics under active safety intervention regimes.
- **Status:** REMEDIATED. Separated $a_{\text{prop}}$ and $a_{\text{exec}}$ in rollout telemetry and buffer.

---

### DEF-06: Preference Smoothness Cross-Objective Dimensional Error
- **Severity:** P0 (Mathematical Correctness)
- **Module:** `DOIM-Traffic-RL/training/interaction_aware_ppo.py`
- **Description:** Preference smoothness was computed as `(weights[:, 1:] - weights[:, :-1])**2`, penalizing differences across adjacent objective dimensions (e.g. waiting time vs. queue length) rather than temporal transitions across consecutive timesteps.
- **Root Cause:** Indexing along feature dimension rather than temporal trajectory dimension.
- **Blast Radius:** Discouraged diverse preference weights across objectives, corrupting multi-objective trade-offs.
- **Status:** REMEDIATED. Restructured to $\frac{\sum M_t \|w_t - w_{t-1}\|_2^2}{\sum M_t}$ with detached target anchors and episode reset masking.

---

### DEF-07: Silent Continuation Following Critical Safety Faults
- **Severity:** P0 (Safety & Reliability)
- **Module:** `DOIM-Traffic-RL/environment/safety_shield.py` & `environment/sumo_env.py`
- **Description:** When no verified clearance path existed (Option C), the shield kept the current phase and logged an intervention, but the episode continued and could enter benchmark results as a valid completed run.
- **Root Cause:** Absence of a typed safety exception and run-invalidation flag.
- **Blast Radius:** Potential inclusion of corrupted or unverifiable signal states in benchmark evaluation tables.
- **Status:** REMEDIATED. Defined `SafetyFaultError`, added `is_valid_run = False` flag in `info`, and enforced immediate episode termination.

---

### DEF-08: Benchmark Eligibility Missing Safety Invalidation Filter
- **Severity:** P1 (Experimental Rigor)
- **Module:** `DOIM-Traffic-RL/analysis/statistical_analyzer.py`
- **Description:** `load_run_summaries()` did not explicitly filter out runs flagged with `safety_fault=True` or `is_valid_run=False`.
- **Root Cause:** Eligibility predicate checked only file existence, step count, and manifest schema.
- **Blast Radius:** Faulted runs could theoretically contaminate statistical comparison tables.
- **Status:** REMEDIATED. Added explicit check for `safety_fault` and `is_valid_run` in `load_run_summaries()`.

