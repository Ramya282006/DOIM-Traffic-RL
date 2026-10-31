# PHASE 11 — EXPERIMENTAL AUDIT REPORT
**Research Project:** Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems  
**Repository:** DOIM-Traffic-RL  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (T-ITS) / Q1 Research  
**Auditor:** Independent Senior Research Engineer & Strict IEEE T-ITS Reviewer  
**Audit Commit:** `6882d3c6d1f118f1b501910fa55f7d9f3eac711f`  
**Current Branch:** `phase10-experimental-validation`  
**Date of Audit:** October 31, 2026  

---

## 1. Executive Summary

This Audit Report provides a comprehensive, rigorous examination of the experimental codebase, raw artifacts, logging mechanisms, statistical pipelines, safety verification infrastructure, and manuscript assertions of the `DOIM-Traffic-RL` repository.

Acting under the strict peer-review standards of IEEE Transactions on Intelligent Transportation Systems (T-ITS), this audit establishes:
1. **Mathematical Soundness:** The underlying DOIM and SCOIL architectures (`models/doim_network.py`, `models/interaction_learner.py`) possess mathematically sound gradient flow, autograd tape connectivity, and Bayesian shrinkage regularization.
2. **Current Experimental Scope:** The committed raw results consist exclusively of **128-step diagnostic exploration rollouts**. They do not constitute asymptotic or converged RL policies.
3. **Core Defect Identified:** The primary benchmark table in the manuscript (`MANUSCRIPT.md:Table I`) is titled as representing `north_south_surge`, but contains cross-scenario contaminated numbers ($70.57\,\text{s}$ for Fixed-Time, $97.57\,\text{s}$ for IA-MOPPO) that blended 3 distinct traffic demand scenarios.
4. **Baseline Asymmetry:** Fixed-Weight PPO and State-Adaptive PPO lack deterministic frozen evaluation functions, whereas IA-MOPPO has an explicit deterministic evaluation harness.
5. **Readiness Determination:** The repository is **not yet ready for submission to IEEE T-ITS**. However, the infrastructure is robust, passes 59/59 regression tests, and is ready for the staged Convergence Pilot (5,000 steps) once baseline evaluation symmetry and manifest backward-compatibility are resolved.

---

## 2. Repository & Branch State

- **Active Branch:** `phase10-experimental-validation`
- **Head Commit:** `6882d3c6d1f118f1b501910fa55f7d9f3eac711f`
- **Working Tree:** Clean (no uncommitted modifications on main source tree)
- **Execution Environment:** Python 3.11.9, PyTorch 2.15.0+cpu, Eclipse SUMO v1.28.0 (Windows 11 x64)
- **Regression Suite:** 59 items collected; 59 passed in $16.82\,\text{s}$ (`pytest tests/ -v`).

---

## 3. Detailed Component Inspections

### 3.1 Training Scripts and Configurable Budgets
- **Script:** `training/interaction_aware_ppo.py`
  - Function: `train_interaction_aware_ppo()`
  - Supported budgets: 128 (smoke), 5,000 (pilot), 10,000 (intermediate), 50,000 (main convergence).
  - Checkpoint interval: Configurable (defaults to 2,500 steps). Saves `model_step_<N>.pt` containing `network_state_dict`, `scoil_state_dict`, `optimizer_state_dict`, `step`.
  - Observation: Periodic checkpointing was verified to save and restore model weights with bitwise exactness ($100\%$ tensor match, verified by `scratch/phase11_validation_harness.py`).
  
### 3.2 Baseline Implementations & Parity
- **Fixed-Time Controller (`baselines/fixed_time/fixed_time_controller.py`):**
  - Uses a deterministic 60-second cycle (27s green N-S, 3s yellow, 27s green E-W, 3s yellow).
  - Cycle time is well-justified for the 3x3 grid network dimensions (link length $283.2\,\text{m}$, speed limit $13.89\,\text{m/s}$).
  - Limitation: Defaults `evaluation_mode` to `"train_rollout"` instead of `"deterministic_eval"`.
- **Fixed-Weight PPO (`baselines/ppo/fixed_weight_ppo.py`):**
  - Architecture: Identical GCN + GRU backbone, identical action heads, fixed weights $\mathbf{w} = [0.2, 0.2, 0.2, 0.2, 0.2]^\top$.
  - Defect (DEF-P11-05): Contains only a training rollout loop (`deterministic=False`). Lacks a dedicated `evaluate_fixed_weight_ppo()` function.
- **State-Adaptive PPO (`baselines/ppo/state_adaptive_ppo.py`):**
  - Architecture: Direct state-to-preference MLP $\mathbf{w}_t = \text{Softmax}(\mathbf{W}_s [\mathbf{h}_t^{\text{temp}}, \mathbf{s}_t] + \mathbf{b}_s)$, bypassing the interaction manifold.
  - Defect (DEF-P11-05): Lacks a dedicated `evaluate_state_adaptive_ppo()` function.

### 3.3 Evaluation Scripts
- **Modern Evaluation (`training/interaction_aware_ppo.py:577-670`):**
  - `evaluate_interaction_aware_ppo()` sets `agent.network.eval()`, `agent.interaction_learner.eval()`, uses `torch.no_grad()`, and enforces `deterministic=True` (argmax actions).
  - Defect (DEF-P11-03): Passes `evaluation_mode="deterministic_evaluation"`, which fails schema validation against `common/experiment.py` (`valid_modes = {"train_rollout", "deterministic_eval", "stochastic_eval", "smoke_test"}`).
- **Legacy Script (`training/evaluate_multiseed.py`):**
  - Still references `stable_baselines3.PPO` and `.zip` checkpoints. Marked as legacy and superseded by `training/interaction_aware_ppo.py`.

### 3.4 Experiment Manifest & Logging System
- **Harness (`common/experiment.py`):**
  - Provides per-step CSV streaming (`metrics_step.csv`), configuration snapshotting (`config_snapshot.yaml`), and run manifests (`manifest.json`).
  - Defect (DEF-P11-02): Enforces 12 mandatory fields. Historical runs in `results/runs/` use legacy keys, causing `validate_experiment_manifest()` to fail on 100% of historical runs.

### 3.5 Statistical Analyzer
- **Implementation (`analysis/statistical_analyzer.py`):**
  - Implements scenario isolation (`aggregate_by_scenario_and_algorithm()`), intra-model evaluation averaging (seed as statistical unit), Student's t-test, Wilcoxon signed-rank test, Welch's t-test, and Holm-Bonferroni correction.
  - Defect (DEF-P11-01): The committed file `results/tables/benchmark_comparison.csv` was generated prior to Phase 10 scenario isolation and mixed all scenarios into a single row per algorithm.
  - Defect (DEF-P11-04): `experiments/run_comprehensive_matrix.py:117` calls `StatisticalAnalyzer(runs_dir=output_root)`, which raises `TypeError` because the parameter name is `runs_root`.

### 3.6 Safety Verifier and SUMO Signal Programs
- **Network XML (`environment/network/research_3x3/grid.net.xml`):**
  - Controllable TLS: `A1`, `B0`, `B1`, `B2`, `C1`.
  - Link count: 12 links for `A1`, `B0`, `B2`, `C1`; 20 links for `B1`.
  - Phase structure: Every controllable intersection has a 4-phase static program (Green 1: 42s, Yellow 1: 3s, Green 2: 42s, Yellow 2: 3s).
  - Geometric conflict validation: Junction `<request foes="...">` bitmasks confirmed zero conflicting major greens ($G+G = 0$) in all legal phases.
- **Safety Shield (`environment/safety_shield.py`):**
  - Enforces minimum green ($10\,\text{s}$), maximum green ($60\,\text{s}$), and yellow clearance ($3\,\text{s}$).
  - During 128-step exploration, the shield intercepted 188 to 226 unsafe switching requests per run, completely preventing illegal transitions.
  - Defect (DEF-P11-08): Hardcodes a cyclic 4-phase structure (`(curr + 1) % 4` for yellow, `(curr + 2) % 4` for green).

---

## 4. Verification of Specific Research Directives

| Audit Requirement | Verification Status | Exact Evidence |
| :--- | :---: | :--- |
| **All algorithms use comparable training budgets** | **PARTIAL** | In 128-step smoke runs, all algorithms ran for 128 steps. However, Fixed-Time executed 128 non-exploratory steps, while RL baselines performed random exploration. Longer budgets (5k, 50k) have not yet been executed. |
| **Training seeds and evaluation seeds are independently controlled** | **VERIFIED** | Centralized PRNG manager (`common/seeding.py`) uses NumPy `SeedSequence` to derive statistically independent evaluation seeds ($r < 0.01$). |
| **Evaluation uses frozen models without updating parameters** | **VERIFIED (IA-MOPPO)**<br>**FAILED (Baselines)** | `evaluate_interaction_aware_ppo()` enforces `eval()`, `torch.no_grad()`, and argmax actions. FW-PPO and SA-PPO lack frozen evaluation loops entirely. |
| **Repeated evaluation episodes are averaged within each trained model** | **VERIFIED** | `StatisticalAnalyzer.aggregate_by_scenario_and_algorithm()` groups repeated evaluation episodes by `train_seed` before cross-model statistics. |
| **Statistics are calculated using independent trained models as units** | **VERIFIED** | Degrees of freedom ($N$) in hypothesis testing strictly equals the count of independent training seeds ($N=5$). |
| **Different traffic scenarios are never mixed** | **FAILED (Committed Table)**<br>**VERIFIED (Analyzer Engine)** | The active analyzer supports strict scenario isolation, but committed table `benchmark_comparison.csv` and Manuscript Table I contain cross-scenario blended numbers. |
| **Missing or incomplete runs are not silently treated as completed** | **VERIFIED** | `validate_experiment_manifest()` strictly checks `completion_status == "COMPLETED"`, rejecting unverified or crashed runs. |
| **Run manifests identify exact source commit and configuration** | **VERIFIED** | Manifest records `git_sha` (or `code_commit`), `config_hash`, and model SHA-256 checksum. |
| **Checkpoints can be resumed correctly** | **VERIFIED** | Validated via `scratch/phase11_validation_harness.py`: restored model weights matched to 16 decimal places and produced identical deterministic actions. |
| **All manuscript results match raw experiment artifacts** | **FAILED** | Manuscript Table I claims to report `north_south_surge` but cites blended numbers ($70.57\,\text{s}$ vs $97.57\,\text{s}$). The true `north_south_surge` values in raw artifacts are $55.13\,\text{s}$ and $82.19\,\text{s}$. |

---

## 5. Peer-Review Verdict & Recommendations

### Verdict: REVISE BEFORE EXPERIMENTAL LAUNCH
The repository exhibits excellent engineering rigor in its core neural and traffic dynamics modules, but suffers from documentation contamination, manifest schema mismatch, and baseline evaluation asymmetry.

### Action Items Before Launching 5,000-Step Pilot:
1. Fix schema compatibility in `common/experiment.py` and `analysis/statistical_analyzer.py` (DEF-P11-02).
2. Fix evaluation mode string mismatch (DEF-P11-03).
3. Fix matrix runner keyword argument (DEF-P11-04).
4. Implement frozen deterministic evaluation functions for Fixed-Weight PPO and State-Adaptive PPO (DEF-P11-05).
5. Regenerate scenario-isolated benchmark tables and correct Manuscript Table I (DEF-P11-01).
6. Proceed to Phase B validation and execute Phase C (Convergence Pilot) only after user approval.
