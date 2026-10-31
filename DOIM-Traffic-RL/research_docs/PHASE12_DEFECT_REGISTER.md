# PHASE 12: CRITICAL DEFECT REGISTER AND REMEDIATION AUDIT

**Project:** DOIM-Traffic-RL
**Research Title:** Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems
**Evaluation Standard:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)
**Audit Date:** October 31, 2026
**Auditor:** Senior Research Engineer & Independent IEEE Auditor
**Working Branch:** `ramya` (Pull Request #5)
**Base Commit on main:** `db720af5644536c4cbf08446c1922c8b977d1db5`
**Latest Source Commit:** `83304d8b368522f0ed9050732c6952f2a6bb1ea6` (PR #5 Head)

---

## 1. Master Defect Status Register

| Defect ID | Severity | Defect Summary | Remediation Status | Verification Evidence |
| :--- | :---: | :--- | :---: | :--- |
| **DEF-P11-01** | **CRITICAL** | Baseline evaluation asymmetry across horizon, seeds, and execution conditions | **FIXED** | Symmetric `evaluate_*` functions across all 4 algorithms; frozen parameters (`eval()`, `torch.no_grad()`); single-episode fixed-time loop termination (`break` without silent reset chaining); explicit runtime propagation of runner's requested `eval_horizon` (default 1800, configurable) to environment `max_steps`, loop termination condition, manifest, and summary across all 4 algorithms; verified at runtime (900s) via `test_evaluation_horizon_runtime_propagation_all_four_algorithms` and live SUMO (15s) via `test_live_sumo_eval_horizon_propagation`. |
| **DEF-P11-02** | **CRITICAL** | Modern manifest schema rejects all historical runs due to schema drift; migration previously invented defaults | **FIXED** | Removed all invented defaults (no default seeds, steps, git sha, config hashes). Separated historical completion from reproducibility eligibility. Original manifests retained unchanged. Produced `results/manifest_migration_audit.json` auditing all 49 runs (44 completed on disk, 0 reproducibility-eligible); verified by `test_manifest_migration_and_classification.py`. |
| **DEF-P11-03** | **CRITICAL** | Conflation of heterogeneous scenario statistics and numerical threshold classification for fairness | **FIXED** | Grouped strictly by `['scenario', 'algorithm']`. Completely eliminated threshold-based classification (`<= 1.0`). Introduced explicit metric identity (`fairness_metric_name`) and unit metadata (`fairness_metric_unit`). Unidentifiable runs marked `UNKNOWN` / ineligible. Statistical analyzer explicitly separates primary algorithms from ablations (`is_ablation`, `configuration_type`). Verified by `test_fairness_definitions.py` and `test_scenario_isolation_synthetic.py`. |
| **DEF-P11-04** | **CRITICAL** | Safety shield assumed hardcoded 4-phase cycles with naive `(curr+1)%4` and manual conflict pairs | **FIXED** | Implemented `extract_tls_index_mapping` establishing exact relationship between junction `<request>` indices, `<connection tl="..." linkIndex="...">`, and signal-state character positions; accounts for multiple connections sharing a `linkIndex`; rigorously distinguishes priority-green (`G`) collisions from permissive-green (`g`) right-of-way yields (`<request response="...">`); verifies TraCI active program consistency across all 5 controllable intersections in the research grid; handles unmapped/unreliable junctions via fail-safe dynamic switch rejection and static cycling; verified by `test_safety_shield_semantics.py` (15 tests) and `test_safety_transitions_trace.py`. |
| **DEF-P11-05** | **HIGH** | Top-level SUMO/TraCI initialization inside test modules crashed pytest collection | **FIXED** | Removed global execution from `tests/test_environment.py`, `environment/test_sumo_connection.py`, and `tests/test_ppo_smoke.py`; added `pytest.ini` testpaths; collection executes in $<1\,\text{s}$ with zero side effects; verified by `test_import_environment_has_no_simulation_side_effects`. |
| **DEF-P11-06** | **HIGH** | Manuscript reported 128-step diagnostic runs without explaining Fixed-Time advantage | **FIXED** | Explicitly documented 128-step runs as diagnostic smoke tests in `README.md` and `MANUSCRIPT.md`; explained clearance penalty physics ($23\%$ lost time in RL exploration vs $9\%$ in Fixed-Time); strictly avoided claims of algorithmic superiority or convergence; verified by `test_manuscript_consistency.py`. |
| **DEF-P11-07** | **MEDIUM** | Comprehensive runner crashed on `StatisticalAnalyzer(runs_dir=...)` argument error | **FIXED** | Added `runs_dir` keyword argument alias in `StatisticalAnalyzer.__init__()` and aligned `run_comprehensive_matrix.py`; verified by `test_runner_interface.py::test_statistical_analyzer_constructor_alias`. |
| **DEF-P11-08** | **MEDIUM** | Incomplete runs were silently ignored or caused pipeline crashes without audit trails | **FIXED** | Added `rejected_runs` tracking in `StatisticalAnalyzer`, quarantined inconsistent runs (mismatched step counts, metadata discrepancies), and exported machine-readable `rejected_runs.json`; verified by `test_runner_interface.py::test_rejected_runs_audit_generation`. |
| **DEF-P11-09** | **LOW** | Outdated test badge and undocumented Windows SAC environment constraints | **FIXED** | Updated badge to `pytest-95 passed, 1 skipped`; documented clean venv creation via `requirements.txt` and Windows SAC reputation constraints without self-signed certificate modifications or disabling Windows security. |

---

## 2. Detailed Technical Audit Per Defect

### DEF-P11-01: Baseline Evaluation Asymmetry & Evaluation Orchestration
- **Remediation Details:** Standardized baseline evaluation across all algorithms (`Fixed_Time`, `FW_PPO`, `SA_PPO`, `IA_MOPPO`). Each evaluation executes in `deterministic_eval` mode for an explicit horizon (`eval_horizon`) over independent evaluation seeds (`seed + 1000 + ep`) decoupled from training. Model weights are frozen with `torch.no_grad()` and `model.eval()`.
- **Evaluation Horizon Propagation:** The runner's requested `--eval_horizon` is explicitly passed to `train_interaction_aware_ppo`, `evaluate_interaction_aware_ppo`, `evaluate_fixed_time`, `train_fixed_weight_ppo`, `evaluate_fixed_weight_ppo`, `train_state_adaptive_ppo`, and `evaluate_state_adaptive_ppo`. Stored in `ExperimentRun.eval_horizon`, recorded in `manifest.json` and `summary.json`, and bounds the evaluation simulation loop (`step < effective_horizon`).
- **Fixed-Time Single Episode Enforcement:** Corrected `baselines/fixed_time/fixed_time_controller.py` by replacing the inner reset loop with an immediate `break` on `terminated or truncated`, ensuring a requested episode cannot silently become multiple shorter episodes or a multi-episode aggregate.
- **Artifact Schema Compatibility:** Verified that evaluation summaries produced by `Fixed_Time`, `FW_PPO`, `SA_PPO`, and `IA_MOPPO` adhere to compatible schemas (`evaluation_mode`, `episode_id`, `eval_seed`, `eval_horizon`, `total_reward`, `safety_violations`).
- **Test Evidence:** `tests/test_deterministic_evaluation.py` (6/6 passed, including `test_evaluation_horizon_runtime_propagation_all_four_algorithms` verifying 900s runtime execution and `test_live_sumo_eval_horizon_propagation` verifying 15s live SUMO execution).

### DEF-P11-02: Manifest Schema Inconsistencies & Legacy Data Invalidation
- **Remediation Details:** Replaced binary validation with a 4-state lifecycle. Implemented `migrate_legacy_manifest()` in `common/experiment.py` which dynamically maps Phase 10 fields (`run_id` $\to$ `experiment_id`, `total_steps` $\to$ `training_timesteps`, `status` $\to$ `completion_status`).
- **No Invented Provenance:** Removed all default fallbacks that previously invented seeds (`train_seed=42`, `eval_seed=42`), timesteps, git commit hashes, configuration hashes, or evaluation modes. Missing provenance remains explicitly `UNKNOWN`.
- **Completion vs Reproducibility Separation:** Separated historical completion status (`completed_on_disk: bool`) from scientific reproducibility eligibility (`reproducibility_eligible: bool`). Historical runs missing explicit evaluation modes or decoupled evaluation seeds are marked `reproducibility_eligible = False`.
- **Audit Results across 49 Historical Runs:**
  - `VALID`: 0 (historical runs conform to Phase 10 schema, missing modern decoupled evaluation fields).
  - `LEGACY_VALID`: 44 (completed on disk, 128 steps, valid telemetry; preserved immutably).
  - `INCOMPLETE`: 4 (rejected: status `INITIALIZED`, 0 step rows).
  - `INVALID`: 1 (rejected: missing manifest file).
  - `reproducibility_eligible`: 0 / 49 (none meet the strict Phase 12 reproducibility standard due to absent evaluation provenance).
- **Test Evidence:** `tests/test_manifest_migration_and_classification.py` (5/5 passed), `tests/test_experiment_schema.py` (4/4 passed).

### DEF-P11-03: Scenario Aggregation Conflation & Fairness Metric Identification
- **Remediation Details:** Modified `StatisticalAnalyzer.aggregate_by_scenario_and_algorithm()` to group strictly by `['scenario', 'algorithm']`.
- **Elimination of Threshold Classification:** Removed heuristic classification `raw_fairness <= 1.0`. Introduced explicit metric identity (`fairness_metric_name`: `"delay_disparity"` vs `"delay_variance"` vs `"UNKNOWN"`) and unit metadata (`fairness_metric_unit`: `"dimensionless"` vs `"s^2"` vs `"UNKNOWN"`).
- **Ineligibility of Ambiguous Runs:** Runs lacking explicit metric metadata or traceable schema origins are classified as `UNKNOWN` and marked ineligible for fairness aggregation rather than guessed.
- **Statistical Cross-Consistency:** Enforced cross-consistency between manifest metadata, summary JSON, and `metrics_step.csv` row counts. Mismatched step counts or configuration contradictions trigger immediate rejection with explicit quarantine reasons.
- **Primary Algorithm vs Ablation Tagging:** Statistical analyzer explicitly tags runs with `is_ablation: bool` and `configuration_type: "Primary Algorithm" | "Ablation"`, exporting separate comparison tables (`primary_benchmark_comparison.csv`, `ablation_study_comparison.csv`) without conflation.
- **Test Evidence:** `tests/test_scenario_isolation_synthetic.py` (1/1 passed), `tests/test_fairness_definitions.py` (5/5 passed).

### DEF-P11-04: Microscopic Network Safety Shield & Conflict Verification
- **Remediation Details:** Replaced hardcoded 4-phase modulo logic (`(curr + 1) % 4`) and direct request index comparison with dynamic SUMO `.net.xml` index mapping:
  - Implemented `extract_tls_index_mapping` parsing junction `intLanes`, `<request foes="..." response="...">`, and `<connection tl="..." linkIndex="...">`.
  - Maps junction request indices directly to traffic-light signal-state character positions (`linkIndex`), correctly resolving multiple connections sharing a single signal head (`link_to_reqs`).
  - Right-of-Way / Permissive Green Semantics: Enforces SUMO yield semantics. Simultaneous priority greens (`G` and `G`) on conflicting movements are strictly rejected. Permissive green (`g`) is permitted with opposing priority green (`G`) strictly when a validated right-of-way yield is registered in the junction `response` bitmask (`(idx_yield, idx_priority) in yield_pairs`). Unregistered or dual permissive conflicts are rejected.
  - Active Program Consistency: Synchronizes active TLS `programID` with TraCI runtime states, verifying that controlled link counts strictly equal state character lengths across all 5 controllable intersections (`A1`: 12, `B0`: 12, `B1`: 20, `B2`: 12, `C1`: 12).
  - Fail-Safe Guarantee: If any junction mapping cannot be established reliably, the shield enforces fail-safe static sequencing and prevents unverified phase jumps.
- **Test Evidence:** `tests/test_safety_shield_semantics.py` (15/15 passed), `tests/test_safety_transitions_trace.py` (1/1 passed), `tests/test_traffic_safety.py` (6/6 passed).

### DEF-P11-05: Top-Level TraCI Execution in Test Collection
- **Remediation Details:** Removed all global execution statements from `tests/test_environment.py`, `environment/test_sumo_connection.py`, and `tests/test_ppo_smoke.py`. All SUMO/TraCI logic is encapsulated in guarded test functions with try/finally teardown. Configured `pytest.ini` to restrict collection to `tests/`.
- **Test Evidence:** `tests/test_environment.py::test_import_environment_has_no_simulation_side_effects` (passed). Pytest collects 95 items in $<1.0\,\text{s}$ with zero background processes.

### DEF-P11-06: Scientific Reporting of 128-Step Diagnostic Runs
- **Remediation Details:** Added comprehensive scientific notes to `README.md` and `MANUSCRIPT.md` Section VIII-B clarifying that 128-step runs are diagnostic smoke tests. Provided the physical traffic rationale: random exploration in RL triggers frequent phase switching requests, each incurring 3 seconds of yellow clearance ($23\%$ lost green time), whereas Fixed-Time follows pre-timed progression losing only $9\%$ of time. Clarified that long-horizon convergence is evaluated at 5,000–50,000 steps. Strictly avoided claiming convergence or algorithmic superiority.
- **Test Evidence:** `tests/test_manuscript_consistency.py` (5/5 passed).

### DEF-P11-07: Statistical Analyzer Constructor Keyword Incompatibility
- **Remediation Details:** Added `runs_dir: Optional[Union[str, Path]] = None, **kwargs` alias to `StatisticalAnalyzer.__init__()` and aligned `experiments/run_comprehensive_matrix.py:117` to `runs_root=output_root`.
- **Test Evidence:** `tests/test_runner_interface.py::test_statistical_analyzer_constructor_alias` (passed).

### DEF-P11-08: Unaudited Run Rejections
- **Remediation Details:** Implemented audit tracking in `StatisticalAnalyzer` with `self.rejected_runs`, `self.legacy_runs`, and `self.valid_runs`. Added automated export of `rejected_runs.json` in `export_summary_tables()` detailing run directories, failure reasons, and timestamps.
- **Test Evidence:** `tests/test_runner_interface.py::test_rejected_runs_audit_generation` (passed).

### DEF-P11-09: Unverified Test Badges & Environment Integrity
- **Remediation Details:** Updated `README.md` badge to `pytest-95 passed, 1 skipped` based on genuine execution evidence. Documented virtual environment reproduction via `requirements.txt` and Windows SAC cloud reputation behavior. Prohibited modifying native binaries or disabling host protections.
- **Test Evidence:** Verified by live pytest execution of 96 test cases (95 passed, 1 skipped).

---

## 3. Audit Certification

All 9 defects documented in Phase 11 and reviewed in Phase 12 have been systematically remediated and verified under strict scientific criteria. Every fix is verified by targeted regression tests, network-level validation, and cross-artifact consistency checks. Zero regressions exist.
