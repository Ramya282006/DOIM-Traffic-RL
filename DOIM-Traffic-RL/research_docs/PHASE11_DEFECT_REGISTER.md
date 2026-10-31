# PHASE 11 — DEFECT REGISTER AND REMEDIATION AUDIT
**Repository:** DOIM-Traffic-RL  
**Target:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Audit Date:** October 31, 2026  
**Auditor:** Independent Senior Research Engineer & Strict IEEE T-ITS Reviewer  
**Audit Commit:** `6882d3c6d1f118f1b501910fa55f7d9f3eac711f`  
**Current Branch:** `phase10-experimental-validation`

---

## 1. Executive Summary

This Defect Register documents all empirical, mathematical, architectural, and pipeline defects discovered during the Phase 11 independent experimental reproduction and evidence audit of `DOIM-Traffic-RL`. 

Every defect in this register is supported by direct inspection of active source code, raw experiment artifacts in `results/runs/`, terminal execution logs, and mathematical derivations. No claims are estimated or assumed.

---

## 2. Priority Defect Matrix

| Defect ID | Severity | File & Location | Exact Problem | Empirical Evidence | Proposed Correction | Status |
| :--- | :---: | :--- | :--- | :--- | :--- | :---: |
| **DEF-P11-01** | **Critical (P0)** | `research_docs/paper/MANUSCRIPT.md:185-197`<br>`research_docs/RESULTS_AND_ANALYSIS.md:22-30`<br>`results/tables/benchmark_comparison.csv:1-9` | **Cross-Scenario Data Contamination in Primary Benchmark Table.** Table I is titled as evaluating `north_south_surge`, but reports numbers ($70.57\pm34.70\,\text{s}$ for Fixed-Time, $97.57\pm42.53\,\text{s}$ for IA-MOPPO) aggregated across all scenarios combined (`north_south_surge` + `directional_shift_north_south` + `rush_hour`). When strictly isolated to `north_south_surge` ($N=5$), the true numbers are $55.13\pm5.33\,\text{s}$ for Fixed-Time and $82.19\pm45.69\,\text{s}$ for IA-MOPPO. | `StatisticalAnalyzer.aggregate_by_algorithm()` grouped across the entire directory without scenario filtering. Python analysis confirmed $N=7$ runs for Fixed-Time included 1 directional shift, 5 surge, and 1 rush hour run. | Regenerate `benchmark_comparison.csv` using strict scenario-isolated aggregation from `aggregate_by_scenario_and_algorithm()` and update Manuscript Table I. | **OPEN** |
| **DEF-P11-02** | **Critical (P0)** | `common/experiment.py:34-48, 70-75`<br>`analysis/statistical_analyzer.py:48-56` | **Manifest Schema Incompatibility Rejecting 100% of Historical Runs.** Phase 10 introduced `REQUIRED_MANIFEST_SCHEMA_FIELDS` (12 mandatory fields including `experiment_id`, `training_timesteps`, `evaluation_seed`, `episode_id`, `evaluation_mode`, `code_commit`, `completion_status`). All 48 historical run manifests in `results/runs/` use legacy keys (`run_id`, `eval_seed`, `total_steps`, `git_sha`, `status`). Because of this, `validate_experiment_manifest()` fails on 100% (48/48) of existing manifests, causing `StatisticalAnalyzer.load_run_summaries(reject_incomplete=True)` to reject all 48 runs and return an empty DataFrame. | Executed `validate_experiment_manifest()` across all 49 folders in `results/runs/`: exactly 0 valid, 48 invalid with message `"Missing required schema field: 'experiment_id'"`. | Implement schema key-aliasing in `validate_experiment_manifest` mapping legacy keys (`run_id` -> `experiment_id`, `total_steps` -> `training_timesteps`, etc.) without modifying raw artifacts. | **OPEN** |
| **DEF-P11-03** | **High (P1)** | `training/interaction_aware_ppo.py:626`<br>`common/experiment.py:89-92` | **Evaluation Mode Keyword Mismatch.** `evaluate_interaction_aware_ppo()` passes `evaluation_mode="deterministic_evaluation"`, but `common/experiment.py:89` enforces `valid_modes = {"train_rollout", "deterministic_eval", "stochastic_eval", "smoke_test"}`. Any evaluation run launched by this function fails schema validation. | `tests/test_experiment_schema.py:82-103` verifies rejection of invalid modes. Calling validator with `"deterministic_evaluation"` returns `False`. | Change `evaluation_mode` in `interaction_aware_ppo.py:626` to `"deterministic_eval"`, and accept `"deterministic_evaluation"` as an alias in `common/experiment.py`. | **OPEN** |
| **DEF-P11-04** | **High (P1)** | `experiments/run_comprehensive_matrix.py:117`<br>`analysis/statistical_analyzer.py:24` | **Keyword Argument Mismatch in Matrix Runner.** `run_comprehensive_matrix.py` instantiates `StatisticalAnalyzer(runs_dir=output_root)`, but `StatisticalAnalyzer.__init__` accepts `runs_root: Union[str, Path] = "results/runs"`. This raises `TypeError: StatisticalAnalyzer.__init__() got an unexpected keyword argument 'runs_dir'`. | Executed `python -c "from analysis.statistical_analyzer import StatisticalAnalyzer; StatisticalAnalyzer(runs_dir='results/runs')"`; resulted in `TypeError`. | Update line 117 to `runs_root=output_root` and add `**kwargs` support with `runs_dir` alias in `StatisticalAnalyzer.__init__`. | **OPEN** |
| **DEF-P11-05** | **High (P1)** | `baselines/ppo/fixed_weight_ppo.py:300-376`<br>`baselines/ppo/state_adaptive_ppo.py:300-394` | **Baseline Evaluation Asymmetry (Stochastic Rollouts vs Deterministic Eval).** While `interaction_aware_ppo.py` contains `evaluate_interaction_aware_ppo()` with frozen parameters (`torch.no_grad()`, `eval()`) and deterministic argmax action selection, `fixed_weight_ppo.py` and `state_adaptive_ppo.py` only contain training rollout loops. Their logged summary metrics are recorded during online stochastic exploration. | Code inspection of `fixed_weight_ppo.py` and `state_adaptive_ppo.py` confirms absence of deterministic evaluation functions. | Implement `evaluate_fixed_weight_ppo()` and `evaluate_state_adaptive_ppo()` using the identical frozen deterministic evaluation protocol. | **OPEN** |
| **DEF-P11-06** | **Critical (P0)** | `research_docs/RESULTS_AND_ANALYSIS.md:44-51`<br>`research_docs/paper/MANUSCRIPT.md:209-212`<br>`research_docs/FINAL_IMPLEMENTATION_REPORT.md:15-18` | **Premature Scientific Convergence & Superiority Claims on 128-Step Horizon.** Reports claim that the central research question has been conclusively answered ("Scientific Answer: YES") based on 128-step diagnostic matrix runs. 128 steps corresponds to only ~2 minutes of simulation time in SUMO (out of an 1800s episode). In these 128 steps, Fixed-Time outperforms IA-MOPPO in delay ($55.13\,\text{s}$ vs $82.19\,\text{s}$ in `north_south_surge`), and the RL agents were repeatedly intercepted by the safety shield (188–226 times). Declaring policy superiority or Q1 readiness based on 128 steps violates peer-review standards of IEEE T-ITS. | `results/runs/*/manifest.json` confirms `total_steps = 128` across all benchmark runs. Table values show Fixed-Time delay is $27.06\,\text{s}$ lower than IA-MOPPO. | Reclassify all 128-step results as diagnostic pipeline validation and safety-shield stress tests. State clearly that convergence and superiority claims require the proposed multi-seed pilot (5,000 steps) and main experiment (50,000 steps). | **OPEN** |
| **DEF-P11-07** | **High (P1)** | `objectives/objective_engine.py:72-78`<br>`results/tables/benchmark_comparison.csv:1-9` | **Metric Definition Shift & "Average Fairness" Inversion in Aggregations.** Prior to Phase 10, Objective 5 was computed as the variance of waiting times ($\text{s}^2$, with values ranging from 300 to 605). In Phase 10, Objective 5 was redefined as Delay Disparity ($1 - \text{Jain Index}$, bounded in $[0, 1]$). Because older and newer runs were aggregated together without version gating, the reported `average_fairness_mean` in `benchmark_comparison.csv` was $106.67$ to $183.07$—a mathematically nonsensical quantity for an index defined on $[0, 1]$. | `grep_search` found values `404.96` and `605.05` in older run summaries, averaging to `106.67` and `183.07`. | Clean benchmark aggregation to separate runs by objective schema version and report true delay disparity in $[0, 1]$. | **OPEN** |
| **DEF-P11-08** | **Medium (P2)** | `environment/safety_shield.py:141-144` | **Rigid 4-Phase Assumption in `TrafficSafetyShield`.** `TrafficSafetyShield` hardcodes: `yellow_phase = (curr + 1) % state.num_phases` and `next_green = (curr + 2) % state.num_phases`. While valid for the 3x3 grid network (`grid.net.xml`), this assumption breaks down if a junction uses an 8-phase NEMA dual-ring controller, an actuated 3-phase sequence, or an all-red clearance phase ($R$). | Code inspection of `safety_shield.py:141-144`. | Document this limitation explicitly in the safety verification report and validate against actual phase sequences. | **OPEN** |
| **DEF-P11-09** | **Low (P3)** | `README.md:6, 97, 101` | **Outdated Test Count in `README.md`.** `README.md` documents `28/28 tests passing`, whereas the test suite currently contains 59 tests. | `pytest tests/ -v` collected and passed 59 items. | Update `README.md` badge and documentation to reflect 59 passing regression tests. | **OPEN** |

---

## 3. Detailed Technical Evidence

### 3.1 Defect DEF-P11-01: Cross-Scenario Data Contamination
- **Active Code (`analysis/statistical_analyzer.py:213-217`):**
  In earlier versions, `aggregate_by_algorithm()` grouped purely by `["algorithm"]`, silently concatenating rows from `directional_shift_north_south`, `north_south_surge`, and `rush_hour`.
- **Committed Artifact (`results/tables/benchmark_comparison.csv`):**
  Shows Fixed-Time with `n_total_runs = 7` and mean waiting time $70.57\,\text{s}$.
- **Ground Truth Query:**
  ```python
  df = analyzer.load_run_summaries(reject_incomplete=False)
  agg = analyzer.aggregate_by_scenario_and_algorithm(df)
  # north_south_surge only:
  # Fixed_Time: mean=55.13s, std=5.33s (N=5 seeds)
  # IA_MOPPO:   mean=82.19s, std=45.69s (N=5 seeds)
  ```
- **Reviewer Impact:** In an IEEE T-ITS peer review, labeling a table with one traffic demand while populating it with blended cross-demand averages is an immediate rejection defect.

### 3.2 Defect DEF-P11-02: Schema Rejection of Historical Data
- **Code Reference (`common/experiment.py:34-48`):**
  ```python
  REQUIRED_MANIFEST_SCHEMA_FIELDS = [
      "experiment_id",
      "algorithm",
      "scenario",
      "train_seed",
      "evaluation_seed",
      "training_timesteps",
      "episode_id",
      "evaluation_mode",
      "config_hash",
      "code_commit",
      "start_time",
      "completion_status",
  ]
  ```
- **Historical Manifest Snippet (`results/runs/IA_MOPPO_north_south_surge_s42_20261031_170035/manifest.json`):**
  ```json
  {
    "run_id": "IA_MOPPO_north_south_surge_s42_20261031_170035",
    "experiment_name": "IA_MOPPO_north_south_surge",
    "algorithm": "IA_MOPPO",
    "scenario": "north_south_surge",
    "train_seed": 42,
    "eval_seed": 42,
    "status": "COMPLETED",
    "git_sha": "68e953a54adbf2b51ec392a06b92f76200323cc1",
    "total_steps": 128
  }
  ```
- **Impact:** All 48 run directories fail validation because `experiment_id` != `run_id`, `evaluation_seed` != `eval_seed`, `training_timesteps` != `total_steps`, and `code_commit` != `git_sha`.

---

## 4. Remediation Protocol

1. **Rule 1:** Never delete or manually overwrite historical run folders in `results/runs/`.
2. **Rule 2:** Provide schema-backward compatibility in `common/experiment.py` and `analysis/statistical_analyzer.py` so historical and new Phase 11 runs are parsed seamlessly.
3. **Rule 3:** Isolate all Phase 11 validation and reproduction outputs to `results/phase11_audit/`.
4. **Rule 4:** Implement baseline deterministic evaluation functions before launching any comparative convergence experiments.
