# PHASE 35: FORENSIC CORRECTIONS AND EVIDENCE INTEGRITY REPORT

**Mission ID:** `DOIM-PPO-PHASE35-37-SCIENTIFIC-VALIDATION`  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Date:** 2026-11-04  
**Audit Standard:** Strict Forensic Reconciliation (Ground-Truth Filesystem Evidence Only)

---

## 1. Executive Summary & Forensic Resolution of Known Inconsistencies

Phase 35 addresses every numerical, methodological, architectural, and provenance inconsistency identified during Phase 34 prior to executing the Phase 36 confirmatory campaign.

### Master Discrepancy Resolution Table:
| Discrepancy Item | Prior Reported Value | Verified Ground Truth | Resolution / Corrective Action | Verified Filesystem Evidence |
| :--- | :---: | :---: | :--- | :--- |
| **1. DOIM-PPO Shield Interventions** | `4,672` in Phase 34.3 report text | **`4,249`** | Corrected in all text and tables. 4,672 was an errant draft transcription; 4,249 is the exact count in raw CSV. | `results/runs/IA_MOPPO_north_south_surge_s42_20261103_195855/metrics_step.csv` (sum = 4,249) |
| **2. Hypervolume vs Ablations** | DOIM: `0.05722`<br>No-SCOIL: `0.06071`<br>Pearson: `0.06090` | **Verified Accurate** | Documented scientific mechanism: No-SCOIL has a 1.7% raw delay edge on seed 1042, but triggers **+5.3% to +16.0% more illegal phase switches** (4,474 and 4,927 overrides vs 4,249). Full DOIM-PPO optimizes phase discipline. | `results/phase35/corrected_pareto_results.json` |
| **3. Waiting Time vs Per-Vehicle Delay** | Ambiguously cited as $510.94\,\text{s}$ vs $7.90\,\text{s}$ | **Distinct Metrics** | Formalized: $W_{\text{net}} = 510.94\,\text{s}$ is the time-averaged network-wide waiting time. $\bar{d}_{\text{veh}} = 7.90\,\text{s/veh}$ is the per-vehicle normalized trip delay ($\bar{N}_{\text{veh}} \approx 64.64$). | `research_docs/phase35/PHASE35_METRIC_RECONCILIATION.md` |
| **4. Fairness Formulation** | Jain's Index vs Delay Disparity | **Dual Formulations** | Standardized: Delay Disparity $\delta_{\text{fair}} = 0.5020$ (minimization in reward); Jain's Index $J(x) = 1.0 - \delta_{\text{fair}} = 0.4980$ (maximization). | `objectives/fairness.py` & `metrics_step.csv` |
| **5. Throughput Accounting** | Instantaneous vs Cumulative | **Cumulative Standard** | Standardized: Instantaneous step throughput ($0-2\,\text{veh/s}$) integrated over 1,800s equals exactly **641.0 completed vehicles** in `summary.json`. | `summary.json` & step sum |
| **6. $\text{CO}_2$ Emission Accounting** | Rate ($\text{mg/s}$) vs Total ($\text{kg}$) | **Both Preserved** | Standardized: Rate is $113,768.5\,\text{mg/s}$ (used in multi-objective vector $[0, 120\text{k}]$); cumulative release is **$204.78\,\text{kg}$**. | `metrics_step.csv` column `raw_co2` |
| **7. Matrix Completeness** | 105 claimed vs 29 verified | **29 Verified (27.6%)** | The missing 76 cells are cataloged as active confirmatory targets for Phase 36 execution. | `results/phase34/confirmatory_matrix_status.csv` |

---

## 2. Epistemological and Methodological Guardrails

1. **Rejection of Fabricated Data:**  
   Under no circumstances will empty cells in the confirmatory matrix be populated by simulation interpolation or synthetic averages.
2. **Predictive Association vs Causal Inference:**  
   The learned interaction matrix $\tilde{M}_t$ in SCOIL represents dynamic predictive cross-objective correlation, not Pearlian do-calculus structural causal identification. All claims in research documents are constrained to predictive coupling.
3. **Safety Shield Characterization:**  
   The safety shield is a formal runtime supervisor enforcing $g_{\min} \ge 10\,\text{s}$ and yellow clearance $\ge 3\,\text{s}$. Max-Pressure's low delay on `grid3x3` is explicitly acknowledged as dependent on the shield acting as an external 13-second bang-bang cycle timer.

---

## 3. Artifact Manifest

- Master research validation ledger: `research_docs/MASTER_RESEARCH_VALIDATION_LEDGER.md`
- Metric reconciliation report: `research_docs/phase35/PHASE35_METRIC_RECONCILIATION.md`
- Statistical correction report: `research_docs/phase35/PHASE35_STATISTICAL_CORRECTION.md`
- Pareto revalidation report: `research_docs/phase35/PHASE35_PARETO_REVALIDATION.md`
- SCOIL attribution audit report: `research_docs/phase35/PHASE35_SCOIL_ATTRIBUTION_AUDIT.md`
- Provenance audit report: `research_docs/phase35/PHASE35_PROVENANCE_AUDIT.md`
- Corrected verified dataset: `results/phase35/corrected_verified_dataset.json`
- Corrected statistical results: `results/phase35/corrected_statistical_results.json`
- Corrected Pareto results: `results/phase35/corrected_pareto_results.json`

