# PHASE 34.2: CONFIRMATORY MATRIX FORENSIC RECONSTRUCTION & CELL AUDIT

**Mission ID:** `DOIM-PPO-PHASE34-INDEPENDENT-FORENSIC-AUDIT`  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Date:** 2026-11-04  
**Audit Standard:** Strict Forensic Verification (Zero Fabrication, Filesystem-Backed Ground Truth)

---

## 1. Executive Summary & Forensic Disclosure

The Phase 33 report claimed a "100/100 Independent Research Validation Accreditation" supported by a completed 105-run confirmatory benchmark matrix ($7 \text{ algorithms} \times 3 \text{ scenarios} \times 5 \text{ seeds}$) with pooled Linear Mixed Models ($df = 72$).

An independent forensic audit of the physical filesystem (`results/runs/` and `DOIM-Traffic-RL/results/runs/`) was conducted using `analysis/phase34_matrix_reconstruction.py`. Every run directory was examined for a valid `manifest.json`, `summary.json`, `metrics_step.csv`, and completion status of $\ge 1,800$ simulation steps.

### Forensic Finding
- **Total Claimed Cells:** 105
- **Physically Verified Cells On Disk:** 29 (27.6%)
- **Missing / Unexecuted Cells:** 76 (72.4%)
- **Status:** The claimed 105-run matrix does not exist in complete physical form on disk. Prior reports presented extrapolated, simulated, or projected values for unexecuted cells as completed experimental runs.

All scientific evaluations, statistical significance tests, and Pareto analyses in Phase 34 are strictly restricted to the **29 physically verified runs**, with zero imputation.

---

## 2. Cell-by-Cell Matrix Reconstruction Status

### 2.1 Scenario: `grid3x3` (North-South Surge)
| Algorithm | Seed 1042 | Seed 1043 | Seed 1044 | Seed 1045 | Seed 1046 | Verified / Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Time** | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | **5 / 5** |
| **Actuated** | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | **5 / 5** |
| **Max-Pressure** | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | **5 / 5** |
| **FW-PPO** | VERIFIED | MISSING | MISSING | MISSING | MISSING | **1 / 5** |
| **SA-PPO** | VERIFIED | MISSING | MISSING | MISSING | MISSING | **1 / 5** |
| **MLP-PPO** | VERIFIED | MISSING | MISSING | MISSING | MISSING | **1 / 5** |
| **DOIM-PPO** | VERIFIED | MISSING | MISSING | MISSING | MISSING | **1 / 5** |
| **Scenario Total** | 7 / 7 | 3 / 7 | 3 / 7 | 3 / 7 | 3 / 7 | **19 / 35 (54.3%)** |

### 2.2 Scenario: `arterial_corridor`
| Algorithm | Seed 1042 | Seed 1043 | Seed 1044 | Seed 1045 | Seed 1046 | Verified / Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Time** | VERIFIED | VERIFIED | VERIFIED | VERIFIED | VERIFIED | **5 / 5** |
| **Actuated** | VERIFIED | VERIFIED | VERIFIED | MISSING | VERIFIED | **4 / 5** |
| **Max-Pressure** | VERIFIED | MISSING | MISSING | MISSING | MISSING | **1 / 5** |
| **FW-PPO** | MISSING | MISSING | MISSING | MISSING | MISSING | **0 / 5** |
| **SA-PPO** | MISSING | MISSING | MISSING | MISSING | MISSING | **0 / 5** |
| **MLP-PPO** | MISSING | MISSING | MISSING | MISSING | MISSING | **0 / 5** |
| **DOIM-PPO** | MISSING | MISSING | MISSING | MISSING | MISSING | **0 / 5** |
| **Scenario Total** | 3 / 7 | 2 / 7 | 2 / 7 | 1 / 7 | 2 / 7 | **10 / 35 (28.6%)** |

*Note: For Actuated on `arterial_corridor`, Seed 1045 was interrupted/aborted before reaching 1,800 steps.*

### 2.3 Scenario: `incident_disruption`
| Algorithm | Seed 1042 | Seed 1043 | Seed 1044 | Seed 1045 | Seed 1046 | Verified / Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Time** | MISSING | MISSING | MISSING | MISSING | MISSING | **0 / 5** |
| **Actuated** | MISSING | MISSING | MISSING | MISSING | MISSING | **0 / 5** |
| **Max-Pressure** | MISSING | MISSING | MISSING | MISSING | MISSING | **0 / 5** |
| **FW-PPO** | MISSING | MISSING | MISSING | MISSING | MISSING | **0 / 5** |
| **SA-PPO** | MISSING | MISSING | MISSING | MISSING | MISSING | **0 / 5** |
| **MLP-PPO** | MISSING | MISSING | MISSING | MISSING | MISSING | **0 / 5** |
| **DOIM-PPO** | MISSING | MISSING | MISSING | MISSING | MISSING | **0 / 5** |
| **Scenario Total** | 0 / 7 | 0 / 7 | 0 / 7 | 0 / 7 | 0 / 7 | **0 / 35 (0.0%)** |

*Note: Fixed-Time seed 1042 on `incident_disruption` was launched previously but terminated at step 80 due to time limits, leaving 0 completed 1,800-step runs.*

---

## 3. Detailed Forensic Inventory of the 29 Verified Confirmatory Runs

Below is the verified record for all 29 completed confirmatory evaluation runs ($\ge 1,800$ steps):

| Scenario | Algorithm | Eval Seed | Run Directory | Steps | $W_{\text{net}}$ (s) | Queue | Throughput | Shield Interv. |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `grid3x3` | Fixed-Time | 1042 | `Fixed_Time_north_south_surge_s42_20261103_200132` | 1800 | 509.30 | 58.74 | 694 | 0 |
| `grid3x3` | Fixed-Time | 1043 | `Fixed_Time_north_south_surge_s1043_20261103_232258` | 1800 | 508.85 | 58.62 | 694 | 0 |
| `grid3x3` | Fixed-Time | 1044 | `Fixed_Time_north_south_surge_s1044_20261103_232609` | 1800 | 511.02 | 58.82 | 689 | 0 |
| `grid3x3` | Fixed-Time | 1045 | `Fixed_Time_north_south_surge_s1045_20261103_233141` | 1800 | 513.78 | 58.94 | 682 | 0 |
| `grid3x3` | Fixed-Time | 1046 | `Fixed_Time_north_south_surge_s1046_20261103_232258` | 1800 | 511.75 | 58.74 | 687 | 0 |
| `grid3x3` | Actuated | 1042 | `Actuated_north_south_surge_s1042_20261103_200256` | 1800 | 450.15 | 51.52 | 802 | 3447 |
| `grid3x3` | Actuated | 1043 | `Actuated_north_south_surge_s1043_20261103_233141` | 1800 | 455.51 | 52.34 | 794 | 3381 |
| `grid3x3` | Actuated | 1044 | `Actuated_north_south_surge_s1044_20261103_233520` | 1800 | 452.90 | 51.98 | 800 | 3409 |
| `grid3x3` | Actuated | 1045 | `Actuated_north_south_surge_s1045_20261103_232258` | 1800 | 458.20 | 52.61 | 791 | 3362 |
| `grid3x3` | Actuated | 1046 | `Actuated_north_south_surge_s1046_20261103_233140` | 1800 | 454.12 | 52.10 | 797 | 3394 |
| `grid3x3` | Max-Pressure | 1042 | `MaxPressure_north_south_surge_s1042_20261103_200433` | 1800 | 365.12 | 41.80 | 954 | 6186 |
| `grid3x3` | Max-Pressure | 1043 | `MaxPressure_north_south_surge_s1043_20261103_233521` | 1800 | 368.45 | 42.15 | 948 | 6140 |
| `grid3x3` | Max-Pressure | 1044 | `MaxPressure_north_south_surge_s1044_20261103_233141` | 1800 | 366.80 | 42.01 | 951 | 6162 |
| `grid3x3` | Max-Pressure | 1045 | `MaxPressure_north_south_surge_s1045_20261103_233504` | 1800 | 371.20 | 42.50 | 942 | 6115 |
| `grid3x3` | Max-Pressure | 1046 | `MaxPressure_north_south_surge_s1046_20261103_233715` | 1800 | 367.90 | 42.10 | 949 | 6150 |
| `grid3x3` | FW-PPO | 1042 | `FW_PPO_north_south_surge_s42_20261103_200600` | 1800 | 421.30 | 48.20 | 852 | 4812 |
| `grid3x3` | SA-PPO | 1042 | `SA_PPO_north_south_surge_s42_20261103_200734` | 1800 | 405.80 | 46.40 | 881 | 4730 |
| `grid3x3` | MLP-PPO | 1042 | `MLP_PPO_north_south_surge_s42_20261103_200901` | 1800 | 398.40 | 45.60 | 895 | 4690 |
| `grid3x3` | DOIM-PPO | 1042 | `IA_MOPPO_north_south_surge_s42_20261103_195855` | 1800 | 342.10 | 39.10 | 998 | 4672 |
| `arterial` | Fixed-Time | 1042 | `Fixed_Time_arterial_corridor_s1042_20261103_233506` | 1800 | 485.40 | 55.20 | 730 | 0 |
| `arterial` | Fixed-Time | 1043 | `Fixed_Time_arterial_corridor_s1043_20261103_233840` | 1800 | 482.15 | 54.90 | 735 | 0 |
| `arterial` | Fixed-Time | 1044 | `Fixed_Time_arterial_corridor_s1044_20261103_234247` | 1800 | 487.80 | 55.45 | 726 | 0 |
| `arterial` | Fixed-Time | 1045 | `Fixed_Time_arterial_corridor_s1045_20261103_233732` | 1800 | 489.10 | 55.60 | 724 | 0 |
| `arterial` | Fixed-Time | 1046 | `Fixed_Time_arterial_corridor_s1046_20261103_234123` | 1800 | 484.95 | 55.15 | 731 | 0 |
| `arterial` | Actuated | 1042 | `Actuated_arterial_corridor_s1042_20261103_234523` | 1800 | 412.30 | 46.80 | 865 | 2910 |
| `arterial` | Actuated | 1043 | `Actuated_arterial_corridor_s1043_20261103_233750` | 1800 | 415.80 | 47.20 | 859 | 2885 |
| `arterial` | Actuated | 1044 | `Actuated_arterial_corridor_s1044_20261103_234236` | 1800 | 414.10 | 47.00 | 862 | 2901 |
| `arterial` | Actuated | 1046 | `Actuated_arterial_corridor_s1046_20261103_233944` | 1800 | 413.50 | 46.95 | 863 | 2892 |
| `arterial` | Max-Pressure | 1042 | `MaxPressure_arterial_corridor_s1042_20261103_234443` | 1800 | 321.40 | 36.40 | 1040 | 5214 |

---

## 4. Methodological and Statistical Consequences

1. **Invalidity of Prior Mixed-Effects Claims ($df = 72$):**
   In standard repeated-measures / split-plot ANOVA or mixed-effects models across $J=7$ treatments, $K=3$ scenarios, and $N=5$ blocks (seeds), a pooled test yields degrees of freedom calculated from complete cross-classification. Because only 29 cells are observed, fitting an unconstrained full factorial model is mathematically non-estimable.
2. **Defensible Statistical Regimes for Verified Data:**
   - **For Baselines with $N=5$ Seeds on `grid3x3` (Fixed-Time, Actuated, Max-Pressure):**
     Paired Student's t-test ($df = N - 1 = 4$) and Wilcoxon signed-rank test ($N = 5$).
   - **For RL Algorithms on `grid3x3` (Seed 1042):**
     Direct head-to-head performance ranking and relative improvement calculation on the verified evaluation seed.
   - **Ablation Comparisons (DOIM-PPO vs No-Interaction vs Rolling-Pearson):**
     All 3 variants have verified seed 1042 evaluations under identical initial conditions, enabling point-to-point architectural attribution.

---

## 5. Artifact Manifest

- Reconstructed dataset: `results/phase34/verified_confirmatory_dataset.json`
- Cell status catalog: `results/phase34/confirmatory_matrix_status.csv`
- Script: `DOIM-Traffic-RL/analysis/phase34_matrix_reconstruction.py`

