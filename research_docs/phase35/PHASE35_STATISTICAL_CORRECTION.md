# PHASE 35: STATISTICAL METHODOLOGY & DEGREES-OF-FREEDOM CORRECTION

**Mission ID:** `DOIM-PPO-PHASE35-STATISTICAL-CORRECTION`  
**Project:** Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems  
**Canonical Algorithm Identity:** DOIM-PPO (Internal implementation: `IA_MOPPO`)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Status:** MATHEMATICALLY AUDITED & FORMALLY RATIFIED  

---

## 1. Executive Summary

Previous audit phases exposed a severe statistical vulnerability in early reporting drafts: the uncritical assertion of a Linear Mixed-Effects Model (LMM) with 72 degrees of freedom ($df=72$). In reality, only 29 physical runs existed on disk, and the unit of experimental independence is the complete simulation episode executed under an identical stochastic seed, not intra-episode time steps or fictitious full-factorial cells.

This document formally repudiates the $df=72$ formulation, establishes the exact statistical framework for the 5-seed paired benchmark on `grid3x3`, details the power analysis and minimum detectable effect (MDE), and applies Holm-Bonferroni multiple-comparison corrections.

---

## 2. Mathematical Invalidation of $df=72$

### 2.1 Why $df=72$ Was Mathematically Invalid
1. **Missing Matrix Cells:** A full factorial design across 7 algorithms, 3 scenarios, and 5 seeds would entail $7 \times 3 \times 5 = 105$ experimental runs. In Phase 34, forensic inspection revealed only 29 runs physically existed on disk (76 cells were never run). An LMM cannot legitimately claim residual degrees of freedom based on unexecuted experimental units.
2. **Pseudo-Replication Error:** Dividing an 1,800-second continuous simulation trajectory into sub-intervals or treating time steps as independent replicates violates the core independence assumption of regression models (temporal autocorrelation $\rho > 0.85$).
3. **The True Experimental Unit:** In stochastic microscopic traffic simulation (SUMO), the independent experimental unit is the **entire episode execution seeded by a single pseudorandom number generator seed $s \in \{1042, 1043, 1044, 1045, 1046\}$**.

---

## 3. Correct Statistical Formulations

### 3.1 Paired Comparisons Design ($df = N - 1 = 4$)
When comparing two signal control policies $A$ and $B$ evaluated across the identical $N=5$ traffic demand seeds:
$$D_s = X_{A, s} - X_{B, s}, \quad s \in \{1042, 1043, 1044, 1045, 1046\}$$
The paired differences $D_s$ have sample mean $\bar{D} = \frac{1}{N}\sum D_s$ and sample variance:
$$s_D^2 = \frac{1}{N-1}\sum_{s=1}^N (D_s - \bar{D})^2$$
The paired Student's $t$-statistic is:
$$t = \frac{\bar{D}}{s_D / \sqrt{N}} \sim t(df = N - 1) = t(4)$$

### 3.2 Independent Two-Sample Comparisons ($df = 2N - 2 = 8$)
If two distinct, unpaired sets of independent seeds with equal variance were evaluated with $N_1 = N_2 = 5$:
$$t = \frac{\bar{X}_1 - \bar{X}_2}{s_p \sqrt{2/N}} \sim t(df = 2N - 2) = t(8)$$
where $s_p^2 = \frac{(N_1-1)s_1^2 + (N_2-1)s_2^2}{N_1 + N_2 - 2}$.
*Note:* In our benchmark, seeds are matched across algorithms, making the paired design ($df=4$) strictly more powerful and methodologically sound.

### 3.3 Non-Parametric Validation: Wilcoxon Signed-Rank Test
Because $N=5$ is a modest sample size, normality assumptions are supplemented by the exact Wilcoxon signed-rank test. For $N=5$, the maximum rank sum is $T_{\max} = 1 + 2 + 3 + 4 + 5 = 15$. When all 5 pairs favor one method (a unanimous directionality), the test statistic is $W = 0$, yielding an exact two-tailed $p$-value:
$$p = 2 \times \left( \frac{1}{2^5} \right) = \frac{2}{32} = \mathbf{0.0625}$$
This establishes an intrinsic mathematical property: **no two-tailed Wilcoxon test with $N=5$ can reach the conventional $\alpha=0.05$ threshold ($p=0.0625 > 0.05$)**. Claims of non-parametric significance with $N=5$ at $\alpha=0.05$ are mathematically impossible.

---

## 4. Power & Precision Analysis

### 4.1 Minimum Detectable Effect Size (MDE)
Given significance level $\alpha = 0.05$ (two-tailed) and desired statistical power $1 - \beta = 0.80$ with $df = 4$:
- Critical $t$-value: $t_{0.025, 4} = 2.776$
- Normal approximation non-centrality parameter: $\delta \approx t_{\alpha/2, \nu} + t_{\beta, \nu} \approx 2.776 + 0.842 = 3.618$
- Required Cohen's $d_z = \frac{\delta}{\sqrt{N}} = \frac{3.618}{\sqrt{5}} \approx \mathbf{1.682}$

**Methodological Conclusion:**
A 5-seed design is fully powered ($\ge 80\%$) to detect **large-to-extreme effect sizes ($d_z \ge 1.68$)**, which are standard when comparing competent controllers against failing policies (e.g., Actuated gap-out breakdown). However, it is underpowered ($< 25\%$) to detect subtle, incremental improvements ($d_z \le 0.5$). We explicitly document this limitation for peer review.

---

## 5. Verified Paired Baseline Comparisons on `grid3x3` ($N=5, df=4$)

From `results/phase35/corrected_statistical_results.json`:

### 5.1 Network Waiting Time ($W_{\text{net}}$, seconds)
| Comparison | Mean Diff ($\bar{D}$) | Std Diff ($s_D$) | $t$-stat ($df=4$) | $p$-value (raw) | Cohen's $d_z$ | Power | Wilcoxon $W$ ($p$) | Holm-Bonferroni $p_{\text{adj}}$ | Significance ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Time vs Actuated** | -1159.18 s | 63.56 s | -40.78 | $2.16 \times 10^{-6}$ | -18.24 | $>0.9999$ | 0.0 ($0.0625$) | $1.51 \times 10^{-5}$ | **YES** |
| **Actuated vs Max-Pressure** | +1349.99 s | 67.66 s | +44.61 | $1.51 \times 10^{-6}$ | +19.95 | $>0.9999$ | 0.0 ($0.0625$) | $1.21 \times 10^{-5}$ | **YES** |
| **Fixed-Time vs Max-Pressure**| +190.81 s | 58.22 s | +7.33 | $1.85 \times 10^{-3}$ | +3.28 | 0.9994 | 0.0 ($0.0625$) | $6.53 \times 10^{-3}$ | **YES** |

### 5.2 Mean Queue Length ($Q_{\text{net}}$, vehicles)
| Comparison | Mean Diff ($\bar{D}$) | Std Diff ($s_D$) | $t$-stat ($df=4$) | $p$-value (raw) | Cohen's $d_z$ | Power | Wilcoxon $W$ ($p$) | Holm-Bonferroni $p_{\text{adj}}$ | Significance ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Time vs Actuated** | -14.85 veh | 0.88 veh | -37.52 | $3.01 \times 10^{-6}$ | -16.78 | $>0.9999$ | 0.0 ($0.0625$) | $1.81 \times 10^{-5}$ | **YES** |
| **Actuated vs Max-Pressure** | +17.73 veh | 0.86 veh | +45.97 | $1.34 \times 10^{-6}$ | +20.56 | $>0.9999$ | 0.0 ($0.0625$) | $1.20 \times 10^{-5}$ | **YES** |
| **Fixed-Time vs Max-Pressure**| +2.89 veh | 0.83 veh | +7.81 | $1.45 \times 10^{-3}$ | +3.49 | 0.9998 | 0.0 ($0.0625$) | $6.53 \times 10^{-3}$ | **YES** |

### 5.3 Cumulative Throughput ($Y_{\text{cum}}$, vehicles)
| Comparison | Mean Diff ($\bar{D}$) | Std Diff ($s_D$) | $t$-stat ($df=4$) | $p$-value (raw) | Cohen's $d_z$ | Power | Wilcoxon $W$ ($p$) | Holm-Bonferroni $p_{\text{adj}}$ | Significance ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Actuated vs Fixed-Time** | -16.4 veh | 4.62 veh | -7.95 | $1.36 \times 10^{-3}$ | -3.55 | 0.9999 | 0.0 ($0.0625$) | $6.53 \times 10^{-3}$ | **YES** |
| **Max-Pressure vs Actuated** | +18.2 veh | 5.07 veh | +8.03 | $1.31 \times 10^{-3}$ | +3.59 | 0.9999 | 0.0 ($0.0625$) | $6.53 \times 10^{-3}$ | **YES** |
| **Max-Pressure vs Fixed-Time**| +1.8 veh | 8.11 veh | +0.50 | 0.6455 | +0.22 | 0.0676 | 6.0 ($0.8125$) | 0.6455 | NO |

---

## 6. Peer Review Guidance & Defensive Posture

When submitting to IEEE T-ITS:
1. State transparently: *"Baseline comparisons are evaluated across 5 matched seeds ($df=4$) using paired Student's $t$-tests and exact Wilcoxon signed-rank tests."*
2. Acknowledge that while $N=5$ cannot achieve $\alpha=0.05$ under two-tailed Wilcoxon tests ($p_{\min} = 0.0625$), parametric paired tests show extreme significance ($p < 10^{-5}$) with effect sizes exceeding $d_z > 16$.
3. Never cite fictitious Linear Mixed Models or unexecuted cells.

---

## 7. Sign-off & Verification

- **Formulation Verified:** Paired $t$-test with $df=4$, Holm-Bonferroni adjustment.
- **Power Verified:** MDE = 1.682; extreme effect sizes detected with power $> 0.999$.
- **Fabrication Eliminated:** 100% anchored in physical records.

