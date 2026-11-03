# Phase 28.2.1: Statistical Design & Degrees-of-Freedom Correction
**Project:** DOIM-PPO (*Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems*)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard:** Formal Statistical Methodology & Experimental Unit Audit  
**Auditing Panel:** Independent Research Validation Team  
**Date:** November 3, 2026  
**Active Branch:** `research/phase28-independent-research-validation` | **Verified HEAD SHA:** `31a4eec`

---

## 1. Executive Summary: The Degrees-of-Freedom Audit

This document formally corrects all degrees-of-freedom ($df$), statistical power, and sample-size derivations from Phase 28.2. It establishes mathematically rigorous distinctions between:
1. **Paired Student's $t$-tests** across matched random seed realizations ($df = N - 1$).
2. **Independent Pooled Two-Sample $t$-tests** between uncorrelated groups ($df = 2N - 2$).
3. **Repeated-Measures / Linear Mixed-Effects Models (LMM)** across multi-scenario blocks.
4. **Non-Parametric Paired Tests** (Wilcoxon Signed-Rank Test) and their mathematical sample-size limits.

---

## 2. Mathematical Taxonomy of Comparative Statistical Tests

### 2.1 Paired Student's $t$-Test ($df = N - 1$)
When comparing DOIM-PPO ($X$) against a baseline ($Y$) evaluated on the **exact same held-out random demand seeds** $s \in \{1042, 1043, 1044, 1045, 1046\}$ in a given scenario, the observations are paired:
$$D_i = X_i - Y_i, \quad i = 1, \dots, N$$
- **Null Hypothesis:** $H_0: \mu_D = 0$ versus $H_1: \mu_D \ne 0$.
- **Test Statistic:**
  $$t = \frac{\bar{D}}{s_D / \sqrt{N}} \sim t_{N-1}$$
- **Exact Degrees of Freedom:**
  $$\mathbf{df_{\text{paired}} = N - 1}$$
  - For $N = 3$ seed pairs: $df = 3 - 1 = \mathbf{2}$.
  - For $N = 5$ seed pairs: $df = 5 - 1 = \mathbf{4}$.

#### Critical Values & Power Analysis (Paired $t$-Test):
Under $\alpha = 0.05$ (two-tailed) and statistical power $1 - \beta = 0.80$:
- **For $N = 3$ pairs ($df = 2$):**
  - $t_{0.025, 2} = 4.303$, $t_{0.20, 2} = 1.061 \implies t_{\alpha/2} + t_\beta = 5.364$.
  - Minimum Detectable Effect (Paired Cohen's $d_z = \mu_D / \sigma_D$):
    $$d_z = \frac{5.364}{\sqrt{3}} = \mathbf{3.097}$$
  - *Finding:* With $N=3$ pairs, the true effect size must exceed $3.1$ standard deviations of the paired difference to achieve $80\%$ power! Subtle or moderate improvements ($d_z \le 1.0$) cannot be statistically detected.
- **For $N = 5$ pairs ($df = 4$):**
  - $t_{0.025, 4} = 2.776$, $t_{0.20, 4} = 0.941 \implies t_{\alpha/2} + t_\beta = 3.717$.
  - Minimum Detectable Effect:
    $$d_z = \frac{3.717}{\sqrt{5}} = \mathbf{1.662}$$
  - *Finding:* Expanding to $N=5$ pairs cuts the required effect size from $3.097$ to $1.662$ (a $46.3\%$ reduction in MDE). However, $d_z = 1.66$ is still a large effect; **five seeds does NOT automatically guarantee statistical significance for modest differences**.

---

### 2.2 Independent Two-Sample Pooled $t$-Test ($df = 2N - 2$)
When comparing two uncoupled algorithms where observations are independent and homoscedastic (equal variances $\sigma_1^2 = \sigma_2^2$):
- **Exact Degrees of Freedom:**
  $$\mathbf{df_{\text{indep}} = (N_1 - 1) + (N_2 - 1) = 2N - 2}$$
  - For $N = 3$ per group: $df = 2(3) - 2 = \mathbf{4}$.
  - For $N = 5$ per group: $df = 2(5) - 2 = \mathbf{8}$.

#### Critical Values & Power Analysis (Independent $t$-Test):
- **For $N = 3$ per group ($df = 4$):**
  - $t_{0.025, 4} = 2.776$, $t_{0.20, 4} = 0.941 \implies t_{\alpha/2} + t_\beta = 3.717$.
  - Standard error: $\text{SE} = \sigma \sqrt{2/3} = 0.816 \sigma$.
  - Minimum Detectable Effect (Independent Cohen's $d$):
    $$d = 3.717 \times \sqrt{2/3} = \mathbf{3.035}$$
- **For $N = 5$ per group ($df = 8$):**
  - $t_{0.025, 8} = 2.306$, $t_{0.20, 8} = 0.889 \implies t_{\alpha/2} + t_\beta = 3.195$.
  - Standard error: $\text{SE} = \sigma \sqrt{2/5} = 0.632 \sigma$.
  - Minimum Detectable Effect:
    $$d = 3.195 \times \sqrt{2/5} = \mathbf{2.021}$$

---

### 2.3 Non-Parametric Paired Comparisons (Wilcoxon Signed-Rank Test)
When normality of paired differences $\Delta_i$ cannot be guaranteed (e.g., severe skewness in heavy congestion):
- **Mathematical Bound on Minimum $p$-Value for Small $N$:**
  For $N$ non-zero paired differences, the maximum signed-rank statistic $W = \sum_{i=1}^N i = \frac{N(N+1)}{2}$.
  The exact two-tailed minimum possible $p$-value occurs when all differences share the same sign:
  $$p_{\min} = 2 \times \left(\frac{1}{2}\right)^N = \frac{1}{2^{N-1}}$$
  - **For $N = 3$:** $p_{\min} = 1 / 4 = \mathbf{0.2500}$.
  - **For $N = 5$:** $p_{\min} = 1 / 16 = \mathbf{0.0625} > 0.05$!
  - **For $N = 6$:** $p_{\min} = 1 / 32 = \mathbf{0.03125} < 0.05$.

> [!WARNING]
> **Fundamental Non-Parametric Constraint for IEEE T-ITS:**  
> For an isolated single-scenario analysis with $N=5$ seeds, a two-tailed Wilcoxon signed-rank test **can never achieve $p < 0.05$**, even if DOIM-PPO wins on all 5 seeds!  
> To conduct valid non-parametric inference, observations must be pooled across the 3 benchmark scenarios (yielding $N = 3 \times 5 = 15$ paired observations, where $p_{\min} = 2 / 2^{15} < 0.0001$).

---

### 2.4 Multi-Scenario Repeated-Measures & Linear Mixed-Effects Model
To overcome the sample size limit of isolated 5-seed tests, the primary statistical synthesis across the 105 confirmatory runs will use a **Linear Mixed-Effects Model (LMM)**:

$$Y_{ijk} = \mu + \alpha_i (\text{Algorithm}) + \beta_j (\text{Scenario}) + (\alpha\beta)_{ij} + u_k (\text{Seed Random Effect}) + \epsilon_{ijk}$$

- **Algorithms ($I = 7$):** Fixed factor.
- **Scenarios ($J = 3$):** Fixed blocking factor.
- **Seed ($K = 5$):** Random intercept accounting for inter-seed correlation.
- **Residual Degrees of Freedom:**
  $$df_{\text{resid}} = (I - 1)(K - 1) \times J = (7 - 1)(5 - 1) \times 3 = \mathbf{72}$$
- *Statistical Advantage:* With $df = 72$, the critical value drops to $t_{0.025, 72} \approx 1.993$, providing sufficient statistical power to detect moderate effect sizes ($d \ge 0.45$) across the entire multi-scenario benchmark.

---

## 3. Corrected Power & Precision Reference Table

| Test Type | Group Size ($N$) | Exact $df$ | Critical $t_{0.025}$ | Critical Sum ($t_\alpha + t_\beta$) | Minimum Detectable Effect | Can Reject $H_0$ at $p < 0.05$? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paired $t$-Test** | $N = 3$ | **2** | $4.303$ | $5.364$ | $d_z = 3.10$ | Only with massive effect |
| **Paired $t$-Test** | $N = 5$ | **4** | $2.776$ | $3.717$ | $d_z = 1.66$ | Only with large effect |
| **Independent $t$-Test** | $N = 3$ | **4** | $2.776$ | $3.717$ | $d = 3.03$ | Only with massive effect |
| **Independent $t$-Test** | $N = 5$ | **8** | $2.306$ | $3.195$ | $d = 2.02$ | Only with large effect |
| **Wilcoxon Paired** | $N = 5$ | — | — | — | Minimum $p = 0.0625$ | **NO (Two-tailed impossible)** |
| **Wilcoxon Pooled** | $N = 15$ | — | — | — | Minimum $p < 0.0001$ | **YES** |
| **LMM Omnibus** | $7 \times 3 \times 5$ | **72** | $1.993$ | $2.835$ | $d = 0.45$ | **YES (Detects moderate effects)** |
