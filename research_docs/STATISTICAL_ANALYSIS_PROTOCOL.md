# DOIM-Traffic-RL: Statistical Analysis & Hypothesis Testing Protocol
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Standard:** Strict Q1 Statistical Rigor & Elimination of Pseudoreplication  
**Reference Code:** `analysis/statistical_analyzer.py`

---

## 1. Statistical Foundations & Primary Unit of Analysis

In multi-agent and traffic reinforcement learning literature, a pervasive flaw is *pseudoreplication*—treating sequential simulation steps or repeated evaluation episodes of a single trained neural network as independent statistical samples ($N = \text{steps}$ or $N = \text{episodes}$). 

### 1.1 Strict Unit Definition
- In `DOIM-Traffic-RL`, the **primary statistical unit** is strictly an **independent training run initialized with a unique random seed**:
  $$s_{\text{train}} \in \{42, 123, 456, 789, 2024\}$$
- Each trained model $\pi_{\theta}^{(s)}$ is evaluated across $M$ disjoint evaluation episodes $\{e_1, \dots, e_M\}$:
  $$\bar{Y}^{(s)} = \frac{1}{M} \sum_{m=1}^M Y^{(s, e_m)}$$
- All cross-algorithm hypothesis tests, effect sizes, and confidence intervals operate exclusively on the vector of model-level aggregates:
  $$\vec{Y} = [\bar{Y}^{(s_1)}, \bar{Y}^{(s_2)}, \dots, \bar{Y}^{(s_N)}]^T, \quad N \ge 5$$

---

## 2. Hypothesis Testing Suite

### 2.1 Paired Comparisons
Because all models are trained on matched training seeds and evaluated on identical demand seeds, we employ paired tests:
1. **Paired Student's t-Test:** For normally distributed differences $D_i = Y_{\text{DOIM}, i} - Y_{\text{Baseline}, i}$:
   $$t = \frac{\bar{D}}{s_D / \sqrt{N}}, \quad \nu = N - 1$$
2. **Wilcoxon Signed-Rank Test:** Non-parametric alternative when normality is rejected by Shapiro-Wilk:
   $$W = \min(W^+, W^-)$$
   where $W^+$ and $W^-$ are the sums of ranks for positive and negative differences.
3. **Welch's Two-Sample t-Test:** Employed when comparing across unequal variances without direct seed pairing:
   $$t = \frac{\bar{Y}_1 - \bar{Y}_2}{\sqrt{s_1^2 / N_1 + s_2^2 / N_2}}$$

### 2.2 Multiple Comparison Corrections
When testing 5 traffic objectives across multiple algorithm pairings, uncorrected testing inflates the family-wise type I error rate. We enforce:
1. **Holm-Bonferroni Step-Down Procedure:**
   Sort raw p-values $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$. Reject $H_{(k)}$ if:
   $$p_{(k)} \le \frac{\alpha}{m - k + 1}$$
   Guarantees strong FWER control at $\alpha = 0.05$.
2. **Benjamini-Hochberg False Discovery Rate (FDR):**
   Controls the expected proportion of false discoveries across exploratory screening:
   $$p_{(k)} \le \frac{k}{m} Q, \quad Q = 0.05$$

---

## 3. Effect Size & Confidence Intervals

### 3.1 Standardized Effect Size (Cohen's d)
We report Cohen's $d$ to quantify practical engineering significance beyond binary $p$-values:
$$d = \frac{\bar{Y}_{\text{DOIM}} - \bar{Y}_{\text{base}}}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(N_1 - 1)s_1^2 + (N_2 - 1)s_2^2}{N_1 + N_2 - 2}}$$
- Small Effect: $|d| \approx 0.2$
- Medium Effect: $|d| \approx 0.5$
- Large Effect: $|d| \ge 0.8$

### 3.2 Confidence Intervals
Every performance metric is accompanied by a two-sided 95% Student-t confidence interval:
$$\text{CI}_{95\%} = \bar{Y} \pm t_{N-1, 0.975} \cdot \frac{s}{\sqrt{N}}$$
Additionally, for skewed metrics (such as queue spillback and maximum delay), we report non-parametric bootstrap confidence intervals (1,000 resamples with replacement).

---

## 4. Rejection Criteria & Run Invalidation

The statistical analyzer strictly rejects:
1. Runs where metrics CSV has step count mismatches with `summary.json`.
2. Runs with `completion_status != "COMPLETED"`.
3. Runs flagged with `safety_fault = True` (critical safety aborts).
4. Runs where evaluation mode is stochastic training rollout rather than deterministic evaluation.
5. Incompatible comparisons where evaluation horizons differ (e.g. comparing 900s to 1800s).

