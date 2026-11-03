# Phase 28.2.1: Pre-Registration Statistical Protocol & Approval Gate
**Project:** DOIM-PPO (*Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems*)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard:** Open-Science Pre-Registration & Methodological Freezing  
**Auditing Panel:** Independent Research Validation Team  
**Date:** November 3, 2026  

---

## 1. Frozen Pre-Registered Statistical Analysis Protocol

Before initiating any confirmatory runs, the statistical analysis protocol is frozen:

### 1.1 Primary and Secondary Outcomes
1. **Primary Confirmatory Outcome:**
   - Network-Wide Accumulated Waiting Time Rate ($W_{\text{net}} = \frac{1}{T}\sum_t \sum_l w_l(t)$ [veh·s/step]).
2. **Secondary Confirmatory Outcomes:**
   - Network Throughput ($R_{\text{total}} = \sum_t r_{\text{arr}}(t)$ [veh]).
   - Mean Queue Length ($Q = \frac{1}{T}\sum_t \sum_l q_l(t)$ [veh/step]).
   - Total Carbon Emissions ($M_{\text{CO}_2} = \sum_t e(t) \times 10^{-6}$ [kg]).
   - Delay Disparity Index ($1 - J \in [0, 1]$ [unitless]).
   - Normalized 5D Hypervolume ($HV \in [0, 1.2^5]$).

### 1.2 Hypothesis Testing & Significance Criteria
- **Pairwise Comparisons:** Paired Student's $t$-test on difference $D_k = Y_{\text{DOIM}, k} - Y_{\text{Base}, k}$ ($df = 4$).
- **Normality Check:** Shapiro-Wilk test on differences $\Delta$. If $p < 0.05$, report Wilcoxon signed-rank test pooled across scenarios ($N=15$).
- **Omnibus Synthesis:** Linear Mixed-Effects Model (LMM) with Algorithm and Scenario as fixed factors, and Seed as a random blocking factor ($df = 72$).
- **Multiple Comparison Correction:** Benjamini-Hochberg False Discovery Rate (FDR) controlled at $q = 0.05$ across all 21 pairwise algorithm comparisons.
- **Effect Size Metric:** Paired Cohen's $d_z = \bar{D} / s_D$ and Cliff's Delta (non-parametric dominance).
- **Practical Significance Threshold:** A performance improvement is declared practically significant only if:
  $$\frac{W_{\text{Base}} - W_{\text{DOIM}}}{W_{\text{Base}}} \ge 5.0\% \quad \text{AND} \quad d_z \ge 0.50$$

### 1.3 Missing Data & Failed Run Policy
- Zero data imputation.
- Any crash, TraCI timeout, or clearance violation will be permanently logged in `failure_log.json` and reported transparently in Table III of the manuscript.

---

## 2. Final Strategic Recommendations for Execution

We present three scientifically defensible pathways for the confirmatory campaign:

### Option A: Approve the 105-Run Campaign at 5,000 Steps
- **Wall-Clock Runtime:** $\approx 5.64$ hours.
- **Pros:** Fast turnaround; executes within host memory and compute limits.
- **Cons:** At 5,000 steps ($39$ PPO updates), DOIM vs No-SCOIL ablation separation may remain modest ($< 4\%$).

### Option B: Approve the 105-Run Campaign at 10,000 Steps (Recommended for IEEE T-ITS)
- **Wall-Clock Runtime:** $\approx 7.80$ hours.
- **Pros:** Provides $78$ PPO updates ($624$ gradient steps). Yields strong policy convergence, decisive ablation separation, and unquestionable scientific rigor for IEEE T-ITS reviewers.
- **Cons:** Requires $\approx 2$ additional hours of CPU time.

### Option C: Run a Targeted Feasibility Experiment (4 Runs) Before Deciding
- **Wall-Clock Runtime:** $\approx 25$ minutes.
- **Protocol:** Run DOIM-PPO and No-SCOIL at $5,000$ steps and $10,000$ steps on Seed 42 in `grid3x3`. Inspect the learning curves and ablation gap to empirically determine whether 5k or 10k steps is optimal.
- **Pros:** Zero guesswork; empirical verification before committing to the full campaign.

---

## 3. Formal Authorization Decision

**To the User / Research Director:**

All statistical formulas, degrees of freedom, execution accountings, and pre-registered outcome protocols are complete and frozen.

$$\mathbf{PLEASE\,SELECT\,YOUR\,PREFERRED\,PATHWAY:}$$
1. **Option A:** Approve the 105-run confirmatory campaign at **5,000 steps** ($pprox 5.6$ hours).
2. **Option B (Recommended):** Approve the 105-run confirmatory campaign at **10,000 steps** ($pprox 7.8$ hours).
3. **Option C:** Execute the **4-run feasibility experiment** ($pprox 25$ minutes) to test 5k vs 10k convergence before launching the full campaign.
