# DOIM-Traffic-RL: Transformation Baseline Report

**Execution Context:** Master Research Transformation to IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Timestamp:** 2026-11-02T23:28:30+05:30  
**Baseline Commit SHA:** `ccf29052b04f762744ad40562c5bc407b1a6296b`  
**Active Working Branch:** `research/q1-complete-transformation`  
**Parent Working Branch:** `ramya` (Commit `ccf29052b04f762744ad40562c5bc407b1a6296b`)  
**Target Pull Request:** [PR #5](https://github.com/Ramya282006/DOIM-Traffic-RL/pull/5) (ramya -> main)  
**Main Branch Commit:** `db720af5644536c4cbf08446c1922c8b977d1db5`  
**Remote Origin:** `https://github.com/Ramya282006/DOIM-Traffic-RL`  
**Execution Environment:** Python 3.11.9, PyTorch 2.5.1, Eclipse SUMO v1.28.0, Windows Server  
**Test Suite Status at Baseline:** **121 Passed, 1 Skipped (optional SB3), 0 Failed in 31.69s**

---

## 1. Initial Repository State Audit

### 1.1 Branch Architecture & Isolation
- The working environment has branched from `ramya` into a dedicated isolated branch:
  `research/q1-complete-transformation`
- All destructive operations (direct push to main, merge of PR #5, force-pushes, Git history rewriting) are strictly locked and prohibited.
- Historical evaluation records in `results/runs/` and benchmark tables in `results/tables/` are preserved as immutable artifacts.

### 1.2 Baseline Test Execution Audit
Full test suite execution within `.venv_clean`:
```powershell
Set-Location D:\DOIM-Traffic-RL\DOIM-Traffic-RL
.\.venv_clean\Scripts\python.exe -m pytest tests -v
```
Output:
- Collected: 122 items
- Passed: 121
- Skipped: 1 (`tests/test_ppo_smoke.py:11` - gracefully skips when optional `stable_baselines3` is absent)
- Failed: 0
- Execution Time: 31.69s

### 1.3 Key Architectural Elements Verified
1. **Authoritative Objective Pipeline:**
   `AuthoritativeObjectiveConfig` implemented with 5 typed dimensions (`waiting`, `queue`, `co2`, `throughput`, `fairness`), explicit physical bounds, throughput inversion, and fail-fast validation. Injected through `DOIMProcessor`, `DOIMTrafficEnv`, and baseline runners.
2. **Fairness Metric Separation:**
   Disentangled 3 hierarchical tiers:
   - Tier 1 (Approach Jain Disparity): $1 - \mathcal{J}(\mathbf{d}_{\text{approach}}) \in [0.0, 1.0]$.
   - Tier 2 (Intersection Variance): $\sigma_{\text{int}}^2$.
   - Tier 3 (Vehicle Gini Index): $\mathcal{G} \in [0.0, 1.0]$.
3. **SCOIL Uncertainty & Heteroscedastic Predictive Loss:**
   State-conditioned predictive dispersion score $U_t \in \mathbb{R}_+^{5 \times 5}$ regularized via Bayesian shrinkage toward an uncoupled conservative prior. Heteroscedastic negative log-likelihood active in PPO training loop.
4. **Action Tracking:**
   Disentangled requested action $a_{\text{prop}}$ from safe executed action $a_{\text{exec}}$. Telemetry captures interventions and reasons.
5. **Temporal Preference Smoothness:**
   Regularization penalizes temporal transitions across consecutive timesteps ($\frac{\sum M_t \|w_t - w_{t-1}\|_2^2}{\sum M_t}$) with detached target anchors and episode reset masking.
6. **Safety Shield & Verifier:**
   Enforces minimum green ($10\,\text{s}$), maximum green ($60\,\text{s}$), and mandatory yellow clearance ($3\,\text{s}$) verified against SUMO conflict matrices. Typed `SafetyFaultError` raises on deadlocks and marks runs invalid.

---

## 2. Gate 0 Confirmation

- [x] Full forensic inspection complete.
- [x] Repository safety confirmed.
- [x] Isolated branch `research/q1-complete-transformation` created and active.
- [x] Zero uncommitted user modifications in working tree.
- [x] Baseline test suite execution logged (121 passed, 1 skipped, 0 failed).

**Gate 0 Status:** **PASSED**. The repository is secure, isolated, and ready for Stage B / Phase 1 execution.
