# DOIM-Traffic-RL: Complete Scientific Reproducibility Package
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Standard:** Open Science, Full Computational Reproducibility & Artifact Verification  
**Repository:** `https://github.com/Ramya282006/DOIM-Traffic-RL`  
**Verified Environment:** Python 3.11.9, Eclipse SUMO 1.28.0, PyTorch 2.4+  

---

## 1. System Requirements & Environment Setup

### 1.1 Software Dependencies
- **Operating System:** Windows 10/11, Ubuntu Linux 22.04 LTS, or macOS 13+.
- **Python Version:** Python 3.11.9 (strict compatibility with torch and gymnasium).
- **Microscopic Traffic Simulator:** Eclipse SUMO v1.28.0+ (with `SUMO_HOME` environment variable set).
- **Core Libraries:**
  ```text
  torch>=2.2.0
  gymnasium>=0.29.0
  numpy>=1.26.0
  scipy>=1.12.0
  pandas>=2.2.0
  pyyaml>=6.0.1
  matplotlib>=3.8.0
  seaborn>=0.13.0
  pytest>=8.0.0
  traci>=1.28.0
  ```

### 1.2 Installation Instructions
```powershell
# 1. Clone repository and checkout research branch
git clone https://github.com/Ramya282006/DOIM-Traffic-RL.git
cd DOIM-Traffic-RL\DOIM-Traffic-RL

# 2. Activate Python 3.11 clean virtual environment
.\.venv_clean\Scripts\Activate.ps1

# 3. Verify SUMO binary availability
python -c "import traci; from environment.sumo_locator import find_sumo_binary; print('SUMO Binary:', find_sumo_binary())"
```

---

## 2. Step-by-Step Reproduction Workflow

### Step 1: Execute Complete Automated Test Suite (120 Tests)
Before executing simulations, verify that all mathematical formulations, safety verifiers, and unit tests pass:
```powershell
python -m pytest tests/ -v
```
*Expected Output:* `119 passed, 1 skipped (optional stable_baselines3), 0 failed`.

### Step 2: Run Diagnostic Smoke Matrix (128 steps)
Verifies TraCI socket connectivity, action shielding, and autograd gradient flow:
```powershell
python experiments/run_smoke_test.py
```
*Expected Output:* `SMOKE TEST PASSED: All 4 algorithms and safety shield verified`.

### Step 3: Run Controlled Pilot Qualification (3 Seeds, 5,000 steps)
Trains across seeds `42, 123, 456` with frozen deterministic evaluation ($T_{\text{eval}} = 1800\text{ s}$):
```powershell
python experiments/run_comprehensive_matrix.py --timesteps 5000 --seeds 42 123 456 --eval_episodes 1 --eval_horizon 1800
```

### Step 4: Run Main Benchmark Convergence Matrix (5 Seeds, 50,000 steps)
Executes primary comparative benchmark across all 4 algorithms and ablations:
```powershell
python experiments/run_comprehensive_matrix.py --timesteps 50000 --seeds 42 123 456 789 2024 --eval_episodes 5 --eval_horizon 1800
```

### Step 5: Generate Statistical Tables & Hypothesis Tests
Aggregates run manifests, verifies cross-artifact consistency, and exports LaTeX comparison tables:
```powershell
python analysis/statistical_analyzer.py
```
*Artifacts Generated:* `results/tables/summary_table.csv`, `results/tables/hypothesis_tests.csv`.

### Step 6: Generate Publication Figures
Generates publication-ready vector graphics:
```powershell
python analysis/plot_generator.py
```
*Figures Generated:*
- `results/figures/learning_curves.png`: Mean reward and waiting time across training seeds with 95% confidence bands.
- `results/figures/interaction_matrix_heatmap.png`: SCOIL asymmetric cross-coupling heatmap.
- `results/figures/fairness_disparity_comparison.png`: Corridor delay disparity violin plots.
- `results/figures/ablation_radar_chart.png`: Multi-objective Pareto trade-off comparisons.

---

## 3. Provenance & Artifact Verification

Every executed experiment automatically records an immutable run ledger in `results/runs/<run_id>/`:
- `config_snapshot.yaml`: Exact configuration hash and hyperparameters.
- `manifest.json`: Git commit SHA, environment versions, seed mapping, and checksums.
- `metrics_step.csv`: Per-step streaming log of physical and normalized objectives, action decisions, preference weights, and shield interventions.
- `summary.json`: Mathematical summary statistics across the complete episode.
- `checkpoints/`: Model weight checkpoints with SHA-256 fingerprints.

Automated manifest verification can be audited at any time:
```powershell
python common/verification.py --runs_root results/runs
```

