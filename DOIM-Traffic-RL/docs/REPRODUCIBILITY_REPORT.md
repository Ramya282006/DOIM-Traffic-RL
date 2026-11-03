# DOIM-Traffic-RL: Reproducibility Guide & Environment Verification (Phase 23)

**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard Compliance:** IEEE Open Science, Code & Data Reproducibility Standards  
**Document Version:** 2.0.0  
**Generated UTC:** 2026-11-03 18:09:00 UTC  

---

## 1. Environment Lock & System Specifications

| Component | Specification | Verification Command / Hash |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Enterprise (AMD64) | `cmd.exe /c ver` |
| **Python Runtime** | Python 3.12.10 (x86_64) | `python --version` |
| **PyTorch Engine** | 2.15.0+cpu | `python -c "import torch; print(torch.__version__)"` |
| **Microscopic Simulator** | Eclipse SUMO v1.27.1 / TraCI v1.28.0 | `sumo --version` |
| **JDK Environment** | OpenJDK 17.0.20.1 LTS | `java -version` |
| **Historical Archive** | 246 Artifacts (SHA-256 Verified) | `results/historical_archive_manifest.json` |

---

## 2. One-Click Setup & Execution Instructions

### 2.1 Virtual Environment Initialization
```powershell
# Create Python 3.12 virtual environment
python -m venv .venv_doim
.\.venv_doim\Scripts\Activate.ps1

# Install PyTorch CPU and simulation dependencies
pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install traci==1.28.0 sumolib==1.28.0 gymnasium==1.4.0 numpy scipy pandas matplotlib seaborn pytest pyyaml
```

### 2.2 Complete Unit & Integration Test Suite
```powershell
cd DOIM-Traffic-RL
$env:PYTHONPATH = "."
..\.venv_doim\Scripts\python.exe -m pytest tests/ -v
```
*Expected Result:* **152 passed, 1 skipped (optional stable_baselines3), 0 failed.**

### 2.3 Executing Multi-Scenario Baseline Matrix
```powershell
cd DOIM-Traffic-RL
$env:PYTHONPATH = "."

# Run rapid verification smoke test (128 steps)
..\.venv_doim\Scripts\python.exe experiments/run_smoke_test.py

# Run standard pilot training (500 steps, gate-approved)
..\.venv_doim\Scripts\python.exe experiments/run_comprehensive_matrix.py --scenarios normal --algorithms DOIM_IA_MOPPO --train-steps 500 --seeds 42

# Run full confirmatory convergence matrix (50,000 steps per seed)
# Note: Requires ~38.8 hours of CPU compute across all 70 runs
..\.venv_doim\Scripts\python.exe experiments/run_comprehensive_matrix.py --tier main_convergence --confirm-compute-budget
```

### 2.4 Statistical Aggregation & Publication Table Export
```powershell
cd DOIM-Traffic-RL
$env:PYTHONPATH = "."
..\.venv_doim\Scripts\python.exe -c "
from analysis.statistical_analyzer import StatisticalAnalyzer
analyzer = StatisticalAnalyzer()
df = analyzer.load_confirmatory_benchmark_runs()
print(f'Certified Confirmatory Runs Loaded: {len(df)}')
analyzer.export_latex_tables(output_markdown='results/tables/confirmatory_summary.md')
"
```
