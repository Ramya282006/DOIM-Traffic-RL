# DOIM-Traffic-RL: Environment Lock & Hardware Provenance Specification (Phase A)

**Document Version:** 1.0.0  
**Generated UTC:** 2026-11-03 17:05:00 UTC  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard Compliance:** IEEE T-ITS Reproducibility & Research Data Guidelines  

---

## 1. System Hardware & Host Architecture

| Component | Specification |
| :--- | :--- |
| **Host Operating System** | Microsoft Windows 11 Enterprise / Pro (AMD64, 64-bit) |
| **CPU Architecture** | x86_64 / AMD64 Multiprocessor |
| **Physical Working Tree** | `D:\DOIM-Traffic-RL` |
| **Active Python Environment** | `D:\DOIM-Traffic-RL\.venv_doim` |
| **Java JDK Runtime** | OpenJDK 17.0.20.1 LTS (`C:\Program Files\Microsoft\jdk-17.0.20.101-hotspot\bin\java.exe`) |

---

## 2. Microscopic Simulator & TraCI Interface

| Package / Tool | Version | Executable / Library Location | Integrity / Status |
| :--- | :---: | :--- | :---: |
| **Eclipse SUMO Engine** | `1.27.1` | `C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe` | Verified Native Executable |
| **`SUMO_HOME` Variable** | — | `C:\Program Files (x86)\Eclipse\Sumo\` | Configured in Environment |
| **TraCI Python Interface** | `1.28.0` | `.venv_doim\Lib\site-packages\traci` | Verified Python Package |
| **SumoLib Package** | `1.28.0` | `.venv_doim\Lib\site-packages\sumolib` | Verified Python Package |

---

## 3. Core Deep Learning & Scientific Python Stack

| Framework / Library | Installed Version | Source / Wheel | Integrity Check |
| :--- | :---: | :--- | :---: |
| **Python** | `3.12.10` | CPython Windows 64-bit | Verified |
| **PyTorch (`torch`)** | `2.15.0+cpu` | Official PyTorch CPU Wheel (`download.pytorch.org/whl/cpu`) | Clean execution, no SAC restriction |
| **Gymnasium** | `1.4.0` | Official PyPI Wheel | Clean execution |
| **NumPy** | `2.5.4` | Official PyPI Wheel | Clean execution |
| **SciPy** | `1.18.1` | Official PyPI Wheel | Clean execution |
| **Pandas** | `3.1.0` | Official PyPI Wheel | Clean execution |
| **PyYAML** | `6.0.3` | Official PyPI Wheel | Clean execution |
| **Matplotlib** | `3.11.2` | Official PyPI Wheel | 300 DPI Publication Quality |
| **Seaborn** | `0.13.2` | Official PyPI Wheel | Clean execution |
| **Pytest** | `9.2.0` | Official PyPI Wheel | 126 passed, 1 skipped |

---

## 4. Dependencies Lockfile Hash

- **`requirements.txt` SHA-256:**
  `86214db21a28a3f5aef34dae5ee9352ef9c7ad6e0ca8fb36c1e55042898953f9`
- **Historical Archive Manifest SHA-256:**
  Calculated dynamically in `results/historical_archive_manifest.json` indexing 246 experiment files.

---

## 5. Security & Platform Isolation Guarantee

1. **Smart App Control (SAC) Compatibility:** Unmodified, official PyPI and PyTorch CPU binary wheels execute cleanly on Windows 11 without security policy overrides or developer certificates.
2. **Immutable Historical Results:** All 49 historical run directories in `results/runs/` are sealed and protected with SHA-256 checksums in `results/historical_archive_manifest.json`.
3. **No In-Place Result Overwrites:** Diagnostic runs ($\le 128$ steps) are segregated from pilot ($\ge 5,000$ steps) and confirmatory ($\ge 10,000$ steps) benchmark matrices.
