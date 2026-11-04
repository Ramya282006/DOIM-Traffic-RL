"""
DOIM-PPO Phase 35: Forensic Investigation of Numerical and Methodological Discrepancies
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

PKG_ROOT = Path("D:/DOIM-Traffic-RL/DOIM-Traffic-RL")
RUNS_DIR = PKG_ROOT / "results/runs"

print("=== DISCREPANCY 1: DOIM-PPO SHIELD INTERVENTIONS ===")
for r in sorted(RUNS_DIR.glob("*IA_MOPPO*")):
    sf = r / "summary.json"
    mf = r / "metrics_step.csv"
    if sf.exists():
        with open(sf) as f:
            s = json.load(f)
        sum_shield = s.get("total_shield_interventions")
        csv_shield = None
        if mf.exists():
            df = pd.read_csv(mf)
            if "shield_interventions" in df.columns:
                csv_shield = int(df["shield_interventions"].sum())
        print(f"Run: {r.name}")
        print(f"  steps={s.get('total_steps')} | mode={s.get('evaluation_mode')} | eval_seed={s.get('eval_seed')} | train_seed={s.get('train_seed')}")
        print(f"  summary.json shield={sum_shield} | metrics_step.csv shield={csv_shield}")

print("\n=== DISCREPANCY 2: HYPERVOLUME VALUES (DOIM vs No-SCOIL vs Rolling Pearson) ===")
# Inspect the ablation runs
ab_runs = {
    "DOIM-PPO (Full)": "IA_MOPPO_north_south_surge_s42_20261103_195855",
    "DOIM No-Interaction": "DOIM_no_interaction_north_south_surge_s42_20261103_201117",
    "DOIM Rolling Pearson": "DOIM_rolling_pearson_north_south_surge_s42_20261103_201452",
    "MLP-PPO": "MLP_PPO_north_south_surge_s42_20261103_200901",
}
for name, r_name in ab_runs.items():
    r_dir = RUNS_DIR / r_name
    if (r_dir / "metrics_step.csv").exists():
        df = pd.read_csv(r_dir / "metrics_step.csv")
        sf = json.load(open(r_dir / "summary.json"))
        print(f"{name:25s}:")
        print(f"  mean_w_net={df['raw_waiting'].mean():.4f}")
        print(f"  mean_queue={df['raw_queue'].mean():.4f}")
        print(f"  mean_co2={df['raw_co2'].mean():.2f}")
        print(f"  throughput={sf.get('total_throughput')}")
        print(f"  delay_disparity={df['raw_fairness'].mean():.4f}")
        print(f"  shield={df['shield_interventions'].sum()}")

print("\n=== DISCREPANCY 3: WAITING TIME & DELAY DEFINITIONS ===")
# Inspect Little's Law ratio W_net / N_bar
df_doim = pd.read_csv(RUNS_DIR / "IA_MOPPO_north_south_surge_s42_20261103_195855/metrics_step.csv")
print("DOIM raw_waiting mean:", df_doim['raw_waiting'].mean())
print("DOIM raw_waiting sum:", df_doim['raw_waiting'].sum())

print("\n=== DISCREPANCY 4: FAIRNESS (Jain vs Disparity) ===")
print("DOIM raw_fairness min/max/mean:", df_doim['raw_fairness'].min(), df_doim['raw_fairness'].max(), df_doim['raw_fairness'].mean())

print("\n=== DISCREPANCY 5: THROUGHPUT (Instantaneous vs Cumulative) ===")
print("DOIM raw_throughput sum:", df_doim['raw_throughput'].sum())
print("DOIM raw_throughput last:", df_doim['raw_throughput'].iloc[-1])

print("\n=== DISCREPANCY 6: CO2 (mg/s rate vs Total kg) ===")
print("DOIM raw_co2 mean (mg/s):", df_doim['raw_co2'].mean())
print("DOIM raw_co2 total (kg):", df_doim['raw_co2'].sum() * 1e-6)

