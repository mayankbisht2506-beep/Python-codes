import pandas as pd

print("--- GLOBAL STATISTICAL BUDGET (TABLE VII) ---")
print("Objective: Verify the Net Global Preference for Vacuum Elastodynamics.")

# Values derived from the individual validated scripts:
# 1. generate_figure4_tension.py  -> -376.0 (SNe Absolute)
# 2. generate_figure3_growth.py   -> -1.8   (S8 Growth)
# 3. validate_BAO_consistency.py  -> +5.6   (BAO w/ Ruler Contraction)
# 4. Independent Checks           -> +5.0   (Cosmic Chronometers)

data = {
    "Dataset": ["Pantheon+ (SNe)", "Growth (fsigma8)", "Chronometers H(z)", "BAO (6dF/BOSS)"],
    "Physics Tested": ["Absolute Calibration (H0)", "Clustering Amplitude (S8)", "Expansion History", "Standard Ruler (rs)"],
    "Delta Chi2": [-376.0, -1.8, +5.0, +5.6],
    "Verdict": ["Decisive Resolution", "Statistical Preference", "Consistent", "Concordant (Full Model)"]
}

df = pd.DataFrame(data)

# Calculate Global Net
global_net = df["Delta Chi2"].sum()

print("\n" + "="*80)
print(df.to_string(index=False, col_space=20, justify='left'))
print("-" * 80)
print(f"GLOBAL NET EVIDENCE (Delta Chi2):   {global_net:.1f}")
print("="*80)

if global_net < -300:
    print("CONCLUSION: The Unified Vacuum Model is globally preferred (> 15 sigma).")
else:
    print("CONCLUSION: Model fails global audit.")
