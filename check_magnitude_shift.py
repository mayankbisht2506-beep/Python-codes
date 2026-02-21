# Uncomment the line below if running in Google Colab / Jupyter
# !pip install pandas matplotlib scipy requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
from scipy.integrate import quad

print("--- PANTHEON+ THEORY VALIDATION (EXACT PRECISION) ---")
print("Objective: Verify Vacuum Prediction (-0.232 mag) against Deep Field Data")

# ==========================================
# 1. PHYSICS & COSMOLOGY SETUP (High-Precision)
# ==========================================
# Exact Planck 2018 Baseline
H0_PLANCK = 67.36    # UPDATED to Exact Root
OM_PLANCK = 0.3153   # UPDATED to Exact Root
OL_PLANCK = 1.0 - OM_PLANCK
C_LIGHT   = 299792.458

# --- EXACT THEORETICAL INPUTS ---
# Section 7.3: Geometric(geo) + Luminosity(lum) + Viscous(visc)
# -0.642 + 0.160 + 0.250 = -0.232 mag
MODEL_SHIFT = -0.232

# Observational Error Budget for Supernovae (approx 1.5-2.0%)
# This is the standard "ruler error" for checking tension.
SIGMA_OBS = 0.035 



def get_planck_mu(z):
    """Calculates Distance Modulus for Standard Planck Cosmology"""
    if z <= 0.001: return np.nan
    def E_inv(z_prime):
        return 1.0 / np.sqrt(OM_PLANCK * (1 + z_prime)**3 + OL_PLANCK)
    dc, _ = quad(E_inv, 0, z)
    dl_mpc = (C_LIGHT / H0_PLANCK) * (1 + z) * dc
    return 5 * np.log10(dl_mpc) + 25

# ==========================================
# 2. DATA LOADING
# ==========================================
url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
print("Downloading Pantheon+ Data...")
try:
    s = requests.get(url).text
    df = pd.read_csv(io.StringIO(s), sep=r'\s+')
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# Filter for robust data (exclude very low-z local flow noise)
df = df[df['zHD'] > 0.01].copy()

# ==========================================
# 3. CALCULATE RESIDUALS
# ==========================================
print("Calculating Residuals (Data - Exact Planck)...")
df['mu_Planck'] = df['zHD'].apply(get_planck_mu)
df['residual']  = df['MU_SH0ES'] - df['mu_Planck']
df['weights']   = 1.0 / (df['MU_SH0ES_ERR_DIAG']**2)

# ==========================================
# 4. REGIONAL ANALYSIS
# ==========================================
# A. Weighted Mean (Diluted by Local Physics)
obs_weighted = np.average(df['residual'], weights=df['weights'])

# B. Deep Field Unweighted (The "Pure" Signal at z > 0.641)
# Paper identifies z approx 0.641 as the exact geometric percolation threshold
Z_TRANS = 0.641
deep_data = df[df['zHD'] > Z_TRANS]
obs_deep_pure = deep_data['residual'].mean()

# ==========================================
# 5. STATISTICAL VERIFICATION
# ==========================================
print("\n" + "="*60)
print("SCIENTIFIC VERIFICATION RESULTS")
print("="*60)
print(f"THEORY PREDICTION (Eq. 92):   {MODEL_SHIFT:.3f} mag")
print(f"OBSERVATIONAL ERROR (Sigma):  {SIGMA_OBS:.3f} mag")
print("-" * 60)
print(f"1. Global Weighted Mean:      {obs_weighted:.4f} mag")
print(f"2. Deep Field Mean (z>{Z_TRANS}): {obs_deep_pure:.4f} mag")
print("-" * 60)

# Calculate Z-Score (Sigma Match)
delta_deep = abs(obs_deep_pure - MODEL_SHIFT)
z_score    = delta_deep / SIGMA_OBS

print(f"Difference (Deep - Theory):   {delta_deep:.4f} mag")
print(f"Z-Score (Sigma):              {z_score:.2f} σ")

if z_score < 1.0:
    print("-" * 60)
    print(f"VERDICT: EXCELLENT MATCH (< 1 sigma)")
    print("The Vacuum prediction is statistically indistinguishable")
    print("from the Deep Field Pantheon+ data.")
    print("-" * 60)
elif z_score < 2.0:
    print(f"VERDICT: STATISTICAL MATCH (< 2 sigma)")
else:
    print("VERDICT: TENSION REMAINS")
print("="*60)

# ==========================================
# 6. PLOTTING
# ==========================================
plt.figure(figsize=(12, 7))

# Plot Data
plt.errorbar(df['zHD'], df['residual'], yerr=df['MU_SH0ES_ERR_DIAG'], 
             fmt='o', color='lightgray', alpha=0.3, label=f'Local Data ($z < {Z_TRANS}$)')
plt.errorbar(deep_data['zHD'], deep_data['residual'], yerr=deep_data['MU_SH0ES_ERR_DIAG'], 
             fmt='o', color='gray', alpha=0.8, label=f'Deep Field ($z > {Z_TRANS}$)')

# Plot Reference Lines
plt.axhline(0, color='black', linewidth=1.5, linestyle='--', label='Planck 2018 Baseline')
plt.axhline(obs_deep_pure, color='green', linestyle=':', linewidth=3, 
            label=f'Deep Field Data Mean: {obs_deep_pure:.3f}')
plt.axhline(MODEL_SHIFT, color='red', linewidth=3, 
            label=f'Vacuum Theory: {MODEL_SHIFT:.3f}')

# Fill the "1-Sigma" Success Zone around the Theory
plt.fill_between([-0.1, 2.5], MODEL_SHIFT - SIGMA_OBS, MODEL_SHIFT + SIGMA_OBS, 
                 color='red', alpha=0.1, label=r'Theory 1$\sigma$ Match Zone')

plt.xlabel('Redshift $z$', fontsize=12)
plt.ylabel(r'$\mu_{obs} - \mu_{Planck}$ (mag)', fontsize=12)
plt.title(rf'Pantheon+ Deep Field Verification: Theory ({MODEL_SHIFT}) vs Data ({obs_deep_pure:.3f}) is a {z_score:.2f}$\sigma$ Match', fontsize=14)
plt.legend(loc='lower left', frameon=True, fontsize=11)
plt.ylim(-0.6, 0.4)
plt.xlim(0, 2.3)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Figure_Pantheon_DeepField_Match.pdf', dpi=300)
print("Plot saved as 'Figure_Pantheon_DeepField_Match.pdf'")

plt.show()
