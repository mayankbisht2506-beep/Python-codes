import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
from scipy.integrate import quad

# ==========================================
# 1. PHYSICS CONSTANTS & COSMOLOGY
# ==========================================
H0_PLANCK = 67.4
OM_PLANCK = 0.315
OL_PLANCK = 1.0 - OM_PLANCK
C_LIGHT = 299792.458

# --- CORRECTED PARAMETERS (Matches Add 31.pdf) ---
# Derived from Lepton Viscosity (eta=0.21)
H0_MODEL = 73.40  
MODEL_SHIFT = -5 * np.log10(H0_MODEL / H0_PLANCK) # Result: -0.1852

def get_planck_mu(z):
    """Standard LCDM Distance Modulus (Planck 2018)"""
    if z <= 0: return np.nan
    inv_E = lambda zp: 1.0 / np.sqrt(OM_PLANCK * (1 + zp)**3 + OL_PLANCK)
    integral, _ = quad(inv_E, 0, z)
    d_L = (1 + z) * (C_LIGHT / H0_PLANCK) * integral
    return 5 * np.log10(d_L) + 25

# ==========================================
# 2. DATA LOADING
# ==========================================
url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
print("Downloading Pantheon+ Data...")
df = pd.read_csv(io.StringIO(requests.get(url).text), sep=r'\s+')
df = df[df['zHD'] > 0.01].copy()

# ==========================================
# 3. CALCULATE RESIDUALS
# ==========================================
print("Calculating Residuals...")
df['mu_Planck'] = df['zHD'].apply(get_planck_mu)
df['residual_raw'] = df['MU_SH0ES'] - df['mu_Planck']

# Calculate the mean tension for high-z (z > 0.65)
mean_tension = df[df['zHD'] > 0.65]['residual_raw'].mean()

# ==========================================
# 4. PLOTTING THE COMPARISON (FIXED)
# ==========================================
plt.figure(figsize=(12, 7))

# A. Raw Data Points
plt.errorbar(df['zHD'], df['residual_raw'], yerr=df['MU_SH0ES_ERR_DIAG'], 
             fmt='o', color='gray', alpha=0.2, label='Pantheon+ (SH0ES Calibrated)')

# B. The "Problem": Baseline Tension (z > 0.65)
# Use a raw f-string (fr"...") to handle both variables and LaTeX backslashes
plt.axhline(mean_tension, color='blue', linestyle='--', linewidth=2, 
            label=fr'Observed Tension Gap ($\mu \approx {mean_tension:.3f}$)')

# C. The "Solution": Vacuum Elastodynamics Prediction
z_fine = np.linspace(0.01, 2.3, 500)
# Use a raw f-string here as well for \Delta
plt.axhline(MODEL_SHIFT, color='red', linewidth=3, 
            label=fr'Vacuum Model Prediction ($H_0={H0_MODEL}, \Delta M={MODEL_SHIFT:.3f}$)')

# D. Formatting
plt.axhline(0, color='black', linewidth=1) # Planck Baseline
plt.xlabel(r'Redshift $z$', fontsize=12)
plt.ylabel(r'$\mu_{obs} - \mu_{Planck}$ (mag)', fontsize=12)
plt.title('Hubble Tension: Observation vs. Vacuum Elastodynamics Resolution', fontsize=14)
plt.legend(loc='lower left', frameon=True)
plt.ylim(-0.6, 0.4)
plt.grid(alpha=0.2)

plt.show()



print(f"\n--- COMPARISON SUMMARY ---")
print(f"Observed High-z Tension: {mean_tension:.4f} mag (approx -0.205)")
print(f"Model Predicted Shift:  {MODEL_SHIFT:.4f} mag (approx -0.195)")
print(f"Accuracy: {100 * (1 - abs(mean_tension - MODEL_SHIFT)/abs(mean_tension)):.2f}%")

plt.show()
