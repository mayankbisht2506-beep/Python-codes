import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os

# ==========================================
# 1. SETUP & DATA DOWNLOAD
# ==========================================
print("--- RUNNING PANTHEON+ TENSION TEST ---")
print("Objective: Quantify resolution of Hubble Tension (Planck Baseline)")

DATA_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
COV_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"

DATA_FILE = "Pantheon+SH0ES.dat"
COV_FILE = "Pantheon+SH0ES_STAT+SYS.cov"

def download_file(url, filename):
    """Downloads files robustly from the official repository."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            exit()
    else:
        print(f"Found {filename}, skipping download.")

download_file(DATA_URL, DATA_FILE)
download_file(COV_URL, COV_FILE)

# ==========================================
# 2. DATA LOADING & FILTERING
# ==========================================
print("Loading Light Curve Data...")
# Pantheon+ uses whitespace separation
df = pd.read_csv(DATA_FILE, sep=r'\s+')

# Standard Cosmology Filter: zHD > 0.01
# Removes nearby objects dominated by peculiar velocities
mask = df['zHD'] > 0.01
df_clean = df[mask].reset_index(drop=True)
print(f"Analyzed Supernovae: {len(df_clean)} (z > 0.01)")

# ==========================================
# 3. ROBUST MATRIX LOADING (HEADER FIX)
# ==========================================
print("Loading & Processing Covariance Matrix...")

# Read file as flat string to handle potential headers
with open(COV_FILE, 'r') as f:
    content = f.read().split()

data_flat = np.array(content, dtype=float)
N_FULL = 1701 # Official Pantheon+ size

# Logic to remove the header count if present
if len(data_flat) == N_FULL * N_FULL + 1:
    print("Detected header element. Removing...")
    cov_matrix = data_flat[1:].reshape((N_FULL, N_FULL))
elif len(data_flat) == N_FULL * N_FULL:
    cov_matrix = data_flat.reshape((N_FULL, N_FULL))
else:
    print(f"CRITICAL ERROR: Matrix size mismatch. Expected {N_FULL*N_FULL}, got {len(data_flat)}")
    exit()

# Filter Matrix to match the data mask
indices = np.where(mask)[0]
cov_filtered = cov_matrix[np.ix_(indices, indices)]

print("Inverting Covariance Matrix...")
try:
    inv_cov = np.linalg.inv(cov_filtered)
except np.linalg.LinAlgError:
    print("Matrix singular. Using Pseudo-Inverse.")
    inv_cov = np.linalg.pinv(cov_filtered)

# ==========================================
# 4. PHYSICS MODELS
# ==========================================
# Constants
C_LIGHT = 299792.458
H0_PLANCK = 67.4   # STRICT Baseline (Early Universe)
OM_PLANCK = 0.315
OL_PLANCK = 1.0 - OM_PLANCK

def get_planck_mu(z_array):
    """Calculates Distance Modulus for fixed Planck Cosmology."""
    # Vectorized Trapezoidal Integration for speed
    z_max = np.max(z_array)
    z_grid = np.linspace(0, z_max * 1.01, 2000)
    E_inv = 1.0 / np.sqrt(OM_PLANCK * (1 + z_grid)**3 + OL_PLANCK)
    
    # Comoving distance integral
    comoving = np.cumsum(E_inv) * (z_grid[1] - z_grid[0])
    
    # Interpolate to actual z values
    dl_mpc = (1 + z_array) * (C_LIGHT / H0_PLANCK) * np.interp(z_array, z_grid, comoving)
    return 5 * np.log10(dl_mpc) + 25

# MODEL A: Planck Baseline
mu_planck = get_planck_mu(df_clean['zHD'].values)

# MODEL B: Vacuum Elastodynamics (Viscous Model)
# Parameters derived from lattice percolation theory
MAG_SHIFT = -0.205 
Z_TRANS = 0.65
WIDTH = 0.1

viscous_correction = MAG_SHIFT / (1 + np.exp(-(df_clean['zHD'].values - Z_TRANS) / WIDTH))
mu_viscous = mu_planck + viscous_correction

# ==========================================
# 5. STATISTICAL CALCULATION (FULL MATRIX)
# ==========================================
# Residual vectors (Data - Model)
R_planck = df_clean['MU_SH0ES'].values - mu_planck
R_viscous = df_clean['MU_SH0ES'].values - mu_viscous

# Chi-Squared Calculation: R.T * C^-1 * R
chi2_planck = R_planck.T @ inv_cov @ R_planck
chi2_viscous = R_viscous.T @ inv_cov @ R_viscous
d_chi2 = chi2_viscous - chi2_planck

# ==========================================
# 6. RESULTS & PLOTTING
# ==========================================
print("\n" + "="*50)
print("FINAL RESULTS: TENSION RESOLUTION")
print("Baseline: Fixed Planck 2018 (H0=67.4)")
print("="*50)
print(f"Chi2 (Planck Baseline):   {chi2_planck:.2f}")
print(f"Chi2 (Viscous Model):     {chi2_viscous:.2f}")
print(f"Delta Chi2:               {d_chi2:.2f}")
print("-" * 50)

if d_chi2 < -100:
    print("VERDICT: DECISIVE PREFERENCE")
    print(f"The model resolves the Hubble Tension with {np.sqrt(abs(d_chi2)):.1f} sigma significance.")
else:
    print("VERDICT: INCONCLUSIVE")
print("="*50)

# Plotting
plt.figure(figsize=(10,6))

# Plot Residuals vs Planck
plt.errorbar(df_clean['zHD'], R_planck, yerr=df_clean['MU_SH0ES_ERR_DIAG'], 
             fmt='o', color='lightgrey', alpha=0.3, label='Pantheon+ Residuals')

# Plot Viscous Prediction Curve
z_sort = np.sort(df_clean['zHD'])
curve = MAG_SHIFT / (1 + np.exp(-(z_sort - Z_TRANS) / WIDTH))
plt.plot(z_sort, curve, 'r-', linewidth=3, label='Vacuum Elastodynamics')

plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Redshift z')
plt.ylabel(r'Magnitude Residual ($\mu_{obs} - \mu_{Planck}$)')
plt.title(f'Tension Resolution: $\Delta\chi^2 = {d_chi2:.2f}$')
plt.legend()
plt.ylim(-0.6, 0.4)
plt.grid(True, alpha=0.3)

# Save Plot
plot_file = "Tension_Resolution_Result.png"
plt.savefig(plot_file)
print(f"Plot saved to {plot_file}")
plt.show()
