import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os

# ==========================================
# 1. SETUP & DATA DOWNLOAD
# ==========================================
print("--- RUNNING PANTHEON+ TENSION TEST (QUADRUPLE CONCORDANCE) ---")
print("Objective: Verify Metric 1 (Absolute Magnitude) for H0 = 73.4")

DATA_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
COV_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"

DATA_FILE = "Pantheon+SH0ES.dat"
COV_FILE = "Pantheon+SH0ES_STAT+SYS.cov"

def download_file(url, filename):
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
            print(f"Error: {e}")
            exit()
    else:
        print(f"Found {filename}, using local copy.")

download_file(DATA_URL, DATA_FILE)
download_file(COV_URL, COV_FILE)

# ==========================================
# 2. DATA PROCESSING
# ==========================================
print("Loading Pantheon+ Data...")
df = pd.read_csv(DATA_FILE, sep=r'\s+')
mask = df['zHD'] > 0.01
df_clean = df[mask].reset_index(drop=True)
print(f"Supernovae (z > 0.01): {len(df_clean)}")

print("Processing Covariance Matrix...")
with open(COV_FILE, 'r') as f:
    content = f.read().split()

data_flat = np.array(content, dtype=float)
N_FULL = 1701 
if len(data_flat) == N_FULL**2 + 1:
    cov_matrix = data_flat[1:].reshape((N_FULL, N_FULL))
else:
    cov_matrix = data_flat.reshape((N_FULL, N_FULL))

indices = np.where(mask)[0]
cov_filtered = cov_matrix[np.ix_(indices, indices)]

print("Inverting Matrix...")
try:
    inv_cov = np.linalg.inv(cov_filtered)
except:
    inv_cov = np.linalg.pinv(cov_filtered)

# ==========================================
# 3. PHYSICS ENGINE (CORRECTED FOR ADD 26.PDF)
# ==========================================
C_LIGHT = 299792.458
H0_PLANCK = 67.4   # Baseline
OM_PLANCK = 0.315
OL_PLANCK = 1.0 - OM_PLANCK

# --- NEW PHYSICS PARAMETERS ---
# The Lepton Viscosity (eta=0.21) predicts H0 = 73.4, not 73.74.
H0_VACUUM = 73.40  
MAG_SHIFT = -5 * np.log10(H0_VACUUM / H0_PLANCK) # Result: approx -0.185

Z_TRANS = 0.65     # Percolation Threshold
WIDTH = 0.15       

print(f"Physics Check:")
print(f"  Target H0: {H0_VACUUM} (Matches Lepton Sum Rule)")
print(f"  Mag Shift: {MAG_SHIFT:.4f} mag")

def get_planck_mu(z_array):
    z_max = np.max(z_array)
    z_grid = np.linspace(0, z_max * 1.01, 2000)
    E_inv = 1.0 / np.sqrt(OM_PLANCK * (1 + z_grid)**3 + OL_PLANCK)
    comoving = np.cumsum(E_inv) * (z_grid[1] - z_grid[0])
    dl_mpc = (1 + z_array) * (C_LIGHT / H0_PLANCK) * np.interp(z_array, z_grid, comoving)
    return 5 * np.log10(dl_mpc) + 25

# MODEL A: Planck Baseline
mu_planck = get_planck_mu(df_clean['zHD'].values)

# MODEL B: Vacuum Elastodynamics (Quadruple Concordance)
# Note: Background expansion uses LINEAR scaling (n=1) as discussed in Section 7.6.1
sigmoid = 1.0 / (1 + np.exp((df_clean['zHD'].values - Z_TRANS) / WIDTH))
viscous_correction = MAG_SHIFT * sigmoid
mu_viscous = mu_planck + viscous_correction

# ==========================================
# 4. STATISTICS
# ==========================================
R_planck = df_clean['MU_SH0ES'].values - mu_planck
R_viscous = df_clean['MU_SH0ES'].values - mu_viscous

chi2_planck = R_planck.T @ inv_cov @ R_planck
chi2_viscous = R_viscous.T @ inv_cov @ R_viscous
d_chi2 = chi2_viscous - chi2_planck

print("\n" + "="*50)
print(f"FINAL METRIC 1 RESULTS (H0={H0_VACUUM})")
print("="*50)
print(f"Chi2 (Planck 67.4):   {chi2_planck:.2f}")
print(f"Chi2 (Vacuum 73.4):   {chi2_viscous:.2f}")
print(f"Delta Chi2:           {d_chi2:.2f}")
print("-" * 50)
if d_chi2 < -50:
    print("VERDICT: SUCCESS. Resolves Hubble Tension.")
else:
    print("VERDICT: FAIL. Tension remains.")

# ==========================================
# 5. PLOTTING
# ==========================================
plt.figure(figsize=(10,6))
plt.errorbar(df_clean['zHD'], R_planck, yerr=df_clean['MU_SH0ES_ERR_DIAG'], 
             fmt='o', color='lightgrey', alpha=0.3, label='Pantheon+ Residuals')

z_sort = np.sort(df_clean['zHD'])
curve = MAG_SHIFT / (1 + np.exp((z_sort - Z_TRANS) / WIDTH))
plt.plot(z_sort, curve, 'r-', linewidth=3, label=f'Vacuum Model (H0={H0_VACUUM})')

plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Redshift z')
plt.ylabel(r'Magnitude Residual $\mu - \mu_{Planck}$')
plt.title(f'Resolution of Hubble Tension\n$\Delta\chi^2 = {d_chi2:.1f}$ (Target H0={H0_VACUUM})')
plt.legend()
plt.ylim(-0.6, 0.4)
plt.grid(True, alpha=0.3)
plt.savefig("Figure4a_Metric1_Corrected.png")
plt.show()
