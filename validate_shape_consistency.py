import numpy as np
import pandas as pd
import requests
import os

print("--- SHAPE CONSISTENCY TEST (STEEL MAN ARGUMENT) ---")
print("Objective: Verify expansion history shape matches data (Marginalized H0).")

# ==========================================
# 1. SETUP & DATA
# ==========================================
DATA_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
COV_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"
DATA_FILE = "Pantheon+SH0ES.dat"
COV_FILE = "Pantheon+SH0ES_STAT+SYS.cov"

def download_file(url, filename):
    if not os.path.exists(filename):
        try:
            r = requests.get(url, stream=True)
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"Error: {e}")

download_file(DATA_URL, DATA_FILE)
download_file(COV_URL, COV_FILE)

# Load Data
df = pd.read_csv(DATA_FILE, sep=r'\s+')
mask = df['zHD'] > 0.01
df_clean = df[mask].reset_index(drop=True)

# Load Covariance
with open(COV_FILE, 'r') as f:
    data_flat = np.array(f.read().split(), dtype=float)
N = 1701
if len(data_flat) == N*N + 1:
    cov_matrix = data_flat[1:].reshape((N, N))
else:
    cov_matrix = data_flat.reshape((N, N))

indices = np.where(mask)[0]
cov_filtered = cov_matrix[np.ix_(indices, indices)]
inv_cov = np.linalg.pinv(cov_filtered)

# ==========================================
# 2. PHYSICS MODELS
# ==========================================
C_LIGHT = 299792.458
OM = 0.315
OL = 1.0 - OM

# PARAMETERS (Theoretical)
Z_TRANS = 0.65    # Percolation Threshold
WIDTH = 0.1       # Transition Width
H0_LATE = 73.04   # SH0ES (Today)
H0_EARLY = 67.4   # Planck (Early)

def integrate_distance_vectorized(z_values, h_func):
    z_max = np.max(z_values)
    z_grid = np.linspace(0, z_max*1.05, 2000)
    h_grid = h_func(z_grid)
    integrand = C_LIGHT / h_grid
    comoving = np.cumsum(integrand) * (z_grid[1] - z_grid[0])
    return np.interp(z_values, z_grid, comoving)

# MODEL A: LCDM (Standard)
def h_lcdm(z):
    return H0_LATE * np.sqrt(OM * (1 + z)**3 + OL)

# MODEL B: Vacuum Elastodynamics (Corrected Direction)
def h_viscous(z):
    # Sigmoid Function:
    # z=0   -> exp(-6.5) ~ 0 -> Sigmoid ~ 1.0
    # z>>1  -> exp(+large)   -> Sigmoid ~ 0.0
    sigmoid = 1.0 / (1.0 + np.exp((z - Z_TRANS) / WIDTH))
    
    # CORRECTED MIXING LOGIC:
    # We want H0_LATE (73) when Sigmoid=1 (Today)
    # We want H0_EARLY (67) when Sigmoid=0 (Past)
    h0_eff = H0_EARLY + (H0_LATE - H0_EARLY) * sigmoid
    
    return h0_eff * np.sqrt(OM * (1 + z)**3 + OL)

# Calculate Distances
dl_lcdm = (1 + df_clean['zHD']) * integrate_distance_vectorized(df_clean['zHD'], h_lcdm)
mu_lcdm = 5 * np.log10(dl_lcdm) + 25

dl_visc = (1 + df_clean['zHD']) * integrate_distance_vectorized(df_clean['zHD'], h_viscous)
mu_visc = 5 * np.log10(dl_visc) + 25

# ==========================================
# 3. STATISTICAL TEST
# ==========================================
def calc_marginalized_chi2(mu_model, mu_data, inv_c):
    residuals = mu_data - mu_model
    # Analytic Marginalization
    W = np.sum(inv_c)
    W_R = np.sum(np.dot(residuals.T, inv_c))
    A = W_R / W
    resid_final = residuals - A
    return resid_final.T @ inv_c @ resid_final

mu_data = df_clean['MU_SH0ES'].values
chi2_lcdm = calc_marginalized_chi2(mu_lcdm, mu_data, inv_cov)
chi2_visc = calc_marginalized_chi2(mu_visc, mu_data, inv_cov)
d_chi2 = chi2_visc - chi2_lcdm

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Standard LCDM Chi2:   {chi2_lcdm:.2f}")
print(f"Viscous Vacuum Chi2:  {chi2_visc:.2f}")
print(f"Delta Chi2:           {d_chi2:.2f}")
print("-" * 50)
if abs(d_chi2) < 2.0:
    print("VERDICT: INDISTINGUISHABLE (Pass).")
    print("The model matches the expansion shape.")
else:
    print("VERDICT: CHECK PARAMETERS.")
print("="*50)
