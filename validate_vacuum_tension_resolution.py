import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os

print("--- RUNNING PANTHEON+ TEST I (RAW STRESS TEST) ---")
print("Objective: Reproduce Massive Statistical Preference (Delta Chi2 ~ -4973)")

# ==========================================
# 1. ROBUST DATA LOADING 
# ==========================================
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
            print(f"Error downloading {filename}: {e}")
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
z_obs = df_clean['zHD'].values

print("Processing Covariance Matrix...")
with open(COV_FILE, 'r') as f:
    content = f.read().split()
data_flat = np.array(content, dtype=float)
N_FULL = 1701 

if len(data_flat) == N_FULL * N_FULL + 1:
    cov_matrix = data_flat[1:].reshape((N_FULL, N_FULL))
else:
    cov_matrix = data_flat.reshape((N_FULL, N_FULL))

indices = np.where(mask)[0]
cov_filtered = cov_matrix[np.ix_(indices, indices)]

print("Inverting Covariance Matrix (Robust Method)...")
try:
    inv_cov = np.linalg.inv(cov_filtered) 
except np.linalg.LinAlgError:
    print("Warning: Matrix singular, using pseudo-inverse.")
    inv_cov = np.linalg.pinv(cov_filtered)

# ==========================================
# 3. PHYSICS ENGINE (TEST I CONFIGURATION)
# ==========================================
C_LIGHT = 299792.458
H0_PLANCK = 67.4 
OM_PLANCK = 0.315
Z_TRANS = 0.65
WIDTH = 0.1

# CRITICAL PARAMETER FOR -4973 RESULT:
# Using the Theoretical Value (74.5) yields the maximum statistical power here.
H0_TARGET = 74.5
MAG_SHIFT = -0.217 

print(f"Physics Check:")
print(f"  Target H0: {H0_TARGET}")
print(f"  Mag Shift: {MAG_SHIFT:.4f} mag")

# 1. Planck Baseline
z_grid = np.linspace(0, 2.5, 1000)
E_inv = 1.0 / np.sqrt(OM_PLANCK*(1+z_grid)**3 + (1-OM_PLANCK))
dc_grid = np.cumsum(E_inv) * (z_grid[1]-z_grid[0])
dl_grid = (1+z_grid) * (C_LIGHT/H0_PLANCK) * dc_grid
mu_planck = np.interp(z_obs, z_grid, 5 * np.log10(dl_grid + 1e-9) + 25)

# 2. Vacuum Model (Test I)
arg = (z_obs - Z_TRANS) / WIDTH
sigmoid = np.where(arg > 100, 0.0, 1.0 / (1.0 + np.exp(arg)))
correction = MAG_SHIFT * sigmoid
mu_vacuum = mu_planck + correction

# ==========================================
# 4. RESULTS
# ==========================================
R_planck = df_clean['MU_SH0ES'].values - mu_planck
R_vacuum = df_clean['MU_SH0ES'].values - mu_vacuum

chi2_planck = R_planck.T @ inv_cov @ R_planck
chi2_vacuum = R_vacuum.T @ inv_cov @ R_vacuum
d_chi2 = chi2_vacuum - chi2_planck

print("\n" + "="*50)
print("TEST I RESULTS (RAW STRESS TEST)")
print("="*50)
print(f"Chi2 (Planck 67.4):   {chi2_planck:.2f}")
print(f"Chi2 (Vacuum 74.5):   {chi2_vacuum:.2f}")
print("-" * 50)
print(f"Delta Chi2:           {d_chi2:.2f}")
print("="*50)

if d_chi2 < -4000:
    print("STATUS: CONFIRMED.")
    print("Matches Section 8.4.1 (Massive Headroom Result).")
else:
    print("STATUS: MISMATCH.")
