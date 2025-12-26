import numpy as np
import pandas as pd
import requests
import os

print("--- GLOBAL POINT-TEST (FULL MATRIX) ---")
print("Objective: Compare Standard Model vs. Vacuum Elastodynamics using rigorous statistics.")

# ==========================================
# 1. LOAD DATA (Full Matrix Support)
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
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            print(f"Error: {e}")
            exit()
    else:
        print(f"Found {filename}.")

download_file(DATA_URL, DATA_FILE)
download_file(COV_URL, COV_FILE)

# Load Data
df = pd.read_csv(DATA_FILE, sep=r'\s+')
mask = df['zHD'] > 0.01
df_clean = df[mask].reset_index(drop=True)
z_sn = df_clean['zHD'].values
mu_sn = df_clean['MU_SH0ES'].values

# Load Matrix (The Heavy Lifting)
print("Processing Covariance Matrix (This fixes the Omega_m bias)...")
with open(COV_FILE, 'r') as f:
    content = f.read().split()
data_flat = np.array(content, dtype=float)
N = 1701
if len(data_flat) == N*N + 1:
    cov_matrix = data_flat[1:].reshape((N, N))
else:
    cov_matrix = data_flat.reshape((N, N))

indices = np.where(mask)[0]
cov_filtered = cov_matrix[np.ix_(indices, indices)]
inv_cov = np.linalg.pinv(cov_filtered) # Robust Inverse

# ==========================================
# 2. PHYSICS MODELS
# ==========================================
c_light = 299792.458
RD_FID = 147.78 # Planck anchor

def get_predictions(params, z_sn_array):
    H0, Om, eta = params
    
    # Hubble Function
    def H_z(z):
        E = np.sqrt(Om*(1+z)**3 + (1-Om))
        # Viscosity Transition
        sigmoid = 1.0 / (1.0 + np.exp((z - 0.65) / 0.1))
        amp = (1.0 - eta) + eta * sigmoid
        return H0 * E * amp
    
    # 1. Supernovae Distances
    z_max = 2.5
    z_grid = np.linspace(0, z_max, 1000)
    H_vals = H_z(z_grid)
    dc_grid = np.cumsum(1.0/H_vals) * (z_grid[1] - z_grid[0]) * c_light
    dc_grid[0] = 0
    dl_vec = (1 + z_sn_array) * np.interp(z_sn_array, z_grid, dc_grid)
    mu_vec = 5 * np.log10(dl_vec + 1e-9) + 25
    
    # 2. BAO Predictions (Consensus Points)
    bao_preds = []
    # 6dF (z=0.106, rs/DV)
    H_0106 = H_z(0.106)
    DC_0106 = np.interp(0.106, z_grid, dc_grid)
    DV_0106 = (0.106 * DC_0106**2 * c_light / H_0106)**(1/3)
    bao_preds.append(RD_FID / DV_0106) 
    
    # SDSS (z=0.15, DV/rs)
    H_015 = H_z(0.15)
    DC_015 = np.interp(0.15, z_grid, dc_grid)
    DV_015 = (0.15 * DC_015**2 * c_light / H_015)**(1/3)
    bao_preds.append(DV_015 / RD_FID)
    
    # BOSS (z=0.38, DM/rs)
    DC_038 = np.interp(0.38, z_grid, dc_grid)
    bao_preds.append(DC_038 / RD_FID)
    
    # BOSS (z=0.51, DM/rs)
    DC_051 = np.interp(0.51, z_grid, dc_grid)
    bao_preds.append(DC_051 / RD_FID)

    # BOSS (z=0.61, DM/rs)
    DC_061 = np.interp(0.61, z_grid, dc_grid)
    bao_preds.append(DC_061 / RD_FID)
    
    return mu_vec, np.array(bao_preds)

# BAO Data Vector
bao_obs = np.array([0.336, 4.466, 1512.39/147.78, 1975.22/147.78, 2306.68/147.78])
bao_err = np.array([0.015, 0.168, 25.0/147.78, 30.0/147.78, 37.0/147.78])

# ==========================================
# 3. RUN THE TEST
# ==========================================
# Scenario A: Standard Model (Planck Priors)
# H0=67.4, Om=0.315, Eta=0 (No Viscosity)
mu_A, bao_A = get_predictions([67.4, 0.315, 0.0], z_sn)

# Scenario B: Vacuum Elastodynamics (Paper Claims)
# H0=73.0, Om=0.315, Eta=0.17 (Viscosity ON)
mu_B, bao_B = get_predictions([73.04, 0.315, 0.17], z_sn)

# Calculate Chi2 (Supernovae) - FULL MATRIX
diff_A = mu_sn - mu_A
diff_B = mu_sn - mu_B
chi2_sn_A = diff_A.T @ inv_cov @ diff_A
chi2_sn_B = diff_B.T @ inv_cov @ diff_B

# Calculate Chi2 (BAO)
chi2_bao_A = np.sum(((bao_obs - bao_A)/bao_err)**2)
chi2_bao_B = np.sum(((bao_obs - bao_B)/bao_err)**2)

total_A = chi2_sn_A + chi2_bao_A
total_B = chi2_sn_B + chi2_bao_B

print("\n" + "="*50)
print("FINAL SCIENTIFIC VERDICT")
print("="*50)
print(f"Scenario A: Standard Planck (H0=67.4, eta=0)")
print(f"  SN Chi2:  {chi2_sn_A:.2f}")
print(f"  BAO Chi2: {chi2_bao_A:.2f}")
print(f"  TOTAL:    {total_A:.2f}")
print("-" * 50)
print(f"Scenario B: Vacuum Model (H0=73.0, eta=0.17)")
print(f"  SN Chi2:  {chi2_sn_B:.2f}")
print(f"  BAO Chi2: {chi2_bao_B:.2f}")
print(f"  TOTAL:    {total_B:.2f}")
print("-" * 50)
print(f"DELTA CHI2: {total_B - total_A:.2f}")
print("="*50)

if total_B < total_A:
    print("CONCLUSION: The Vacuum Model is statistically preferred.")
    print("This confirms the paper's claim without MCMC artifacts.")
else:
    print("CONCLUSION: The Standard Model is preferred.")
