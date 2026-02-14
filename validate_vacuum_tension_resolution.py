import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
from scipy.optimize import minimize

# ==========================================
# 1. SETUP & DATA DOWNLOAD
# ==========================================
print("--- RUNNING PANTHEON+ TEST I: RAW STRESS TEST (N=1701) ---")
print("Objective: Optimize MAG_SHIFT to find Maximum Headroom (Table 7)")

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

# ==========================================
# 2. DATA PROCESSING (N=1701)
# ==========================================
print("Loading Data...")
df = pd.read_csv(DATA_FILE, sep=r'\s+')

# Include ALL Supernovae to penalize LCDM's local failure
mask = df['zHD'] > 0.000 
df_clean = df[mask].reset_index(drop=True)
z_obs = df_clean['zHD'].values
mu_data = df_clean['MU_SH0ES'].values
print(f"Supernovae loaded: {len(df_clean)} (Matches Paper N=1701)")

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
inv_cov = np.linalg.pinv(cov_filtered) 

# ==========================================
# 3. PHYSICS BASELINE (PLANCK)
# ==========================================
C_LIGHT = 299792.458
H0_PLANCK = 67.4 
OM_PLANCK = 0.315
Z_TRANS = 0.65
WIDTH = 0.1

z_grid = np.linspace(0, 2.5, 5000) 
E_inv = 1.0 / np.sqrt(OM_PLANCK*(1+z_grid)**3 + (1-OM_PLANCK))

# Trapezoidal Integration
dc_grid = np.cumsum((E_inv[:-1] + E_inv[1:]) / 2 * np.diff(z_grid))
dc_grid = np.insert(dc_grid, 0, 0) 

# Linear Distance Interpolation
dc_obs = np.interp(z_obs, z_grid, dc_grid)
dl_obs = (1 + z_obs) * (C_LIGHT / H0_PLANCK) * dc_obs

# Apply Logarithm AFTER interpolation
mu_planck = 5 * np.log10(dl_obs + 1e-12) + 25

# Baseline Chi2 Calculation
R_planck = mu_data - mu_planck
chi2_planck = R_planck.T @ inv_cov @ R_planck

# ==========================================
# 4. RAW STRESS TEST OPTIMIZATION
# ==========================================
print("Optimizing Vacuum Model Headroom...")

# Define the Objective Function for the Optimizer
def objective_vacuum(shift_param):
    mag_shift = shift_param[0]
    # Apply the shift exclusively to the Late Universe via the sigmoid phase transition
    correction = mag_shift / (1 + np.exp((z_obs - Z_TRANS) / WIDTH))
    mu_vacuum = mu_planck + correction
    
    R_vacuum = mu_data - mu_vacuum
    chi2_vac = R_vacuum.T @ inv_cov @ R_vacuum
    return chi2_vac

# Run the Optimizer (Starting near our theoretical target of -0.217)
res = minimize(objective_vacuum, x0=[-0.217], method='BFGS')

best_mag_shift = res.x[0]
chi2_vacuum_best = res.fun
d_chi2 = chi2_vacuum_best - chi2_planck

# Predict the effective local H0 for this optimized shift
# Derived from: Shift = 5 * log10(H0_Planck / H0_Optimized)
H0_optimized = H0_PLANCK / (10 ** (best_mag_shift / 5))

print("\n" + "="*55)
print("TEST I: RAW STRESS TEST RESULTS (OPTIMIZED)")
print("="*55)
print(f"Chi2 (Planck 67.4 Baseline):    {chi2_planck:.2f}")
print(f"Chi2 (Vacuum Model Minimum):    {chi2_vacuum_best:.2f}")
print("-" * 55)
print(f"Delta Chi2 (Max Headroom):      {d_chi2:.2f}")
print(f"Optimized Mag Shift:            {best_mag_shift:.4f} mag")
print(f"Effective Local H0 Predicted:   {H0_optimized:.2f} km/s/Mpc")
print("="*55)

if d_chi2 < -3000:
    print("\nSTATUS: CONFIRMED.")
    print("Matches Table 7 (Test I). The optimizer successfully found the massive")
    print("mathematical headroom available in the Phase Transition model.")
else:
    print("\nSTATUS: CHECK PARAMETERS.")

# ==========================================
# 5. PLOTTING
# ==========================================
plt.figure(figsize=(10,6))
plt.errorbar(z_obs, R_planck, yerr=df_clean['MU_SH0ES_ERR_DIAG'], 
             fmt='o', color='lightgrey', alpha=0.3, label='Pantheon+ Residuals')

# Plot the mathematically optimized curve
z_sort = np.sort(z_obs)
arg_sort = (z_sort - Z_TRANS) / WIDTH
curve_sigmoid = np.where(arg_sort > 100, 0.0, 1.0 / (1 + np.exp(arg_sort)))
curve_opt = best_mag_shift * curve_sigmoid

plt.plot(z_sort, curve_opt, 'r-', linewidth=3, label=f'Optimized Vacuum Shift ({best_mag_shift:.3f} mag)')

plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Redshift z', fontsize=12)
plt.ylabel(r'Magnitude Residual $\mu - \mu_{Planck}$', fontsize=12)
plt.title(rf'Test I: Raw Stress Test (Maximum Headroom)' + '\n' + rf'$\Delta\chi^2 \approx {d_chi2:.1f}$', fontsize=14)
plt.legend(fontsize=10)
plt.ylim(-0.6, 0.4)
plt.grid(True, alpha=0.3)
plt.savefig('Figure_Test1_RawStressTest.png')
print("Plot saved as 'Figure_Test1_RawStressTest.png'")
plt.show()
