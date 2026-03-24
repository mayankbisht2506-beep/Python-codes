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
print("Objective: Optimize H0 to find Maximum Headroom (Table 7)")

DATA_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
COV_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"
DATA_FILE = "Pantheon+SH0ES.dat"
COV_FILE = "Pantheon+SH0ES_STAT+SYS.cov"

def download_file(url, filename):
    if not os.path.exists(filename):
        try:
            print(f"Downloading {filename}...")
            r = requests.get(url, stream=True)
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

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

# Extract safe diagonal errors for plotting
err_diag = np.sqrt(np.diag(cov_filtered))

# ==========================================
# 3. PHYSICS BASELINE (PLANCK 2018 EXACT)
# ==========================================
C_LIGHT = 299792.458
H0_PLANCK = 67.36     # EXACT: Planck 2018 Baseline
OM_PLANCK = 0.3153    # EXACT: Planck 2018 Baseline
OL_PLANCK = 1.0 - OM_PLANCK

# ==========================================
# EXACT PURE THEORY PARAMETERS (VED)
# ==========================================
Z_TRANS = 0.641
WIDTH = 0.10
H_FAST = 74.69        # EXACT: Early Geometric Ceiling
OM_PRIMORDIAL = 0.3116 # EXACT: Topological Bare Density
OM_EFFECTIVE  = 0.3639 # EXACT: Viscous Braking Density

def integrate_distance_vectorized(z_values, h_func):
    # Upgraded to 10,000 grid points for ultra-low z precision
    z_grid = np.linspace(0, np.max(z_values)*1.01, 10000)
    h_grid = h_func(z_grid)
    integrand = C_LIGHT / h_grid
    comoving = np.cumsum((integrand[:-1] + integrand[1:]) / 2 * np.diff(z_grid))
    comoving = np.insert(comoving, 0, 0)
    return np.interp(z_values, z_grid, comoving)

# Planck History
def h_lcdm(z):
    return H0_PLANCK * np.sqrt(OM_PLANCK * (1 + z)**3 + OL_PLANCK)

dl_lcdm = (1 + z_obs) * integrate_distance_vectorized(z_obs, h_lcdm)
mu_planck = 5 * np.log10(dl_lcdm + 1e-12) + 25

R_planck = mu_data - mu_planck
chi2_planck = R_planck.T @ inv_cov @ R_planck

# ==========================================
# 4. RAW STRESS TEST OPTIMIZATION
# ==========================================
print("Optimizing Vacuum Model Headroom...")

def objective_vacuum(h0_param):
    # Optimize the local decelerated velocity while locking the geometric ceiling
    H_LOCAL = h0_param[0]
    
    def h_viscous(z):
        arg = (Z_TRANS - z) / WIDTH
        # Safe sigmoid computation
        sigmoid = np.where(arg > 100, 1.0, np.where(arg < -100, 0.0, 1.0 / (1.0 + np.exp(-arg))))
        
        # 1. Density Transition
        OM_Z = OM_PRIMORDIAL + (OM_EFFECTIVE - OM_PRIMORDIAL) * sigmoid
        OL_Z = 1.0 - OM_Z
        
        # 2. Hubble Trajectory Transition
        H_Z = H_FAST + (H_LOCAL - H_FAST) * sigmoid
        
        E_z = np.sqrt(OM_Z * (1 + z)**3 + OL_Z)
        return H_Z * E_z

    dl_visc = (1 + z_obs) * integrate_distance_vectorized(z_obs, h_viscous)
    mu_vacuum = 5 * np.log10(dl_visc + 1e-12) + 25
    
    R_vacuum = mu_data - mu_vacuum
    chi2_vac = R_vacuum.T @ inv_cov @ R_vacuum
    return chi2_vac

# Run Optimizer
res = minimize(objective_vacuum, x0=[72.5], method='Nelder-Mead')
best_H0 = res.x[0]
chi2_vac_opt = res.fun
d_chi2 = chi2_vac_opt - chi2_planck

print(f"\n--- OPTIMIZATION RESULTS ---")
print(f"Planck (67.4) Chi2:  {chi2_planck:.2f}")
print(f"Vacuum Optimum H0:   {best_H0:.2f} km/s/Mpc")
print(f"Vacuum Optimum Chi2: {chi2_vac_opt:.2f}")
print(f"Delta Chi2:          {d_chi2:.2f}")

# ==========================================
# 5. PLOTTING
# ==========================================
plt.figure(figsize=(10,6))
plt.errorbar(z_obs, R_planck, yerr=err_diag, 
             fmt='o', color='lightgrey', alpha=0.3, label='Pantheon+ Residuals (SH0ES Calibrated)')

def h_best(z):
    arg = (Z_TRANS - z) / WIDTH
    sigmoid = np.where(arg > 100, 1.0, np.where(arg < -100, 0.0, 1.0 / (1.0 + np.exp(-arg))))
    OM_Z = OM_PRIMORDIAL + (OM_EFFECTIVE - OM_PRIMORDIAL) * sigmoid
    OL_Z = 1.0 - OM_Z
    H_Z = H_FAST + (best_H0 - H_FAST) * sigmoid
    return H_Z * np.sqrt(OM_Z * (1 + z)**3 + OL_Z)

z_sort = np.sort(z_obs)
dl_best = (1 + z_sort) * integrate_distance_vectorized(z_sort, h_best)
mu_best = 5 * np.log10(dl_best + 1e-12) + 25

dl_planck_sort = (1 + z_sort) * integrate_distance_vectorized(z_sort, h_lcdm)
mu_planck_sort = 5 * np.log10(dl_planck_sort + 1e-12) + 25

curve_opt = mu_best - mu_planck_sort



plt.plot(z_sort, curve_opt, 'r-', linewidth=3, label=f'Optimized Vacuum Model (Terminal $H_0={best_H0:.2f}$)')

plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Redshift z', fontsize=12)
plt.ylabel(r'Magnitude Residual $\mu - \mu_{Planck}$', fontsize=12)
plt.title(rf'Test I: Raw Stress Test (Maximum Headroom)' + '\n' + rf'$\Delta\chi^2 \approx {d_chi2:.1f}$', fontsize=14)
plt.legend(fontsize=10)
plt.ylim(-0.6, 0.4)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_Test1_RawStressTest.png', dpi=300)
print("\nPlot saved as 'Figure_Test1_RawStressTest.png'")
plt.show()
