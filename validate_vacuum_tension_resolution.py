# Uncomment the line below if running in Google Colab / Jupyter
# !pip install scipy numpy matplotlib pandas requests

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
print("Engine: Exact Covariant Geometry (z_metric truncation + continuous penalties)")

DATA_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
COV_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"
DATA_FILE = "Pantheon+SH0ES.dat"
COV_FILE = "Pantheon+SH0ES_STAT+SYS.cov"

def download_file(url, filename):
    if not os.path.exists(filename):
        try:
            print(f"Downloading {filename}...")
            r = requests.get(url, stream=True)
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

def integrate_distance_vectorized(z_values, h_func):
    """Vectorized numerical integration of comoving distance."""
    z_max = np.max(z_values)
    if z_max <= 0: z_max = 0.01
    
    z_grid = np.linspace(0, z_max * 1.05, 10000)
    h_grid = h_func(z_grid)
    integrand = C_LIGHT / h_grid
    comoving = np.cumsum((integrand[:-1] + integrand[1:]) / 2 * np.diff(z_grid))
    comoving = np.insert(comoving, 0, 0)
    return np.interp(z_values, z_grid, comoving)

# --- LCDM Baseline ---
def h_lcdm(z):
    return H0_PLANCK * np.sqrt(OM_PLANCK * (1 + z)**3 + OL_PLANCK)

dl_lcdm = (1 + z_obs) * integrate_distance_vectorized(z_obs, h_lcdm)
mu_planck = 5 * np.log10(dl_lcdm + 1e-12) + 25

R_planck = mu_data - mu_planck
chi2_planck = R_planck.T @ inv_cov @ R_planck

# ==========================================
# 4. EXACT COVARIANT OPTIMIZATION (VED)
# ==========================================
# EXACT PURE THEORY PARAMETERS
Z_TRANS = 0.641
WIDTH = 0.10
H_FAST = 74.69         # EXACT: Early Geometric Ceiling
OM_PRIMORDIAL = 0.3116 # EXACT: Topological Bare Density
OM_EFFECTIVE  = 0.3639 # EXACT: Viscous Braking Density

print("Optimizing Vacuum Model Headroom (Exact Covariant Engine)...")

def get_exact_mu_visc(z_array, h_local_opt):
    """Exact Covariant Distance Modulus Engine parameterized for optimization."""
    # 1. Evaluate S(z)
    arg_obs = (Z_TRANS - z_array) / WIDTH
    S_z = np.where(arg_obs > 100, 1.0, np.where(arg_obs < -100, 0.0, 1.0 / (1.0 + np.exp(-arg_obs))))
    
    # 2. Continuous Early Gravity Field G(z)
    G_z = 1.0 + 0.2177 * (1.0 - S_z)
    
    # 3. Dynamically Shifted Metric Boundary
    z_metric = (1.0 + z_array) / np.sqrt(G_z) - 1.0
    z_metric = np.maximum(z_metric, 0.0) # Safety against negative bounds
    
    # 4. Define optimized H(z) inside the engine
    def h_viscous_opt(z):
        arg = (Z_TRANS - z) / WIDTH
        sigmoid = np.where(arg > 100, 1.0, np.where(arg < -100, 0.0, 1.0 / (1.0 + np.exp(-arg))))
        OM_Z = OM_PRIMORDIAL + (OM_EFFECTIVE - OM_PRIMORDIAL) * sigmoid
        OL_Z = 1.0 - OM_Z
        H_Z = H_FAST + (h_local_opt - H_FAST) * sigmoid
        E_z = np.sqrt(OM_Z * (1 + z)**3 + OL_Z)
        return H_Z * E_z
    
    # 5. Exact Covariant Integration (Truncated at z_metric)
    comoving_at_z_metric = integrate_distance_vectorized(z_metric, h_viscous_opt)
    dl_mpc = (1.0 + z_array) * comoving_at_z_metric
    
    # 6. Flux dilution standard prefactor
    dl_mpc_safe = np.maximum(dl_mpc, 1e-10)
    mu_raw = 5.0 * np.log10(dl_mpc_safe) + 25.0
    
    # 7. Superimpose Continuous Source Penalties (+0.410 max)
    penalties = 0.410 * (1.0 - S_z)
    
    return mu_raw + penalties

def objective_vacuum(h0_param):
    """Objective function to minimize Chi2 by tuning H_LOCAL."""
    H_LOCAL_OPT = h0_param[0]
    
    # Generate exact covariant magnitudes
    mu_vacuum = get_exact_mu_visc(z_obs, H_LOCAL_OPT)
    
    # Calculate Chi2
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

# Plot Baseline Residuals
plt.errorbar(z_obs, R_planck, yerr=err_diag, 
             fmt='o', color='lightgrey', alpha=0.3, label='Pantheon+ Residuals (SH0ES Calibrated)')

# Generate Sorted Curves for Smooth Plotting
z_sort = np.sort(z_obs)

# 1. Exact Covariant Optimized Curve
mu_best_sort = get_exact_mu_visc(z_sort, best_H0)

# 2. Planck Baseline Sorted Curve
dl_planck_sort = (1 + z_sort) * integrate_distance_vectorized(z_sort, h_lcdm)
mu_planck_sort = 5 * np.log10(dl_planck_sort + 1e-12) + 25

# 3. Residual Difference Curve
curve_opt = mu_best_sort - mu_planck_sort

plt.plot(z_sort, curve_opt, 'r-', linewidth=3, label=f'Optimized Covariant Theory (Terminal $H_0={best_H0:.2f}$)')

plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Redshift z', fontsize=12)
plt.ylabel(r'Magnitude Residual $\mu - \mu_{Planck}$', fontsize=12)
plt.title(rf'Test I: Raw Stress Test (Maximum Headroom)' + '\n' + rf'Exact Covariant Engine: $\Delta\chi^2 \approx {d_chi2:.1f}$', fontsize=14)
plt.legend(fontsize=10)
plt.ylim(-0.6, 0.4)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_Test1_ExactCovariant_StressTest.png', dpi=300)
print("\nPlot saved as 'Figure_Test1_ExactCovariant_StressTest.png'")
plt.show()
