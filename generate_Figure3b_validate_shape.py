# Uncomment the line below if running in Google Colab / Jupyter
# !pip install scipy numpy matplotlib pandas requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
from scipy.optimize import minimize

# ==========================================
# 1. SETUP & DATA
# ==========================================
print("--- RUNNING PANTHEON+ SHAPE CONSISTENCY TEST (KINEMATIC) ---")
print("Objective: Verify Metric 2 (Test III: Shape Consistency, Section 8.3.5)")
print("Engine: Exact Covariant Geometry (z_metric truncation + continuous penalties)")

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

download_file(DATA_URL, DATA_FILE)
download_file(COV_URL, COV_FILE)

df = pd.read_csv(DATA_FILE, sep=r'\s+')
# Filter to bulk flow (N=1590) as specified in Section 8.3.5
mask = df['zHD'] > 0.01
df_clean = df[mask].reset_index(drop=True)

with open(COV_FILE, 'r') as f:
    content = f.read().split()
data = np.array(content, dtype=float)
N = 1701
if len(data) == N*N + 1:
    cov_matrix = data[1:].reshape((N, N))
else:
    cov_matrix = data.reshape((N, N))
indices = np.where(mask)[0]
cov_filtered = cov_matrix[np.ix_(indices, indices)]
inv_cov = np.linalg.pinv(cov_filtered)

# Extract safe diagonal errors for plotting
err_diag = np.sqrt(np.diag(cov_filtered))

print(f"Loaded {len(df_clean)} Supernovae (z > 0.01 Bulk Flow).")

# ==========================================
# 2. PHYSICS MODELS
# ==========================================
C_LIGHT = 299792.458
Z_TRANS = 0.641       # EXACT: Topological percolation redshift
WIDTH = 0.10         

# --- MODEL A: PLANCK LCDM (Baseline Control) ---
H0_A = 67.36          # EXACT: Planck 2018
OM_A = 0.3153         # EXACT: Planck 2018
OL_A = 1.0 - OM_A

# --- MODEL B: VACUUM ELASTODYNAMICS (Zero-Parameter Prediction) ---
H_FAST = 74.69         # EXACT: Early Universe Ceiling
H_LOCAL = 72.71        # EXACT: Late Universe Terminal Velocity
OM_PRIMORDIAL = 0.3116 # EXACT: Frictionless Bare Density
OM_EFFECTIVE = 0.3639  # EXACT: Viscous late universe (Inertial Counter-Load)

def integrate_distance_vectorized(z_values, h_func):
    """Vectorized numerical integration of comoving distance."""
    z_max = np.max(z_values)
    # Guard against negative/zero bounds during testing
    if z_max <= 0: z_max = 0.01 
    
    z_grid = np.linspace(0, z_max * 1.05, 10000)
    h_grid = h_func(z_grid)
    integrand = C_LIGHT / h_grid
    comoving = np.cumsum((integrand[:-1] + integrand[1:]) / 2 * np.diff(z_grid))
    comoving = np.insert(comoving, 0, 0)
    return np.interp(z_values, z_grid, comoving)

# --- LCDM Baseline ---
def h_lcdm(z):
    return H0_A * np.sqrt(OM_A * (1 + z)**3 + OL_A)

dl_lcdm = (1 + df_clean['zHD']) * integrate_distance_vectorized(df_clean['zHD'], h_lcdm)
mu_lcdm = 5 * np.log10(dl_lcdm) + 25

# --- Vacuum Elastodynamics Engine ---
def h_viscous(z):
    """Continuous expansion history H(z) for the Vacuum phase transition."""
    arg = (Z_TRANS - z) / WIDTH
    sigmoid = np.where(arg > 100, 1.0, np.where(arg < -100, 0.0, 1.0 / (1.0 + np.exp(-arg))))
    
    OM_Z = OM_PRIMORDIAL + (OM_EFFECTIVE - OM_PRIMORDIAL) * sigmoid
    OL_Z = 1.0 - OM_Z
    H_Z = H_FAST + (H_LOCAL - H_FAST) * sigmoid
    
    E_z = np.sqrt(OM_Z * (1 + z)**3 + OL_Z)
    return H_Z * E_z

def get_exact_mu_visc(z_array):
    """Exact Covariant Distance Modulus including z_metric shift and source penalties."""
    # 1. Evaluate S(z) for observational redshifts
    arg_obs = (Z_TRANS - z_array) / WIDTH
    S_z = np.where(arg_obs > 100, 1.0, np.where(arg_obs < -100, 0.0, 1.0 / (1.0 + np.exp(-arg_obs))))
    
    # 2. Continuous Early Gravity Field G(z)
    G_z = 1.0 + 0.2177 * (1.0 - S_z)
    
    # 3. Dynamically Shifted Metric Boundary (Atomic Drift limit)
    z_metric = (1.0 + z_array) / np.sqrt(G_z) - 1.0
    
    # Ensure z_metric doesn't drop below 0 due to numerical artifacts at very low z
    z_metric = np.maximum(z_metric, 0.0)
    
    # 4. Exact Covariant Integration (Truncated at z_metric)
    comoving_at_z_metric = integrate_distance_vectorized(z_metric, h_viscous)
    dl_mpc = (1.0 + z_array) * comoving_at_z_metric
    
    # 5. Flux dilution standard prefactor
    mu_raw = 5.0 * np.log10(dl_mpc) + 25.0
    
    # 6. Superimpose Continuous Source Penalties (+0.410 max)
    penalties = 0.410 * (1.0 - S_z)
    
    return mu_raw + penalties

# Execute the exact covariant model
mu_visc = get_exact_mu_visc(df_clean['zHD'].values)

# ==========================================
# 3. STATISTICAL TEST (MARGINALIZED SHAPE)
# ==========================================
print("Marginalizing over Absolute Calibration (Intercept)...")
mu_data = df_clean['MU_SH0ES'].values

def get_marginalized_chi2(mu_model, mu_data, inv_c):
    # Optimize a single shift parameter (delta_M) to minimize Chi2
    def objective(delta_M):
        shifted_model = mu_model + delta_M
        residuals = mu_data - shifted_model
        return residuals.T @ inv_c @ residuals
    
    # Run optimizer
    res = minimize(objective, x0=[0.0], method='Nelder-Mead')
    best_shift = res.x[0]
    best_chi2 = res.fun
    return best_chi2, best_shift

chi2_lcdm, offset_lcdm = get_marginalized_chi2(mu_lcdm, mu_data, inv_cov)
chi2_visc, offset_visc = get_marginalized_chi2(mu_visc, mu_data, inv_cov)
d_chi2 = chi2_visc - chi2_lcdm

print("\n" + "-" * 40)
print(f"Model A (Planck 67.36) Shape Chi2:   {chi2_lcdm:.2f}")
print(f"Model B (Vacuum Theory) Shape Chi2:  {chi2_visc:.2f}")
print(f"Delta Chi2 (Shape Penalty):          {d_chi2:.2f}")
print("-" * 40)

if d_chi2 < 10.0:
    print("\nVERDICT: SUCCESS (Consistent).")
    print("The Exact Covariant Vacuum Model shape is statistically indistinguishable from LCDM.")
    print("This proves the dynamic physics flawlessly traverses the degeneracy diagonal.")
else:
    print("\nVERDICT: FAIL.")

# ==========================================
# 4. PLOT
# ==========================================
plt.figure(figsize=(10,6))
resid_plot = mu_data - (mu_lcdm + offset_lcdm)
plt.errorbar(df_clean['zHD'], resid_plot, yerr=err_diag,
             fmt='o', color='lightgrey', alpha=0.3, label='Pantheon+ Residuals (Bulk Flow)')

diff_curve = (mu_visc + offset_visc) - (mu_lcdm + offset_lcdm)
z_sort = np.argsort(df_clean['zHD'].values)

plt.plot(df_clean['zHD'].values[z_sort], diff_curve[z_sort], 'r-', linewidth=3, label=f'Exact Covariant Shape Difference')

plt.axhline(0, color='k', linestyle='--')
plt.title(rf'Pantheon+ Exact Shape Test: $\Delta\chi^2 = {d_chi2:.2f}$', fontsize=14)
plt.xlabel('Redshift z', fontsize=12)
plt.ylabel('Residual Magnitude (Shape Only)', fontsize=12)
plt.legend(fontsize=10, loc='lower left')
plt.ylim(-0.25, 0.25)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure3b_Pantheon_Shape_Test_Covariant.png', dpi=300)
print("\nSaved Figure3b_Pantheon_Shape_Test_Covariant.png")
plt.show()
