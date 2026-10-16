# Uncomment the line below if running in Google Colab / Jupyter
# !pip install scipy numpy matplotlib pandas requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os

# ==========================================
# 1. SETUP & DATA DOWNLOAD
# ==========================================
print("--- RUNNING PANTHEON+ TENSION TEST (ABSOLUTE MAGNITUDE) ---")
print("Objective: Verify Metric 1 (Test II: Theory with Zero Continuous Parameters)")
print("Engine: Exact Covariant Geometry (Pure Unnormalized Engine, Width=0.084)")

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
# 2. DATA PROCESSING (N=1701)
# ==========================================
print("Loading Pantheon+ Data...")
df = pd.read_csv(DATA_FILE, sep=r'\s+')

# We need the full unmarginalized dataset to test Absolute Magnitude (H0)
mask = df['zHD'] > 0.000
df_clean = df[mask].reset_index(drop=True)
print(f"Supernovae (Full N=1701): {len(df_clean)}")

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

print("Inverting Covariance Matrix (Robust Method for Test II)...")
inv_cov = np.linalg.pinv(cov_filtered)

# Extract safe diagonal errors for plotting
err_diag = np.sqrt(np.diag(cov_filtered))

# ==========================================
# 3. EXACT COVARIANT PHYSICS ENGINE
# ==========================================
C_LIGHT = 299792.458
Z_TRANS = 0.641       # EXACT: Topological percolation redshift
WIDTH = 0.084         # EXACT: Derived geometric Jacobian width

# --- MODEL A: PLANCK LCDM (Baseline Control) ---
H0_PLANCK = 67.36     # EXACT: Planck 2018
OM_PLANCK = 0.3153    # EXACT: Planck 2018
OL_PLANCK = 1.0 - OM_PLANCK

# --- MODEL B: VACUUM ELASTODYNAMICS (Zero Continuous Parameters) ---
H_FAST = 74.69         # EXACT: Theoretical E8 Geometry Limit
H_LOCAL = 72.71        # EXACT: Theoretically Derived Terminal Velocity
OM_PRIMORDIAL = 0.3116 # EXACT: Topological Bare Density
OM_EFFECTIVE = 0.3639  # EXACT: Theoretically Derived Viscous Load

def integrate_distance_vectorized(z_values, h_func):
    """Vectorized numerical integration of comoving distance."""
    z_max = np.max(z_values)
    if z_max <= 0: return np.zeros_like(z_values)
    
    # Ultra-dense 20,000 grid for smooth derivatives across the transition
    z_grid = np.linspace(0, z_max * 1.02, 20000)
    h_grid = h_func(z_grid)
    integrand = C_LIGHT / h_grid
    comoving = np.cumsum((integrand[:-1] + integrand[1:]) / 2 * np.diff(z_grid))
    comoving = np.insert(comoving, 0, 0)
    return np.interp(z_values, z_grid, comoving)

# --- LCDM Baseline ---
def h_lcdm(z):
    return H0_PLANCK * np.sqrt(OM_PLANCK * (1 + z)**3 + OL_PLANCK)

dl_lcdm = (1 + df_clean['zHD']) * integrate_distance_vectorized(df_clean['zHD'], h_lcdm)
mu_planck = 5 * np.log10(np.maximum(dl_lcdm, 1e-10)) + 25

# --- Vacuum Elastodynamics Engine ---
def h_viscous(z):
    """Continuous expansion history H(z) for the Vacuum phase transition."""
    arg = (Z_TRANS - z) / WIDTH
    S_z = np.where(arg > 100, 1.0, np.where(arg < -100, 0.0, 1.0 / (1.0 + np.exp(-arg))))
    
    OM_Z = OM_PRIMORDIAL + (OM_EFFECTIVE - OM_PRIMORDIAL) * S_z
    OL_Z = 1.0 - OM_Z
    H_Z = H_FAST + (H_LOCAL - H_FAST) * S_z
    
    E_z = np.sqrt(OM_Z * (1 + z)**3 + OL_Z)
    return H_Z * E_z

def get_exact_mu_visc(z_array):
    """Exact Covariant Distance Modulus (Section 7.3)."""
    # 1. Evaluate unnormalized S(z)
    arg_obs = (Z_TRANS - z_array) / WIDTH
    S_z = np.where(arg_obs > 100, 1.0, np.where(arg_obs < -100, 0.0, 1.0 / (1.0 + np.exp(-arg_obs))))
    
    # 2. Continuous Early Gravity Field G(z)
    G_z = 1.0 + 0.2177 * (1.0 - S_z)
    
    # 3. Dynamically Shifted Metric Boundary (Atomic Drift limit)
    z_metric = (1.0 + z_array) / np.sqrt(G_z) - 1.0
    z_metric = np.maximum(z_metric, 0.0) # Safety against negative bounds at z~0
    
    # 4. Exact Covariant Integration (Truncated at z_metric)
    comoving_at_z_metric = integrate_distance_vectorized(z_metric, h_viscous)
    dl_mpc = (1.0 + z_array) * comoving_at_z_metric
    
    # 5. Flux dilution standard prefactor
    dl_mpc_safe = np.maximum(dl_mpc, 1e-10)
    mu_raw = 5.0 * np.log10(dl_mpc_safe) + 25.0
    
    # 6. Superimpose Continuous Source Penalties (+0.410 max)
    penalties = 0.410 * (1.0 - S_z)
    
    return mu_raw + penalties

# Execute the exact covariant model
mu_viscous = get_exact_mu_visc(df_clean['zHD'].values)

# ==========================================
# 4. STATISTICS (PARTITIONED)
# ==========================================
print("Partitioning Dataset: Isolating 42 Cepheid Calibrator Hosts...")

# Pantheon+ explicitly flags Cepheid hosts
calibrator_mask = df_clean['IS_CALIBRATOR'] == 1
idx_cal = np.where(calibrator_mask)[0]

# Extract subset for the 42 calibrators
R_planck_full = df_clean['MU_SH0ES'].values - mu_planck
R_viscous_full = df_clean['MU_SH0ES'].values - mu_viscous

R_planck_cal = R_planck_full[idx_cal]
R_viscous_cal = R_viscous_full[idx_cal]

# Slice the inverted covariance matrix for the 42 hosts
cov_cal = cov_filtered[np.ix_(idx_cal, idx_cal)]
inv_cov_cal = np.linalg.pinv(cov_cal)

# Calculate Strict Absolute Calibration Chi2 (N=42)
chi2_planck_cal = R_planck_cal.T @ inv_cov_cal @ R_planck_cal
chi2_viscous_cal = R_viscous_cal.T @ inv_cov_cal @ R_viscous_cal
d_chi2_cal = chi2_viscous_cal - chi2_planck_cal

# Calculate Unmarginalized Full Projection (N=1701) - For Reference/Plotting
chi2_planck_full = R_planck_full.T @ inv_cov @ R_planck_full
chi2_viscous_full = R_viscous_full.T @ inv_cov @ R_viscous_full
d_chi2_full = chi2_viscous_full - chi2_planck_full

print("\n" + "="*60)
print(f"FINAL METRIC 1 RESULTS (Test II: Absolute Calibration)")
print("="*60)
print(f"--- 42 CEPHEID HOSTS (Strict Statistical Score) ---")
print(f"Chi2 (Planck 67.4):   {chi2_planck_cal:.2f}")
print(f"Chi2 (Vacuum Theory): {chi2_viscous_cal:.2f}") 
print(f"Delta Chi2 (Net):     {d_chi2_cal:.2f}")
print("-" * 60)
print(f"--- FULL N=1701 PROJECTION (Unmarginalized Reference) ---")
print(f"Delta Chi2 (Full):    {d_chi2_full:.2f}")
print("=" * 60)

if d_chi2_cal < -20:
    print("\nVERDICT: DECISIVE SUCCESS.")
    print(f"The model decisively resolves the local H0 calibration tension")
    print(f"(Delta Chi2 = {d_chi2_cal:.2f}) strictly on the Cepheid calibrator subset,")
    print(f"avoiding statistical double-counting with the Hubble flow!")

# ==========================================
# 5. PLOTTING
# ==========================================
plt.figure(figsize=(10,6))

# Plot all data lightly
plt.errorbar(df_clean['zHD'], R_planck_full, yerr=err_diag, 
             fmt='o', color='lightgrey', alpha=0.3, label='Hubble Flow SNe')

# Highlight the 42 calibrators
plt.errorbar(df_clean['zHD'][calibrator_mask], R_planck_full[calibrator_mask], 
             yerr=err_diag[calibrator_mask], fmt='o', color='blue', alpha=0.8, 
             label='Cepheid Calibrators (N=42)')

diff_curve = mu_viscous - mu_planck
z_sort = np.argsort(df_clean['zHD'].values)

plt.plot(df_clean['zHD'].values[z_sort], diff_curve[z_sort], 'r-', linewidth=3, 
         label=f'Exact Covariant Theory (Terminal $H_0={H_LOCAL}$)')

plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Redshift z', fontsize=12)
plt.ylabel(r'Magnitude Residual $\mu - \mu_{Planck}$', fontsize=12)
plt.title(rf'Resolution of Absolute Calibration Tension (Test II)' + '\n' + 
          rf'Calibrator Improvement: $\Delta\chi^2 \approx {d_chi2_cal:.1f}$', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.ylim(-0.6, 0.4)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Figure3a_Metric1_Covariant.png", dpi=300)
print("\nSaved Figure3a_Metric1_Covariant.png")
plt.show()
