import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os

# ==========================================
# 1. SETUP & DATA DOWNLOAD
# ==========================================
print("--- RUNNING PANTHEON+ TENSION TEST (ABSOLUTE MAGNITUDE) ---")
print("Objective: Verify Metric 1 (Test II: Theoretical Verification) for H0 = 74.5")

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

# CORRECTION: Do not cut out the Cepheid anchors!
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

# ==========================================
# 3. PHYSICS ENGINE (THE UNIFIED MODEL)
# ==========================================
C_LIGHT = 299792.458
Z_TRANS = 0.65   
WIDTH = 0.10     

# --- MODEL A: PLANCK LCDM (Baseline Control) ---
H0_PLANCK = 67.4   
OM_PLANCK = 0.315
OL_PLANCK = 1.0 - OM_PLANCK

# --- MODEL B: VACUUM ELASTODYNAMICS ---
H_FAST = 74.5         # Theoretical E8 Geometry Limit
OM_PRIMORDIAL = 0.315 # Frictionless early universe
OM_EFFECTIVE = 0.366  # CORRECTION: Viscous late universe (Inertial Counter-Load)

def integrate_distance_vectorized(z_values, h_func):
    z_grid = np.linspace(0, np.max(z_values)*1.01, 2000)
    h_grid = h_func(z_grid)
    integrand = C_LIGHT / h_grid
    comoving = np.cumsum((integrand[:-1] + integrand[1:]) / 2 * np.diff(z_grid))
    comoving = np.insert(comoving, 0, 0)
    return np.interp(z_values, z_grid, comoving)

# Planck History
def h_lcdm(z):
    return H0_PLANCK * np.sqrt(OM_PLANCK * (1 + z)**3 + OL_PLANCK)

dl_lcdm = (1 + df_clean['zHD']) * integrate_distance_vectorized(df_clean['zHD'], h_lcdm)
mu_planck = 5 * np.log10(dl_lcdm) + 25

# Vacuum History
def h_viscous(z):
    arg = (z - Z_TRANS) / WIDTH
    sigmoid = np.where(arg > 100, 0.0, 1.0 / (1.0 + np.exp(arg)))
    
    OM_Z = OM_PRIMORDIAL + (OM_EFFECTIVE - OM_PRIMORDIAL) * sigmoid
    OL_Z = 1.0 - OM_Z
    
    E_z = np.sqrt(OM_Z * (1 + z)**3 + OL_Z)
    return H_FAST * E_z

dl_visc = (1 + df_clean['zHD']) * integrate_distance_vectorized(df_clean['zHD'], h_viscous)
mu_viscous = 5 * np.log10(dl_visc) + 25

# ==========================================
# 4. STATISTICS (NO MARGINALIZATION)
# ==========================================
R_planck = df_clean['MU_SH0ES'].values - mu_planck
R_viscous = df_clean['MU_SH0ES'].values - mu_viscous

chi2_planck = R_planck.T @ inv_cov @ R_planck
chi2_viscous = R_viscous.T @ inv_cov @ R_viscous
d_chi2 = chi2_viscous - chi2_planck

print("\n" + "="*50)
print(f"FINAL METRIC 1 RESULTS (H0={H_FAST})")
print("="*50)
print(f"Chi2 (Planck 67.4):   {chi2_planck:.2f}")
print(f"Chi2 (Vacuum {H_FAST}):   {chi2_viscous:.2f}") 
print(f"Delta Chi2:           {d_chi2:.2f}")
print("-" * 50)

if d_chi2 < -2000:
    print("VERDICT: DECISIVE SUCCESS.")
    print("The fast global trajectory organically brightens the luminosity distance,")
    print("perfectly resolving the SH0ES absolute magnitude tension!")

# ==========================================
# 5. PLOTTING
# ==========================================
plt.figure(figsize=(10,6))
plt.errorbar(df_clean['zHD'], R_planck, yerr=df_clean['MU_SH0ES_ERR_DIAG'], 
             fmt='o', color='lightgrey', alpha=0.3, label='Pantheon+ Residuals (SH0ES Calibrated)')

diff_curve = mu_viscous - mu_planck
z_sort = np.argsort(df_clean['zHD'])

plt.plot(df_clean['zHD'][z_sort], diff_curve[z_sort], 'r-', linewidth=3, label=f'Vacuum Model (H0={H_FAST})')

plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Redshift z')
plt.ylabel(r'Magnitude Residual $\mu - \mu_{Planck}$')
plt.title(rf'Resolution of Hubble Tension (Test II)' + '\n' + rf'$\Delta\chi^2 \approx {d_chi2:.1f}$ (Target H0={H_FAST})')
plt.legend()
plt.ylim(-0.6, 0.4)
plt.grid(True, alpha=0.3)
plt.savefig("Figure3a_Metric1_Corrected.png")
print("Saved Figure3a_Metric1_Corrected.png")
plt.show()
