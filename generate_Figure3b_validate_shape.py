import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os

# ==========================================
# 1. SETUP & DATA
# ==========================================
print("--- RUNNING PANTHEON+ SHAPE CONSISTENCY TEST ---")
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
            print(f"Error downloading {filename}: {e}")

download_file(DATA_URL, DATA_FILE)
download_file(COV_URL, COV_FILE)

# Load Data
df = pd.read_csv(DATA_FILE, sep=r'\s+')
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
print(f"Loaded {len(df_clean)} Supernovae.")

# ==========================================
# 2. PHYSICS MODELS
# ==========================================
C_LIGHT = 299792.458
OM = 0.315
OL = 1.0 - OM
Z_TRANS = 0.65   # Percolation Threshold (Eq. 11)
WIDTH = 0.10     # Phase Transition Width

# PARAMETER UPDATE: Matching Section 7.2 "Gravity Boost"
# "This specific trajectory predicts a local H0 ~ 74.5"
H0_EARLY = 67.4  
H0_LATE = 74.5   

def integrate_distance_vectorized(z_values, h_func):
    z_grid = np.linspace(0, np.max(z_values)*1.01, 2000)
    h_grid = h_func(z_grid)
    integrand = C_LIGHT / h_grid
    comoving = np.cumsum(integrand) * (z_grid[1] - z_grid[0])
    return np.interp(z_values, z_grid, comoving)

# MODEL A: Planck LCDM
def h_lcdm(z):
    return H0_EARLY * np.sqrt(OM * (1 + z)**3 + OL)

dl_lcdm = (1 + df_clean['zHD']) * integrate_distance_vectorized(df_clean['zHD'], h_lcdm)
mu_lcdm = 5 * np.log10(dl_lcdm) + 25

# MODEL B: Viscous Vacuum (Shape Test)
def h_viscous(z):
    E_z = np.sqrt(OM * (1 + z)**3 + OL)
    
    # Boost Logic: 
    # Sigmoid smoothly scales H0 from 67.4 (Early) to 74.5 (Late)
    # Implements the relaxation profile
    boost_amp = H0_LATE / H0_EARLY
    
    arg = (z - Z_TRANS) / WIDTH
    # Numerical Stability (Consistent with S8/Hz scripts)
    sigmoid = np.where(arg > 100, 0.0, 1.0 / (1.0 + np.exp(arg)))
    
    effective_boost = 1.0 + (boost_amp - 1.0) * sigmoid
    return H0_EARLY * E_z * effective_boost

dl_visc = (1 + df_clean['zHD']) * integrate_distance_vectorized(df_clean['zHD'], h_viscous)
mu_visc = 5 * np.log10(dl_visc) + 25

# ==========================================
# 3. STATISTICAL TEST
# ==========================================
def calc_marginalized_chi2(mu_model, mu_data, inv_c):
    # This function removes the absolute magnitude offset
    # to test ONLY the shape consistency.
    residuals = mu_data - mu_model
    W = np.sum(inv_c)
    W_R = np.sum(np.dot(residuals.T, inv_c))
    A = W_R / W 
    resid_final = residuals - A
    return resid_final.T @ inv_c @ resid_final, A

mu_data = df_clean['MU_SH0ES'].values

chi2_lcdm, offset_lcdm = calc_marginalized_chi2(mu_lcdm, mu_data, inv_cov)
chi2_visc, offset_visc = calc_marginalized_chi2(mu_visc, mu_data, inv_cov)
d_chi2 = chi2_visc - chi2_lcdm

print("-" * 40)
print(f"Model A (Planck 67.4) Chi2:  {chi2_lcdm:.2f}")
print(f"Model B (Vacuum {H0_LATE}) Chi2:  {chi2_visc:.2f}")
print(f"Delta Chi2:                  {d_chi2:.2f}")
print("-" * 40)

if d_chi2 < 0:
    print("VERDICT: SUCCESS (Shape Verified).")
    print("The Vacuum Model transition fits the SNeIa shape slightly better than LCDM.")
elif d_chi2 < 5:
    print("VERDICT: SUCCESS (Consistent).")
    print("The Vacuum Model preserves the standard expansion history shape.")
else:
    print("VERDICT: FAIL. The phase transition distorts the shape too much.")

# ==========================================
# 4. PLOT
# ==========================================
plt.figure(figsize=(10,6))
resid_plot = mu_data - (mu_lcdm + offset_lcdm)
plt.errorbar(df_clean['zHD'], resid_plot, yerr=df_clean['MU_SH0ES_ERR_DIAG'],
             fmt='o', color='lightgrey', alpha=0.3, label='Pantheon+ Residuals')

diff_curve = (mu_visc + offset_visc) - (mu_lcdm + offset_lcdm)
z_sort = np.argsort(df_clean['zHD'])
plt.plot(df_clean['zHD'][z_sort], diff_curve[z_sort], 'r-', linewidth=3, label=f'Vacuum Model Difference')

plt.axhline(0, color='k', linestyle='--')
plt.title(rf'Pantheon+ Shape Consistency Test: $\Delta\chi^2 = {d_chi2:.2f}$ (Target {H0_LATE})', fontsize=14)
plt.xlabel('Redshift z', fontsize=12)
plt.ylabel('Residual Magnitude (Shape Only)', fontsize=12)
plt.legend(fontsize=10, loc='lower left')
plt.ylim(-0.25, 0.25)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure4_Pantheon_Shape_Test.png')
print("Plot saved as 'Figure2b_Pantheon_Shape_Test.png'")
plt.show()
