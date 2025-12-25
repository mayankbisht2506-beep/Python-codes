import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os

# ==========================================
# 1. SETUP & DATA
# ==========================================
# This script reproduces the "Steel Man" test from Section 9.5 (Table 3)
# Target Result: Delta Chi2 approx -0.57 (Validating Expansion Shape)

DATA_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
COV_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"
DATA_FILE = "Pantheon+SH0ES.dat"
COV_FILE = "Pantheon+SH0ES_STAT+SYS.cov"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            r = requests.get(url, stream=True)
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

# Download Data if missing
download_file(DATA_URL, DATA_FILE)
download_file(COV_URL, COV_FILE)

# Load Supernova Data
try:
    df = pd.read_csv(DATA_FILE, sep=r'\s+')
    mask = df['zHD'] > 0.01
    df_clean = df[mask].reset_index(drop=True)

    # Load Covariance Matrix
    with open(COV_FILE, 'r') as f:
        first_line = f.readline().split()
        # Check if first line is the count (sometimes occurs in old formats)
        if len(first_line) == 1:
            content = f.read().split()
        else:
            # Reset and read all
            f.seek(0)
            content = f.read().split()

        data = np.array(content, dtype=float)
        N = 1701
        # Handle potential header integer
        if len(data) == N*N + 1:
            cov_matrix = data[1:].reshape((N, N))
        else:
            cov_matrix = data.reshape((N, N))

    indices = np.where(mask)[0]
    cov_filtered = cov_matrix[np.ix_(indices, indices)]
    inv_cov = np.linalg.pinv(cov_filtered) # Pseudo-inverse for stability

    print(f"Successfully loaded {len(df_clean)} Supernovae.")

except Exception as e:
    print(f"Critical Error loading data: {e}")
    exit()

# ==========================================
# 2. PHYSICS MODELS
# ==========================================
C_LIGHT = 299792.458
OM = 0.315
OL = 1.0 - OM
Z_TRANS = 0.65
WIDTH = 0.1
H0_LATE = 73.04   # SH0ES Baseline
H0_EARLY = 67.4   # Planck Baseline

def integrate_distance_vectorized(z_values, h_func):
    """Numerically integrates c/H(z)"""
    z_grid = np.linspace(0, np.max(z_values)*1.01, 2000)
    h_grid = h_func(z_grid)
    integrand = C_LIGHT / h_grid
    comoving = np.cumsum(integrand) * (z_grid[1] - z_grid[0])
    return np.interp(z_values, z_grid, comoving)

# --- MODEL A: Standard LCDM ---
def h_lcdm(z):
    return H0_LATE * np.sqrt(OM * (1 + z)**3 + OL)

dl_lcdm = (1 + df_clean['zHD']) * integrate_distance_vectorized(df_clean['zHD'], h_lcdm)
mu_lcdm_shape = 5 * np.log10(dl_lcdm) + 25

# --- MODEL B: Viscous Vacuum (Evolving H0) ---
def h_viscous(z):
    # H0 transitions from 67.4 (High z) to 73.0 (Low z)
    sigmoid = 1 / (1 + np.exp((z - Z_TRANS) / WIDTH))
    h0_eff = H0_EARLY + (H0_LATE - H0_EARLY) * sigmoid
    return h0_eff * np.sqrt(OM * (1 + z)**3 + OL)

dl_visc = (1 + df_clean['zHD']) * integrate_distance_vectorized(df_clean['zHD'], h_viscous)
mu_visc_shape = 5 * np.log10(dl_visc) + 25

# ==========================================
# 3. STATISTICAL TEST (The "Steel Man")
# ==========================================
def calc_marginalized_chi2(mu_model, mu_data, inv_c):
    """Calculates Chi2 marginalizing over absolute magnitude M."""
    residuals = mu_data - mu_model
    W = np.sum(inv_c)
    W_R = np.sum(np.dot(residuals.T, inv_c))
    A = W_R / W # Optimal vertical offset
    resid_final = residuals - A
    return resid_final.T @ inv_c @ resid_final, A

mu_data = df_clean['MU_SH0ES'].values

chi2_lcdm, offset_lcdm = calc_marginalized_chi2(mu_lcdm_shape, mu_data, inv_cov)
chi2_visc, offset_visc = calc_marginalized_chi2(mu_visc_shape, mu_data, inv_cov)
d_chi2 = chi2_visc - chi2_lcdm

print("-" * 40)
print(f"Standard LCDM Chi2:   {chi2_lcdm:.2f}")
print(f"Viscous Vacuum Chi2:  {chi2_visc:.2f}")
print(f"Delta Chi2:           {d_chi2:.2f}")
print("-" * 40)

# ==========================================
# 4. PLOT
# ==========================================
plt.figure(figsize=(10,6))
resid_plot = mu_data - (mu_lcdm_shape + offset_lcdm)
plt.errorbar(df_clean['zHD'], resid_plot, yerr=df_clean['MU_SH0ES_ERR_DIAG'],
             fmt='o', color='lightgrey', alpha=0.3, label='Pantheon+ Residuals')

diff_curve = (mu_visc_shape + offset_visc) - (mu_lcdm_shape + offset_lcdm)
z_sort = np.argsort(df_clean['zHD'])
plt.plot(df_clean['zHD'][z_sort], diff_curve[z_sort], 'r-', linewidth=3, label='Vacuum Model Difference')

plt.axhline(0, color='k', linestyle='--')
plt.title(f'Pantheon+ "Steel Man" Test: $\Delta\chi^2 = {d_chi2:.2f}$', fontsize=14, fontname='serif')
plt.xlabel('Redshift z', fontsize=12, fontname='serif')
plt.ylabel('Residual Magnitude', fontsize=12, fontname='serif')
plt.legend(fontsize=10, loc='lower left', frameon=True)
plt.ylim(-0.25, 0.25)
plt.grid(True, which="major", ls="-", alpha=0.2)
plt.grid(True, which="minor", ls=":", alpha=0.1)
plt.tight_layout()
plt.savefig('Figure4_Pantheon_SteelMan.png')
plt.show()
