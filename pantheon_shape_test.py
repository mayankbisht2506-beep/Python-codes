import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os

# ==========================================
# 1. SETUP & DATA
# ==========================================
print("--- RUNNING 'STEEL MAN' SCIENTIFIC TEST ---")
print("Baseline: Optimized LCDM (Best Fit H0)")
print("Challenger: Viscous Integral (Evolving H0)")

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
            print(f"Error: {e}")

download_file(DATA_URL, DATA_FILE)
download_file(COV_URL, COV_FILE)

# Load Data
df = pd.read_csv(DATA_FILE, sep=r'\s+')
mask = df['zHD'] > 0.01
df_clean = df[mask].reset_index(drop=True)

# Load Matrix
print("Processing Covariance Matrix...")
with open(COV_FILE, 'r') as f:
    content = f.read().split()
    data = np.array(content, dtype=float)
    N = 1701
    cov_matrix = data[1:].reshape((N, N)) if len(data) == N*N+1 else data.reshape((N,N))
    indices = np.where(mask)[0]
    cov_filtered = cov_matrix[np.ix_(indices, indices)]
    try:
        inv_cov = np.linalg.inv(cov_filtered)
    except:
        inv_cov = np.linalg.pinv(cov_filtered)

# ==========================================
# 2. PHYSICS MODELS
# ==========================================
# Constants
C_LIGHT = 299792.458
OM = 0.315
OL = 1.0 - OM
# Transition Parameters (Fixed by Theory)
Z_TRANS = 0.65
WIDTH = 0.1
H0_LATE = 73.04   # SH0ES (Soft Vacuum)
H0_EARLY = 67.4   # Planck (Stiff Vacuum)

# --- HELPER: VECTORIZED INTEGRATION ---
def integrate_distance_vectorized(z_values, h_func):
    """Numerically integrates c/H(z) for an array of redshifts"""
    z_grid = np.linspace(0, np.max(z_values)*1.01, 2000)
    # Calculate H(z) on the grid
    h_grid = h_func(z_grid)
    integrand = C_LIGHT / h_grid
        
    # Cumulative Trapezoidal Integration
    comoving = np.cumsum(integrand) * (z_grid[1] - z_grid[0])
    # Interpolate to data points
    return np.interp(z_values, z_grid, comoving)

# --- MODEL A: OPTIMIZED LCDM (The "Steel Man") ---
# We calculate the shape for H0=73 (arbitrary), then marginalize M to find BEST FIT.
def h_lcdm(z):
    return H0_LATE * np.sqrt(OM * (1 + z)**3 + OL)

dl_lcdm = (1 + df_clean['zHD']) * integrate_distance_vectorized(df_clean['zHD'], h_lcdm)
mu_lcdm_shape = 5 * np.log10(dl_lcdm) + 25

# --- MODEL B: VISCOUS TRANSITION (The Challenger) ---
# H(z) evolves from 73 -> 67.4 via lattice relaxation
def h_viscous(z):
    # Sigmoid for H0 evolution
    # Low z: sigmoid ~ 1 -> h0_eff ~ H0_LATE
    # High z: sigmoid ~ 0 -> h0_eff ~ H0_EARLY
    sigmoid = 1 / (1 + np.exp((z - Z_TRANS) / WIDTH))
    h0_eff = H0_EARLY + (H0_LATE - H0_EARLY) * sigmoid
        
    # Standard expansion factor E(z)
    E_z = np.sqrt(OM * (1 + z)**3 + OL)
    return h0_eff * E_z

dl_visc = (1 + df_clean['zHD']) * integrate_distance_vectorized(df_clean['zHD'], h_viscous)
mu_visc_shape = 5 * np.log10(dl_visc) + 25

# ==========================================
# 3. STATISTICAL TEST (MARGINALIZED M)
# ==========================================
# We use the Conley et al. method to analytically find the best 'M' (vertical offset)
# for both models. This ensures we compare best-vs-best.

def calc_marginalized_chi2(mu_model, mu_data, inv_c):
    residuals = mu_data - mu_model
    # Weights sum
    W = np.sum(inv_c)
    # Weighted residuals sum
    W_R = np.sum(np.dot(residuals.T, inv_c))
    # Optimal Offset A
    A = W_R / W
        
    # Corrected Residuals
    resid_final = residuals - A
        
    # Final Chi2
    chi2 = resid_final.T @ inv_c @ resid_final
    return chi2, A

# Calculate
mu_data = df_clean['MU_SH0ES'].values
chi2_lcdm, offset_lcdm = calc_marginalized_chi2(mu_lcdm_shape, mu_data, inv_cov)
chi2_visc, offset_visc = calc_marginalized_chi2(mu_visc_shape, mu_data, inv_cov)

# ==========================================
# 4. RESULTS
# ==========================================
d_chi2 = chi2_visc - chi2_lcdm

print("\n" + "="*60)
print("FINAL 'STEEL MAN' RESULTS")
print("Both models optimized for absolute calibration (Marginalized M).")
print("Comparing SHAPE of expansion history only.")
print("="*60)
print(f"Chi2 (Optimized LCDM):      {chi2_lcdm:.2f}")
print(f"Chi2 (Viscous Integral):    {chi2_visc:.2f}")
print(f"Delta Chi2:                 {d_chi2:.2f}")
print("-" * 60)

if d_chi2 < -4:
    print("VERDICT: SUCCESS.")
    print(f"The Viscous Transition fits the SHAPE of the data better by {abs(d_chi2):.2f} points.")
    print("This proves the 'Kink' at z=0.65 is real, regardless of H0 calibration.")
elif d_chi2 < 0:
    print("VERDICT: MARGINAL.")
    print("Slight preference, but statistically weak.")
else:
    print("VERDICT: FAILURE.")
    print("Standard LCDM fits the shape just as well or better.")

# Plot
plt.figure(figsize=(10,6))
# Plot Residuals relative to Optimized LCDM
resid_plot = mu_data - (mu_lcdm_shape + offset_lcdm)
plt.errorbar(df_clean['zHD'], resid_plot, yerr=df_clean['MU_SH0ES_ERR_DIAG'], fmt='o', color='lightgrey', alpha=0.3, label='Data Residuals (vs Best LCDM)')

# Plot Viscous Difference
# We plot (Viscous_Best - LCDM_Best) to show where the improvement comes from
diff_curve = (mu_visc_shape + offset_visc) - (mu_lcdm_shape + offset_lcdm)
z_sort = np.argsort(df_clean['zHD'])
plt.plot(df_clean['zHD'][z_sort], diff_curve[z_sort], 'r-', linewidth=3, label='Viscous Model Difference')

plt.axhline(0, color='k', linestyle='--')
plt.title(f'Shape Comparison: $\Delta\chi^2 = {d_chi2:.2f}$')
plt.xlabel('Redshift z')
plt.ylabel('Residual Magnitude')
plt.legend()
plt.ylim(-0.3, 0.3)
plt.savefig('Steel_Man_Test_Result.png')
plt.show()
