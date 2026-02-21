# Uncomment the line below if running in Google Colab / Jupyter
!pip install emcee corner pandas
import numpy as np
import pandas as pd
import emcee
import corner
import requests
import os
import matplotlib.pyplot as plt

print("--- VACUUM ELASTODYNAMICS: FULL JOINT MCMC (N=1701 + 31) ---")
print("MODE: STRICT CALIBRATION (Unified Phase Transition Engine)")

# ==========================================
# 1. AUTO-DOWNLOADER
# ==========================================
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

download_file(DATA_URL, DATA_FILE)
download_file(COV_URL, COV_FILE)

# ==========================================
# 2. LOAD DATA
# ==========================================
# A. Cosmic Chronometers (Full N=31 Dataset)
hz_data = np.array([
    [0.07, 69.0, 19.6], [0.09, 69.0, 12.0], [0.12, 68.6, 26.2], [0.17, 83.0, 8.0],
    [0.179, 75.0, 4.0], [0.199, 75.0, 5.0], [0.20, 72.9, 29.6], [0.27, 77.0, 14.0],
    [0.28, 88.8, 36.6], [0.352, 83.0, 14.0], [0.3802, 83.0, 13.5], [0.4, 95.0, 17.0],
    [0.4004, 77.0, 10.2], [0.4247, 87.1, 11.2], [0.4497, 92.8, 12.9], [0.47, 89.0, 23.0],
    [0.4783, 80.9, 9.0], [0.48, 97.0, 62.0], [0.593, 104.0, 13.0], [0.68, 92.0, 8.0],
    [0.781, 105.0, 12.0], [0.875, 125.0, 17.0], [0.88, 90.0, 40.0], [0.9, 117.0, 23.0],
    [1.037, 154.0, 20.0], [1.3, 168.0, 17.0], [1.363, 160.0, 33.6], [1.43, 177.0, 18.0],
    [1.53, 140.0, 14.0], [1.75, 202.0, 40.0], [1.965, 186.5, 50.4]
])

# B. Pantheon+ (N=1701)
print("Loading Pantheon+ Data...")
df = pd.read_csv(DATA_FILE, sep=r'\s+')

# CORRECTION: The paper uses the FULL dataset including local calibrators.
mask = df['zHD'] > -1.00 
z_sn = df[mask]['zHD'].values
mu_sn = df[mask]['MU_SH0ES'].values

print(f"Pantheon+ Data Loaded: {len(z_sn)} points (Target: 1701)")

# C. Covariance Matrix
print("Inverting Covariance Matrix...")
with open(COV_FILE, 'r') as f:
    content = f.read().split()
data_flat = np.array(content, dtype=float)
if len(data_flat) == 1701**2 + 1:
    cov_matrix = data_flat[1:].reshape((1701, 1701))
else:
    cov_matrix = data_flat.reshape((1701, 1701))
    
indices = np.where(mask)[0]
cov_filtered = cov_matrix[np.ix_(indices, indices)]
# Using pseudo-inverse for stability
inv_cov_sn = np.linalg.pinv(cov_filtered)

# ==========================================
# 3. UNIFIED PHYSICS ENGINE (UPDATED TO EXACT 0.03% INTEGRATION)
# ==========================================
c_light = 299792.458
Z_TRANS = 0.641 # Theoretically derived Section 2.4
WIDTH = 0.10
OM_PRIMORDIAL = 0.3116 # The frictionless baseline Theoretically derived Section 7.1.2

# STRICT GEOMETRIC CEILING (Matches exact Ab Initio Integration)
H_FAST = 74.70         

def hubble_model(z, params):
    # Free Parameters: Local Decelerated H0, and the Effective Late-Time Density
    H0_local, Om_effective = params

    # The phase transition activates at z=0.641
    arg = (Z_TRANS - z) / WIDTH
    sigmoid = np.where(arg > 100, 1.0, np.where(arg < -100, 0.0, 1.0 / (1.0 + np.exp(-arg))))
    
    # 1. Density Transition
    OM_Z = OM_PRIMORDIAL + (Om_effective - OM_PRIMORDIAL) * sigmoid
    OL_Z = 1.0 - OM_Z
    
    # 2. Hubble Trajectory Transition
    H_Z = H_FAST + (H0_local - H_FAST) * sigmoid
    
    # Standard Friedmann Expansion parameter using the dynamic density
    E_z = np.sqrt(OM_Z * (1 + z)**3 + OL_Z)
    
    return H_Z * E_z

def get_dist_mod(z_array, params):
    # Calculates the distance modulus by integrating the inverse Hubble parameter
    z_max = np.max(z_array) * 1.01
    z_grid = np.linspace(0, z_max, 10000)
    H_vals = hubble_model(z_grid, params)
    integrand = 1.0 / H_vals
    
    # Trapezoidal Integration
    comoving = np.cumsum((integrand[:-1] + integrand[1:]) / 2 * np.diff(z_grid))
    comoving = np.insert(comoving, 0, 0)
    dl_mpc = c_light * np.interp(z_array, z_grid, comoving)
    
    return 5.0 * np.log10((1+z_array) * dl_mpc) + 25.0

# ==========================================
# 4. LIKELIHOOD
# ==========================================
def log_likelihood(params):
    H0_local, Om_effective = params
    
    # Broad Priors
    if not (60.0 < H0_local < 80.0 and 0.2 < Om_effective < 0.5):
        return -np.inf
    
    # 1. Cosmic Chronometers
    model_hz = hubble_model(hz_data[:,0], params)
    chi2_hz = np.sum(((hz_data[:,1] - model_hz) / hz_data[:,2])**2)
    
    # 2. Pantheon+
    model_mu = get_dist_mod(z_sn, params)
    residuals = mu_sn - model_mu
    chi2_sn = residuals.T @ inv_cov_sn @ residuals 

    return -0.5 * (chi2_hz + chi2_sn)

# ==========================================
# 5. RUN MCMC
# ==========================================
ndim = 2   
nwalkers = 32

# p0: TRULY BLIND INITIALIZATION
p0 = np.random.uniform(low=[60.0, 0.2], high=[80.0, 0.5], size=(nwalkers, ndim))

print("Running Chain for 10,000 steps (may take ~20 mins)...")
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood)
sampler.run_mcmc(p0, 10000, progress=True)

# --- NEW: Autocorrelation Diagnostic ---
print("\n--- CONVERGENCE DIAGNOSTICS ---")
try:
    tau = sampler.get_autocorr_time()
    print(f"Autocorrelation time (tau): {tau}")
    print(f"Chain length is {10000 / np.mean(tau):.1f} times the autocorrelation time.")
    if (10000 / np.mean(tau)) > 100:
        print("STATUS: Convergence is mathematically rigorous (>100x tau).")
except emcee.autocorr.AutocorrError as e:
    print(f"Autocorrelation Warning: {e}")

# Results
flat_samples = sampler.get_chain(discard=2000, thin=15, flat=True)
labels = [r"$H_0^{local}$", r"$\Omega_{m}^{eff}$"]

print("\n--- FINAL UNIFIED MCMC RESULTS ---")
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    print(f"{labels[i]}: {mcmc[1]:.3f}  +{np.diff(mcmc)[1]:.3f} / -{np.diff(mcmc)[0]:.3f}")

# Plot
# STRICT GEOMETRIC TERMINAL (Matches exact Ab Initio output)
H_OBS_THEORY = 72.72
OM_EFF_THEORY = 0.3639

fig = corner.corner(
    flat_samples, 
    labels=labels, 
    truths=[H_OBS_THEORY, OM_EFF_THEORY], 
    truth_color="#ff4444"
)
plt.savefig("Joint_MCMC_Unified_10k.png", dpi=300)
print("Saved 10,000-step corner plot.")
