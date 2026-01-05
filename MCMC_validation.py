# Uncomment the line below if running in Google Colab / Jupyter
# !pip install emcee corner
import numpy as np
import pandas as pd
import emcee
import corner
import requests
import os
import matplotlib.pyplot as plt

print("--- VACUUM ELASTODYNAMICS: FULL JOINT MCMC (N=1701 + 31) ---")
print("MODE: STRICT CALIBRATION (Reproducing H0 ~ 74.5)")

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
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)

download_file(DATA_URL, DATA_FILE)
download_file(COV_URL, COV_FILE)

# ==========================================
# 2. LOAD DATA
# ==========================================
# A. Cosmic Chronometers
hz_data = np.array([
    [0.07, 69.0, 19.6], [0.12, 68.6, 26.2], [0.20, 72.9, 29.6],
    [0.28, 88.8, 36.6], [0.40, 95.0, 17.0], [0.47, 89.0, 50.0],
    [0.48, 97.0, 62.0], [0.593, 104.0, 13.0], [0.68, 92.0, 8.0],
    [0.781, 105.0, 12.0], [0.875, 125.0, 17.0], [0.88, 90.0, 40.0],
    [0.90, 117.0, 23.0], [1.037, 154.0, 20.0], [1.30, 168.0, 17.0],
    [1.363, 160.0, 33.6], [1.43, 177.0, 18.0], [1.53, 140.0, 14.0],
    [1.75, 202.0, 40.0], [1.965, 186.5, 50.4]
])

# B. Pantheon+ (N=1701)
print("Loading Pantheon+ Data...")
df = pd.read_csv(DATA_FILE, sep=r'\s+')
mask = df['zHD'] > 0.01
z_sn = df[mask]['zHD'].values
mu_sn = df[mask]['MU_SH0ES'].values

# C. Covariance Matrix
print("Inverting Covariance Matrix...")
with open(COV_FILE, 'r') as f:
    content = f.read().split()
data_flat = np.array(content, dtype=float)
if len(data_flat) == 1701**2 + 1:
    cov_matrix = data_flat[1:].reshape((1701, 1701))
else:
    cov_matrix = data_flat.reshape((1701, 1701))
    
# Apply mask and invert
indices = np.where(mask)[0]
cov_filtered = cov_matrix[np.ix_(indices, indices)]
inv_cov_sn = np.linalg.inv(cov_filtered)

# ==========================================
# 3. PHYSICS ENGINE
# ==========================================
c_light = 299792.458
FIXED_Z_TRANS = 0.65
DELTA_GEO_IDEAL = 0.229

def hubble_model(z, params):
    H0_late, Om, eta = params
    
    # Physics: Viscosity dampens relaxation
    delta_eff = DELTA_GEO_IDEAL * (1.0 - eta)
    suppression = np.sqrt(1.0 - delta_eff)
    
    # Sigmoid Transition
    sigmoid = 1.0 / (1.0 + np.exp((z - FIXED_Z_TRANS) / 0.1))
    amp = suppression + (1.0 - suppression) * sigmoid
    
    E_z = np.sqrt(Om * (1 + z)**3 + (1 - Om))
    return H0_late * E_z * amp

def get_dist_mod(z_array, params):
    # Vectorized Integration
    z_max = np.max(z_array) * 1.01
    z_grid = np.linspace(0, z_max, 1000)
    H_vals = hubble_model(z_grid, params)
    integrand = 1.0 / H_vals
    comoving = np.cumsum((integrand[:-1] + integrand[1:]) / 2 * np.diff(z_grid))
    comoving = np.insert(comoving, 0, 0)
    dl_mpc = c_light * np.interp(z_array, z_grid, comoving)
    return 5.0 * np.log10((1+z_array) * dl_mpc) + 25.0

# ==========================================
# 4. LIKELIHOOD (STRICT CALIBRATION)
# ==========================================
def log_likelihood(params):
    H0, Om, eta = params
    
    if not (60 < H0 < 80 and 0.2 < Om < 0.4 and 0.0 < eta < 0.5):
        return -np.inf

    # Priors
    lp_Om = -0.5 * ((Om - 0.315) / 0.05)**2
    lp_eta = -0.5 * ((eta - 0.21) / 0.1)**2 
    
    # 1. Cosmic Chronometers
    model_hz = hubble_model(hz_data[:,0], params)
    chi2_hz = np.sum(((hz_data[:,1] - model_hz) / hz_data[:,2])**2)
    
    # 2. Pantheon+ (STRICT MODE)
    model_mu = get_dist_mod(z_sn, params)
    residuals = mu_sn - model_mu
    
    # --- CRITICAL FIX: DO NOT MARGINALIZE M ---
    # We force the model to hit the SH0ES calibration directly.
    # This anchors H0 to ~73-74.
    chi2_sn = residuals.T @ inv_cov_sn @ residuals 
    # ------------------------------------------

    return lp_Om + lp_eta - 0.5 * (chi2_hz + chi2_sn)

# ==========================================
# 5. RUN MCMC
# ==========================================
ndim = 3   
nwalkers = 32
p0 = [74.5, 0.30, 0.19] + 1e-2 * np.random.randn(nwalkers, ndim)

print("Running Chain (may take 10-15 mins)...")
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood)
sampler.run_mcmc(p0, 4000, progress=True)

# Results
flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
labels = [r"$H_0$", r"$\Omega_m$", r"$\eta$"]

print("\n--- FINAL CORRECTED RESULTS ---")
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    print(f"{labels[i]}: {mcmc[1]:.3f}  +{np.diff(mcmc)[1]:.3f} / -{np.diff(mcmc)[0]:.3f}")

# Plot
fig = corner.corner(flat_samples, labels=labels, truths=[74.5, 0.30, 0.19], truth_color="#ff4444")
plt.savefig("Joint_MCMC_Corrected.png", dpi=300)
print("Saved corner plot.")
