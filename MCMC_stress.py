# Uncomment below if needed
!pip install emcee corner
import numpy as np
import pandas as pd
import emcee
import corner
import requests
import os
import matplotlib.pyplot as plt

print("--- VACUUM ELASTODYNAMICS: STABILITY STRESS TEST ---")
print("MODE: FIXED H0 (74.5) + STABLE VISCOSITY (0.16 - 0.21)")

# ==========================================
# 1. DATA LOADING
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

# CC Data
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

# Pantheon+
df = pd.read_csv(DATA_FILE, sep=r'\s+')
mask = df['zHD'] > 0.01
z_sn = df[mask]['zHD'].values
mu_sn = df[mask]['MU_SH0ES'].values

with open(COV_FILE, 'r') as f:
    content = f.read().split()
data_flat = np.array(content, dtype=float)
cov_matrix = data_flat[1:].reshape((1701, 1701)) if len(data_flat) == 1701**2 + 1 else data_flat.reshape((1701, 1701))
indices = np.where(mask)[0]
inv_cov_sn = np.linalg.inv(cov_matrix[np.ix_(indices, indices)])

# ==========================================
# 2. PHYSICS (FIXED H0 = 74.5)
# ==========================================
c_light = 299792.458
FIXED_Z_TRANS = 0.65
DELTA_GEO_IDEAL = 0.229
FIXED_H0 = 74.5

def hubble_model(z, params):
    Om, eta = params
    delta_eff = DELTA_GEO_IDEAL * (1.0 - eta)
    suppression = np.sqrt(1.0 - delta_eff)
    sigmoid = 1.0 / (1.0 + np.exp((z - FIXED_Z_TRANS) / 0.1))
    amp = suppression + (1.0 - suppression) * sigmoid
    E_z = np.sqrt(Om * (1 + z)**3 + (1 - Om))
    return FIXED_H0 * E_z * amp


# ==============================================================================
# PHYSICS NOTE: EFFECTIVE METRIC RECONSTRUCTION
# ==============================================================================
# This function models the "Effective Expansion History" H_eff(z) required to
# match the net luminosity distance D_L.
#
# In the analytic theory:
#   1. Gravity Boost (G_early > G0) INCREASES H(z) -> Brightens SNe.
#   2. Viscous Damping (Opacity/Drag) DIMS SNe.
#
# Computationally, fitting two canceling parameters creates degeneracy.
# Therefore, this code models the NET effective trajectory:
#   H_eff(z) = H_boost(z) * Damping_Factor
#
# The 'suppression' term below represents the phenomenological net result:
# transitioning from the high-H0 local vacuum to the Planck-compatible
# background without requiring a separate "magnitude bias" parameter.
# ==============================================================================


def get_dist_mod(z_array, params):
    z_max = np.max(z_array) * 1.01
    z_grid = np.linspace(0, z_max, 1000)
    H_vals = hubble_model(z_grid, params)
    integrand = 1.0 / H_vals
    comoving = np.cumsum((integrand[:-1] + integrand[1:]) / 2 * np.diff(z_grid))
    comoving = np.insert(comoving, 0, 0)
    dl_mpc = c_light * np.interp(z_array, z_grid, comoving)
    return 5.0 * np.log10((1+z_array) * dl_mpc) + 25.0

# ==========================================
# 3. LIKELIHOOD (HARD PRIORS ON ETA)
# ==========================================
def log_likelihood(params):
    Om, eta = params
    
    # --- HARD CONSTRAINT ---
    # We strictly forbid eta from leaving the stable zone [0.16, 0.21]
    if not (0.16 <= eta <= 0.21): 
        return -np.inf
    if not (0.1 < Om < 0.6):
        return -np.inf

    # Priors
    # We remove the Gaussian prior on eta because the Uniform box [0.16, 0.21] is the test.
    # We keep a weak prior on Om to guide it.
    lp_Om = -0.5 * ((Om - 0.315) / 0.1)**2
    
    # Chi2
    model_hz = hubble_model(hz_data[:,0], params)
    chi2_hz = np.sum(((hz_data[:,1] - model_hz) / hz_data[:,2])**2)
    
    model_mu = get_dist_mod(z_sn, params)
    residuals = mu_sn - model_mu
    chi2_sn = residuals.T @ inv_cov_sn @ residuals 

    return lp_Om - 0.5 * (chi2_hz + chi2_sn)

# ==========================================
# 4. RUN MCMC
# ==========================================
ndim = 2   
nwalkers = 32
# Start walkers inside the box
p0 = np.zeros((nwalkers, ndim))
p0[:, 0] = 0.30 + 0.01 * np.random.randn(nwalkers) # Om starts near 0.30
p0[:, 1] = 0.18 + 0.01 * np.random.randn(nwalkers) # Eta starts near 0.18

print(f"Stress Testing: H0={FIXED_H0}, Eta=[0.16, 0.21]...")
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood)
sampler.run_mcmc(p0, 4000, progress=True)

flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
labels = [r"$\Omega_m$", r"$\eta$"]

print("\n--- RESULTS FOR CONSTRAINED TEST ---")
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    print(f"{labels[i]}: {mcmc[1]:.3f}  +{np.diff(mcmc)[1]:.3f} / -{np.diff(mcmc)[0]:.3f}")

# Plot
fig = corner.corner(flat_samples, labels=labels, truths=[0.315, 0.21], truth_color="#ff4444")
plt.savefig("Stress_Test_MCMC.png", dpi=300)
print("Saved corner plot.")
