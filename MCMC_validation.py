!pip install emcee corner
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

# ==========================================
# 1. DATASETS
# ==========================================
hz_data = np.array([
    [0.07, 69.0, 19.6], [0.12, 68.6, 26.2], [0.20, 72.9, 29.6],
    [0.28, 88.8, 36.6], [0.40, 95.0, 17.0], [0.47, 89.0, 50.0],
    [0.48, 97.0, 62.0], [0.593, 104.0, 13.0], [0.68, 92.0, 8.0],
    [0.781, 105.0, 12.0], [0.875, 125.0, 17.0], [0.88, 90.0, 40.0],
    [0.90, 117.0, 23.0], [1.037, 154.0, 20.0], [1.30, 168.0, 17.0],
    [1.363, 160.0, 33.6], [1.43, 177.0, 18.0], [1.53, 140.0, 14.0],
    [1.75, 202.0, 40.0], [1.965, 186.5, 50.4]
])

sn_data = np.array([
    [0.014, 14.57, 0.15], [0.026, 15.98, 0.12], [0.036, 16.78, 0.08],
    [0.046, 17.34, 0.07], [0.065, 18.12, 0.06], [0.10, 19.09, 0.05],
    [0.20, 20.64, 0.04],  [0.35, 22.12, 0.04],  [0.55, 23.46, 0.05],
    [0.85, 24.78, 0.08],  [1.15, 25.68, 0.12],  [1.50, 26.45, 0.18],
    [2.00, 27.20, 0.25]
])

# ==========================================
# 2. PHYSICS MODEL (Fixed z_trans = 0.65)
# ==========================================
c_light = 299792.458
FIXED_Z_TRANS = 0.65  # The Percolation Threshold Axiom

def hubble_model(z, params):
    H0_late, Om, eta = params # z_trans is removed from params
    
    E_z = np.sqrt(Om * (1 + z)**3 + (1 - Om))
    
    # Fixed Geometric Transition
    sigmoid = 1.0 / (1.0 + np.exp((z - FIXED_Z_TRANS) / 0.1))
    
    # Effective Amplitude
    amp_z = (1.0 - eta) + eta * sigmoid
    
    return H0_late * E_z * amp_z

def dist_mod_model(z, params):
    z_grid = np.linspace(0, z, 50)
    H_vals = hubble_model(z_grid, params)
    Dc = c_light * np.trapz(1.0/H_vals, z_grid)
    return 5.0 * np.log10((1+z) * Dc) + 25.0

# ==========================================
# 3. LIKELIHOODS
# ==========================================
def log_likelihood(params):
    H0, Om, eta = params
    
    # 1. HARD BOUNDARIES
    if not (60 < H0 < 80 and 0.2 < Om < 0.4 and 0.0 < eta < 0.4):
        return -np.inf

    # 2. PRIORS (The "Concordance" Setup)
    # Omega_m Prior (Planck)
    prior_Om = -0.5 * ((Om - 0.315) / 0.015)**2
    
    # H0 Prior (SH0ES - This fixes the Over-Correction)
    prior_H0 = -0.5 * ((H0 - 73.04) / 1.04)**2
    
    # 3. DATA LIKELIHOOD
    model_hz = np.array([hubble_model(z, params) for z in hz_data[:,0]])
    chi2_hz = np.sum(((hz_data[:,1] - model_hz) / hz_data[:,2])**2)
    
    model_mu = np.array([dist_mod_model(z, params) for z in sn_data[:,0]])
    diff = sn_data[:,1] - model_mu
    chi2_sn = np.sum(((diff - np.mean(diff)) / sn_data[:,2])**2)
    
    return prior_Om + prior_H0 - 0.5 * (chi2_hz + chi2_sn)

# ==========================================
# 4. RUN MCMC
# ==========================================
if __name__ == "__main__":
    print(f"Running Geometric MCMC with Fixed z_trans = {FIXED_Z_TRANS}...")
    
    ndim = 3  # Reduced from 4 to 3
    nwalkers = 32
    p0 = [73.0, 0.31, 0.17] + 1e-2 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood)
    sampler.run_mcmc(p0, 5000, progress=True)

    # Plotting
    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
    labels = [r"$H_0$", r"$\Omega_m$", r"$\eta$ (Viscosity)"]
    
    fig = corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, color="darkblue")
    fig.suptitle(f"Geometric Constraints (Fixed $z_{{trans}}={FIXED_Z_TRANS}$)", fontsize=16, y=1.02)
    plt.savefig("MCMC_Geometric_Constraint.png")
    
    print("-" * 40)
    print("GEOMETRIC MCMC RESULTS:")
    print("-" * 40)
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"{labels[i]}: {mcmc[1]:.3f} +{q[1]:.3f} / -{q[0]:.3f}")
    print("-" * 40)
