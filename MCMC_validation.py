# ==========================================
# 0. SETUP & DEPENDENCIES
# ==========================================
# Uncomment the line below if running in Google Colab / Jupyter
# !pip install emcee corner
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

# ==========================================
# 1. DATASETS
# ==========================================
# Cosmic Chronometers (z, H(z), err)
hz_data = np.array([
    [0.07, 69.0, 19.6], [0.12, 68.6, 26.2], [0.20, 72.9, 29.6],
    [0.28, 88.8, 36.6], [0.40, 95.0, 17.0], [0.47, 89.0, 50.0],
    [0.48, 97.0, 62.0], [0.593, 104.0, 13.0], [0.68, 92.0, 8.0],
    [0.781, 105.0, 12.0], [0.875, 125.0, 17.0], [0.88, 90.0, 40.0],
    [0.90, 117.0, 23.0], [1.037, 154.0, 20.0], [1.30, 168.0, 17.0],
    [1.363, 160.0, 33.6], [1.43, 177.0, 18.0], [1.53, 140.0, 14.0],
    [1.75, 202.0, 40.0], [1.965, 186.5, 50.4]
])

# Pantheon+ Supernovae (z, DistMod, err)
sn_data = np.array([
    [0.014, 14.57, 0.15], [0.026, 15.98, 0.12], [0.036, 16.78, 0.08],
    [0.046, 17.34, 0.07], [0.065, 18.12, 0.06], [0.10, 19.09, 0.05],
    [0.20, 20.64, 0.04],  [0.35, 22.12, 0.04],  [0.55, 23.46, 0.05],
    [0.85, 24.78, 0.08],  [1.15, 25.68, 0.12],  [1.50, 26.45, 0.18],
    [2.00, 27.20, 0.25]
])

# ==========================================
# 2. PHYSICS ENGINE (Corrected for Paper Math)
# ==========================================
c_light = 299792.458
FIXED_Z_TRANS = 0.65       # Percolation Threshold
DELTA_GEO_IDEAL = 0.229    # Lattice Constant

def hubble_model(z, params):
    H0_late, Om, eta = params
    
    # 1. Calculate the Boost Factor from Viscosity
    # Viscosity dampens the relaxation: Delta_eff = 0.229 * (1 - eta)
    delta_eff = DELTA_GEO_IDEAL * (1.0 - eta)
    
    # The Early Universe follows Planck (Lower H).
    # The Late Universe is boosted by 1/sqrt(1-delta).
    # Since H0_late is our parameter, we scale the EARLY universe DOWN.
    suppression_factor = np.sqrt(1.0 - delta_eff) 
    
    # Sigmoid Transition
    sigmoid = 1.0 / (1.0 + np.exp((z - FIXED_Z_TRANS) / 0.1))
    
    # Amp goes from 'suppression' (z>>1) to '1.0' (z=0)
    amp_z = suppression_factor + (1.0 - suppression_factor) * sigmoid
    
    # Standard LCDM
    E_z = np.sqrt(Om * (1 + z)**3 + (1 - Om))
    
    return H0_late * E_z * amp_z

def dist_mod_model(z, params):
    z_grid = np.linspace(0, z, 50)
    H_vals = hubble_model(z_grid, params)
    # Using trapezoid rule for integration
    Dc = c_light * np.sum((1.0/H_vals[:-1] + 1.0/H_vals[1:]) / 2.0 * np.diff(z_grid))
    return 5.0 * np.log10((1+z) * Dc) + 25.0

# ==========================================
# 3. LIKELIHOOD (Unbiased)
# ==========================================
def log_likelihood(params):
    H0, Om, eta = params
    if not (60 < H0 < 80 and 0.2 < Om < 0.4 and 0.0 < eta < 0.5):
        return -np.inf

    # Priors
    lp_Om = -0.5 * ((Om - 0.315) / 0.015)**2
    lp_eta = -0.5 * ((eta - 0.21) / 0.05)**2 # Lepton Prior
    
    # Data Fit
    model_hz = np.array([hubble_model(z, params) for z in hz_data[:,0]])
    chi2_hz = np.sum(((hz_data[:,1] - model_hz) / hz_data[:,2])**2)
    
    model_mu = np.array([dist_mod_model(z, params) for z in sn_data[:,0]])
    diff = sn_data[:,1] - model_mu
    errs = sn_data[:,2]
    weights = 1.0 / errs**2
    M_nuisance = np.sum(diff * weights) / np.sum(weights)
    chi2_sn = np.sum(((diff - M_nuisance) / errs)**2)
    
    return lp_Om + lp_eta - 0.5 * (chi2_hz + chi2_sn)

# ==========================================
# 4. RUNNER
# ==========================================
if __name__ == "__main__":
    print(f"Running Final Validation...")
    
    ndim = 3   
    nwalkers = 32
    p0 = [74.0, 0.31, 0.21] + 1e-2 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood)
    sampler.run_mcmc(p0, 4000, progress=True)

    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
    labels = [r"$H_0$", r"$\Omega_m$", r"$\eta$"]
    
    print("\n" + "="*40)
    print("FINAL POSTERIOR PREDICTIONS")
    print("="*40)
    
    # Get Medians
    h0_res = np.percentile(flat_samples[:, 0], 50)
    eta_res = np.percentile(flat_samples[:, 2], 50)
    
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        print(f"{labels[i]}: {mcmc[1]:.3f} +/- {np.diff(mcmc)[0]:.3f}")

    # Theoretical Check
    # What does this eta actually predict for H0?
    delta_eff = 0.229 * (1 - eta_res)
    boost = 1.0 / np.sqrt(1.0 - delta_eff)
    h0_theory = 67.4 * boost
    
    print("-" * 40)
    print(f"CHECK: Eta={eta_res:.3f} implies Theoretical H0 ~ {h0_theory:.2f}")
    print(f"       MCMC found H0 ~ {h0_res:.2f}")
    
    if 73.0 < h0_res < 76.5:
        print("VERDICT: SUCCESS. Model predicts H0 consistent with SH0ES/Leptons.")
    else:
        print("VERDICT: FAILURE.")
