import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# ==========================================
# 1. SETUP PARAMETERS (Unified Add 33)
# ==========================================
h = 0.674  # Planck H0/100
Om0 = 0.315
rho_crit_0 = 2.775e11 * h**2 # M_sun / Mpc^3
rho_m_0 = Om0 * rho_crit_0

# PHYSICS PARAMETERS
# 1. Gravity Boost (H0=74.5 vs 67.4)
# G_early / G_0 = (74.5/67.4)^2 = 1.22
G_BOOST = 1.22       
# 2. Lepton Viscosity (Fixed)
ETA_MICRO = 0.21     
Z_TRANS = 0.65       
WIDTH = 0.15

# ==========================================
# 2. NUMERICALLY SAFE FUNCTIONS
# ==========================================
def get_visc_step(z):
    """
    Sigmoid transition: Active at Late Times (Low z), Inactive at Early Times.
    """
    arg = (z - Z_TRANS) / WIDTH
    # Clip for safety: 1/(1+exp(60)) is 0.
    arg = np.clip(arg, -100, 100)
    return 1.0 / (1.0 + np.exp(arg))

# ==========================================
# 3. GROWTH FUNCTION SOLVER
# ==========================================
def growth_equation(y, a, model='lcdm'):
    delta, delta_prime = y
    z = 1.0/a - 1.0

    # Hubble Function E(z)
    E = np.sqrt(Om0 * (1+z)**3 + (1-Om0))

    # Standard Friction
    dE_da = -1.5 * Om0 * (a**-4) / E
    friction = 3.0/a + dE_da/E

    # Viscosity (Inactive at z=10, but included for consistency)
    if model == 'viscous':
        step = get_visc_step(z)
        # Scaled macroscopic drag (approx 7.4 * 0.21)
        # We use the explicit Lepton value here
        friction += (7.4 * ETA_MICRO * step) / a

    # Source Term (Gravity)
    source = 1.5 * Om0 / (a**5 * E**2)

    # "Turbocharger" Modification (Early Universe G Boost)
    # The Gravity Boost is active in the Early Universe (z > Z_TRANS)
    if model == 'viscous' and z > Z_TRANS:
        source *= G_BOOST 

    return [delta_prime, -friction * delta_prime + source * delta]

# Solve for Growth Factor D(z)
# a=0.001 corresponds to z=999
a_grid = np.linspace(0.001, 1.0, 1000)

sol_lcdm = odeint(growth_equation, [1e-3, 1.0], a_grid, args=('lcdm',))
sol_visc = odeint(growth_equation, [1e-3, 1.0], a_grid, args=('viscous',))

# Normalize D(z) so D(z=0) = 1 for LCDM
D_lcdm = sol_lcdm[:, 0] / sol_lcdm[-1, 0]
D_visc = sol_visc[:, 0] / sol_lcdm[-1, 0] # Normalize to same baseline

func_D_lcdm = interp1d(1/a_grid - 1, D_lcdm)
func_D_visc = interp1d(1/a_grid - 1, D_visc)

# ==========================================
# 4. HALO MASS FUNCTION (Sheth-Tormen)
# ==========================================
def get_sigma(M, z, D_func):
    """Approximate RMS density fluctuation sigma(M, z)."""
    M8 = 6e14 / h
    # Simplified slope for high-z galaxies sigma(M) ~ M^(-0.1)
    sigma_0 = 0.811 * (M / M8)**(-0.1)
    return sigma_0 * D_func(z)

def sheth_tormen_nm(M, z, D_func):
    """Returns dn/dlnM [Mpc^-3]"""
    sigma = get_sigma(M, z, D_func)
    A = 0.322; p = 0.3; q = 0.707; delta_c = 1.686
    nu = delta_c / sigma
    f_nu = A * np.sqrt(2*q/np.pi) * (1 + (q*nu**2)**-p) * nu * np.exp(-q*nu**2 / 2)
    return (rho_m_0 / M) * f_nu * abs(-0.1)

# ==========================================
# 5. RUN TEST AT z = 10 (JWST ERA)
# ==========================================
z_target = 10.0
masses = np.logspace(9, 12, 50) 

n_lcdm = []
n_visc = []

for M in masses:
    n_lcdm.append(sheth_tormen_nm(M, z_target, func_D_lcdm))
    n_visc.append(sheth_tormen_nm(M, z_target, func_D_visc))

# Cumulative Density n(>M)
cum_lcdm = np.cumsum(n_lcdm[::-1])[::-1]
cum_visc = np.cumsum(n_visc[::-1])[::-1]

# Stats
def get_enhancement(target_mass):
    idx = (np.abs(masses - target_mass)).argmin()
    return cum_visc[idx] / cum_lcdm[idx]

print(f"--- JWST PREDICTION (z={z_target}) ---")
print(f"Gravity Boost: {G_BOOST}x")
print(f"Enhancement at 10^10 M_sun: {get_enhancement(1e10):.1f}x")
print(f"Enhancement at 10^11 M_sun: {get_enhancement(1e11):.1f}x")

# ==========================================
# 6. PLOTTING
# ==========================================
plt.figure(figsize=(10, 6))

plt.loglog(masses, cum_lcdm, 'k--', linewidth=2, label='Standard LCDM')
plt.loglog(masses, cum_visc, 'r-', linewidth=3, label=f'Vacuum Model (G={G_BOOST})')

# JWST Data Point (Labbé et al. 2023 approx)

plt.errorbar([10**11.5], [1e-4], yerr=[[0.5e-4], [2e-4]], fmt='o', color='blue', label='JWST Observation', capsize=5)

plt.title(f'Resolution of JWST "Impossible" Galaxies at z={z_target}', fontsize=14)
plt.xlabel(r'Halo Mass ($M_\odot$)' )
plt.ylabel(r'Cumulative Number Density $n(>M)$ [$Mpc^{-3}$]')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(1e10, 1e12)
plt.ylim(1e-9, 1e-2)

plt.savefig('Figure_JWST_Prediction_Corrected.png')
plt.show()
