import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d

print("--- JWST 'IMPOSSIBLE GALAXIES' TEST (Corrected) ---")

# ==========================================
# 1. COSMOLOGICAL PARAMETERS
# ==========================================
# Standard Planck 2018
h = 0.674  
Om0 = 0.315
Ol0 = 1.0 - Om0
rho_crit_0 = 2.775e11 * h**2 # M_sun / Mpc^3
rho_m_0 = Om0 * rho_crit_0

# VACUUM ELASTODYNAMICS PARAMETERS (Add 46)
# 1. Gravity Boost: Active in Early Universe (z > 0.65)
#    G_early = G_0 * (H0_vac / H0_std)^2
G_BOOST = (74.5 / 67.4)**2  # approx 1.22

# 2. Viscosity: Active in Late Universe (z < 0.65)
#    Using the physical 'Proton Load' floor
ETA_PHYSICAL = 0.157 

# 3. Phase Transition Geometry
Z_TRANS = 0.65
WIDTH = 0.15

# ==========================================
# 2. PHYSICS ENGINE (GROWTH SOLVER)
# ==========================================

def get_visc_activation(z):
    """
    Sigmoidal activation of viscosity.
    Active (1.0) at low z, Inactive (0.0) at high z.
    """
    arg = (z - Z_TRANS) / WIDTH
    # Numerical safety clip
    arg = np.clip(arg, -100, 100)
    return 1.0 / (1.0 + np.exp(arg))

def growth_ode(y, a, model='lcdm'):
    """
    Solves for the linear growth factor D(a).
    y = [D, D'] (Growth and its derivative wrt scale factor a)
    """
    D, D_prime = y
    z = 1.0/a - 1.0
    
    # Hubble Expansion E(a)
    E = np.sqrt(Om0 * a**-3 + Ol0)
    
    # --- FRICTION TERM (Damping) ---
    # Standard Hubble Drag
    dE_da = -1.5 * Om0 * a**-4 / E
    friction = 3.0/a + dE_da/E
    
    # Add Vacuum Viscosity (Only for Vacuum Model at Late Times)
    if model == 'vac':
        activation = get_visc_activation(z)
        # Friction adds directly: eta / a
        friction += (ETA_PHYSICAL * activation) / a

    # --- SOURCE TERM (Gravity) ---
    # Standard Gravity Source
    source = 1.5 * Om0 / (a**5 * E**2)
    
    # Add Gravity Boost (Only for Vacuum Model at Early Times)
    # The "Turbocharger" effect: G is stronger before the transition.
    if model == 'vac':
        # If z > Z_TRANS, we are in the "Stiff/High-G" phase
        # We use (1-activation) to smoothly apply the boost early on
        boost_profile = 1.0 + (G_BOOST - 1.0) * (1.0 - get_visc_activation(z))
        source *= boost_profile

    return [D_prime, -friction * D_prime + source * D]

# ==========================================
# 3. SOLVE GROWTH HISTORY
# ==========================================
# Solve from z=1000 (a=0.001) to z=0 (a=1.0)
a_grid = np.linspace(0.001, 1.0, 1000)
y0 = [a_grid[0], 1.0] # Initial condition: D ~ a in matter dominance

# 1. Solve Standard Model
sol_lcdm = odeint(growth_ode, y0, a_grid, args=('lcdm',))
D_lcdm_raw = sol_lcdm[:, 0]

# 2. Solve Vacuum Model
sol_vac = odeint(growth_ode, y0, a_grid, args=('vac',))
D_vac_raw = sol_vac[:, 0]

# Normalize relative to early times (CMB baseline)
norm_factor = 1.0 / D_lcdm_raw[-1]
D_lcdm = D_lcdm_raw * norm_factor
D_vac  = D_vac_raw  * norm_factor

# Interpolation functions for calculation
func_D_lcdm = interp1d(1/a_grid - 1, D_lcdm)
func_D_vac  = interp1d(1/a_grid - 1, D_vac)

# ==========================================
# 4. HALO MASS FUNCTION (Sheth-Tormen)
# ==========================================
def get_sigma(M, z, D_func):
    """
    RMS density fluctuation sigma(M, z).
    Uses a power-law approximation valid for galaxy scales.
    sigma(M) ~ M^(-alpha)
    """
    M8 = 6e14 / h 
    sigma8 = 0.811
    alpha = 0.1  # Spectral index slope approximation
    
    sigma_M = sigma8 * (M / M8)**(-alpha)
    return sigma_M * D_func(z)

def sheth_tormen_number_density(M, z, D_func):
    """
    Calculates differential number density dn/dlnM.
    """
    sigma = get_sigma(M, z, D_func)
    
    # Sheth-Tormen Parameters
    A = 0.322
    p = 0.3
    q = 0.707
    delta_c = 1.686 
    
    nu = delta_c / sigma
    f_nu = A * np.sqrt(2*q/np.pi) * (1 + (q*nu**2)**-p) * nu * np.exp(-q*nu**2 / 2)
    
    return (rho_m_0 / M) * f_nu * abs(-0.1) # Jacobian for dlnM

# ==========================================
# 5. EXECUTE TEST AT z=10
# ==========================================
z_target = 10.0
mass_range = np.logspace(9, 11.5, 50) 

n_lcdm = []
n_vac = []

for M in mass_range:
    n_lcdm.append(sheth_tormen_number_density(M, z_target, func_D_lcdm))
    n_vac.append(sheth_tormen_number_density(M, z_target, func_D_vac))

# --- FIX: Convert lists to NumPy Arrays before math ---
n_lcdm = np.array(n_lcdm)
n_vac = np.array(n_vac)

# Convert to Cumulative Density n(>M)
dlnM = np.log(mass_range[1]) - np.log(mass_range[0])
cum_lcdm = np.cumsum((n_lcdm * dlnM)[::-1])[::-1]
cum_vac  = np.cumsum((n_vac  * dlnM)[::-1])[::-1]

# Check Enhancement
idx_check = (np.abs(mass_range - 1e10)).argmin()
enhancement = cum_vac[idx_check] / cum_lcdm[idx_check]

print(f"Target Redshift: z = {z_target}")
print(f"Enhancement at M = 10^10 M_sun: {enhancement:.1f}x")

# ==========================================
# 6. PLOT
# ==========================================
plt.figure(figsize=(10, 7))

# Plot Models
plt.loglog(mass_range, cum_lcdm, 'k--', linewidth=2, label='Standard LCDM')
plt.loglog(mass_range, cum_vac, 'r-', linewidth=3, label=f'Vacuum Elastodynamics\n(G_early = {G_BOOST:.2f} G0)')

# Plot JWST Data Approximation (Labbé et al. 2023)
jwst_mass = [1e10, 1e11]
# Rough bounds of the tension
plt.fill_between(jwst_mass, [0.5e-5, 1e-7], [5e-4, 1e-5], color='blue', alpha=0.2, label='JWST Tension Region')

plt.xlabel(r'Halo Mass ($M_\odot$)', fontsize=12)
plt.ylabel(r'Cumulative Number Density $n(>M)$ [$Mpc^{-3}$]', fontsize=12)
plt.title(f'Resolution of JWST "Impossible Galaxies" (z={z_target})', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlim(1e9, 3e11)
plt.ylim(1e-8, 1e-2)

plt.tight_layout()
plt.savefig('JWST_Test_Corrected.png')
print("Plot saved as 'JWST_Test_Corrected.png'")
plt.show()
