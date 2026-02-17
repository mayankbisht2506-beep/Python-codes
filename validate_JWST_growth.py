import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d

print("--- JWST 'IMPOSSIBLE GALAXIES' TEST: LUMINOSITY RESOLUTION ---")
print("Objective: Quantify the mass correction due to VED Thermodynamic Brightening.")

# ==========================================
# 1. COSMOLOGICAL PARAMETERS (Standard LambdaCDM Baseline)
# ==========================================
# We use Planck 2018 parameters to define the "Limit" we are testing against.
h = 0.674
Om0 = 0.315
Ol0 = 1.0 - Om0
Ob0 = 0.049 # Baryon Density
rho_crit_0 = 2.775e11 * h**2 # M_sun / Mpc^3
rho_m_0 = Om0 * rho_crit_0
fb = Ob0 / Om0 # Cosmic Baryon Fraction (~0.156)

# VED PARAMETERS (For the Correction)
G_RATIO = 1.22  # G_early / G_0 derived from your Hubble solution

# Stellar Physics Scaling: L ~ G^alpha
# Conservative range for Main Sequence stars (opacity/pressure limited)
ALPHA_LOW = 4.0
ALPHA_HIGH = 7.0

# ==========================================
# 2. PHYSICS ENGINE: STANDARD GROWTH (Baseline)
# ==========================================
# We calculate the standard LCDM halo abundance to define the "Impossibility Line"
def growth_ode(y, a):
    D, D_prime = y
    E = np.sqrt(Om0 * a**-3 + Ol0)
    dE_da = -1.5 * Om0 * a**-4 / E
    friction = 3.0/a + dE_da/E
    source = 1.5 * Om0 / (a**5 * E**2)
    return [D_prime, -friction * D_prime + source * D]

# Solve Growth from z=1000 to z=10
a_grid = np.linspace(0.001, 0.1, 500) # z=1000 to z=9
y0 = [a_grid[0], 1.0]
sol = odeint(growth_ode, y0, a_grid)
D_raw = sol[:, 0]
# Normalize to z=0 (approximate for high-z comparison)
D_z0_approx = D_raw[-1] * (1.0/0.1) # simplistic growth extrapolation for normalization
D_norm = D_raw / D_z0_approx
func_D = interp1d(1/a_grid - 1, D_norm)

# ==========================================
# 3. HALO MASS FUNCTION (Sheth-Tormen)
# ==========================================
def get_abundance(M_halo, z):
    # Standard LCDM Sigma8 and Spectal Index
    sigma8 = 0.811
    n_s = 0.965

    # Sigma(M) approximation
    R = (3 * M_halo / (4 * np.pi * rho_m_0))**(1.0/3.0)
    # Approximate sigma(R) slope for high-z galaxies
    sigma_M = sigma8 * (M_halo / (6e14/h))**(-0.1) * func_D(z)

    # Sheth-Tormen multiplicity
    A, p, q = 0.322, 0.3, 0.707
    delta_c = 1.686
    nu = delta_c / sigma_M
    f_nu = A * np.sqrt(2*q/np.pi) * (1 + (q*nu**2)**-p) * nu * np.exp(-q*nu**2 / 2)

    # Differential density dn/dlnM
    dn_dlnM = (rho_m_0 / M_halo) * f_nu * abs(-0.1)
    return dn_dlnM

# ==========================================
# 4. DATA & CORRECTION LOGIC
# ==========================================
# Labbé et al. (2023) Data Points (Approximate from Figure 1 of their paper)
# (Observed Stellar Mass, Cumulative Density)
jwst_data = [
    (1e10, 2e-4),   # Lower mass bin
    (1e11, 4e-6)    # The "Impossible" massive candidates
]

# Calculate Luminosity Boost Factor
# L_ved = L_std * (G_ratio)^alpha
# Mass_true = Mass_obs / (G_ratio)^alpha
boost_factor_low = G_RATIO**ALPHA_LOW   # Conservative (alpha=4)
boost_factor_high = G_RATIO**ALPHA_HIGH # Aggressive (alpha=7)

print(f"\nVED High-G Factor: {G_RATIO:.2f}x")
print(f"Luminosity Boost (alpha={ALPHA_LOW}): {boost_factor_low:.2f}x")
print(f"Luminosity Boost (alpha={ALPHA_HIGH}): {boost_factor_high:.2f}x")
print("-" * 40)

# ==========================================
# 5. GENERATE PLOT DATA
# ==========================================
z_target = 10
mass_grid = np.logspace(9, 12, 100)

# Calculate Theoretical Limit (Cumulative Number Density)
# Limit = Baryon Fraction * Halo Mass Function
# i.e., Assuming 100% efficiency of converting Baryons to Stars (Absolute Max)
n_cumulative = []
for M in mass_grid:
    # Get n(>M_halo) where M_halo = M_star / fb
    M_halo_req = M / fb
    # Integrate density above this mass
    M_integ = np.logspace(np.log10(M_halo_req), 14, 50)
    dn = [get_abundance(m, z_target) for m in M_integ]
    # FIX: Using np.trapezoid to avoid deprecation warning
    n_cum = np.trapezoid(dn, np.log(M_integ))
    n_cumulative.append(n_cum)

n_cumulative = np.array(n_cumulative)

# ==========================================
# 6. VISUALIZATION
# ==========================================
plt.figure(figsize=(10, 7))

# 1. The "Impossible" Barrier (LCDM limit)
plt.plot(mass_grid, n_cumulative, 'k--', linewidth=2, label=r'$\Lambda$CDM Limit ($\epsilon=100\%$ Baryon Conv.)')
plt.fill_between(mass_grid, n_cumulative, 1, color='gray', alpha=0.1)
plt.text(1.5e11, 1e-8, 'Forbidden Region', fontsize=12, color='gray')

# 2. Original JWST Data (In Tension)
masses_obs = [p[0] for p in jwst_data]
densities = [p[1] for p in jwst_data]
plt.errorbar(masses_obs, densities, yerr=[[1e-4, 2e-6], [1e-4, 2e-6]], fmt='ko', markersize=8, capsize=5, label='JWST Observed (Labbé et al. 2023)')

# 3. VED Corrected Data
# Shift masses to the LEFT (True Mass is lower)
masses_corr_low = [m / boost_factor_low for m in masses_obs]
masses_corr_high = [m / boost_factor_high for m in masses_obs]

plt.plot(masses_corr_low, densities, 'bo', markersize=8, alpha=0.6, label=f'VED Corrected ($\\alpha={ALPHA_LOW}$)')
plt.plot(masses_corr_high, densities, 'go', markersize=8, alpha=0.6, label=f'VED Corrected ($\\alpha={ALPHA_HIGH}$)')

# Draw arrows connecting Original -> Corrected
for i in range(len(masses_obs)):
    plt.arrow(masses_obs[i], densities[i], masses_corr_low[i] - masses_obs[i], 0,
              color='b', alpha=0.3, length_includes_head=True, head_width=densities[i]*0.1)
    plt.arrow(masses_obs[i], densities[i], masses_corr_high[i] - masses_obs[i], 0,
              color='g', alpha=0.3, length_includes_head=True, head_width=densities[i]*0.1)

plt.xscale('log')
plt.yscale('log')
plt.xlim(1e9, 5e11)
plt.ylim(1e-7, 1e-2)
plt.xlabel(r'Stellar Mass ($M_\odot$)', fontsize=14)
plt.ylabel(r'Cumulative Number Density ($Mpc^{-3}$)', fontsize=14)
plt.title(f'Resolution of JWST Anomaly via Thermodynamic Brightening (z={z_target})', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig('JWST_Resolution_Luminosity.png')
plt.show()
