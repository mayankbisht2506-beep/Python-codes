# Uncomment the line below if running in Google Colab / Jupyter
# !pip install -q scipy numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, trapezoid
from scipy.interpolate import interp1d

print("--- JWST 'IMPOSSIBLE GALAXIES' TEST: LUMINOSITY RESOLUTION ---")
print("Objective: Quantify the mass correction due to VED Thermodynamic Brightening.")
print("-" * 65)

# ==========================================
# 1. COSMOLOGICAL PARAMETERS (VED Bare Baseline)
# ==========================================
h = 0.7267       # EXACT: Updated VED Terminal Velocity (H_local)
Om0 = 0.3116     # VED Bare Primordial Density
Ol0 = 1.0 - Om0  # Exact Flat Spacetime Baseline
Ob0 = 0.04959    # VED Geometric Baryon Density (p_c * gamma_crit)
rho_crit_0 = 2.775e11 * h**2 # M_sun / Mpc^3
rho_m_0 = Om0 * rho_crit_0
fb = Ob0 / Om0   # Cosmic Baryon Fraction (~0.159)

# VED PARAMETERS (For the Correction)
G_RATIO = 1.2165 # EXACT: Updated G_early / G_0 topological derivation

# ==========================================
# 2. PHYSICS ENGINE: EXACT GROWTH INTEGRATION
# ==========================================
def growth_ode(y, a):
    D, D_prime = y
    E = np.sqrt(Om0 * a**-3 + Ol0)
    dE_da = -1.5 * Om0 * a**-4 / E
    friction = 3.0 / a + dE_da / E
    source = 1.5 * Om0 / (a**5 * E**2)
    return [D_prime, -friction * D_prime + source * D]

# Full integration from z=1000 (a=0.001) all the way to z=0 (a=1.0)
a_grid = np.linspace(0.001, 1.0, 2000)
y0 = [a_grid[0], 1.0]
sol = odeint(growth_ode, y0, a_grid)
D_raw = sol[:, 0]

# Exact normalization to present day (D(z=0) = 1.0)
D_norm = D_raw / D_raw[-1]
func_D = interp1d(1.0 / a_grid - 1.0, D_norm)

# ==========================================
# 3. HALO MASS FUNCTION (Sheth-Tormen)
# ==========================================
def get_abundance(M_halo, z):
    sigma8 = 0.812 # EXACT: Updated standard unsuppressed baseline

    # High-z effective spectral index scaling
    sigma_M = sigma8 * (M_halo / (6e14 / h))**(-0.13) * func_D(z)

    # Sheth-Tormen multiplicity
    A, p, q = 0.322, 0.3, 0.707
    delta_c = 1.686
    nu = delta_c / sigma_M
    f_nu = A * np.sqrt(2 * q / np.pi) * (1 + (q * nu**2)**-p) * nu * np.exp(-q * nu**2 / 2)

    # Differential density dn/dlnM
    dn_dlnM = (rho_m_0 / M_halo) * f_nu * 0.13
    return dn_dlnM

# ==========================================
# 4. DATA & CORRECTION LOGIC
# ==========================================
# Labbé et al. (2023) Data Points
jwst_data = [
    (1e10, 2e-4),   # Lower mass bin
    (1e11, 4e-6)    # The "Impossible" massive candidates
]

# Calculate Luminosity Boost Factor (Thomson Scattering for Pop III)
ALPHA_THOMSON = 4.0
boost_factor = G_RATIO**ALPHA_THOMSON   

print(f"VED High-G Factor: {G_RATIO:.4f}x")
print(f"Luminosity Boost (Thomson, alpha={ALPHA_THOMSON}): {boost_factor:.2f}x")
print("-" * 65)

# ==========================================
# 5. GENERATE PLOT DATA
# ==========================================
z_target = 10
mass_grid = np.logspace(9, 12, 100)

# Calculate Theoretical Limit (Cumulative Number Density)
n_cumulative = []
for M in mass_grid:
    M_halo_req = M / fb  # Maximum possible stellar mass assuming epsilon = 100%
    M_integ = np.logspace(np.log10(M_halo_req), 14, 60)
    dn = [get_abundance(m, z_target) for m in M_integ]
    n_cum = trapezoid(dn, np.log(M_integ)) 
    n_cumulative.append(n_cum)

n_cumulative = np.array(n_cumulative)

# ==========================================
# 6. VISUALIZATION
# ==========================================
plt.figure(figsize=(10, 7))

# 1. The "Impossible" Barrier (LCDM limit)
plt.plot(mass_grid, n_cumulative, 'k--', linewidth=2, label=r'$\Lambda$CDM Limit ($\epsilon=100\%$ Baryon Conversion)')
plt.fill_between(mass_grid, n_cumulative, 1, color='gray', alpha=0.1)
plt.text(1.2e11, 1e-8, r'Forbidden Region ($\epsilon > 100\%$)', fontsize=11, color='gray', style='italic')

# 2. Original JWST Data (In Tension)
masses_obs = [p[0] for p in jwst_data]
densities = [p[1] for p in jwst_data]
plt.errorbar(masses_obs, densities, yerr=[[1e-4, 2e-6], [1e-4, 2e-6]], fmt='ko', markersize=8, capsize=5, label='JWST Inferred (Labbé et al. 2023)')

# 3. VED Corrected Data (Strict Thomson Limit)
masses_corr = [m / boost_factor for m in masses_obs]
plt.plot(masses_corr, densities, 'ro', markersize=8, label=r'VED Physical Stellar Mass ($M_{\text{true}} = M_{\text{obs}} / 2.19$)')

# Draw Transition Vectors
for i in range(len(masses_obs)):
    plt.annotate('', xy=(masses_corr[i], densities[i]), xytext=(masses_obs[i], densities[i]),
                 arrowprops=dict(arrowstyle="->", color='red', lw=2))

plt.xscale('log')
plt.yscale('log')
plt.xlim(1e9, 5e11)
plt.ylim(1e-7, 1e-2)
plt.xlabel(r'Stellar Mass ($M_\odot$)', fontsize=13)
plt.ylabel(r'Cumulative Number Density ($\text{Mpc}^{-3}$)', fontsize=13)
plt.title(f'Resolution of JWST Anomaly via Thermodynamic Brightening ($z={z_target}$)', fontsize=14)
plt.legend(fontsize=11, loc='lower left')
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig('Figure_JWST_Thermodynamic_Resolution.pdf', dpi=300)
print("Saved plot as 'Figure_JWST_Thermodynamic_Resolution.pdf'")
plt.show()
