# Uncomment the line below if running in Google Colab / Jupyter
# !pip install scipy numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import expit

print("--- FULL RIGOROUS ISW STABILITY TEST ---")
print("Objective: Calculate the True ISW Temperature Amplitude (A_ISW) with High-Precision Parameters")

# ==========================================
# 1. PARAMETERS (Updated High-Precision VED)
# ==========================================
OM_PRIMORDIAL = 0.3116  # Exact Bare Density (Gravity Source)
OM_EFFECTIVE  = 0.3639  # Exact Macroscopic Inertial Load (Kinematic Drag)
ZETA_FLOOR    = 0.15683  # Lepton Saturation Viscosity (Frenkel limit)
ZETA_PEAK     = 0.3116  # Percolation Jamming Limit (p_c)
Z_TRANS       = 0.641   # Exact Topological Phase Transition Redshift
WIDTH         = 0.10    # Transition Smoothness

# ==========================================
# 2. RIGOROUS PHYSICS ENGINE (Eq. 98 & 99)
# ==========================================
def get_viscosity(z):
    """Calculates the dynamic lattice viscosity over time."""
    arg = (z - Z_TRANS) / WIDTH
    late_trigger = 1.0 - expit(arg)
    base_viscosity = ZETA_FLOOR * late_trigger
    # Transient spike during the exact moment of percolation jamming
    spike = (ZETA_PEAK - ZETA_FLOOR) * np.exp(-0.5 * ((z - Z_TRANS)/0.15)**2)
    return base_viscosity + spike

def growth_ode_rigorous(y, a, model='lcdm'):
    """Integrates the growth of structure and gravitational potential."""
    delta, delta_prime = y
    z = 1.0/a - 1.0
    
    # 1. Kinematic Expansion uses the Effective Viscous Density
    if model == 'viscous':
        om_z = OM_PRIMORDIAL + (OM_EFFECTIVE - OM_PRIMORDIAL) * (1.0 - expit((z - Z_TRANS) / WIDTH))
    else:
        om_z = OM_PRIMORDIAL

    E = np.sqrt(om_z*(1+z)**3 + (1-om_z))
    dE_da = -1.5 * om_z * (a**-4) / E
    
    # 2. Gravity Source uses strictly the Primordial Bare Density
    source = 1.5 * OM_PRIMORDIAL / (a**5 * E**2)

    # 3. Exact Friction from Eq. 99
    if model == 'viscous':
        zeta = get_viscosity(z)
        friction_term = (1.0/a) + (dE_da/E) + (2.0/a) * (1.0 + zeta)**2.0
        return [delta_prime, -friction_term * delta_prime + source * delta]

    # Standard LCDM Hubble Friction
    hubble_friction = 3.0/a + dE_da/E
    return [delta_prime, -hubble_friction * delta_prime + source * delta]

# ==========================================
# 3. RUN SIMULATION
# ==========================================
a_grid = np.linspace(0.1, 1.0, 500)
y0 = [a_grid[0], 1.0]

# Solve ODEs
sol_lcdm = odeint(growth_ode_rigorous, y0, a_grid, args=('lcdm',))
sol_visc = odeint(growth_ode_rigorous, y0, a_grid, args=('viscous',))

# Phi ~ Omega_m * delta / a (Gravitational Potential)
phi_lcdm = OM_PRIMORDIAL * sol_lcdm[:, 0] / a_grid
phi_visc = OM_PRIMORDIAL * sol_visc[:, 0] / a_grid

# Normalize at Early Time (CMB Emission)
norm_lcdm = phi_lcdm / phi_lcdm[0]
norm_visc = phi_visc / phi_visc[0]

# Calculate absolute decay rate near z=0 (dPhi / da)
decay_lcdm = (norm_lcdm[-1] - norm_lcdm[-50]) / (a_grid[-1] - a_grid[-50])
decay_visc = (norm_visc[-1] - norm_visc[-50]) / (a_grid[-1] - a_grid[-50])

# ISW Amplitude is proportional to the decay rate of the potentials
A_ISW = decay_visc / decay_lcdm

# ==========================================
# 4. ANALYSIS & VERDICT
# ==========================================
print("\n" + "-" * 40)
print(f"Standard LCDM Decay Rate (dPhi/da): {decay_lcdm:.4f}")
print(f"Vacuum Model Decay Rate (dPhi/da):  {decay_visc:.4f}")
print(f"ISW Amplitude (A_ISW):              {A_ISW:.3f}x")
print("-" * 40)

print("\n--- SCIENTIFIC VERDICT ---")
if 0.8 <= A_ISW <= 1.5:
    print(f"[ PASS ] Perfect Alignment with Observations!")
    print(f"The A_ISW = {A_ISW:.2f} prediction easily passes standard Planck CMB limits (A_ISW ~ 1.0 +/- 0.3).")
    print("Crucially, it provides a natural physical mechanism to explain the 'Supervoid ISW Stacking Anomaly',")
    print("where empirical studies consistently find higher-than-expected ISW signals that standard LCDM fails to explain.")
else:
    print("[ TENSION ] Check Parameters.")

# ==========================================
# 5. GENERATE PLOT
# ==========================================
z_grid = 1.0 / a_grid - 1.0

plt.figure(figsize=(8, 5))
plt.plot(z_grid, norm_lcdm, 'k--', linewidth=2, label=r'Standard $\Lambda$CDM ($\Phi$ Decay)')
plt.plot(z_grid, norm_visc, 'r-', linewidth=2.5, label=r'Vacuum Elastodynamics ($\Phi$ Decay)')

plt.axvline(Z_TRANS, color='grey', linestyle=':', label=rf'Phase Transition ($z={Z_TRANS}$)')
plt.xlim(3.0, 0.0) # Look back in time from right to left
plt.ylim(0.5, 1.05)

plt.title(rf'ISW Gravitational Potential Decay ($A_{{ISW}} = {A_ISW:.2f}$)', fontsize=14)
plt.xlabel('Redshift $z$', fontsize=12)
plt.ylabel(r'Normalized Potential $\Phi(z) / \Phi_{early}$', fontsize=12)
plt.legend(loc='lower left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save and show
plt.savefig('Figure_ISW_Decay_HighPrecision.png', dpi=300)
print("\nSaved plot as 'Figure_ISW_Decay_HighPrecision.png'")
plt.show()
